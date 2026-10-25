"""The pipeline run report: banners, glossary, status and estimates."""

import logging

import pytest

from staticsbfl.utils import paths
from staticsbfl.log.console import Step, format_duration
from staticsbfl.variation import EPOCHS_BY_MODE, PHASE_NAMES, build_phase
from staticsbfl.log import pipeline as report


@pytest.fixture
def empty_debug_tree(tmp_path, monkeypatch):
    """Empty population trees for every mode, so estimates start from nothing.

    All three are redirected, not just debug: an estimate falls back to the
    other modes' measurements, and must not read a real run's cache.
    """
    for name, folder in (("DEBUG_VARIANTS_DIR", "variants-debug"),
                         ("VARIANTS_DIR", "variants"),
                         ("HEAVY_VARIANTS_DIR", "variants-heavy")):
        monkeypatch.setattr(paths, name, tmp_path / folder)
    return tmp_path


@pytest.mark.parametrize("seconds, text", [
    (4.25, "4.2s"), (42, "42s"), (425, "7m 05s"), (7980, "2h 13m"), (-3, "0.0s"),
])
def test_durations_read_at_the_precision_a_reader_needs(seconds, text):
    assert format_duration(seconds) == text


def test_every_phase_has_a_title_what_it_varies_and_how_to_read_it():
    assert set(report.PHASES) == set(PHASE_NAMES)
    for info in report.PHASES.values():
        assert info.title and info.varies and info.reading


@pytest.mark.parametrize("step", [name for name, _ in report.STEPS])
def test_every_step_says_what_it_does_and_what_it_writes(step):
    spec = build_phase("phase1", "debug")
    assert report.step_description(step, spec, epochs=1)
    assert report.step_outputs(step, "phase1", "debug", "cifar10")
    assert report.STEP_SUMMARY[step]


def test_the_file_to_read_first_is_listed_first():
    first, _note = report.central_outputs("phase2", "light", "cifar10")[0]
    assert first.name == "correlations.log"
    assert "FIRST" in _note


def test_the_glossary_explains_every_column_of_the_predictor_table():
    for column in ("rho/model", "wtau", "top-k", "chance", "rho/layer", "r2",
                   "n_inact", "scope all", "scope active", "layer_mean", "random",
                   "a_f / a_s", "tarantula", "ochiai", "dstar", "rank_model"):
        assert column in report.GLOSSARY, column


def test_an_empty_phase_is_incomplete_and_its_estimate_counts_every_model(empty_debug_tree):
    status = report.phase_status("phase1", "debug", "cifar10")
    assert not status.complete
    assert status.trained == status.scored == 0

    estimate = report.estimate_phase("phase1", "debug", "cifar10", status)
    expected = (status.n_models * EPOCHS_BY_MODE["debug"]
                * report.REFERENCE["train_seconds_per_model_epoch"])
    assert estimate.seconds["train"] == pytest.approx(expected)
    # Nothing has been measured yet, so every figure says it is the reference.
    assert all(report.REFERENCE_NOTE in basis for basis in estimate.basis.values())


def test_a_recorded_timing_replaces_the_reference(empty_debug_tree):
    report.record_timing("phase1", "debug", "cifar10", "score", 12.0, 6, "model")
    estimate = report.estimate_phase("phase1", "debug", "cifar10")
    status = report.phase_status("phase1", "debug", "cifar10")
    assert estimate.seconds["score"] == pytest.approx(status.n_models * 2.0)
    assert "this phase" in estimate.basis["score"]


def test_a_run_that_did_no_work_records_no_rate(empty_debug_tree):
    """A rate of zero would make every later estimate optimistic."""
    report.record_timing("phase1", "debug", "cifar10", "score", 0.4, 0, "model")
    assert not paths.variant_timings("phase1", "debug", "cifar10").exists()


def test_a_failed_step_says_so_and_the_error_propagates(caplog):
    log = logging.getLogger("test-step")
    with caplog.at_level(logging.INFO, logger="test-step"):
        with pytest.raises(RuntimeError):
            with Step(log, 2, 4, "score", "does a thing"):
                raise RuntimeError("boom")
    text = caplog.text
    assert "Step 2/4 · score" in text and "FAILED" in text


# --------------------------------------------------------------------------
# One budget per population
# --------------------------------------------------------------------------


def test_a_model_recorded_under_another_budget_is_flagged():
    from staticsbfl.cli.phase_train import budget_mismatches

    rows = {"a": {"epochs": "3"}, "b": {"epochs": "10"}, "c": {"epochs": ""}}
    assert budget_mismatches(["a", "b", "c", "d"], rows, 10) == ["a"]
    # A row recovered from a checkpoint holds an int, not the CSV's string.
    assert budget_mismatches(["e", "f"], {"e": {"epochs": 10}, "f": {"epochs": 3}}, 10) == ["f"]


def test_a_lost_models_csv_row_is_rebuilt_from_its_checkpoint():
    """An interrupted run used to leave checkpoints nobody recorded."""
    from staticsbfl.persistence import ModelCheckpoint
    from staticsbfl.variation import build_model
    from staticsbfl.cli.phase_train import MODEL_FIELDS, row_from_checkpoint

    config = build_phase("phase1", "debug").configs()[0]
    model = build_model(config, num_classes=10, input_shape=(3, 32, 32))
    checkpoint = ModelCheckpoint(
        key=config.key, state_dict=model.state_dict(), dataset="cifar10",
        input_shape=(3, 32, 32), num_classes=10,
        metadata={"epochs": 3, "optimizer": "Adam", "test_accuracy": 0.5,
                  "test_loss": 1.25, "macro_f1": 0.4},
    )
    row = row_from_checkpoint(config, checkpoint)
    assert set(row) == set(MODEL_FIELDS)
    assert row["epochs"] == 3 and row["test_accuracy"] == "0.500000"
    assert row["n_parameters"] > 0
    # Unknown, so blank -- and therefore excluded from every runtime estimate.
    assert row["train_seconds"] == ""


def test_a_heavy_estimate_uses_light_measurements_until_heavy_has_its_own(empty_debug_tree):
    """A model-epoch costs the same in both modes; the reference is the last resort."""
    report.record_timing("phase2", "light", "cifar10", "score", 30.0, 10, "model")
    estimate = report.estimate_phase("phase2", "heavy", "cifar10")
    assert "light run" in estimate.basis["score"]
    status = report.phase_status("phase2", "heavy", "cifar10")
    assert estimate.seconds["score"] == pytest.approx(status.n_models * 3.0)

    report.record_timing("phase2", "heavy", "cifar10", "score", 10.0, 10, "model")
    assert "light" not in report.estimate_phase("phase2", "heavy", "cifar10").basis["score"]


def test_the_disk_projection_is_dominated_by_the_weights():
    """phase1 is 256-128-64 on CIFAR-10: about 0.83 M parameters per model."""
    from staticsbfl.variation.phases import POPULATION_SIZES
    sizes = POPULATION_SIZES["debug"]
    per_model = report.expected_phase_bytes("phase1", "debug", "cifar10") / (sizes["fit"] + sizes["eval"])
    assert 3.2e6 < per_model < 3.6e6


def test_cache_sizes_count_files_and_skip_what_is_excluded(tmp_path):
    from staticsbfl.cli.cache_size import human, tree_size

    (tmp_path / "sub").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 1000)
    (tmp_path / "sub" / "b.bin").write_bytes(b"x" * 500)
    assert tree_size(tmp_path) == (1500, 2)
    assert tree_size(tmp_path, include=lambda p: p.is_file()) == (1000, 1)
    assert tree_size(tmp_path / "missing") == (0, 0)
    assert human(1500) == "1.5 kB" and human(999) == "999 B"
