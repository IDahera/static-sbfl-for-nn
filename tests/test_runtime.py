"""Runtime plumbing: the cost report, the record layout, the per-run aggregation."""

import csv
import json

import pytest

from staticsbfl.utils import paths
from staticsbfl.evaluation import costs
from staticsbfl.evaluation.runtime import design_of, geometric_mean, per_model_summary
from staticsbfl.cli.runtime_benchmark import run_tag


@pytest.fixture
def light_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "VARIANTS_DIR", tmp_path / "variants")
    return tmp_path / "variants" / "cifar10"


def _models_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "split", "epochs", "train_seconds", "test_accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def test_training_cost_per_population_and_whole_phase(light_root):
    _models_csv(light_root / "phase4" / "models.csv", [
        {"key": "a", "split": "sampled", "epochs": 3, "train_seconds": 6.0, "test_accuracy": 0.5},
        {"key": "b", "split": "sampled", "epochs": 3, "train_seconds": 8.0, "test_accuracy": 0.5},
        {"key": "c", "split": "designed", "epochs": 3, "train_seconds": "", "test_accuracy": 0.5},
    ])
    rows = {r["population"]: r for r in costs.training_rows("light", "cifar10", "phase4")}
    assert rows["sampled"]["total_seconds"] == 14.0
    assert rows["sampled"]["mean_seconds"] == 7.0
    assert rows["sampled"]["seconds_per_model_epoch"] == pytest.approx(7.0 / 3)
    assert rows["*"]["n_models"] == 3 and rows["*"]["n_timed"] == 2
    assert "without a training time" in rows["designed"]["note"]


def test_mixed_epoch_budgets_are_flagged(light_root):
    _models_csv(light_root / "phase1" / "models.csv", [
        {"key": "a", "split": "train", "epochs": 3, "train_seconds": 6.0, "test_accuracy": 0.5},
        {"key": "b", "split": "train", "epochs": 10, "train_seconds": 20.0, "test_accuracy": 0.5},
    ])
    row = next(r for r in costs.training_rows("light", "cifar10", "phase1") if r["population"] == "*")
    assert "MIXED BUDGETS" in row["note"]
    assert row["seconds_per_model_epoch"] is None


def test_missing_and_broken_files_are_reported_not_raised(light_root):
    assert costs.training_rows("light", "cifar10", "phase2")[0]["status"] == "missing"
    (light_root / "phase2").mkdir(parents=True)
    (light_root / "phase2" / "timings.json").write_text("{not json")
    assert {r["status"] for r in costs.step_rows("light", "cifar10", "phase2")} == {"unreadable"}


def test_fit_seconds_are_read_from_the_predict_timing(light_root):
    (light_root / "phase4").mkdir(parents=True)
    (light_root / "phase4" / "timings.json").write_text(json.dumps({
        "predict": {"seconds": 100.0, "units": 10, "unit": "neuron x target",
                    "fit_seconds": 80.0, "fit_seconds_by_predictor": {"mlp": 30.0, "gradient_boosting": 50.0}},
    }))
    predict = next(r for r in costs.step_rows("light", "cifar10", "phase4") if r["step"] == "predict")
    assert predict["fit_seconds"] == 80.0
    assert json.loads(predict["fit_seconds_by_predictor"])["mlp"] == 30.0


def _phase(name, seconds, before, sampled, hwm_after, hwm_increase):
    return {"name": name, "seconds": seconds, "bytes_before": before, "sampled_peak_bytes": sampled,
            "lifetime_max_after_bytes": hwm_after, "lifetime_max_increase_bytes": hwm_increase}


def test_per_model_cost_excludes_session_setup():
    phases = [
        _phase("import", 2.0, 10, 300, 300, 290),
        _phase("data", 0.5, 300, 600, 600, 300),   # transient copy, freed afterwards
        _phase("model_load", 0.1, 400, 420, 600, 0),
        _phase("spectrum", 3.0, 420, 550, 600, 0),  # below the earlier high-water mark
    ]
    summary = per_model_summary(phases)
    assert summary["setup_seconds"] == 2.5
    assert summary["per_model_seconds"] == pytest.approx(3.1)
    # The sampled peak still attributes 150 bytes to the analysis, although the
    # lifetime mark (set during 'data') did not move.
    assert summary["per_model_peak_above_start_bytes"] == 150


def test_records_of_a_smoke_run_never_share_a_directory_with_the_full_run():
    assert run_tag({"samples": "all", "max_samples": None}) == "all"
    assert run_tag({"samples": "all", "max_samples": 500}) == "all_first500"
    assert run_tag({"samples": "3", "max_samples": None}) == "class3"


def test_design_names_and_geometric_mean():
    assert design_of("p4_designed_wide_mlp_s3") == "wide_mlp"
    assert design_of("p4_designed_lecun_300_100_s0") == "lecun_300_100"
    assert geometric_mean([1.0, 100.0]) == pytest.approx(10.0)
    assert geometric_mean([]) is None
