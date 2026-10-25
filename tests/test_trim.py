"""Trimming a shrunk population: only the surplus goes, every kept model stays untouched."""

import csv
import json

import pytest

from staticsbfl.utils import paths
from staticsbfl.variation import build_phase
from staticsbfl.cli import trim


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A debug phase-1 tree holding the current population plus 3 surplus fit and 2 surplus eval models."""
    monkeypatch.setattr(paths, "DEBUG_VARIANTS_DIR", tmp_path / "variants-debug")
    spec = build_phase("phase1", "debug")
    keys = {split: [c.key for c in spec.configs() if c.split == split] for split in spec.split_names}
    extra = {"train": [f"p1_train_{i:03d}" for i in range(900, 903)], "test": [f"p1_test_{i:03d}" for i in range(900, 902)]}
    for split in spec.split_names:
        directory = paths.variant_dir("phase1", split, "debug", "cifar10")
        (directory / "spectra").mkdir(parents=True)
        for key in keys[split] + extra[split]:
            (directory / f"{key}.pt").write_bytes(key.encode())
            (directory / "spectra" / f"{key}.pt").write_bytes(b"s" + key.encode())
    root = paths.variants_root("debug", "cifar10") / "phase1"
    all_keys = [k for split in spec.split_names for k in keys[split] + extra[split]]
    with open(root / "models.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "split", "train_seconds"])
        writer.writeheader()
        writer.writerows({"key": k, "split": "x", "train_seconds": 2.0} for k in all_keys)
    with open(root / "activation_rates.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "layer"])
        writer.writeheader()
        writer.writerows({"key": k, "layer": layer} for k in all_keys for layer in ("fc0", "fc1"))
    (root / "design.json").write_text(json.dumps({
        "populations": {"train": {"n_models": 999}, "test": {"n_models": 999}}, "n_models_total": 1998,
        "generation_cost": {"total_training_seconds": 999.0}}))
    (root / "timings.json").write_text(json.dumps({
        "score": {"seconds": 100.0, "units": len(all_keys)}, "neurons": {"seconds": 1.0}, "predict": {"seconds": 9.0}}))
    (root / "neurons.npz").write_bytes(b"old table")
    (root / "predictions").mkdir()
    (root / "predictions" / "phase1__ochiai__ochiai.csv").write_text("old")
    return keys, extra, all_keys, root


def test_the_dry_run_changes_nothing(tree):
    keys, extra, all_keys, root = tree
    plan = trim.plan_phase("phase1", "debug", "cifar10")
    assert plan["n_surplus"] == 5
    assert plan["surplus"] == extra
    assert (root / "neurons.npz").exists() and len(list(root.rglob("*.pt"))) == 2 * len(all_keys)


def test_apply_removes_exactly_the_surplus_and_keeps_every_kept_file_byte_identical(tree):
    keys, extra, all_keys, root = tree
    trim.apply_phase(trim.plan_phase("phase1", "debug", "cifar10"), "debug", "cifar10")
    for split, kept in keys.items():
        directory = paths.variant_dir("phase1", split, "debug", "cifar10")
        assert sorted(p.stem for p in directory.glob("*.pt")) == sorted(kept)
        assert sorted(p.stem for p in (directory / "spectra").glob("*.pt")) == sorted(kept)
        for key in kept:
            assert (directory / f"{key}.pt").read_bytes() == key.encode()
    kept_keys = {k for v in keys.values() for k in v}
    assert {r["key"] for r in csv.DictReader(open(root / "models.csv"))} == kept_keys
    rates = list(csv.DictReader(open(root / "activation_rates.csv")))
    assert {r["key"] for r in rates} == kept_keys and len(rates) == 2 * len(kept_keys)
    design = json.loads((root / "design.json").read_text())
    assert design["populations"]["train"]["n_models"] == len(keys["train"])
    assert design["n_models_total"] == len(kept_keys)
    assert design["generation_cost"]["total_training_seconds"] == pytest.approx(2.0 * len(kept_keys))
    assert design["trimmed"]["removed_models"] == 5
    timings = json.loads((root / "timings.json").read_text())
    assert "neurons" not in timings and "predict" not in timings
    assert timings["score"]["seconds"] == pytest.approx(100.0 * len(kept_keys) / len(all_keys), abs=0.01)
    assert not (root / "neurons.npz").exists() and not (root / "predictions").exists()


def test_a_population_without_surplus_is_left_alone(tree):
    keys, extra, all_keys, root = tree
    trim.apply_phase(trim.plan_phase("phase1", "debug", "cifar10"), "debug", "cifar10")
    (root / "neurons.npz").write_bytes(b"new table")
    plan = trim.plan_phase("phase1", "debug", "cifar10")
    assert plan["n_surplus"] == 0 and plan["derived"] == [] and plan["bytes"] == 0
