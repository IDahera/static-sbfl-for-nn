"""What the whole run cost, as data: training, scoring, table building, fitting.

Read only. For every mode (light and heavy kept apart -- they are separate
populations under separate budgets), every dataset and every phase it reads

    models.csv     the measured training time of each model (``train_seconds``)
    timings.json   the measured time of the score / neurons / predict steps,
                   and the predictor fit time on its own (``fit_seconds``)

Each file is read under its own try/except: a phase that never ran, a file
from an interrupted run or a row without a time (recovered from a checkpoint)
becomes a row with a ``status`` and a ``note``, and the rest goes on. Training
times are given in total and per model (mean, sd, min, max) and per
model-epoch, because light and heavy differ in epochs; a phase whose models
were trained under more than one epoch count is flagged.

Printing is :func:`staticsbfl.log.evaluation.print_costs`; the CLI is
``staticsbfl/cli/costs.py`` (``make costs``).
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from staticsbfl.log.console import display_path
from staticsbfl.utils.paths import RUNTIME_DIR, ensure_dir, variant_models_csv, variant_timings
from staticsbfl.variation import PHASE_NAMES

MODES = ("light", "heavy")
DATASETS = ("cifar10", "fashion_mnist")
STEPS = ("score", "neurons", "predict")
COLUMNS = (
    "mode", "dataset", "phase", "population", "step", "status", "n_models", "n_timed",
    "total_seconds", "mean_seconds", "sd_seconds", "min_seconds", "max_seconds",
    "epochs", "seconds_per_model_epoch", "fit_seconds", "fit_seconds_by_predictor",
    "mean_test_accuracy", "note",
)


def costs_csv() -> Path:
    """Where the cost report is written: ``cache/runtime/costs.csv``."""
    return RUNTIME_DIR / "costs.csv"


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"total_seconds": None, "mean_seconds": None, "sd_seconds": None,
                "min_seconds": None, "max_seconds": None}
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return {"total_seconds": sum(values), "mean_seconds": mean, "sd_seconds": sd,
            "min_seconds": min(values), "max_seconds": max(values)}


def training_rows(mode: str, dataset: str, phase: str) -> List[dict]:
    """One row per population of *phase*, plus one for the whole phase."""
    base = {"mode": mode, "dataset": dataset, "phase": phase, "step": "train"}
    path = variant_models_csv(phase, mode, dataset)
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            models = list(csv.DictReader(handle))
    except FileNotFoundError:
        return [{**base, "population": "*", "status": "missing", "note": f"no {display_path(path)}"}]
    except (OSError, csv.Error, UnicodeDecodeError) as error:
        return [{**base, "population": "*", "status": "unreadable", "note": f"{type(error).__name__}: {error}"}]

    groups: Dict[str, List[dict]] = {}
    for row in models:
        groups.setdefault(row.get("split", "?"), []).append(row)
    rows = []
    for population, members in sorted(groups.items()) + [("*", models)]:
        times, epochs, accuracies, untimed = [], set(), [], 0
        for row in members:
            try:
                times.append(float(row["train_seconds"]))
            except (KeyError, TypeError, ValueError):
                untimed += 1
            try:
                epochs.add(int(row["epochs"]))
            except (KeyError, TypeError, ValueError):
                pass
            try:
                accuracies.append(float(row["test_accuracy"]))
            except (KeyError, TypeError, ValueError):
                pass
        notes = []
        if untimed:
            notes.append(f"{untimed} model(s) without a training time (rebuilt from a checkpoint)")
        if len(epochs) > 1:
            notes.append(f"MIXED BUDGETS: epochs {sorted(epochs)} -- the mean mixes them")
        stats = _stats(times)
        single_epoch = next(iter(epochs)) if len(epochs) == 1 else None
        rows.append({
            **base, "population": population, "status": "ok" if times else "untimed",
            "n_models": len(members), "n_timed": len(times), **stats,
            "epochs": "/".join(str(e) for e in sorted(epochs)),
            "seconds_per_model_epoch": (stats["mean_seconds"] / single_epoch)
            if stats["mean_seconds"] is not None and single_epoch else None,
            "mean_test_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
            "note": "; ".join(notes),
        })
    return rows


def step_rows(mode: str, dataset: str, phase: str) -> List[dict]:
    """The score / neurons / predict timings recorded in timings.json."""
    base = {"mode": mode, "dataset": dataset, "phase": phase, "population": "*"}
    path = variant_timings(phase, mode, dataset)
    try:
        timings = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [{**base, "step": step, "status": "missing", "note": f"no {display_path(path)}"} for step in STEPS]
    except (OSError, ValueError) as error:
        return [{**base, "step": step, "status": "unreadable", "note": f"{type(error).__name__}: {error}"}
                for step in STEPS]
    rows = []
    for step in STEPS:
        entry = timings.get(step)
        if not isinstance(entry, dict):
            rows.append({**base, "step": step, "status": "missing", "note": "step never recorded (not run or interrupted)"})
            continue
        try:
            units = int(entry.get("units") or 0)
            seconds = float(entry["seconds"])
            per_model = seconds / units if entry.get("unit") == "model" and units else None
            note = f"{units} {entry.get('unit', 'unit')}(s), recorded {entry.get('recorded_at', '?')}"
            if step == "predict" and "fit_seconds" not in entry:
                note += "; fit time not recorded separately (run predates fit timing)"
            rows.append({
                **base, "step": step, "status": "ok", "n_models": units if entry.get("unit") == "model" else None,
                "total_seconds": seconds, "mean_seconds": per_model,
                "fit_seconds": entry.get("fit_seconds"),
                "fit_seconds_by_predictor": json.dumps(entry["fit_seconds_by_predictor"])
                if "fit_seconds_by_predictor" in entry else None,
                "note": note,
            })
        except (KeyError, TypeError, ValueError) as error:
            rows.append({**base, "step": step, "status": "unreadable", "note": f"{type(error).__name__}: {error}"})
    return rows


def collect(modes: Iterable[str], datasets: Iterable[str]) -> List[dict]:
    """Every training and step row of the given modes and datasets; a broken phase becomes an error row."""
    rows = []
    for mode in modes:
        for dataset in datasets:
            for phase in PHASE_NAMES:
                try:
                    rows += training_rows(mode, dataset, phase)
                    rows += step_rows(mode, dataset, phase)
                except Exception as error:  # one broken phase must not end the report
                    rows.append({"mode": mode, "dataset": dataset, "phase": phase, "population": "*",
                                 "step": "*", "status": "error", "note": f"{type(error).__name__}: {error}"})
    return rows


def write(rows: List[dict]) -> Path:
    """Write *rows* to :func:`costs_csv`; returns the path."""
    path = costs_csv()
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else (f"{row[k]:.4f}" if isinstance(row.get(k), float) else row[k])
                             for k in COLUMNS})
    return path


def summarise_mode(rows: List[dict], mode: str, dataset: str) -> dict:
    """Per phase one line, then totals, for one (mode, dataset); plain data.

    ``phases`` holds, per phase, the whole-phase training row and the three
    step rows; ``totals`` sums what was recorded (an untimed model or a missing
    step adds nothing and is listed in ``flags``); ``mean_train_seconds`` is
    the total training time over the timed models.
    """
    mine = [r for r in rows if r.get("mode") == mode and r.get("dataset") == dataset]
    summary = {"mode": mode, "dataset": dataset, "recorded": not all(r.get("status") == "missing" for r in mine),
               "phases": [], "flags": []}
    totals = {"train": 0.0, "score": 0.0, "neurons": 0.0, "predict": 0.0, "fit": 0.0}
    n_models = n_timed = 0
    for phase in PHASE_NAMES:
        train = next((r for r in mine if r["phase"] == phase and r["step"] == "train" and r["population"] == "*"), {})
        steps = {r["step"]: r for r in mine if r["phase"] == phase and r["step"] in STEPS}
        entry = {"phase": phase, "train": train, "steps": steps}
        summary["phases"].append(entry)
        if train.get("status") in ("missing", "unreadable", "error"):
            summary["flags"].append(f"{phase}: training {train.get('status')} ({train.get('note', '')})")
            continue
        for key in ("score", "neurons", "predict"):
            totals[key] += steps.get(key, {}).get("total_seconds") or 0.0
        totals["fit"] += steps.get("predict", {}).get("fit_seconds") or 0.0
        totals["train"] += train.get("total_seconds") or 0.0
        n_models += train.get("n_models") or 0
        n_timed += train.get("n_timed") or 0
        if train.get("note"):
            summary["flags"].append(f"{phase}: {train['note']}")
        for key in STEPS:
            if steps.get(key, {}).get("status") != "ok":
                summary["flags"].append(f"{phase}: {key} {steps.get(key, {}).get('status', 'missing')}")
    summary["totals"] = totals
    summary["n_models"] = n_models
    summary["n_timed"] = n_timed
    summary["mean_train_seconds"] = totals["train"] / n_timed if n_timed else None
    summary["grand_total_seconds"] = totals["train"] + totals["score"] + totals["neurons"] + totals["predict"]
    return summary
