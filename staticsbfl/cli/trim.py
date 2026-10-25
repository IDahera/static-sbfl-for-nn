"""Shrink a cached population to the current sizes, without retraining anything.

    make trim VMODE=heavy                 # dry run: what would go, and how much disk
    make trim VMODE=heavy APPLY=--apply   # do it

Every population is a prefix of a fixed pool layout (``staticsbfl.variation.phases``),
so after the sizes shrink -- heavy 500/100 -> 400/80 on 2026-10-22 -- the models
the new sizes need are exactly the first ones already trained, bit for bit. This
removes the rest and everything derived from the old population size:

    removed   surplus checkpoints and their spectra; their rows in models.csv and
              activation_rates.csv
    rewritten design.json (model counts, generation cost over the kept models,
              and a ``trimmed`` record of what was removed and when)
    removed   neurons.npz, predictions/ (results, correlation log, estimators):
              built on the old population, rebuilt by the next make phase
    adjusted  timings.json: neurons/predict dropped (rerun anyway), the score
              time scaled to the kept models and flagged as such

Nothing is deleted without ``--apply``. Checkpoints and spectra of the kept
models are never touched, so the next ``make heavy`` trains and scores
nothing and only rebuilds the neuron tables and refits the predictors.
"""

import csv
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List

import click

from staticsbfl.log.console import configure_logging, display_path, human_bytes, phase_banner, plain, style
from staticsbfl.utils.paths import (
    variant_activation_rates, variant_design, variant_dir, variant_models_csv, variant_neurons,
    variant_predictions_dir, variant_timings, variants_root,
)
from staticsbfl.variation import PHASE_NAMES, build_phase
from staticsbfl.variation.phases import POPULATION_SIZES

logger = logging.getLogger("trim")


def _size(paths: List[Path]) -> int:
    total = 0
    for path in paths:
        if path.is_dir():
            total += sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        elif path.exists():
            total += path.stat().st_size
    return total


def plan_phase(phase: str, mode: str, dataset: str) -> Dict[str, object]:
    """What trimming *phase* would remove and rewrite; changes nothing."""
    spec = build_phase(phase, mode)
    keep = {c.key for c in spec.configs()}
    files: List[Path] = []
    surplus: Dict[str, List[str]] = {}
    for split in spec.split_names:
        directory = variant_dir(phase, split, mode, dataset)
        if not directory.exists():
            continue
        for checkpoint in sorted(directory.glob("*.pt")):
            if checkpoint.stem not in keep:
                surplus.setdefault(split, []).append(checkpoint.stem)
                files.append(checkpoint)
                spectrum = directory / "spectra" / checkpoint.name
                if spectrum.exists():
                    files.append(spectrum)
    derived = [p for p in (variant_neurons(phase, mode, dataset), variant_predictions_dir(phase, mode, dataset))
               if p.exists()] if surplus else []
    return {"phase": phase, "keep": keep, "surplus": surplus, "files": files, "derived": derived,
            "n_surplus": sum(len(v) for v in surplus.values()),
            "bytes": (_size(files) + _size(derived)) if surplus else 0}


def _filter_csv(path: Path, keep: set) -> int:
    """Drop rows whose ``key`` is not kept; returns how many were dropped."""
    if not path.exists():
        return 0
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    kept = [r for r in rows if r.get("key") in keep]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    return len(rows) - len(kept)


def apply_phase(plan: Dict[str, object], mode: str, dataset: str) -> None:
    """Carry out *plan*: delete, filter, rewrite. Kept models are never touched."""
    phase, keep = plan["phase"], plan["keep"]
    before_models = None
    models_csv = variant_models_csv(phase, mode, dataset)
    if models_csv.exists():
        with open(models_csv, newline="", encoding="utf-8") as handle:
            before_models = sum(1 for _ in csv.DictReader(handle))
    for path in plan["files"]:
        path.unlink()
    for path in plan["derived"]:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    dropped_rows = _filter_csv(models_csv, keep)
    _filter_csv(variant_activation_rates(phase, mode, dataset), keep)

    design_path = variant_design(phase, mode, dataset)
    if design_path.exists():
        design = json.loads(design_path.read_text())
        spec = build_phase(phase, mode)
        counts = {split: sum(1 for c in spec.configs() if c.split == split) for split in spec.split_names}
        for split, info in design.get("populations", {}).items():
            if split in counts:
                info["n_models"] = counts[split]
        design["n_models_total"] = sum(counts.values())
        times = []
        if models_csv.exists():
            with open(models_csv, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    try:
                        times.append(float(row["train_seconds"]))
                    except (KeyError, TypeError, ValueError):
                        pass
        if times:
            design["generation_cost"].update({
                "models_recorded": len(times), "total_training_seconds": round(sum(times), 1),
                "mean_seconds_per_model": round(sum(times) / len(times), 2),
                "min_seconds_per_model": round(min(times), 2), "max_seconds_per_model": round(max(times), 2),
                "note": "recomputed over the kept models after trimming; wall clock is that of the original run",
            })
        design["trimmed"] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "removed_models": plan["n_surplus"],
                             "sizes_now": POPULATION_SIZES[mode], "removed_keys": plan["surplus"]}
        design_path.write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")

    timings_path = variant_timings(phase, mode, dataset)
    if timings_path.exists():
        timings = json.loads(timings_path.read_text())
        for step in ("neurons", "predict"):
            timings.pop(step, None)
        score = timings.get("score")
        if score and before_models:
            ratio = (before_models - dropped_rows) / before_models
            score.update({"seconds": round(score["seconds"] * ratio, 2), "units": round(score["units"] * ratio),
                          "scaled_by_trim": round(ratio, 4)})
        timings_path.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")


@click.command()
@click.option("--mode", type=click.Choice(sorted(POPULATION_SIZES)), required=True)
@click.option("--dataset", type=click.Choice(("cifar10", "fashion_mnist")), default="cifar10", show_default=True)
@click.option("--phase", "phases", multiple=True, type=click.Choice(PHASE_NAMES), help="Default: every phase.")
@click.option("--apply", "apply_", is_flag=True, help="Actually delete; without it, only report.")
def main(mode, dataset, phases, apply_):
    """Remove the models a population no longer uses after its size shrank."""
    configure_logging()
    sizes = POPULATION_SIZES[mode]
    phase_banner(logger, f"StaticSBFL · TRIM · {mode} · {dataset} · to {sizes['fit']} fit / {sizes['eval']} eval",
                 [("Tree", [display_path(variants_root(mode, dataset))]),
                  ("Mode", [style("APPLY: files will be deleted", "yellow") if apply_ else
                            "dry run -- nothing is changed; add --apply (make: APPLY=--apply)"])])
    total_models = total_bytes = 0
    for phase in phases or PHASE_NAMES:
        plan = plan_phase(phase, mode, dataset)
        total_models += plan["n_surplus"]
        total_bytes += plan["bytes"]
        per_split = ", ".join(f"{split} {len(keys)} ({keys[0]} … {keys[-1]})" for split, keys in plan["surplus"].items())
        plain(logger, f"  {phase}: {plan['n_surplus']:>4} surplus model(s){': ' + per_split if per_split else ''}")
        if plan["n_surplus"]:
            plain(logger, style(f"           also stale: {', '.join(display_path(p) for p in plan['derived']) or 'nothing'}"
                                f"  ·  frees {human_bytes(plan['bytes'])}", "dim"))
            if apply_:
                apply_phase(plan, mode, dataset)
                plain(logger, style(f"           trimmed: {plan['n_surplus']} models removed, design.json / "
                                    "models.csv / activation_rates.csv / timings.json updated", "green"))
    plain(logger, "")
    verb = "removed" if apply_ else "would remove"
    plain(logger, f"  {verb} {total_models} model(s), {human_bytes(total_bytes)}."
          + ("" if apply_ or not total_models else "  Rerun with --apply to do it."))
    if apply_ and total_models:
        plain(logger, style(f"  Next: make {mode}  (trains and scores nothing; rebuilds tables and predictors)", "yellow"))


if __name__ == "__main__":
    main()
