"""Generate, train and cache the architecture-variation populations.

    train-variants --phase phase1 --mode light
    train-variants --phase phase3 --mode light --device mps

Each phase produces one population that a predictor is fitted on and one or more
that it is evaluated on, cached under ``cache/variants/<phase>/<split>/``, plus
one manifest CSV per phase holding every model's configuration alongside the
quality it reached. Phase 3 has a ``base`` population plus one validation group
per varied axis (``deeper``, ``wider``, ``saturating``), each holding the other
axes at the base region so a drop can be attributed to the axis it moved.

The split is **by model, never by neuron**: neurons of one model share its
architecture, its initialisation and its training run, so splitting below the
model level leaks nearly everything that distinguishes them. The partition
therefore happens over configurations, before any model is built.

Already-cached models are skipped, so an interrupted run resumes.
"""

import csv
import json
import logging
import time
from typing import Dict, List

import click

from staticsbfl.data import IMAGE_DATASET_NAMES, load_dataset
from staticsbfl.models import count_parameters, parameter_bytes
from staticsbfl.persistence import ModelCheckpoint, load_model, save_model
from staticsbfl.training import OPTIMIZER_NAME, evaluate, train_epochs
from staticsbfl.utils.devices import device_option, resolve_device
from staticsbfl.log.console import (
    Step, configure_logging, display_path, format_duration, human_count, plain, style,
)
from staticsbfl.utils.paths import (
    ensure_dir, variant_design, variant_model_path, variant_models_csv,
)
from staticsbfl.utils.seeding import set_seed
from staticsbfl.variation import (
    EPOCHS_BY_MODE,
    MODELS_PER_SPLIT,
    PHASE_NAMES,
    PhaseSpec,
    build_model,
    build_phase,
)
from staticsbfl.log.pipeline import (
    announce_phase, step_context, step_description, step_index, step_outputs,
)

logger = logging.getLogger("variants")


MODEL_FIELDS = (
    "key", "phase", "split", "family", "seed",
    "depth", "hidden_widths", "activation", "activation_id",
    "min_width", "max_width", "mean_width", "total_hidden_units", "width_profile",
    "n_parameters", "size_bytes", "optimizer", "epochs",
    "test_accuracy", "test_loss", "macro_f1", "train_seconds",
)

def write_models_csv(phase: str, mode: str, dataset: str, rows: Dict[str, dict]) -> None:
    """One row per model: what was built, and how good it turned out.

    The *realisation* half of the record. Rows for models this run skipped are
    carried over from the existing file, so the CSV always describes the whole
    population on disk rather than only what the last invocation touched.
    """
    path = variant_models_csv(phase, mode, dataset)
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MODEL_FIELDS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])
    logger.info("Wrote %s (%d model(s))", display_path(path), len(rows))

def write_design(
    spec: PhaseSpec, *, mode: str, epochs: int, dataset: str, device: str,
    cost: Dict[str, float] = None,
) -> None:
    """Write the phase *specification*: the question and how it was populated.

    This is the half of the record that the model files cannot supply -- which
    region each group was drawn from, which axis it moved, which group is fitted
    and which are evaluated, and what budget every model got. Machine-readable,
    so a later run can check it is extending the same design rather than a
    different one, and so a table in the paper can be generated rather than
    transcribed.

    Deliberately *not* here: one line per model. That is ``models.csv``, and
    duplicating it in a second file was how the two drifted apart before.
    """
    path = variant_design(spec.name, mode, dataset)
    ensure_dir(path.parent)
    existing = json.loads(path.read_text()) if path.exists() else {}
    design = {
        "phase": spec.name,
        "question": spec.question,
        "mode": mode,
        "dataset": dataset,
        "epochs_per_model": epochs,
        "identical_budget": True,
        "device": device,
        "fit_population": spec.train_split,
        "evaluation_populations": list(spec.evaluation_splits),
        "populations": {
            name: {
                "role": "fit" if name == spec.train_split else "evaluation",
                "n_models": len(spec.populations[name]),
                "region": spec.regions[name].describe(),
            }
            for name in spec.split_names
        },
        "n_models_total": spec.n_models,
        "generation_cost": cost or existing.get("generation_cost", {}),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote the phase design to %s", display_path(path))

def read_models_csv(phase: str, mode: str, dataset: str) -> Dict[str, dict]:
    """Existing rows, so models this run skips keep the record they already had."""
    path = variant_models_csv(phase, mode, dataset)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["key"]: row for row in csv.DictReader(handle)}

def row_from_checkpoint(config, checkpoint: ModelCheckpoint) -> dict:
    """A ``models.csv`` row rebuilt from a cached checkpoint.

    ``models.csv`` used to be written once, at the end of a run, so a run that
    was interrupted left trained checkpoints without rows -- and every later run
    skipped those models and never recorded them. The checkpoint carries the
    configuration, the budget and the test quality, so the row can be recovered
    exactly; only the training time is lost, and it is left blank rather than
    guessed, which also keeps it out of every runtime estimate.
    """
    model = build_model(
        config, num_classes=checkpoint.num_classes, input_shape=checkpoint.input_shape
    )
    model.load_state_dict(checkpoint.state_dict)
    meta = checkpoint.metadata or {}
    number = lambda key: "" if meta.get(key) is None else f"{meta[key]:.6f}"
    return {
        **config.as_row(),
        "n_parameters": count_parameters(model),
        "size_bytes": parameter_bytes(model),
        "optimizer": meta.get("optimizer", ""),
        "epochs": meta.get("epochs", ""),
        "test_accuracy": number("test_accuracy"),
        "test_loss": number("test_loss"),
        "macro_f1": number("macro_f1"),
        "train_seconds": "",
    }


def budget_mismatches(keys, rows: Dict[str, dict], epochs: int) -> List[str]:
    """Keys whose recorded epoch count is not *epochs*.

    A model without a recorded count is not judged here; rows are recovered
    from the checkpoints before this is asked, so that case is rare. Counts are
    compared as text: rows read from the CSV hold strings, rows recovered from a
    checkpoint hold integers.
    """
    recorded = lambda key: str(rows.get(key, {}).get("epochs", "") or "")
    return [key for key in keys if recorded(key) not in ("", str(epochs))]


@click.command()
@click.option(
    "--phase",
    type=click.Choice(PHASE_NAMES),
    required=True,
    help="phase1 = reseeding only, phase2 = sampled architectures, "
         "phase3 = disjoint train/test regions.",
)
@click.option(
    "--mode",
    type=click.Choice(sorted(MODELS_PER_SPLIT)),
    default="light",
    show_default=True,
    help="debug = 4 fit models (pipeline check only, not a result), "
         "light = 100, heavy = 500.",
)
@device_option()
@click.option(
    "--dataset", type=click.Choice(IMAGE_DATASET_NAMES), default="cifar10", show_default=True,
    help="fashion_mnist is the control population: a dense model reaches ~88% there "
         "but plateaus near 50% on CIFAR-10, which separates 'the task is too hard "
         "for this capacity' from 'deep narrow ReLU stacks lose units anyway'.",
)
@click.option("--epochs", type=int, default=None, help="Override the mode's epoch count.")
@click.option("--batch-size", type=int, default=256, show_default=True)
@click.option(
    "--max-train-samples", type=int, default=None, help="Cap the training split."
)
@click.option(
    "--force/--skip-existing", default=False, show_default=True,
    help="Retrain models that are already cached.",
)
@click.option(
    "--retrain-mismatched", is_flag=True,
    help="Retrain only the cached models whose recorded epoch count differs from "
         "this run's, and keep every other cached model. Their spectra are rescored "
         "by the next make score, which treats a spectrum older than its model as stale.",
)
@click.option("--dry-run", is_flag=True, help="Print the population and stop.")
@click.option("--verbose", is_flag=True, help="Debug-level logging.")
def main(
    phase, mode, device_name, dataset, epochs, batch_size,
    max_train_samples, force, retrain_mismatched, dry_run, verbose,
):
    """Build, train and cache one phase's train and test populations."""
    configure_logging(verbose)
    device = resolve_device(device_name)
    epochs = epochs if epochs is not None else EPOCHS_BY_MODE[mode]

    spec: PhaseSpec = build_phase(phase, mode)

    if dry_run:
        # The preview is the phase banner plus one line per model: everything a
        # run would do, nothing it would write.
        announce_phase(phase, mode, dataset, glossary=False)
        plain(logger, style(f"── Population of {spec.name}: {spec.n_models} models ", "bold"))
        for config in spec.configs():
            logger.info(
                "    %-24s %-10s depth %2d  widths %-28s %-10s seed %d",
                config.key, config.split, config.depth,
                "-".join(str(w) for w in config.hidden_widths)[:28],
                config.activation, config.seed,
            )
        plain(logger, style("Dry run: nothing was trained or written.", "dim"))
        return

    number, of = step_index("train")
    with Step(
        logger, number, of, "train",
        step_description("train", spec, epochs=epochs),
        outputs=step_outputs("train", phase, mode, dataset),
        context=step_context(phase, mode, dataset),
    ) as step:
        if mode == "debug":
            logger.warning(
                "DEBUG MODE: %d fit models. This checks that the pipeline runs and "
                "writes the files it should. It cannot support a predictor, and every "
                "number downstream of it is noise -- do not read the results.",
                len(spec.train),
            )
        for line in spec.describe().split("\n"):
            logger.info("%s", line)
        logger.info(
            "SCOPE: dense (MLP) models only -- convolutional and recurrent "
            "variants are not implemented yet."
        )
        logger.info(
            "%d models total, %d epochs each, %s on %s",
            spec.n_models, epochs, dataset, device,
        )

        step.section("Checking the cache")
        rows = read_models_csv(spec.name, mode, dataset)
        cached = [
            config for config in spec.configs()
            if variant_model_path(spec.name, config.split, config.key, mode, dataset).exists()
        ]
        unrecorded = [c for c in cached if c.key not in rows]
        for config in unrecorded:
            checkpoint = load_model(
                variant_model_path(spec.name, config.split, config.key, mode, dataset)
            )
            rows[config.key] = row_from_checkpoint(config, checkpoint)
        if unrecorded:
            logger.info(
                "Recovered %d models.csv row(s) from their checkpoints -- trained by a "
                "run that stopped before writing them. Their training time is unknown "
                "and left blank.", len(unrecorded),
            )

        # Every model of a population gets the same budget. A cached model trained
        # under another one would be skipped and silently kept, so the run stops
        # instead of producing a population whose design.json claims otherwise.
        other_budget = budget_mismatches([c.key for c in cached], rows, epochs)
        if other_budget and retrain_mismatched and not force:
            logger.warning(
                "%d cached model(s) were trained for another epoch count than %d and "
                "will be retrained: %s", len(other_budget), epochs,
                ", ".join(other_budget[:5]) + (" ..." if len(other_budget) > 5 else ""),
            )
        elif other_budget and not force:
            raise click.ClickException(
                f"{len(other_budget)} cached model(s) in {display_path(variant_models_csv(spec.name, mode, dataset).parent)} "
                f"were trained for another epoch count than {epochs} "
                f"({', '.join(other_budget[:3])}{' ...' if len(other_budget) > 3 else ''}). "
                "Skipping them would mix two training budgets in one population. "
                "Retrain just those with TRAIN_ARGS=--retrain-mismatched (Makefile) / "
                "--retrain-mismatched, or everything with FORCE=--force."
            )
        logger.info("%d of %d models already cached.", len(cached), spec.n_models)

        write_design(spec, mode=mode, epochs=epochs, dataset=dataset, device=str(device))

        step.section("Loading the dataset")
        bundle = load_dataset(
            dataset, batch_size=batch_size, max_train_samples=max_train_samples
        )
        trained, skipped = 0, 0
        run_started = time.perf_counter()

        retrain = set(other_budget) if retrain_mismatched else set()
        to_train = spec.n_models - (0 if force else len(cached) - len(retrain))
        step.section(
            f"Training {to_train} of {spec.n_models} models"
            + ("" if force or not cached else
               f" ({len(cached) - len(retrain)} already cached"
               + (f", {len(retrain)} retrained for the budget)" if retrain else ")"))
        )

        # Written however the loop ends, so an interrupted run leaves a row for
        # every checkpoint it saved rather than checkpoints nobody recorded.
        try:
            for index, config in enumerate(spec.configs(), start=1):
                destination = variant_model_path(spec.name, config.split, config.key, mode, dataset)
                if destination.exists() and not force and config.key not in retrain:
                    skipped += 1
                    continue

                # Seed before construction so the initialisation belongs to the config.
                set_seed(config.seed)
                model = build_model(
                    config, num_classes=bundle.num_classes, input_shape=bundle.input_shape
                )
                n_params = count_parameters(model)

                started = time.perf_counter()
                for _epoch in train_epochs(
                    model, bundle.train_loader, epochs=epochs, device=device
                ):
                    pass
                train_seconds = time.perf_counter() - started

                result = evaluate(
                    model, bundle.test_loader, device=device, num_classes=bundle.num_classes
                )
                trained += 1
                remaining = (time.perf_counter() - run_started) / trained * (to_train - trained)
                logger.info(
                    "[%3d/%d] %-24s %-10s depth %2d  %-10s %8s params  acc %.4f  "
                    "(%.1fs, ~%s left)",
                    index, spec.n_models, config.key, config.split, config.depth,
                    config.activation, human_count(n_params), result.accuracy, train_seconds,
                    format_duration(remaining),
                )

                save_model(
                    destination,
                    ModelCheckpoint(
                        key=config.key,
                        state_dict=model.state_dict(),
                        dataset=dataset,
                        input_shape=bundle.input_shape,
                        num_classes=bundle.num_classes,
                        metadata={
                            "config": config.as_row(),
                            "hidden_widths": list(config.hidden_widths),
                            "epochs": epochs,
                            "optimizer": OPTIMIZER_NAME,
                            "test_accuracy": result.accuracy,
                            "test_loss": result.loss,
                            "macro_f1": result.macro_f1,
                        },
                    ),
                )

                rows[config.key] = {
                    **config.as_row(),
                    "n_parameters": n_params,
                    "size_bytes": parameter_bytes(model),
                    "optimizer": OPTIMIZER_NAME,
                    "epochs": epochs,
                    "test_accuracy": f"{result.accuracy:.6f}",
                    "test_loss": f"{result.loss:.6f}",
                    "macro_f1": f"{result.macro_f1:.6f}",
                    "train_seconds": f"{train_seconds:.2f}",
                }
        finally:
            step.section("Writing the records")
            write_models_csv(spec.name, mode, dataset, rows)

        elapsed = time.perf_counter() - run_started
        per_model = [float(r["train_seconds"]) for r in rows.values() if r.get("train_seconds")]
        if per_model:
            # Generation cost is itself an experimental result: it bounds how many
            # models a study of this kind can afford, so it is reported and stored
            # rather than left in the scrollback.
            logger.info(
                "Generation cost: %d model(s) total, %.1fs of training "
                "(mean %.1fs, min %.1fs, max %.1fs per model); wall clock %.1fs "
                "including dataset preparation and evaluation.",
                len(per_model), sum(per_model), sum(per_model) / len(per_model),
                min(per_model), max(per_model), elapsed,
            )
            # Generation cost belongs with the design it describes, not in a third
            # file: it is a property of "this many models at this budget", which is
            # exactly what design.json records.
            write_design(
                spec, mode=mode, epochs=epochs, dataset=dataset, device=str(device),
                cost={
                    "models_recorded": len(per_model),
                    "total_training_seconds": round(sum(per_model), 1),
                    "mean_seconds_per_model": round(sum(per_model) / len(per_model), 2),
                    "min_seconds_per_model": round(min(per_model), 2),
                    "max_seconds_per_model": round(max(per_model), 2),
                    "wall_clock_seconds_this_run": round(elapsed, 1),
                    "note": "wall clock covers this invocation only; skipped models excluded",
                },
            )
        step.summary = f"{trained} trained, {skipped} already cached"


if __name__ == "__main__":
    main()
