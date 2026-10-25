"""Hit spectrum and suspiciousness for the variation populations.

    spectrum-variants --phase phase1 --mode light
    spectrum-variants --phase phase3 --split width --device mps

Runs the same single counting pass as ``compute-suspiciousness``, but over every
model of a phase, and caches the result next to that model under
``cache/variants/<phase>/<split>/spectra/``. It also appends the per-layer
activation rates to a phase-level CSV, because with hundreds of architectures
the rates are no longer something to read off a log line -- a population whose
activation rates drift with depth would make later comparisons meaningless.

The train/test split is inherited from the population and never re-derived here.
"""

import csv
import logging
import time
from typing import List

import click
import torch

from staticsbfl.data import IMAGE_DATASET_NAMES, load_dataset
from staticsbfl.persistence import (
    SuspiciousnessBundle,
    load_model,
    save_suspiciousness,
)
from staticsbfl.suspiciousness import (
    DEFAULT_STAR,
    compute_hit_spectrum,
    score_spectrum,
    summarize_activity,
)
from staticsbfl.utils.devices import device_option, resolve_device
from staticsbfl.log.console import Step, configure_logging, display_path
from staticsbfl.utils.paths import (
    variant_activation_rates,
    ensure_dir,
    variant_model_path,
    variant_spectrum_path,
)
from staticsbfl.utils.seeding import set_seed
from staticsbfl.variation import (
    MODELS_PER_SPLIT,
    PHASE_NAMES,
    build_model,
    build_phase,
)
from staticsbfl.log.pipeline import (
    record_timing, step_context, step_description, step_index, step_outputs,
)

logger = logging.getLogger("variant-spectrum")

ACTIVITY_FIELDS = (
    "key", "phase", "split", "layer", "layer_index", "depth", "width",
    "n_neurons", "mean_rate", "median_rate", "saturated_fraction", "dead_fraction",
    "tarantula_spread", "accuracy", "n_failing", "n_successful",
)


@click.command()
@click.option("--phase", type=click.Choice(PHASE_NAMES), required=True)
@click.option(
    "--mode", type=click.Choice(sorted(MODELS_PER_SPLIT)), default="light",
    show_default=True,
)
@device_option()
@click.option("--star", type=int, default=DEFAULT_STAR, show_default=True)
@click.option("--threshold", type=float, default=0.0, show_default=True)
@click.option("--batch-size", type=int, default=512, show_default=True)
@click.option("--max-test-samples", type=int, default=None, help="Cap the test split.")
@click.option(
    "--force/--skip-existing", default=False, show_default=True,
    help="Rescore models whose spectrum is already cached.",
)
@click.option(
    "--dataset", type=click.Choice(IMAGE_DATASET_NAMES), default="cifar10", show_default=True,
    help="Which population tree to read. Must match the dataset the models were "
         "trained on; every cache path is namespaced by it.",
)
@click.option("--verbose", is_flag=True, help="Debug-level logging.")
def main(
    phase, mode, device_name, dataset, star, threshold,
    batch_size, max_test_samples, force, verbose,
):
    """Count and score every model of one phase."""
    configure_logging(verbose)
    device = resolve_device(device_name)
    spec = build_phase(phase, mode)

    number, of = step_index("score")
    with Step(
        logger, number, of, "score",
        step_description("score", spec),
        outputs=step_outputs("score", phase, mode, dataset),
        context=step_context(phase, mode, dataset),
    ) as step:
        logger.info(
            "%d model(s) of %s on %s · D*^%d · a neuron counts as active when its "
            "response is > %g",
            spec.n_models, spec.name, device, star, threshold,
        )

        step.section("Loading the test set")
        bundle = load_dataset(
            dataset, batch_size=batch_size, max_test_samples=max_test_samples
        )
        activity_rows: List[dict] = []
        scored, skipped, stale, missing = 0, 0, 0, []
        run_started = time.perf_counter()

        step.section(f"Counting and scoring {spec.n_models} models (progress every 10)")
        for index, config in enumerate(spec.configs(), start=1):
            weights = variant_model_path(spec.name, config.split, config.key, mode, dataset)
            if not weights.exists():
                missing.append(config.key)
                continue

            destination = variant_spectrum_path(spec.name, config.split, config.key, mode, dataset)
            # A spectrum older than its weights was counted from a model that no
            # longer exists -- the checkpoint was retrained since -- so it is
            # recounted rather than skipped.
            outdated = destination.exists() and destination.stat().st_mtime < weights.stat().st_mtime
            if destination.exists() and not force and not outdated:
                skipped += 1
                continue
            stale += outdated

            set_seed(config.seed)
            checkpoint = load_model(weights)
            model = build_model(
                config,
                num_classes=checkpoint.num_classes,
                input_shape=checkpoint.input_shape,
            )
            model.load_state_dict(checkpoint.state_dict)

            hit_spectrum = compute_hit_spectrum(
                model, bundle.test_loader, device=device, threshold=threshold
            )
            scores = score_spectrum(hit_spectrum, star=star)
            activity = summarize_activity(hit_spectrum, scores)

            for layer_index, (name, summary) in enumerate(activity.items()):
                activity_rows.append(
                    {
                        "key": config.key,
                        "phase": spec.name,
                        "split": config.split,
                        "layer": name,
                        "layer_index": layer_index,
                        "depth": config.depth,
                        "width": config.hidden_widths[layer_index],
                        "n_neurons": summary.n_neurons,
                        "mean_rate": f"{summary.mean_rate:.6f}",
                        "median_rate": f"{summary.median_rate:.6f}",
                        "saturated_fraction": f"{summary.saturated_fraction:.6f}",
                        "dead_fraction": f"{summary.dead_fraction:.6f}",
                        "tarantula_spread": f"{summary.score_spread:.6f}",
                        "accuracy": f"{hit_spectrum.accuracy:.6f}",
                        "n_failing": hit_spectrum.n_failing,
                        "n_successful": hit_spectrum.n_successful,
                    }
                )

            save_suspiciousness(
                destination,
                SuspiciousnessBundle(
                    key=config.key,
                    hit_spectrum=hit_spectrum.to(torch.device("cpu")),
                    scores={
                        m: {k: v.cpu() for k, v in per_layer.items()}
                        for m, per_layer in scores.items()
                    },
                    metadata={
                        "config": config.as_row(),
                        "phase": spec.name,
                        "split": config.split,
                        "dataset": dataset,
                        "star": star,
                        "threshold": threshold,
                        "activation_summary": {
                            n: vars(summary) for n, summary in activity.items()
                        },
                        "model_metadata": checkpoint.metadata,
                    },
                ),
            )
            scored += 1
            if index % 10 == 0 or index == spec.n_models:
                dead = sum(s.dead_fraction * s.n_neurons for s in activity.values())
                logger.info(
                    "  [%3d/%d] %-24s acc %.4f  %5d neurons  %4.1f%% dead",
                    index, spec.n_models, config.key,
                    hit_spectrum.accuracy, hit_spectrum.n_neurons,
                    100 * dead / max(1, hit_spectrum.n_neurons),
                )

        if skipped:
            logger.info("%d model(s) already had a cached spectrum and were skipped.", skipped)
        if stale:
            logger.info("%d spectrum/spectra were older than their retrained model and were recounted.", stale)
        if missing:
            logger.warning(
                "%d model(s) have no cached weights; run make train first: %s",
                len(missing), ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""),
            )

        if activity_rows:
            step.section("Writing the activation rates")
            path = variant_activation_rates(spec.name, mode, dataset)
            ensure_dir(path.parent)
            existing = []
            if path.exists() and not force:
                with open(path, newline="", encoding="utf-8") as handle:
                    existing = [r for r in csv.DictReader(handle)]
            done = {(r["key"], r["layer"]) for r in activity_rows}
            merged = [r for r in existing if (r["key"], r["layer"]) not in done] + activity_rows
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=ACTIVITY_FIELDS)
                writer.writeheader()
                writer.writerows(merged)
            logger.info("Wrote %s (%d layer rows)", display_path(path), len(merged))

        record_timing(
            spec.name, mode, dataset, "score", time.perf_counter() - run_started,
            scored, "model", device=str(device),
        )
        step.summary = f"{scored} scored, {skipped} already cached" + (
            f", {len(missing)} MISSING weights" if missing else ""
        )


if __name__ == "__main__":
    main()
