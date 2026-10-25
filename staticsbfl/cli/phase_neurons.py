"""Build the neuron-level table from cached variant models and their spectra.

    build-neuron-dataset --phase phase1
    build-neuron-dataset --phase phase3 --mode light

Reads ``cache/variants/<phase>/<split>/`` -- the trained models and the hit
spectra computed from them -- and writes one table to
``cache/variants/<phase>/neurons.npz``.

One row per hidden neuron. Features are computable without executing the model;
targets come from the spectrum. Each row records the model it came from, which
is the group for a ranking objective and the unit for any split.

LIGHT PATH ONLY: dense models. Convolutional and recurrent variants are not
built yet and are skipped with an explicit notice.
"""

import logging
import time

import click
import numpy as np

from staticsbfl.data import IMAGE_DATASET_NAMES
from staticsbfl.dataset import FEATURE_NAMES, TARGET_NAMES, combine, build_model_rows
from staticsbfl.persistence import load_model, load_suspiciousness
from staticsbfl.log.console import Step, configure_logging, display_path
from staticsbfl.utils.paths import (
    variant_neurons,
    ensure_dir,
    variant_model_path,
    variant_spectrum_path,
)
from staticsbfl.variation import MODELS_PER_SPLIT, PHASE_NAMES, build_model, build_phase
from staticsbfl.log.pipeline import (
    record_timing, step_context, step_description, step_index, step_outputs,
)

logger = logging.getLogger("neuron-dataset")

#: Families the builder can currently handle.
SUPPORTED_FAMILIES = ("mlp",)


@click.command()
@click.option("--phase", type=click.Choice(PHASE_NAMES), required=True)
@click.option(
    "--mode", type=click.Choice(sorted(MODELS_PER_SPLIT)), default="light",
    show_default=True,
)
@click.option(
    "--dataset", type=click.Choice(IMAGE_DATASET_NAMES), default="cifar10", show_default=True,
    help="Which population tree to read. Must match the dataset the models were "
         "trained on; every cache path is namespaced by it.",
)
@click.option("--verbose", is_flag=True, help="Debug-level logging.")
def main(phase, mode, dataset, verbose):
    """Assemble the neuron table for one phase."""
    configure_logging(verbose)
    spec = build_phase(phase, mode)

    number, of = step_index("neurons")
    with Step(
        logger, number, of, "neurons",
        step_description("neurons", spec),
        outputs=step_outputs("neurons", phase, mode, dataset),
        context=step_context(phase, mode, dataset),
    ) as step:
        logger.info(
            "SCOPE: dense (MLP) models only. Convolutional and recurrent variants "
            "are not implemented yet and any non-dense model will be skipped."
        )
        started = time.perf_counter()
        parts = []
        skipped_family, missing = [], []

        step.section(f"Reading {spec.n_models} models and their spectra")
        for config in spec.configs():
            if config.family not in SUPPORTED_FAMILIES:
                skipped_family.append(config.key)
                continue

            weights = variant_model_path(spec.name, config.split, config.key, mode, dataset)
            spectrum = variant_spectrum_path(spec.name, config.split, config.key, mode, dataset)
            if not weights.exists() or not spectrum.exists():
                missing.append(config.key)
                continue

            checkpoint = load_model(weights)
            bundle = load_suspiciousness(spectrum)
            model = build_model(
                config,
                num_classes=checkpoint.num_classes,
                input_shape=checkpoint.input_shape,
            )
            model.load_state_dict(checkpoint.state_dict)

            input_size = int(np.prod(checkpoint.input_shape))
            features, targets, layers = build_model_rows(
                model, config, bundle,
                input_size=input_size, num_classes=checkpoint.num_classes,
            )
            parts.append((features, targets, layers, config.key, config.split))

        if skipped_family:
            logger.warning(
                "SKIPPED %d non-dense model(s); only %s is supported so far: %s",
                len(skipped_family), "/".join(SUPPORTED_FAMILIES),
                ", ".join(skipped_family[:5]) + (" ..." if len(skipped_family) > 5 else ""),
            )
        if missing:
            logger.warning(
                "%d model(s) lack cached weights or a cached spectrum; run "
                "make train and make score first: %s",
                len(missing), ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""),
            )
        if not parts:
            raise click.ClickException("Nothing to build. Train and score the models first.")

        step.section("Assembling the table")
        table = combine(parts)
        logger.info(
            "%d rows from %d models · %d static features · %d targets",
            len(table), table.n_models, len(FEATURE_NAMES), len(TARGET_NAMES),
        )
        for split in spec.split_names:
            subset = table.for_split(split)
            if len(subset):
                role = "fit " if split == spec.train_split else "eval"
                logger.info(
                    "    [%s] %-12s %7d rows  %3d models  mean %d neurons/model",
                    role, split, len(subset), subset.n_models,
                    len(subset) // max(1, subset.n_models),
                )

        path = variant_neurons(spec.name, mode, dataset)
        ensure_dir(path.parent)
        table.save(path)
        logger.info("Wrote %s (%.1f MB)", display_path(path), path.stat().st_size / 1e6)

        # No CSV preview is written. It duplicated the first 200 rows of the table
        # in a second format, was never read by anything, and grew with every run.
        # `NeuronDataset.load(path)` in a REPL answers the same question.

        record_timing(
            spec.name, mode, dataset, "neurons", time.perf_counter() - started,
            len(parts), "model",
        )
        step.summary = f"{len(table):,} neurons from {table.n_models} models"


if __name__ == "__main__":
    main()
