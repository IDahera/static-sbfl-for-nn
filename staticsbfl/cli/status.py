"""Banners, status and runtime estimates of the pipeline.

    make status
    python -m staticsbfl.cli.status status --mode heavy
    python -m staticsbfl.cli.status start --phase phase2 --mode light --dataset cifar10

The command-line face of :mod:`staticsbfl.log.pipeline`, which holds the logic; the
Makefile calls ``start`` and ``end`` around every ``make phase``.
"""

import click

from staticsbfl.log.pipeline import (
    DATASETS, REAL_MODES, announce_experiment, announce_phase, close_phase, logger, print_status,
)
from staticsbfl.log.console import configure_logging, plain, style
from staticsbfl.utils.paths import CACHE_DIR, VARIANTS_DIR
from staticsbfl.variation import PHASE_NAMES
from staticsbfl.variation.phases import POPULATION_SIZES

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

_mode = click.option("--mode", type=click.Choice(sorted(POPULATION_SIZES)), default="light", show_default=True)
_dataset = click.option("--dataset", type=click.Choice(DATASETS), default="cifar10", show_default=True)
_phase = click.option("--phase", type=click.Choice(PHASE_NAMES), required=True)


@click.group()
def main():
    """Banners, status and runtime estimates for the pipeline."""
    configure_logging()


@main.command()
@_mode
@_dataset
def overview(mode, dataset):
    """The experiment banner: four phases, their status, the glossary."""
    announce_experiment(mode, dataset)


@main.command()
@_phase
@_mode
@_dataset
@click.option("--glossary/--no-glossary", default=True, show_default=True)
def start(phase, mode, dataset, glossary):
    """The banner that opens one phase."""
    announce_phase(phase, mode, dataset, glossary)


@main.command()
@_phase
@_mode
@_dataset
@click.option("--since", type=float, default=None, help="Unix time the phase started.")
def end(phase, mode, dataset, since):
    """The banner that closes one phase, with a check of what it wrote."""
    close_phase(phase, mode, dataset, since)


@main.command()
@click.option("--mode", type=click.Choice(REAL_MODES + ("debug",)), multiple=True,
              help="Default: light and heavy.")
@click.option("--dataset", type=click.Choice(DATASETS), multiple=True,
              help="Default: every dataset.")
def status(mode, dataset):
    """What is done, what is missing, how long the rest takes, and how much disk."""
    from staticsbfl.cli.cache_size import human, tree_size

    import shutil

    coming = 0
    for name in dataset or DATASETS:
        for each in mode or REAL_MODES:
            coming += print_status(each, name)[1]

    plain(logger, "")
    cache, _files = tree_size(CACHE_DIR)
    free = shutil.disk_usage(CACHE_DIR if CACHE_DIR.exists() else CACHE_DIR.parent).free
    plain(logger, style("Disk", "bold") + f"   cache/ {human(cache)} in total · {human(free)} free · "
          f"≈ {human(coming)} still to come for the trees above")
    if coming > free:
        plain(logger, style(
            "  ! the trees above would not fit on this disk when complete.", "yellow"))
    legacy = sorted(p.name for p in VARIANTS_DIR.glob("phase*") if p.is_dir())
    if legacy:
        plain(logger, style(
            f"  note: cache/variants/{{{','.join(legacy)}}} use the old layout (no dataset "
            "folder) and are read by nothing.", "dim",
        ))
    plain(logger, style("  make cache-size breaks down all of cache/, with what deletes each part.", "dim"))
    plain(logger, "")


if __name__ == "__main__":
    main()
