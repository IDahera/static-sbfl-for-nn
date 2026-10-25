"""What the whole run cost: training, scoring, table building and fitting.

    make costs
    python -m staticsbfl.cli.costs --mode light --mode heavy

Read only; light and heavy are reported apart, each per dataset and phase,
with total and per-model training time (mean, sd) and per model-epoch. Every
file is read under its own try/except, so a missing or broken phase is reported
and the rest goes on. Writes ``cache/runtime/costs.csv`` and ``costs.log``.

The data is :mod:`staticsbfl.evaluation.costs`, the printing
:func:`staticsbfl.log.evaluation.print_costs`.
"""

import click

from staticsbfl.evaluation import costs
from staticsbfl.log import evaluation as report
from staticsbfl.log.console import add_file_log, configure_logging, display_path, phase_banner, plain, style
from staticsbfl.utils.paths import RUNTIME_DIR


@click.command()
@click.option("--mode", "modes", multiple=True, type=click.Choice(costs.MODES), help="Default: light and heavy.")
@click.option("--dataset", "datasets", multiple=True, type=click.Choice(costs.DATASETS), help="Default: both.")
def main(modes, datasets):
    """Training and step times of the whole run, light and heavy apart."""
    configure_logging()
    add_file_log(RUNTIME_DIR / "costs.log", mode="w", timestamps=False)
    modes, datasets = tuple(modes) or costs.MODES, tuple(datasets) or costs.DATASETS
    phase_banner(report.logger, "StaticSBFL · COST OF THE WHOLE RUN · " + " / ".join(modes), [
        ("Reads", ["<root>/<phase>/models.csv (train_seconds per model)",
                   "<root>/<phase>/timings.json (score, neurons, predict, fit)",
                   style("read only; every file is tried on its own and a missing one is reported", "dim")]),
        ("Columns", ["per model = mean training time of a timed model; sd across models",
                     "per m-ep = per model and epoch -- comparable between light and heavy",
                     "(fit) = predictor fitting alone, part of predict"]),
    ], outputs=[(costs.costs_csv(), "one row per mode/dataset/phase/population/step"), RUNTIME_DIR / "costs.log"])
    rows = costs.collect(modes, datasets)
    report.print_costs(rows, modes, datasets)
    path = costs.write(rows)
    plain(report.logger, "")
    plain(report.logger, "  " + style(f"▸ {display_path(path)}", "yellow"))


if __name__ == "__main__":
    main()
