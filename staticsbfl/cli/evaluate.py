"""The final evaluation -- read only, run it last.

    make evaluate                                  # light and heavy, CIFAR-10, scope active
    python -m staticsbfl.cli.evaluate --mode light --phase phase3 --scope all
    python -m staticsbfl.cli.evaluate --model-type tree_ensemble --metric ochiai
    python -m staticsbfl.cli.evaluate --predictor mlp --no-runtime

Reads what every earlier step wrote -- the prediction results of the four
phases, ``design.json`` and ``models.csv``, the cost report and the runtime
records -- and recomputes nothing. It prints the essentials (legend, best case
per phase and population with its baselines, overview, light vs heavy, caveats,
costs, the runtime comparison) and writes:

    <root>/summary_<scope>.csv                  overview per mode, for paper tables
    cache/runtime/<dataset>/<mode>/summary.csv  per tool, + by_design.csv
    cache/evaluation/<dataset>/report.html      EVERY row, in tabs, highlighted, filterable
    cache/evaluation/<dataset>/statistics.csv   per group: count, undefined, min, ..., max
    cache/runtime/costs.csv                     the cost report (also make costs)

Light and heavy are always both shown unless ``--mode`` narrows it; a phase,
mode or runtime run that does not exist yet is reported as missing and appears
in the HTML as a placeholder naming the command that produces it.

Filters (``--predictor``, ``--model-type``, ``--metric``, ``--target``,
``--population``) apply to the printout, the CSVs and the HTML alike; the
baselines are kept on every remaining case. The HTML additionally filters in
the browser.

The computation is :mod:`staticsbfl.evaluation`, the printing :mod:`staticsbfl.log.evaluation`,
the page :mod:`staticsbfl.viz.html_report`.
"""

import click

from staticsbfl.evaluation import accuracy as acc
from staticsbfl.evaluation import costs as cost_data
from staticsbfl.evaluation import runtime as runtime_data
from staticsbfl.log import evaluation as report
from staticsbfl.log.pipeline import DATASETS
from staticsbfl.log.console import add_file_log, configure_logging, display_path, phase_banner, plain, style
from staticsbfl.utils.paths import CACHE_DIR, runtime_root, variant_predictions_dir, variants_root
from staticsbfl.variation import PHASE_NAMES
from staticsbfl.viz import html_report

MODES = ("light", "heavy", "debug")
PREDICTORS = ("gradient_boosting", "random_forest", "mlp")


def evaluation_dir(dataset: str):
    return CACHE_DIR / "evaluation" / dataset


@click.command()
@click.option("--mode", "modes", multiple=True, type=click.Choice(MODES),
              help="Default: light and heavy (whichever has results).")
@click.option("--dataset", type=click.Choice(DATASETS), default="cifar10", show_default=True)
@click.option("--phase", "phases", multiple=True, type=click.Choice(PHASE_NAMES), help="Default: every phase.")
@click.option("--scope", type=click.Choice(("active", "all")), default="active", show_default=True,
              help="Which evaluation scope the headlines use; the other is shown beside it.")
@click.option("--predictor", "predictors", multiple=True, type=click.Choice(PREDICTORS),
              help="Keep only these fitted predictors (repeatable).")
@click.option("--model-type", "model_types", multiple=True, type=click.Choice(("tree_ensemble", "neural_network")),
              help="Keep only predictors of this type: tree_ensemble = gradient boosting (and the random "
                   "forest of results fitted before 2026-10-22), "
                   "neural_network = MLP.")
@click.option("--metric", "metrics", multiple=True, type=click.Choice(acc.METRICS))
@click.option("--target", "targets", multiple=True, help="e.g. ochiai_rank_model (repeatable).")
@click.option("--population", "populations", multiple=True, help="e.g. designed, deeper (repeatable).")
@click.option("--legend/--no-legend", default=True, show_default=True)
@click.option("--write/--no-write", default=True, show_default=True, help="Write the CSV summaries.")
@click.option("--runtime/--no-runtime", default=True, show_default=True,
              help="Add the runtime comparison (reads cache/runtime/).")
@click.option("--html/--no-html", default=True, show_default=True, help="Write the HTML report.")
def main(modes, dataset, phases, scope, predictors, model_types, metrics, targets, populations,
         legend, write, runtime, html):
    """Summarise every result without recomputing anything."""
    configure_logging()
    directory = evaluation_dir(dataset)
    add_file_log(directory / "evaluation.log", mode="w", timestamps=False)
    flt = acc.ResultFilter(tuple(predictors), tuple(model_types), tuple(metrics), tuple(targets), tuple(populations))
    # Without --mode, light and heavy are both shown; what does not exist yet is a placeholder.
    modes = tuple(modes) or ("light", "heavy")
    phase_banner(report.logger, f"StaticSBFL · EVALUATION · {' / '.join(modes)} · {dataset}", [
        ("Reads", ["<root>/<phase>/predictions/*.csv  (+ design.json, models.csv)",
                   "cache/runtime/  (costs.csv, <dataset>/<mode>/runs.csv, fidelity.csv)",
                   style("read only -- nothing is trained, scored or fitted", "dim")]),
        ("Scope", [f"headlines on scope '{scope}'" + ("  (dead units removed)" if scope == "active" else "")]),
        ("Filter", [flt.describe()]),
    ], outputs=[(directory / "report.html", "every row, highlighted per phase, filterable"),
                (directory / "evaluation.log", "this output")])
    if legend:
        report.print_legend()
        report.print_glossary()

    results, missing = [], []
    for mode in modes:
        for phase in phases or PHASE_NAMES:
            result = acc.load_phase(phase, mode, dataset, flt)
            if result is None:
                missing.append(f"{mode}/{phase}")
                report.print_phase_missing(phase, mode, variant_predictions_dir(phase, mode, dataset))
            else:
                results.append(result)
                report.print_phase(result, scope)

    table = acc.overview_rows(results, scope)
    statistics = acc.statistics_table(results)
    if table:
        report.print_overview(table, scope)
        report.print_mode_comparison(table)
        report.print_statistics(results, scope)
        report.print_caveats(results, missing, scope, flt)
        if write:
            for mode in modes:
                path = acc.write_summary([r for r in table if r["mode"] == mode], mode, dataset, scope)
                if path:
                    plain(report.logger, "  " + style(f"▸ {display_path(path)}", "yellow")
                          + style("   the overview as CSV, for tables in the paper", "dim"))
    else:
        plain(report.logger, "")
        plain(report.logger, style("Nothing to summarise yet. Run make light (or make phase) first.", "yellow"))

    if write and statistics:
        path = acc.write_statistics(statistics, directory / "statistics.csv")
        plain(report.logger, "  " + style(f"▸ {display_path(path)}", "yellow")
              + style("   count, undefined, min, quartiles, mean, max, sd, range per group", "dim"))

    summaries = []
    cost_rows = cost_data.collect([m for m in modes if m != "debug"], [dataset])
    if write:
        cost_data.write(cost_data.collect(cost_data.MODES, cost_data.DATASETS))
    if runtime:
        plain(report.logger, "")
        plain(report.logger, style("── Cost of the whole run " + "─" * 53, "bold", "cyan"))
        report.print_costs(cost_rows, modes, [dataset])
        tools = _tools_for(flt)
        for mode in modes:
            summary = runtime_data.summarise(mode, dataset, scope, tools=tools)
            report.print_runtime(summary, mode, dataset, runtime_root(mode, dataset) / "runs.csv")
            if summary:
                summaries.append(summary)
                if write:
                    for path in runtime_data.write_summary(summary):
                        plain(report.logger, "  " + style(f"▸ {display_path(path)}", "yellow"))

    if html:
        path = html_report.write_report(
            directory / "report.html", dataset=dataset, modes=modes, scope=scope, results=results,
            overview=table, runtime=summaries, costs=cost_rows, statistics=statistics,
            filter_text=flt.describe(), missing=missing)
        plain(report.logger, "")
        plain(report.logger, "  " + style(f"▸ {display_path(path)}", "yellow")
              + style("   open in a browser: every row, highlighted, with filters", "dim"))
    plain(report.logger, "")


def _tools_for(flt: acc.ResultFilter):
    """The runtime tools a predictor/model-type filter keeps (dynamic tools always stay)."""
    if not (flt.predictors or flt.model_types):
        return ()
    keep = [t for t in runtime_data.DYNAMIC_TOOLS]
    for tool in runtime_data.STATIC_TOOLS:
        predictor = tool[len("staticsbfl_"):]
        if flt.predictors and predictor not in flt.predictors:
            continue
        if flt.model_types and acc.PREDICTOR_TYPES.get(predictor) not in flt.model_types:
            continue
        keep.append(tool)
    return tuple(keep)


if __name__ == "__main__":
    main()
