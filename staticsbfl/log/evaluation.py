"""Console and log rendering of the final evaluation.

Every function here takes the plain data of :mod:`staticsbfl.evaluation` and prints
it; none of them reads a file or computes a figure that is not already in that
data. What is printed is the essential part -- headlines, overview, caveats,
costs, the runtime comparison; the HTML report (:mod:`staticsbfl.viz.html_report`)
lists every row.
"""

import logging
from typing import List, Optional, Sequence

from staticsbfl.benchmark.footprint import MEASUREMENT_NOTE
from staticsbfl.evaluation import accuracy as acc
from staticsbfl.evaluation import costs as cost_data
from staticsbfl.evaluation.runtime import SHORT_LABELS, TOOL_LABELS, RuntimeSummary
from staticsbfl.log.pipeline import PHASES, phase_title
from staticsbfl.log.console import display_path, format_duration, human_bytes, log_table, plain, style, wrap_text

logger = logging.getLogger("evaluate")


# --------------------------------------------------------------------------
# Formatting of single values
# --------------------------------------------------------------------------


def fmt(value: Optional[float], digits: int = 3, sign: bool = False) -> str:
    """A number with *digits* decimals, ``n/a`` for undefined."""
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def fmt_seconds(value) -> str:
    """Seconds as ms or a duration."""
    if value is None:
        return "-"
    return f"{value * 1000:.0f} ms" if value < 1 else format_duration(value)


def fmt_bytes(value) -> str:
    """Bytes in human units."""
    return "-" if value is None else human_bytes(int(value))


# --------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------


def print_legend(log: logging.Logger = logger) -> None:
    """Every reported quantity: range, better direction, meaning."""
    plain(log, style("── What the numbers mean " + "─" * 53, "bold"))
    plain(log, style(f"  {'quantity':<14}{'range':<14}{'better':<13}meaning", "bold"))
    rows = [(q.label, q.range, q.better, q.meaning) for q in acc.QUANTITIES] + list(acc.EXTRA_TERMS)
    for label, value_range, better, meaning in rows:
        wrapped = wrap_text(meaning, 50)
        plain(log, f"  {label:<14}{value_range:<14}{better:<13}{wrapped[0]}")
        for line in wrapped[1:]:
            plain(log, f"  {'':<41}{line}")
    plain(log, "")


def headline(best: acc.Best, label: str) -> str:
    """The one green line: value, case, and what the baselines reach on it."""
    row = best.row
    if best.column in ("top_k", "top_k_05"):
        chance = row.get(f"{best.column}_chance")
        factor = f", {best.value / chance:.1f}x chance" if chance else ""
        value = f"{fmt(best.value)}  (chance {fmt(chance)}{factor})"
    else:
        sd_key = {"spearman_model": "spearman_model_sd", "weightedtau_model": "weightedtau_model_sd"}.get(best.column)
        sd = row.get(sd_key) if sd_key else None
        value = fmt(best.value) + (f" ± {fmt(sd)}" if sd is not None else "")
    text = f"★ best {label:<10} {value:<34} {best.case}"
    tail = f"   (layer_mean {fmt(best.layer_mean)} · random {fmt(best.random)} · margin {fmt(best.margin, sign=True)})"
    return style(text, "green", "bold") + style(tail, "dim")


def print_phase_missing(phase: str, mode: str, hint_path) -> None:
    """The banner of a phase that has no results yet, with its command."""
    title = phase_title(phase)
    plain(logger, "")
    plain(logger, style(f"── {title} " + "─" * max(3, 74 - len(title)), "bold", "cyan"))
    plain(logger, style(f"  not computed yet -- nothing in {display_path(hint_path)}/. Run: make phase "
                        f"PHASE={phase}" + ("" if mode == "light" else f" VMODE={mode}"), "yellow"))


def print_phase(result: acc.PhaseResult, scope: str) -> None:
    """Per population the best rho/model, wtau and top-k, and per metric; then the phase best."""
    title = phase_title(result.phase)
    plain(logger, "")
    plain(logger, style(f"── {title} · {result.mode} " + "─" * max(3, 70 - len(title) - len(result.mode)), "bold", "cyan"))
    design = result.design
    fit = design.get("fit_population", "?")
    fit_models = design.get("populations", {}).get(fit, {}).get("n_models", "?")
    plain(logger, f"  question  {design.get('question', PHASES[result.phase].reading)}")
    plain(logger, f"  fitted on '{fit}' ({fit_models} models) · read as: {PHASES[result.phase].reading}")
    if not result.complete:
        plain(logger, style(f"  ! only {len(result.files)} of {result.expected_files} result files -- the predict "
                            f"step did not finish. Rerun: make predict PHASE={result.phase}", "yellow"))
    for population in result.populations:
        sample = next((r for r in result.rows if r["population"] == population and r["scope"] == scope), None)
        if sample is None:
            continue
        dead = sample.get("inactive_fraction")
        accuracy = result.accuracy.get(population)
        context = (f"{sample['n_models']} models" + (f" · {100 * dead:.1f}% dead units" if dead is not None else "")
                   + (f" · mean accuracy {accuracy:.3f}" if accuracy is not None else ""))
        plain(logger, "")
        plain(logger, "  " + style(f"population {population}", "bold") + style(f"   {context}", "dim"))
        for column, label in acc.HEADLINE_COLUMNS:
            best = acc.best_of(result.rows, column, population=population, scope=scope)
            if best:
                plain(logger, "    " + headline(best, label))
        per_metric = acc.best_per_metric(result.rows, population, scope)
        plain(logger, "  " + style("  per metric (best rho/model): "
                                   + " · ".join(f"{m} {fmt(v)}" for m, v in per_metric.items()), "dim"))
    if len(result.populations) > 1:
        plain(logger, "")
        plain(logger, "  " + style("over the whole phase", "bold"))
        for column, label in acc.HEADLINE_COLUMNS:
            best = acc.best_of(result.rows, column, population=None, scope=scope)
            if best:
                plain(logger, "    " + headline(best, label) + style(f"   ← population '{best.row['population']}'", "yellow"))


def print_overview(table: Sequence[dict], scope: str) -> None:
    other = "all" if scope == "active" else "active"
    plain(logger, "")
    plain(logger, style("── Overview: best fitted predictor per population " + "─" * 28, "bold", "cyan"))
    rows = []
    for r in table:
        factor = r.get("top_k_x_chance")
        rows.append([r["mode"], r["phase"], r["population"], fmt(r["best_rho_model"]), fmt(r["layer_mean_rho"]),
                     fmt(r["margin_over_layer_mean"], 2, sign=True), fmt(r.get(f"best_rho_model_{other}")),
                     fmt(r["best_wtau"]), fmt(r["best_top_k"]), f"{factor:.1f}x" if factor else "n/a", r["n_models"]])
    log_table(logger, ["mode", "phase", "population", "rho/model", "layer_mean", "margin", f"rho {other}",
                       "wtau", "top-k", "×chance", "models"], rows, align="lll" + "r" * 8, indent="  ",
              highlight=[(r["margin_over_layer_mean"] or 0) <= 0 for r in table])
    plain(logger, style("  highlighted: margin <= 0 -- the predictor knows no more than the layer", "dim"))


def print_mode_comparison(table: Sequence[dict]) -> None:
    """Light next to heavy: best rho/model per (phase, population) in both modes."""
    order = ("light", "heavy", "debug")
    present = {r["mode"] for r in table}
    modes = [m for m in order if m in present] + sorted(present - set(order))
    if len(modes) < 2:
        return
    plain(logger, "")
    plain(logger, style("── Light vs heavy: best rho/model per population " + "─" * 29, "bold", "cyan"))
    cases = sorted({(r["phase"], r["population"]) for r in table})
    rows = []
    for phase, population in cases:
        values = {r["mode"]: r["best_rho_model"] for r in table if r["phase"] == phase and r["population"] == population}
        diff = values[modes[1]] - values[modes[0]] if all(m in values for m in modes) else None
        rows.append([phase, population] + [fmt(values.get(m)) for m in modes] + [fmt(diff, sign=True)])
    log_table(logger, ["phase", "population"] + modes + [f"{modes[1]} - {modes[0]}"], rows, align="ll" + "r" * (len(modes) + 1), indent="  ")


def print_caveats(results: Sequence[acc.PhaseResult], missing: Sequence[str], scope: str,
                  flt: acc.ResultFilter = acc.NO_FILTER) -> None:
    plain(logger, "")
    plain(logger, style("── Read these numbers with ─" + "─" * 50, "bold"))
    counts = sorted({acc.n_candidates(r.rows, p, scope) for r in results for p in r.populations})
    uncertainty = acc.typical_uncertainty(results, scope)
    notes = [
        f"Each 'best' is the maximum over {'/'.join(map(str, counts)) or '?'} fitted cases "
        "(predictors x targets). A maximum over many candidates is optimistic; quote "
        "the per-metric lines or one pre-chosen predictor beside it.",
        "Tarantula, Ochiai and D* are separate results. The best case usually comes from "
        "one of them; the per-metric line shows how far the others are behind.",
        "Phase 1 is the reference: a predictor cannot be more consistent across models "
        "than suspiciousness itself is across reseeding. Read phases 2-4 against it.",
        "The margin over layer_mean, not the raw rho, says whether a predictor learned "
        "anything about the neuron rather than about its layer.",
        "sd is the spread across evaluation models; the uncertainty of a mean is about 2·sd/sqrt(n) ("
        + (f"typically ±{uncertainty[0]:.3f} here: sd {uncertainty[1]:.2f} over {uncertainty[2]} models"
           if uncertainty else "no sd recorded") + "). Differences inside that band are noise.",
        "Dense MLP populations only: the study has no convolutional or recurrent variants.",
    ]
    if flt.active:
        notes.append(f"FILTERED: {flt.describe()}. Baselines are kept on every remaining case.")
    if missing:
        notes.append(f"Not computed yet: {', '.join(missing)} -- the overview covers the rest only.")
    for note in notes:
        wrapped = wrap_text(note, 74)
        plain(logger, f"  • {wrapped[0]}")
        for line in wrapped[1:]:
            plain(logger, f"    {line}")


# --------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------


def print_costs(rows: List[dict], modes: Sequence[str], datasets: Sequence[str]) -> None:
    """Per mode and dataset: one line per phase, then the totals and every flag."""
    for mode in modes:
        for dataset in datasets:
            summary = cost_data.summarise_mode(rows, mode, dataset)
            if not summary["recorded"]:
                plain(logger, style(f"\n  {mode} · {dataset}: nothing recorded yet", "dim"))
                continue
            plain(logger, "")
            plain(logger, style(f"  {mode.upper()} · {dataset}", "bold"))
            table = []
            for entry in summary["phases"]:
                train, steps = entry["train"], entry["steps"]
                if train.get("status") in ("missing", "unreadable", "error"):
                    table.append([entry["phase"], "-", "-", "-", train.get("status"), "", "", "", "", "", "", ""])
                    continue
                table.append([
                    entry["phase"], train.get("n_models") or 0, train.get("n_timed") or 0, train.get("epochs") or "-",
                    fmt_seconds(train.get("total_seconds")), fmt_seconds(train.get("mean_seconds")),
                    fmt_seconds(train.get("sd_seconds")), fmt_seconds(train.get("seconds_per_model_epoch")),
                    fmt_seconds(steps.get("score", {}).get("total_seconds")),
                    fmt_seconds(steps.get("neurons", {}).get("total_seconds")),
                    fmt_seconds(steps.get("predict", {}).get("total_seconds")),
                    fmt_seconds(steps.get("predict", {}).get("fit_seconds")),
                ])
            totals = summary["totals"]
            table.append(["total", summary["n_models"], summary["n_timed"], "", fmt_seconds(totals["train"]),
                          fmt_seconds(summary["mean_train_seconds"]), "", "", fmt_seconds(totals["score"]),
                          fmt_seconds(totals["neurons"]), fmt_seconds(totals["predict"]),
                          fmt_seconds(totals["fit"] or None)])
            log_table(logger, ["phase", "models", "timed", "epochs", "train total", "per model", "sd", "per m-ep",
                               "score", "neurons", "predict", "(fit)"], table, indent="  ",
                      highlight=[False] * (len(table) - 1) + [True])
            plain(logger, f"  whole {mode} run on {dataset}: {format_duration(summary['grand_total_seconds'])} "
                          "measured (train + score + neurons + predict; untimed models not included)")
            for flag in summary["flags"]:
                plain(logger, style(f"  ! {flag}", "yellow"))


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------


TIME_NOTE = (
    "Time: wall clock (perf_counter) per phase inside each process; accelerator work is "
    "synchronised at every phase boundary. 'per model' = load the model + analyse it "
    "(static: features + predict; dynamic: forward passes + counting + scoring); 'setup' = "
    "import + test data or predictor load, paid once per session. A value per model is the "
    "median over its repeats; 'median'/'max' are then taken over the models."
)


def print_measurement_note(environment: dict) -> None:
    plain(logger, style("  How this was measured", "bold"))
    if environment:
        plain(logger, f"    {environment.get('cpu', '?')} · {environment.get('platform', '?')} · "
                      f"{environment.get('logical_cpus', '?')} logical CPUs · python {environment.get('python', '?')}")
        for key, value in environment.items():
            if key.startswith("versions_"):
                plain(logger, f"    {key[len('versions_'):]}: {value}")
    for line in wrap_text(TIME_NOTE + " " + MEASUREMENT_NOTE):
        plain(logger, style(f"    {line}", "dim"))


def print_runtime(summary: Optional[RuntimeSummary], mode: str, dataset: str, hint_path) -> None:
    plain(logger, "")
    plain(logger, style(f"  ═══ Runtime and memory · phase 4 designed models · {mode} · {dataset} ═══", "bold"))
    if summary is None:
        plain(logger, style(f"  No runtime runs at {display_path(hint_path)}. Run make runtime VMODE={mode} "
                            f"DATASET={dataset} first.", "yellow"))
        return
    print_measurement_note(summary.environment)

    plain(logger, "")
    plain(logger, style("  Per tool (median over models; max with its model)", "bold"))
    log_table(logger, ["tool", "models", "per model", "max", "total", "setup", "mem/model", "max",
                       "process", "gpu", "runs"],
              [[t.label, t.models, fmt_seconds(t.seconds_median), fmt_seconds(t.seconds_max),
                fmt_seconds(t.seconds_total), fmt_seconds(t.setup_median), fmt_bytes(t.memory_median),
                fmt_bytes(t.memory_max), fmt_bytes(t.process_median), fmt_bytes(t.gpu_median),
                t.as_dict()["status"]] for t in summary.tools], align="l" + "r" * 9 + "l")
    for t in summary.tools:
        if t.seconds_max_model:
            plain(logger, style(f"    {t.label}: slowest model {t.seconds_max_model}, "
                                f"largest memory {t.memory_max_model}", "dim"))
        for row in summary.runs:
            if row["tool"] == t.tool and row.get("status") in ("failed", "timeout", "unreadable"):
                plain(logger, style(f"      ! {row['model']} r{row['repeat']}: {row['status']} {row.get('detail', '')[:120]}", "yellow"))
    plain(logger, style("    mem/model = footprint above the set-up process while one model is analysed; "
                        "process = whole-process high-water mark; gpu = MPS allocator share (sampled)", "dim"))

    if summary.designs:
        plain(logger, "")
        plain(logger, style("  Per design: median per-model time over its seeds", "bold"))
        tools = [t.tool for t in summary.tools]
        log_table(logger, ["design", "neurons"] + [SHORT_LABELS.get(t, t) for t in tools],
                  [[d["design"], d["neurons"]] + [fmt_seconds(d.get(t)) for t in tools] for d in summary.designs])

    if summary.speedups:
        plain(logger, "")
        plain(logger, style("  Speed-up per model (dynamic / static time on the same model): geometric mean [min-max]", "bold"))
        log_table(logger, ["static", "vs dynamic", "speed-up", "range", "models"],
                  [[TOOL_LABELS[s.static], TOOL_LABELS[s.dynamic], f"{s.geometric_mean:.1f}x",
                    f"{s.minimum:.1f}x - {s.maximum:.1f}x", s.models] for s in summary.speedups],
                  align="llrrr", highlight=[s.geometric_mean < 1 for s in summary.speedups])
        plain(logger, style("    highlighted: the static predictor is SLOWER per model than this dynamic tool", "dim"))

    cost = summary.cost
    plain(logger, "")
    plain(logger, style("  One-time cost of the static approach (phase 4)", "bold"))
    for label, key in (("train fit population", "train_seconds"), ("score it (pro rata)", "score_seconds"),
                       ("neuron table", "neurons_seconds")):
        plain(logger, f"    {label:<28}{fmt_seconds(cost.get(key)):>12}")
    if cost.get("train_note"):
        plain(logger, style(f"    ! {cost['train_note']}", "yellow"))
    for entry in summary.amortization:
        if entry.dynamic is None:
            plain(logger, f"    {TOOL_LABELS[entry.static]}: " + style("phase-4 predict or training not recorded yet", "yellow"))
            continue
        label = "whole predict step (fit not recorded)" if entry.fit_is_upper_bound_of_predict else "fit, all fitted targets"
        if entry.models_to_pay_off is None:
            verdict = "never pays off (not faster per model)"
        else:
            verdict = (f"saves {fmt_seconds(entry.saving_per_model)}/model -> pays off after {entry.models_to_pay_off:,} models "
                       f"({entry.models_to_pay_off_without_training:,} if the fit population was trained anyway)")
        plain(logger, f"    {TOOL_LABELS[entry.static]} vs {TOOL_LABELS[entry.dynamic]}: one-time "
                      f"{fmt_seconds(entry.one_time_seconds)} ({label} {fmt_seconds(entry.fit_seconds)}, upper bound) · {verdict}")

    if summary.accuracy:
        plain(logger, "")
        plain(logger, style("  What the static ranking is worth on the same models (rho/model, designed, rank_model)", "bold"))
        for tool, values in summary.accuracy.items():
            plain(logger, f"    {TOOL_LABELS[tool]:<32}" + "  ".join(f"{m} {fmt(v)}" for m, v in values.items()))
        plain(logger, style("    the dynamic tools are the reference: they compute the spectrum itself", "dim"))

    if summary.fidelity:
        plain(logger, "")
        plain(logger, style("  Fidelity: does every dynamic tool compute the same spectrum? (same model, same data)", "bold"))
        log_table(logger, ["tool", "reference", "identical", "neurons off", "max diff", "min rho T / O / D*"],
                  [[TOOL_LABELS.get(f.tool, f.tool), f.reference, f"{f.identical}/{f.models}",
                    f"{f.neurons_differing}/{f.neurons}", f.max_count_difference,
                    " / ".join("-" if v is None else f"{v:.7f}" for v in f.min_rho.values())] for f in summary.fidelity],
                  align="llrrrr")
        plain(logger, style("    a difference of 1 count is a pre-activation within float rounding of the threshold "
                            "(other device or framework); ICSoft@pt counts Ochiai with its own (targeted) formula", "dim"))


# --------------------------------------------------------------------------
# Glossary and statistics
# --------------------------------------------------------------------------


def print_glossary() -> None:
    """What the populations, scopes, modes, predictors, metrics and targets are."""
    from staticsbfl.evaluation.glossary import LOG_KINDS, TERMS

    plain(logger, style("── What the names mean " + "─" * 55, "bold"))
    for kind, heading in LOG_KINDS:
        plain(logger, style(f"  {heading}", "bold"))
        for value, text in TERMS[kind].items():
            wrapped = wrap_text(text, 56)
            plain(logger, f"    {value:<20}{wrapped[0]}")
            for line in wrapped[1:]:
                plain(logger, f"    {'':<20}{line}")
    plain(logger, "")


def print_statistics(results: Sequence[acc.PhaseResult], scope: str, column: str = "spearman_model") -> None:
    """Per mode and phase: rho/model statistics by model type -- the spread behind each 'best'."""
    plain(logger, "")
    plain(logger, style(f"── Spread of {column} by model type (scope {scope}) " + "─" * 30, "bold", "cyan"))
    rows = []
    for result in results:
        subset = [r for r in result.rows if r["scope"] == scope]
        for entry in acc.group_statistics(subset, column, "model_type"):
            rows.append([result.mode, result.phase, entry["group"], entry["rows"], entry["undefined"],
                         fmt(entry["min"]), fmt(entry["median"]), fmt(entry["mean"]), fmt(entry["max"]), fmt(entry["sd"])])
    log_table(logger, ["mode", "phase", "model type", "rows", "undef", "min", "median", "mean", "max", "sd"],
              rows, align="lll" + "r" * 7, indent="  ", highlight=[r[2] == "*" for r in rows])
    plain(logger, style("  every row of a phase in exactly one model type; '*' = all of them. Full table: "
                        "statistics.csv and the Statistics tab of the report", "dim"))
