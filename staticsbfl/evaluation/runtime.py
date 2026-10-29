"""The runtime comparison as data: records, aggregates, speed-ups, amortization.

Two halves:

**Records** -- what one measured process wrote (``<key>__r<k>.json`` from
:class:`staticsbfl.benchmark.footprint.Recorder`, plus ``.status.json`` from the
orchestrator) becomes one flat row (:func:`flatten`); the per-model cost is
split from the one-off session setup (:func:`per_model_summary`); two spectra
are compared count by count (:func:`compare_spectra`).

**Aggregates** -- from those rows: per tool the median and the maximum over the
models (each model's value is the median over its repeats), per design, the
speed-up of every static predictor over every dynamic tool (geometric mean,
min, max over models), the one-time cost of the static approach and the number
of analysed models after which it pays off. All of it is plain data, so a test
can check every figure against a hand-made table; printing is
:mod:`staticsbfl.log.evaluation`, the HTML is :mod:`staticsbfl.viz.html_report`.
"""

import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from staticsbfl.utils.paths import runtime_root, variant_predictions_dir

PHASE = "phase4"
EVAL_SPLIT = "designed"
METRICS = ("tarantula", "ochiai", "dstar")

#: Every tool the benchmark knows, in reporting order, with its printed name.
TOOL_LABELS: Dict[str, str] = {
    "staticsbfl_gradient_boosting": "StaticSBFL (gradient boosting)",
    "staticsbfl_mlp": "StaticSBFL (MLP)",
    "hs_pt_cpu": "HS@pt (this repository), CPU",
    "hs_pt_mps": "HS@pt (this repository), MPS",
    "icsoft_pt_cpu": "ICSoft@pt, CPU",
    "icsoft_pt_mps": "ICSoft@pt, MPS",
    "deepfault_tf1": "DF@tf1 (graph mode)",
    "deepfault_tf2": "DF@tf2 (eager)",
}
#: Column headers for narrow tables.
SHORT_LABELS: Dict[str, str] = {
    "staticsbfl_gradient_boosting": "static GB", "staticsbfl_mlp": "static MLP",
"hs_pt_cpu": "HS CPU", "hs_pt_mps": "HS MPS",
    "icsoft_pt_cpu": "ICSoft CPU", "icsoft_pt_mps": "ICSoft MPS",
    "deepfault_tf1": "DF@tf1", "deepfault_tf2": "DF@tf2",
}
STATIC_TOOLS = tuple(t for t in TOOL_LABELS if t.startswith("staticsbfl_"))
DYNAMIC_TOOLS = tuple(t for t in TOOL_LABELS if not t.startswith("staticsbfl_"))

#: Phases a record may contain, in reporting order.
PHASE_COLUMNS = (
    "import", "data", "predictor_load", "model_load", "test_model",
    "collect", "spectrum", "score", "score_tarantula", "score_ochiai", "score_dstar",
    "features", "predict",
)

#: Paid once per session, not per model: a deployed static predictor is loaded
#: once and applied to many models; a dynamic tool loads its framework and the
#: test set once. Speed-up and amortization use the per-model cost only.
SETUP_PHASES = ("import", "data", "predictor_load")

FIDELITY_COLUMNS = (
    "tool", "model", "reference", "n_neurons", "neurons_differing", "max_count_difference",
    "spearman_tarantula", "spearman_ochiai", "spearman_dstar", "identical",
)


def number(value) -> Optional[float]:
    """A CSV cell as a float; blank, nan or text means undefined (``None``)."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def design_of(key: str) -> str:
    """``p4_designed_wide_mlp_s2`` -> ``wide_mlp``."""
    return re.sub(r"_s\d+$", "", key.replace("p4_designed_", ""))


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


def per_model_summary(phases: Sequence[dict]) -> dict:
    """Seconds and memory of the per-model phases of one record.

    The memory figure is the highest footprint reached during those phases
    (sampled peak, or the exact high-water mark where it rose within them)
    minus the footprint before the first of them: what analysing one more model
    costs on top of a process that is already set up.
    """
    analysis = [p for p in phases if p["name"] not in SETUP_PHASES]
    setup = [p for p in phases if p["name"] in SETUP_PHASES]
    if not analysis:
        return {}
    start = analysis[0]["bytes_before"]
    peak = max(max(p["sampled_peak_bytes"], p["lifetime_max_after_bytes"] if p["lifetime_max_increase_bytes"] else 0)
               for p in analysis)
    return {
        "setup_seconds": sum(p["seconds"] for p in setup),
        "per_model_seconds": sum(p["seconds"] for p in analysis),
        "per_model_start_bytes": start,
        "per_model_peak_above_start_bytes": max(0, peak - start),
    }


RECORD_FIELDS = (
    "n_neurons", "n_layers", "n_samples", "accuracy", "memory_source",
    "process_start_bytes", "process_lifetime_max_bytes", "process_max_rss_bytes",
    "wall_seconds_in_runner", "estimator_bytes_on_disk", "selected_class",
    "tensorflow", "torch", "sklearn", "python", "executing_eagerly", "torch_threads", "exact_malloc",
)


def flatten(tool: str, key: str, repeat: int, directory: Path) -> dict:
    """One CSV row from a run's record and status files."""
    row = {"tool": tool, "model": key, "repeat": repeat}
    status_path = directory / f"{key}__r{repeat}.status.json"
    record_path = directory / f"{key}__r{repeat}.json"
    try:
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
    except (OSError, ValueError):
        status = {"status": "unreadable"}
    row["status"] = status.get("status", "ok" if record_path.exists() else "missing")
    row["detail"] = status.get("detail", "")
    row["process_wall_seconds"] = status.get("process_wall_seconds")
    if not record_path.exists():
        return row
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError) as error:
        row.update({"status": "unreadable", "detail": f"{type(error).__name__}: {error}"})
        return row
    for name in RECORD_FIELDS:
        row[name] = record.get(name)
    gpu = record.get("gpu_reading") or {}
    row["gpu_peak_above_baseline_bytes"] = gpu.get("peak_above_baseline_bytes")
    row["gpu_driver_delta_bytes"] = gpu.get("driver_delta_bytes")
    row.update(per_model_summary(record.get("phases", [])))
    for phase in record.get("phases", []):
        name = phase["name"]
        row[f"{name}_seconds"] = phase["seconds"]
        row[f"{name}_peak_bytes"] = phase["lifetime_max_increase_bytes"]
        row[f"{name}_sampled_bytes"] = phase["sampled_peak_above_start_bytes"]
    return row


def run_columns() -> List[str]:
    """The column order of ``runs.csv``."""
    base = ["tool", "model", "repeat", "status", "detail", "process_wall_seconds", *RECORD_FIELDS,
            "gpu_peak_above_baseline_bytes", "gpu_driver_delta_bytes",
            "setup_seconds", "per_model_seconds", "per_model_start_bytes", "per_model_peak_above_start_bytes"]
    for phase in PHASE_COLUMNS:
        base += [f"{phase}_seconds", f"{phase}_peak_bytes", f"{phase}_sampled_bytes"]
    return base


def compare_spectra(got: Dict[str, List], ref: Dict[str, List]) -> dict:
    """Count and score agreement of two spectra given as per-layer arrays.

    Both arguments map ``a_f``, ``a_s`` and every metric to a list of per-layer
    arrays. The result counts neurons whose counts differ at all, the largest
    count difference, and Spearman rho per metric (``None`` where one side is
    constant, i.e. rho is undefined).
    """
    import numpy as np
    from scipy.stats import spearmanr

    differing, max_difference, n = 0, 0, 0
    for got_af, got_as, ref_af, ref_as in zip(got["a_f"], got["a_s"], ref["a_f"], ref["a_s"]):
        got_af, got_as = np.asarray(got_af, dtype=np.int64), np.asarray(got_as, dtype=np.int64)
        ref_af, ref_as = np.asarray(ref_af, dtype=np.int64), np.asarray(ref_as, dtype=np.int64)
        n += len(ref_af)
        differing += int(((got_af != ref_af) | (got_as != ref_as)).sum())
        max_difference = max(max_difference, int(np.abs(got_af - ref_af).max()), int(np.abs(got_as - ref_as).max()))
    row = {"n_neurons": n, "neurons_differing": differing, "max_count_difference": max_difference,
           "identical": differing == 0}
    for metric in METRICS:
        if metric not in got or metric not in ref:
            row[f"spearman_{metric}"] = None
            continue
        a = np.concatenate([np.asarray(x, dtype=np.float64) for x in got[metric]])
        b = np.concatenate([np.asarray(x, dtype=np.float64) for x in ref[metric]])
        finite_a, finite_b = a[np.isfinite(a)], b[np.isfinite(b)]
        if not len(finite_a) or not len(finite_b) or np.ptp(finite_a) == 0 or np.ptp(finite_b) == 0:
            row[f"spearman_{metric}"] = None
            continue
        row[f"spearman_{metric}"] = float(spearmanr(np.nan_to_num(a, posinf=1e300), np.nan_to_num(b, posinf=1e300)).statistic)
    return row


# --------------------------------------------------------------------------
# Reading the tables
# --------------------------------------------------------------------------


def read_csv(path: Path) -> List[dict]:
    """Every row of a CSV, or ``[]`` if it is missing or broken."""
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def read_runs(mode: str, dataset: str) -> List[dict]:
    """``runs.csv`` of one runtime run."""
    return read_csv(runtime_root(mode, dataset) / "runs.csv")


def read_fidelity(mode: str, dataset: str) -> List[dict]:
    """``fidelity.csv`` of one runtime run."""
    return read_csv(runtime_root(mode, dataset) / "fidelity.csv")


def read_environment(mode: str, dataset: str) -> dict:
    """``environment.json`` of one runtime run, or ``{}``."""
    try:
        return json.loads((runtime_root(mode, dataset) / "environment.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------
# Aggregates
# --------------------------------------------------------------------------


def per_model_medians(runs: Sequence[dict], column: str) -> Dict[str, Dict[str, float]]:
    """``{tool: {model: median over repeats of column}}``, successful runs only."""
    collected: Dict[str, Dict[str, List[float]]] = {}
    for row in runs:
        if row.get("status") not in ("ok", "cached"):
            continue
        value = number(row.get(column))
        if value is None:
            continue
        collected.setdefault(row["tool"], {}).setdefault(row["model"], []).append(value)
    return {tool: {model: statistics.median(v) for model, v in models.items()}
            for tool, models in collected.items()}


def geometric_mean(values: Sequence[float]) -> Optional[float]:
    """Geometric mean of the positive values, ``None`` if there are none."""
    values = [v for v in values if v and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def _median(values) -> Optional[float]:
    values = list(values)
    return statistics.median(values) if values else None


def _max(values) -> Optional[float]:
    values = list(values)
    return max(values) if values else None


def _mean(values) -> Optional[float]:
    values = list(values)
    return statistics.fmean(values) if values else None


def _min(values) -> Optional[float]:
    values = list(values)
    return min(values) if values else None


@dataclass
class ToolRow:
    """Per-tool aggregates over the models of one runtime run."""
    tool: str
    models: int
    seconds_mean: Optional[float]
    seconds_median: Optional[float]
    seconds_min: Optional[float]
    seconds_max: Optional[float]
    seconds_max_model: Optional[str]
    seconds_total: Optional[float]
    setup_median: Optional[float]
    memory_mean: Optional[float]
    memory_median: Optional[float]
    memory_min: Optional[float]
    memory_max: Optional[float]
    memory_max_model: Optional[str]
    process_median: Optional[float]
    process_max: Optional[float]
    gpu_median: Optional[float]
    status: Dict[str, int] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """The printed name of the tool."""
        return TOOL_LABELS.get(self.tool, self.tool)

    def as_dict(self) -> dict:
        """The row as plain data, status flattened to text."""
        data = dict(vars(self))
        data["status"] = ", ".join(f"{n} {k}" for k, n in sorted(self.status.items()))
        data["label"] = self.label
        return data


def _argmax(values: Dict[str, float]) -> Optional[str]:
    return max(values, key=values.get) if values else None


def tool_rows(runs: Sequence[dict]) -> List[ToolRow]:
    """Per tool: mean, median, minimum and maximum over models (each model = median of its repeats).

    The mean and the median are both kept because they differ here: a tool whose
    cost grows with the model (DeepFault) has a mean well above its median.
    """
    seconds = per_model_medians(runs, "per_model_seconds")
    setup = per_model_medians(runs, "setup_seconds")
    memory = per_model_medians(runs, "per_model_peak_above_start_bytes")
    process = per_model_medians(runs, "process_lifetime_max_bytes")
    gpu = per_model_medians(runs, "gpu_peak_above_baseline_bytes")
    status: Dict[str, Dict[str, int]] = {}
    for row in runs:
        bucket = status.setdefault(row["tool"], {})
        bucket[row.get("status", "?")] = bucket.get(row.get("status", "?"), 0) + 1
    present = {row["tool"] for row in runs}
    ordered = [t for t in TOOL_LABELS if t in present] + sorted(present - set(TOOL_LABELS))
    out = []
    for tool in ordered:
        s, m = seconds.get(tool, {}), memory.get(tool, {})
        out.append(ToolRow(
            tool=tool, models=len(s),
            seconds_mean=_mean(s.values()), seconds_median=_median(s.values()),
            seconds_min=_min(s.values()), seconds_max=_max(s.values()), seconds_max_model=_argmax(s),
            seconds_total=sum(s.values()) if s else None,
            setup_median=_median(setup.get(tool, {}).values()),
            memory_mean=_mean(m.values()), memory_median=_median(m.values()),
            memory_min=_min(m.values()), memory_max=_max(m.values()), memory_max_model=_argmax(m),
            process_median=_median(process.get(tool, {}).values()),
            process_max=_max(process.get(tool, {}).values()),
            gpu_median=_median(gpu.get(tool, {}).values()),
            status=status.get(tool, {}),
        ))
    return out


def by_design(runs: Sequence[dict]) -> List[dict]:
    """Per design (architecture): median per-model time of every tool over its seeds."""
    seconds = per_model_medians(runs, "per_model_seconds")
    neurons: Dict[str, int] = {}
    for row in runs:
        value = number(row.get("n_neurons"))
        if value:
            neurons[design_of(row["model"])] = int(value)
    designs = sorted({design_of(m) for models in seconds.values() for m in models},
                     key=lambda d: (neurons.get(d, 0), d))
    rows = []
    for design in designs:
        row = {"design": design, "neurons": neurons.get(design)}
        for tool, models in seconds.items():
            row[tool] = _median(v for m, v in models.items() if design_of(m) == design)
        rows.append(row)
    return rows


@dataclass
class Speedup:
    """Dynamic / static per-model time over common models."""
    static: str
    dynamic: str
    geometric_mean: float
    minimum: float
    maximum: float
    models: int


def speedups(runs: Sequence[dict]) -> List[Speedup]:
    """Dynamic per-model time / static per-model time, on the models both measured."""
    seconds = per_model_medians(runs, "per_model_seconds")
    out = []
    for static in [t for t in STATIC_TOOLS if t in seconds]:
        for dynamic in [t for t in DYNAMIC_TOOLS if t in seconds]:
            common = sorted(set(seconds[static]) & set(seconds[dynamic]))
            ratios = [seconds[dynamic][m] / seconds[static][m] for m in common if seconds[static][m] > 0]
            if ratios:
                out.append(Speedup(static, dynamic, geometric_mean(ratios), min(ratios), max(ratios), len(ratios)))
    return out


def one_time_cost(mode: str, dataset: str, cost_rows: Optional[Sequence[dict]] = None) -> Dict[str, object]:
    """Seconds the static approach spends once, before it predicts anything.

    Training and scoring the phase-4 fit population ('sampled'), building the
    neuron table, and fitting the predictor. The score and neurons steps are
    timed per phase, so the fit population's share of the score step is taken
    pro rata by model count. Fit time per predictor exists for runs since fit
    timing was added; before it, the whole predict step stands in as an upper
    bound. ``make predict`` fits both target kinds, the deployed predictor needs
    one, so the fit time is an upper bound in any case.
    """
    if cost_rows is None:
        from staticsbfl.evaluation import costs
        cost_rows = costs.collect([mode], [dataset])
    mine = [r for r in cost_rows if r.get("phase") == PHASE and r.get("mode", mode) == mode
            and r.get("dataset", dataset) == dataset]
    train = next((r for r in mine if r["step"] == "train" and r["population"] == "sampled"), {})
    everything = next((r for r in mine if r["step"] == "train" and r["population"] == "*"), {})
    steps = {r["step"]: r for r in mine if r["step"] in ("score", "neurons", "predict")}
    share = None
    if train.get("n_models") and everything.get("n_models"):
        share = number(train["n_models"]) / number(everything["n_models"])
    score = number(steps.get("score", {}).get("total_seconds"))
    predict = steps.get("predict", {})
    by_predictor = predict.get("fit_seconds_by_predictor") or {}
    if isinstance(by_predictor, str):
        try:
            by_predictor = json.loads(by_predictor)
        except ValueError:
            by_predictor = {}
    return {
        "train_seconds": number(train.get("total_seconds")),
        "train_note": train.get("note", "") or "",
        "score_seconds": score * share if score is not None and share else None,
        "neurons_seconds": number(steps.get("neurons", {}).get("total_seconds")),
        "predict_seconds": number(predict.get("total_seconds")),
        "fit_by_predictor": {k: number(v) for k, v in by_predictor.items()},
        "n_fit_models": train.get("n_models"),
    }


@dataclass
class Amortization:
    """One-time cost of a static predictor and its break-even point against one dynamic tool."""
    static: str
    dynamic: Optional[str]
    one_time_seconds: Optional[float]
    fit_seconds: Optional[float]
    fit_is_upper_bound_of_predict: bool
    saving_per_model: Optional[float] = None
    models_to_pay_off: Optional[int] = None
    models_to_pay_off_without_training: Optional[int] = None


def amortization(runs: Sequence[dict], cost: Dict[str, object]) -> List[Amortization]:
    """After how many analysed models the static one-time cost has paid for itself.

    ``models = ceil(one_time / median over models of (dynamic - static))``;
    "without training" leaves out the training of the fit population, for the
    case where those models exist anyway. A dynamic tool that is not slower
    per model gets ``saving_per_model <= 0`` and no break-even point.
    """
    seconds = per_model_medians(runs, "per_model_seconds")
    parts = [cost.get("train_seconds"), cost.get("score_seconds"), cost.get("neurons_seconds")]
    known = [v for v in parts if v is not None]
    out = []
    for static in [t for t in STATIC_TOOLS if t in seconds]:
        predictor = static[len("staticsbfl_"):]
        fit = (cost.get("fit_by_predictor") or {}).get(predictor)
        from_predict = fit is None
        if from_predict:
            fit = cost.get("predict_seconds")
        once = sum(known) + fit if known and fit is not None else None
        base = dict(static=static, one_time_seconds=once, fit_seconds=fit, fit_is_upper_bound_of_predict=from_predict)
        dynamics = [t for t in DYNAMIC_TOOLS if t in seconds]
        if once is None or not dynamics:
            out.append(Amortization(dynamic=None, **base))
            continue
        for dynamic in dynamics:
            common = set(seconds[static]) & set(seconds[dynamic])
            if not common:
                continue
            saving = statistics.median(seconds[dynamic][m] - seconds[static][m] for m in common)
            entry = Amortization(dynamic=dynamic, saving_per_model=saving, **base)
            if saving > 0:
                entry.models_to_pay_off = math.ceil(once / saving)
                entry.models_to_pay_off_without_training = math.ceil(
                    max(0.0, once - (cost.get("train_seconds") or 0.0)) / saving)
            out.append(entry)
    return out


def static_accuracy(mode: str, dataset: str, predictor: str, target_kind: str = "rank_model",
                    scope: str = "active") -> Dict[str, Optional[float]]:
    """rho/model of *predictor* on phase 4 'designed', per metric -- accuracy next to speed."""
    out = {}
    directory = variant_predictions_dir(PHASE, mode, dataset)
    for metric in METRICS:
        target = metric if target_kind == "raw" else f"{metric}_{target_kind}"
        value = None
        for row in read_csv(directory / f"{PHASE}__{metric}__{target}.csv"):
            if (row.get("population") == EVAL_SPLIT and row.get("predictor") == predictor
                    and row.get("scope") == scope and row.get("target", target) == target):
                value = number(row.get("spearman_model"))
        out[metric] = value
    return out


@dataclass
class FidelityRow:
    """Spectrum agreement of one dynamic tool with one reference, over all models."""
    tool: str
    reference: str
    models: int
    identical: int
    neurons_differing: int
    neurons: int
    max_count_difference: int
    min_rho: Dict[str, Optional[float]]


def fidelity_summary(rows: Sequence[dict]) -> List[FidelityRow]:
    """Per (tool, reference): how many spectra are identical, and the worst case."""
    out = []
    pairs = sorted({(r["tool"], r["reference"]) for r in rows},
                   key=lambda p: (list(TOOL_LABELS).index(p[0]) if p[0] in TOOL_LABELS else 99, p[1]))
    for tool, reference in pairs:
        mine = [r for r in rows if r["tool"] == tool and r["reference"] == reference]
        rho = {}
        for metric in METRICS:
            values = [number(r.get(f"spearman_{metric}")) for r in mine]
            values = [v for v in values if v is not None]
            rho[metric] = min(values) if values else None
        out.append(FidelityRow(
            tool=tool, reference=reference, models=len(mine),
            identical=sum(str(r.get("identical")) == "True" for r in mine),
            neurons_differing=sum(int(number(r.get("neurons_differing")) or 0) for r in mine),
            neurons=sum(int(number(r.get("n_neurons")) or 0) for r in mine),
            max_count_difference=max(int(number(r.get("max_count_difference")) or 0) for r in mine),
            min_rho=rho,
        ))
    return out


@dataclass
class RuntimeSummary:
    """Everything the evaluation shows about one (mode, dataset) runtime run."""

    mode: str
    dataset: str
    runs: List[dict]
    tools: List[ToolRow]
    designs: List[dict]
    speedups: List[Speedup]
    cost: Dict[str, object]
    amortization: List[Amortization]
    accuracy: Dict[str, Dict[str, Optional[float]]]
    fidelity: List[FidelityRow]
    environment: dict


def summarise(mode: str, dataset: str, scope: str = "active", *, tools: Sequence[str] = ()) -> Optional[RuntimeSummary]:
    """Read and aggregate one runtime run; ``None`` if it has no runs yet.

    *tools* restricts every table to those tools (empty: all).
    """
    runs = read_runs(mode, dataset)
    if tools:
        runs = [r for r in runs if r["tool"] in tools]
    if not runs:
        return None
    cost = one_time_cost(mode, dataset)
    statics = [t for t in STATIC_TOOLS if any(r["tool"] == t for r in runs)]
    fidelity = read_fidelity(mode, dataset)
    if tools:
        fidelity = [r for r in fidelity if r["tool"] in tools]
    return RuntimeSummary(
        mode=mode, dataset=dataset, runs=runs, tools=tool_rows(runs), designs=by_design(runs),
        speedups=speedups(runs), cost=cost, amortization=amortization(runs, cost),
        accuracy={t: static_accuracy(mode, dataset, t[len("staticsbfl_"):], scope=scope) for t in statics},
        fidelity=fidelity_summary(fidelity), environment=read_environment(mode, dataset),
    )


def write_summary(summary: RuntimeSummary) -> List[Path]:
    """``summary.csv`` (per tool) and ``by_design.csv`` next to the runs."""
    root = runtime_root(summary.mode, summary.dataset)
    written = []
    rows = []
    for tool in summary.tools:
        row = tool.as_dict()
        for s in summary.speedups:
            if s.static == tool.tool:
                row[f"speedup_vs_{s.dynamic}"] = s.geometric_mean
        for a in summary.amortization:
            if a.static == tool.tool and a.dynamic:
                row[f"models_to_pay_off_vs_{a.dynamic}"] = a.models_to_pay_off
        for metric, value in summary.accuracy.get(tool.tool, {}).items():
            row[f"rho_model_{metric}"] = value
        rows.append(row)
    if rows:
        fields = []
        for row in rows:
            fields += [k for k in row if k not in fields]
        path = root / "summary.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)
    if summary.designs:
        fields = ["design", "neurons"] + [t.tool for t in summary.tools]
        path = root / "by_design.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary.designs)
        written.append(path)
    return written
