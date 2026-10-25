"""The prediction results as data: read, filter, pick the best, mark the highs.

Everything here works on the CSVs that ``make predict`` wrote and returns plain
data -- no printing (that is :mod:`staticsbfl.log.evaluation`) and no drawing (that
is :mod:`staticsbfl.viz.html_report`). So every number a summary or a report shows
can be checked by a unit test against a hand-made table.

Reading rules
-------------
* A blank or ``nan`` cell is *undefined* and stays ``None``; it is never read
  as 0 and never wins a maximum.
* "Best" is always the maximum over **fitted** predictors only; the two
  baselines (``baseline:layer_mean``, ``baseline:random``) are reported next to
  it on the *same case* (target, population, scope), never as the best.
* Highlights are relative to one phase: the maximum of a column among the
  fitted rows of that phase, and every fitted row at or above the phase's 90th
  percentile of that column.
"""

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from staticsbfl.utils.paths import (
    variant_design, variant_models_csv, variant_predictions_dir, variants_root,
)

LAYER_MEAN = "baseline:layer_mean"
RANDOM = "baseline:random"
METRICS = ("tarantula", "ochiai", "dstar")
TARGET_KINDS = ("rank_model", "raw")

#: Predictor -> model type, for filtering by family rather than by name.
PREDICTOR_TYPES: Dict[str, str] = {
    "gradient_boosting": "tree_ensemble",
    "random_forest": "tree_ensemble",
    "mlp": "neural_network",
    LAYER_MEAN: "baseline",
    RANDOM: "baseline",
}
MODEL_TYPES = ("tree_ensemble", "neural_network", "baseline")

#: Share of a phase's fitted rows a value must beat to be marked "high".
HIGH_QUANTILE = 0.90


@dataclass(frozen=True)
class Quantity:
    """One reported column: label, range, better direction, meaning."""
    column: str      # column in the result CSV
    label: str       # as printed
    range: str
    better: str
    meaning: str


QUANTITIES: Tuple[Quantity, ...] = (
    Quantity("spearman_model", "rho/model", "-1 … 1", "higher",
             "Spearman rank correlation of predicted vs measured suspiciousness, "
             "within each model, then the mean over models. 1 = same order, 0 = "
             "unrelated, -1 = reversed."),
    Quantity("weightedtau_model", "wtau", "-1 … 1", "higher",
             "Weighted Kendall tau: like rho, but a swap near the top of the "
             "ranking costs far more than one near the bottom -- matches how a "
             "candidate list is read."),
    Quantity("top_k_05", "top-5%", "0 … 1", "higher",
             "Share of the truly top-5% most suspicious neurons that are also in "
             "the predicted top-5%, per model. Chance about 0.05."),
    Quantity("top_k", "top-10%", "0 … 1", "higher",
             "Share of the truly top-10% most suspicious neurons that are also in "
             "the predicted top-10%, per model."),
    Quantity("top_k_chance", "top-k chance", "0 … 1", "— (the bar)",
             "What a random ranking reaches on top-k: k/n, about 0.10. Not a "
             "result -- top-k has to clear it. 'x chance' = top-k / chance."),
    Quantity("gt_gain_05", "gain@5%", "… 1", "higher",
             "How much MEASURED suspiciousness the predicted top-5% carries: (mean of the "
             "predicted top - mean of all) / (mean of the true top - mean of all). 0 = a random "
             "pick, 1 = as good as the true top-5%; scale-free, so comparable across metrics."),
    Quantity("gt_gain_10", "gain@10%", "… 1", "higher",
             "The same for the top 10%. High gain with modest overlap means: the predicted "
             "candidates are not exactly the true top, but they are highly suspicious ones."),
    Quantity("gt_lift_10", "lift@10%", "≥ 0", "higher",
             "Mean measured suspiciousness of the predicted top-10% divided by that of the "
             "remaining 90%. 1 = no separation."),
    Quantity("spearman_layer", "rho/layer", "-1 … 1", "higher",
             "Spearman within each layer only: what is left once 'which layer is "
             "the neuron in' is taken away."),
)

EXTRA_TERMS: Tuple[Tuple[str, str, str, str], ...] = (
    ("margin", "-2 … 2", "higher",
     "best predictor minus baseline:layer_mean on the same case. <= 0: the "
     "predictor knows no more than the layer a neuron sits in."),
    ("± sd", "≥ 0", "lower",
     "spread of the value ACROSS evaluation models -- not a standard error. "
     "The standard error of the mean is sd / sqrt(n_models)."),
    ("scope", "all | active", "—",
     "'active' drops neurons that never fired (they score 0 under every metric, "
     "so recognising them is easy and useless). 'active' is the honest headline."),
)

#: The columns a headline or a highlight is computed for, with their labels.
HEADLINE_COLUMNS = (("spearman_model", "rho/model"), ("weightedtau_model", "wtau"), ("top_k_05", "top-5%"),
                    ("top_k", "top-10%"), ("gt_gain_10", "gain@10%"))

#: Numeric columns converted on reading; everything else stays a string.
NUMERIC_COLUMNS = tuple(q.column for q in QUANTITIES) + (
    "spearman_model_sd", "weightedtau_model_sd", "inactive_fraction", "r2",
    # since the top-k extension; absent (None) in tables written before it
    "top_k_05", "top_k_05_chance", "top_k_20", "top_k_20_chance",
    "top_k_near", "top_k_near_chance", "top_k_true_percentile",
    # since the ground-truth separation (2026-10-22)
    "gt_top_05", "gt_rest_05", "gt_ideal_05", "gt_gain_05",
    "gt_top_10", "gt_rest_10", "gt_ideal_10", "gt_gain_10",
)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def number(value) -> Optional[float]:
    """A CSV cell as a float; blank or nan means undefined, never zero."""
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def read_results(phase: str, mode: str, dataset: str) -> Tuple[List[dict], List[str]]:
    """Every result row of *phase*, and the names of the files it came from."""
    directory = variant_predictions_dir(phase, mode, dataset)
    files = sorted(directory.glob(f"{phase}__*__*.csv")) if directory.exists() else []
    rows: List[dict] = []
    for path in files:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for column in NUMERIC_COLUMNS:
                    row[column] = number(row.get(column))
                row["n_models"] = int(number(row.get("n_models")) or 0)
                row["model_type"] = PREDICTOR_TYPES.get(row.get("predictor", ""), "other")
                for suffix in ("05", "10"):
                    top, rest = row.get(f"gt_top_{suffix}"), row.get(f"gt_rest_{suffix}")
                    row[f"gt_lift_{suffix}"] = top / rest if top is not None and rest else None
                rows.append(row)
    return rows, [p.name for p in files]


def read_design(phase: str, mode: str, dataset: str) -> dict:
    """A phase's ``design.json``, or ``{}``."""
    path = variant_design(phase, mode, dataset)
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def mean_accuracy(phase: str, mode: str, dataset: str) -> Dict[str, float]:
    """Mean test accuracy per population, from ``models.csv``."""
    path = variant_models_csv(phase, mode, dataset)
    if not path.exists():
        return {}
    by_split: Dict[str, List[float]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = number(row.get("test_accuracy"))
            if value is not None:
                by_split.setdefault(row.get("split", ""), []).append(value)
    return {split: sum(v) / len(v) for split, v in by_split.items() if v}


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultFilter:
    """Which rows to keep. An empty field keeps everything on that axis.

    Baselines are kept whenever their *case* is kept, whatever the predictor
    filter says: a best value without its layer-mean baseline cannot be read.
    """

    predictors: Tuple[str, ...] = ()
    model_types: Tuple[str, ...] = ()
    metrics: Tuple[str, ...] = ()
    targets: Tuple[str, ...] = ()
    populations: Tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        """Whether any axis is filtered."""
        return any((self.predictors, self.model_types, self.metrics, self.targets, self.populations))

    def describe(self) -> str:
        """The filter in words, ``none`` without one."""
        parts = [f"{name} {'/'.join(values)}" for name, values in (
            ("predictor", self.predictors), ("model type", self.model_types), ("metric", self.metrics),
            ("target", self.targets), ("population", self.populations)) if values]
        return " · ".join(parts) if parts else "none"

    def keeps(self, row: dict) -> bool:
        """Whether *row* passes; baselines pass whenever their case does."""
        if self.metrics and row.get("metric") not in self.metrics:
            return False
        if self.targets and row.get("target") not in self.targets:
            return False
        if self.populations and row.get("population") not in self.populations:
            return False
        if str(row.get("predictor", "")).startswith("baseline:"):
            return True
        if self.predictors and row.get("predictor") not in self.predictors:
            return False
        if self.model_types and PREDICTOR_TYPES.get(row.get("predictor", ""), "other") not in self.model_types:
            return False
        return True

    def apply(self, rows: Iterable[dict]) -> List[dict]:
        """The rows that pass."""
        return [row for row in rows if self.keeps(row)]


NO_FILTER = ResultFilter()


# --------------------------------------------------------------------------
# Selecting the best case
# --------------------------------------------------------------------------


@dataclass
class Best:
    """The best fitted row of a column, with both baselines on its case."""
    column: str
    row: dict
    layer_mean: Optional[float]
    random: Optional[float]

    @property
    def value(self) -> float:
        """The best value."""
        return self.row[self.column]

    @property
    def margin(self) -> Optional[float]:
        """Best minus layer-mean baseline, ``None`` without a baseline."""
        return None if self.layer_mean is None else self.value - self.layer_mean

    @property
    def case(self) -> str:
        """``predictor · target`` of the best row."""
        return f"{self.row['predictor']} · {self.row['target']}"


def is_fitted(row: dict) -> bool:
    """Whether *row* is a fitted predictor rather than a baseline."""
    return not str(row.get("predictor", "")).startswith("baseline:")


def baseline_value(rows: Sequence[dict], like: dict, predictor: str, column: str) -> Optional[float]:
    """*predictor*'s value of *column* on the same case (target, population, scope) as *like*."""
    for row in rows:
        if (row["predictor"] == predictor and row["target"] == like["target"]
                and row["population"] == like["population"] and row["scope"] == like["scope"]):
            return row[column]
    return None


def best_of(rows: Sequence[dict], column: str, *, population: Optional[str], scope: str) -> Optional[Best]:
    """The fitted predictor with the highest *column*, with both baselines on its case."""
    candidates = [
        r for r in rows
        if is_fitted(r) and r["scope"] == scope
        and (population is None or r["population"] == population)
        and r.get(column) is not None
    ]
    if not candidates:
        return None
    top = max(candidates, key=lambda r: r[column])
    return Best(column, top, baseline_value(rows, top, LAYER_MEAN, column),
                baseline_value(rows, top, RANDOM, column))


def n_candidates(rows: Sequence[dict], population: str, scope: str) -> int:
    """Fitted rows on (population, scope): how many cases a 'best' is the maximum of."""
    return sum(1 for r in rows if is_fitted(r) and r["population"] == population and r["scope"] == scope)


def best_per_metric(rows: Sequence[dict], population: str, scope: str) -> Dict[str, Optional[float]]:
    """Best rho/model per suspiciousness metric -- they are separate results."""
    out = {}
    for metric in METRICS:
        best = best_of([r for r in rows if r["metric"] == metric], "spearman_model",
                       population=population, scope=scope)
        out[metric] = best.value if best else None
    return out


# --------------------------------------------------------------------------
# One phase, and the overview over phases
# --------------------------------------------------------------------------


@dataclass
class PhaseResult:
    """One phase's rows, files, design, accuracies and ordered populations."""
    phase: str
    mode: str
    dataset: str
    rows: List[dict]
    files: List[str]
    design: dict = field(default_factory=dict)
    accuracy: Dict[str, float] = field(default_factory=dict)
    populations: List[str] = field(default_factory=list)

    @property
    def expected_files(self) -> int:
        """Result files a complete phase has."""
        return len(METRICS) * len(TARGET_KINDS)

    @property
    def complete(self) -> bool:
        """Whether every result file exists."""
        return len(self.files) >= self.expected_files


def load_phase(phase: str, mode: str, dataset: str, flt: ResultFilter = NO_FILTER) -> Optional[PhaseResult]:
    """One phase's result rows (filtered), its design and accuracies; ``None`` if none exist."""
    rows, files = read_results(phase, mode, dataset)
    if not rows:
        return None
    rows = flt.apply(rows)
    design = read_design(phase, mode, dataset)
    order = list(design.get("evaluation_populations", []))
    populations = sorted({r["population"] for r in rows},
                         key=lambda p: (order.index(p) if p in order else len(order), p))
    return PhaseResult(phase, mode, dataset, rows, files, design, mean_accuracy(phase, mode, dataset), populations)


def overview_rows(results: Iterable[PhaseResult], scope: str) -> List[dict]:
    """One row per (phase, population): the numbers a paper table needs."""
    other = "all" if scope == "active" else "active"
    table: List[dict] = []
    for result in results:
        for population in result.populations:
            rho = best_of(result.rows, "spearman_model", population=population, scope=scope)
            if rho is None:
                continue
            rho_other = best_of(result.rows, "spearman_model", population=population, scope=other)
            tau = best_of(result.rows, "weightedtau_model", population=population, scope=scope)
            top = best_of(result.rows, "top_k", population=population, scope=scope)
            chance = top.row["top_k_chance"] if top else None
            table.append({
                "mode": result.mode, "phase": result.phase, "population": population, "scope": scope,
                "best_rho_model": rho.value, "best_rho_case": rho.case,
                "rho_sd": rho.row.get("spearman_model_sd"), "layer_mean_rho": rho.layer_mean,
                "margin_over_layer_mean": rho.margin,
                f"best_rho_model_{other}": rho_other.value if rho_other else None,
                "best_wtau": tau.value if tau else None, "best_wtau_case": tau.case if tau else "",
                "best_top_k": top.value if top else None, "top_k_chance": chance,
                "top_k_x_chance": (top.value / chance) if top and chance else None,
                "best_top_k_case": top.case if top else "", "n_models": rho.row["n_models"],
                "n_candidates": n_candidates(result.rows, population, scope),
            })
    return table


def typical_uncertainty(results: Sequence[PhaseResult], scope: str) -> Optional[Tuple[float, float, int]]:
    """``(2·sd/sqrt(n), sd, n)`` for the median best case, or ``None`` without sds."""
    pairs = [
        (best.row["spearman_model_sd"], best.row["n_models"])
        for r in results for p in r.populations
        for best in [best_of(r.rows, "spearman_model", population=p, scope=scope)]
        if best and best.row.get("spearman_model_sd") is not None and best.row["n_models"]
    ]
    if not pairs:
        return None
    sd = sorted(v for v, _ in pairs)[len(pairs) // 2]
    n = sorted(n for _, n in pairs)[len(pairs) // 2]
    return 2 * sd / math.sqrt(n), sd, n


# --------------------------------------------------------------------------
# Highlights
# --------------------------------------------------------------------------


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default), without numpy."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile of nothing")
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def phase_highlights(rows: Sequence[dict], columns: Sequence[str] = tuple(c for c, _ in HEADLINE_COLUMNS),
                     *, high_quantile: float = HIGH_QUANTILE) -> Dict[Tuple[int, str], str]:
    """``{(row index, column): "best" | "high"}`` within one phase's *rows*.

    Per scope, since 'all' and 'active' are different questions: the maximum of
    a column among fitted rows is "best" (every row that ties it), a fitted row
    at or above the *high_quantile* of that column is "high". Baselines and
    undefined cells are never marked.
    """
    marks: Dict[Tuple[int, str], str] = {}
    for scope in sorted({r["scope"] for r in rows}):
        for column in columns:
            indexed = [(i, r[column]) for i, r in enumerate(rows)
                       if r["scope"] == scope and is_fitted(r) and r.get(column) is not None]
            if not indexed:
                continue
            values = [v for _, v in indexed]
            top, bar = max(values), quantile(values, high_quantile)
            for index, value in indexed:
                if value == top:
                    marks[(index, column)] = "best"
                elif value >= bar:
                    marks[(index, column)] = "high"
    return marks


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def write_summary(table: List[dict], mode: str, dataset: str, scope: str) -> Optional[Path]:
    """The overview as ``summary_<scope>.csv`` next to the phases."""
    if not table:
        return None
    path = variants_root(mode, dataset) / f"summary_{scope}.csv"
    fields = list(table[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in table:
            writer.writerow({
                k: ("" if v is None else f"{v:.6f}" if isinstance(v, float) else v)
                for k, v in row.items()
            })
    return path


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------

#: Columns statistics are reported for, in order.
STATISTIC_COLUMNS = ("spearman_model", "weightedtau_model", "top_k_05", "top_k",
                     "gt_gain_05", "gt_gain_10", "gt_lift_10", "spearman_layer")

#: What rows can be grouped by.
GROUPINGS = ("model_type", "predictor", "metric", "population", "scope", "target", "phase", "mode")


def describe_values(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """Count, undefined count, min, quartiles, median, mean, max, sd and range.

    ``rows`` counts the defined values, ``undefined`` the rest; the two always
    add up to the number of values given, so nothing is silently lost.
    Quartiles use linear interpolation (numpy's default); sd is the population
    standard deviation (ddof 0).
    """
    defined = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    out: Dict[str, Optional[float]] = {"rows": len(defined), "undefined": len(values) - len(defined)}
    if not defined:
        out.update({k: None for k in ("min", "q25", "median", "mean", "q75", "max", "sd", "range")})
        return out
    mean = sum(defined) / len(defined)
    out.update({
        "min": min(defined), "q25": quantile(defined, 0.25), "median": quantile(defined, 0.5),
        "mean": mean, "q75": quantile(defined, 0.75), "max": max(defined),
        "sd": math.sqrt(sum((v - mean) ** 2 for v in defined) / len(defined)),
        "range": max(defined) - min(defined),
    })
    return out


def group_statistics(rows: Sequence[dict], column: str, group_by: str) -> List[dict]:
    """:func:`describe_values` of *column* for every value of *group_by*, plus ``*`` for all rows.

    Every row lands in exactly one group, so the groups' ``rows + undefined``
    sum to the ``*`` line's -- the property the tests pin.
    """
    groups: Dict[str, List[Optional[float]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_by, "")), []).append(row.get(column))
    out = [{"group_by": group_by, "group": group, "column": column, **describe_values(values)}
           for group, values in sorted(groups.items())]
    out.append({"group_by": group_by, "group": "*", "column": column,
                **describe_values([row.get(column) for row in rows])})
    return out


def statistics_table(results: Sequence[PhaseResult], *, columns: Sequence[str] = STATISTIC_COLUMNS,
                     groupings: Sequence[str] = GROUPINGS) -> List[dict]:
    """Per (mode, phase, scope): statistics of every column under every grouping.

    Scopes are never pooled ('all' and 'active' are different questions); a
    grouping by scope is therefore reported over both scopes of a phase.
    """
    out = []
    for result in results:
        for scope in sorted({r["scope"] for r in result.rows}) + ["*"]:
            rows = result.rows if scope == "*" else [r for r in result.rows if r["scope"] == scope]
            for group_by in groupings:
                if (group_by == "scope") != (scope == "*"):
                    continue
                for column in columns:
                    for entry in group_statistics(rows, column, group_by):
                        out.append({"mode": result.mode, "phase": result.phase, "scope": scope, **entry})
    return out


def write_statistics(statistics: Sequence[dict], path: Path) -> Path:
    """The full statistics table as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["mode", "phase", "scope", "group_by", "group", "column", "rows", "undefined",
              "min", "q25", "median", "mean", "q75", "max", "sd", "range"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in statistics:
            writer.writerow({k: ("" if row.get(k) is None else f"{row[k]:.6f}" if isinstance(row.get(k), float) else row[k])
                             for k in fields})
    return path
