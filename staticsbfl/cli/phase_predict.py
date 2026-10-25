"""Fit static predictors of neuron suspiciousness and evaluate them as rankers.

    staticsbfl-predict --phase phase1
    staticsbfl-predict --phase phase3 --metric ochiai --target rank_layer

Reads the neuron table built by ``staticsbfl-neurons``, fits each predictor
on the phase's training population, and evaluates it on every other population.
**All three suspiciousness metrics are fitted by default**, each with its own
estimators: a predictor of Tarantula is not a predictor of D*, and reporting one
of them as "the" result hides that the three rank neurons differently.

Evaluation is ranking-first. What a localization result is judged on is whether
the neurons at the top of the predicted ordering are the right ones, not the
squared error of a suspiciousness value, so the reported quantities are:

    per-model Spearman   rank correlation computed WITHIN each model, then
                         averaged over models. Pooling all neurons of all models
                         into one correlation is dominated by between-model
                         variance, which is the easy part.
    per-model weighted   the same pairs under Vigna's hyperbolic weighting, which
                         charges a swap near the top of the ranking far more than
                         one near the bottom. A spectrum is consumed as a
                         candidate list, so this is the coefficient that matches
                         the use; Spearman weights rank 2000 like rank 2.
    top-k retrieval      the share of the truly most suspicious k% recovered in
                         the predicted top k%, per model, against its own chance
                         level k/n rather than a flat 0.10.
    per-layer Spearman   the same correlation restricted to one layer at a time,
                         summarised within a model before models are averaged.

Two baselines are always reported. **Layer mean** predicts every neuron of a
layer as that layer's mean target: a predictor that does not beat it has learned
only which layer a neuron sits in. **Random** is the chance level.

R^2 is also reported, for continuity with a regression framing, but it is not
the headline -- and for D*, whose raw values span nine orders of magnitude, it is
not interpretable at all.

**Every result is reported twice**, over all neurons and over active ones only.
A neuron that never fired scores exactly 0 under all three metrics, so a
predictor that has merely learned to recognise dead units is rewarded as if it
had ranked live ones. On the populations measured so far that was worth up to
19% of the reported correlation, and it varies by phase, which makes the phases
incomparable until the two are separated.

LIGHT PATH ONLY: dense models, since that is what the table currently contains.
"""

import csv
import logging
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import click
import numpy as np

from staticsbfl.data import IMAGE_DATASET_NAMES
from staticsbfl.dataset import (
    ACTIVITY_TARGET,
    ARCHITECTURE,
    FEATURE_NAMES,
    FEATURES,
    NEURON_STATIC,
    NeuronDataset,
    TARGET_METRICS,
)
from staticsbfl.log.console import (
    configure_logging, display_path, format_duration, plain, style, Step,
)
from staticsbfl.utils.paths import (
    ensure_dir, variant_correlation_log, variant_neurons,
    variant_predictions_csv, variant_predictions_dir,
)
from staticsbfl.utils.seeding import DEFAULT_SEED
from staticsbfl.variation import PHASE_NAMES, build_phase
from staticsbfl.variation.phases import POPULATION_SIZES
from staticsbfl.log.pipeline import (
    TABLE_LEGEND, record_timing, step_context, step_description, step_index,
    step_outputs,
)

logger = logging.getLogger("predict")

#: Which target flavour to fit. "raw" is the suspiciousness value itself;
#: "rank_model" and "rank_layer" are its normalised rank.
TARGET_KINDS = ("rank_model", "rank_layer", "raw")

#: The fitted predictors, in the order make_predictors returns them.
PREDICTOR_NAMES = ("gradient_boosting", "mlp")

#: Feature-group ablations, so the contribution of each half is separable.
FEATURE_SETS = {
    "all": FEATURE_NAMES,
    "neuron_static": tuple(f.name for f in FEATURES if f.group == NEURON_STATIC),
    "architecture": tuple(f.name for f in FEATURES if f.group == ARCHITECTURE),
}

#: Fraction of a model's neurons counted as "most suspicious" for top-k.
TOP_K_FRACTION = 0.10

#: Further top-k sizes, so a result does not hinge on one arbitrary cut-off:
#: column ``top_k_05`` is the top 5 %, ``top_k_20`` the top 20 %.
EXTRA_TOP_K = {"top_k_05": 0.05, "top_k_20": 0.20}

#: The two evaluation scopes, always reported side by side.
SCOPES = ("all", "active")

#: Columns of the results CSV, in reporting order.
RESULT_FIELDS: Tuple[str, ...] = (
    "phase", "mode", "metric", "target", "population", "predictor", "scope",
    "n_models", "n_rows", "n_inactive", "inactive_fraction",
    "spearman_model", "spearman_model_sd", "spearman_model_n", "spearman_undefined",
    "weightedtau_model", "weightedtau_model_sd",
    "top_k", "top_k_chance",
    "top_k_05", "top_k_05_chance", "top_k_20", "top_k_20_chance",
    "top_k_near", "top_k_near_chance", "top_k_true_percentile",
    "gt_top_05", "gt_rest_05", "gt_ideal_05", "gt_gain_05",
    "gt_top_10", "gt_rest_10", "gt_ideal_10", "gt_gain_10",
    "spearman_layer", "spearman_layer_models", "spearman_layer_undefined",
    "r2",
)


def make_predictors(seed: int) -> Dict[str, object]:
    """One tree ensemble and one neural network.

    Gradient boosting is the usual strong baseline on tabular data; the MLP is
    included because it is the obvious alternative and its absence would be the
    first question asked. A random forest was fitted until 2026-10-22 and
    dropped: on every light phase it was within a few hundredths of gradient
    boosting, it was the most expensive fit by far, and a fitted forest is
    about 5 GB per target on the light phase-4 table (more than the machine's
    memory for the three heavy ones), which rules it out as a deployable
    static predictor.

    Both use early stopping on an internal validation split, so their
    capacity is bounded by the data rather than by a hand-picked iteration
    count. Each target is fitted by its own instance -- a regressor predicts one
    quantity, so the raw-value and rank fits never share parameters, and neither
    do two suspiciousness metrics.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=800, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=20, l2_regularization=1.0, early_stopping=True,
            n_iter_no_change=30, validation_fraction=0.15, random_state=seed,
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(256, 128, 64), max_iter=1500,
                learning_rate_init=1e-3, alpha=1e-4, batch_size=256,
                early_stopping=True, n_iter_no_change=25, validation_fraction=0.15,
                random_state=seed,
            ),
        ),
    }


# --------------------------------------------------------------------------
# Coefficients
# --------------------------------------------------------------------------


def spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Spearman rank correlation, or ``None`` where it is undefined.

    A constant vector has no ranking, so the correlation against it does not
    exist. Returning ``None`` and counting the occurrences is the convention the
    method requires: silently substituting zero reports "no association" for a
    case where nothing was measured, and it is the layer-mean baseline -- constant
    within every layer by construction -- that suffers most from the confusion.
    """
    from scipy.stats import spearmanr

    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    result = spearmanr(a, b).statistic
    return None if np.isnan(result) else float(result)


def weighted_tau(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Vigna's weighted Kendall tau: the same pairs, the top weighted more.

    Hyperbolic rank weighting, tie-aware, symmetric in its two arguments. On a
    50-element ranking, swapping the top two costs roughly 37 times what swapping
    the bottom two costs, where Spearman charges the two identically. Since a
    suspiciousness ranking is consumed as a candidate list -- the top few are
    inspected and the tail is never read -- this is the coefficient that matches
    what the ranking is for.
    """
    from scipy.stats import weightedtau

    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    result = weightedtau(a, b).statistic
    return None if np.isnan(result) else float(result)


def top_k_overlap(
    truth: np.ndarray, prediction: np.ndarray, fraction: float
) -> Tuple[Optional[float], Optional[float]]:
    """Overlap of the top *fraction*, and the chance level for this size.

    Chance is ``k / n``, not the nominal fraction: ``k`` is rounded to a whole
    number of neurons, so a layer of 7 units has a top-10% of 1 unit and a chance
    level of 1/7, not 0.10. Reporting the nominal value would understate chance
    for small groups and overstate it for large ones.
    """
    n = len(truth)
    if n < 2:
        return None, None
    k = max(1, int(round(fraction * n)))
    best_truth = set(np.argsort(truth)[-k:])
    best_prediction = set(np.argsort(prediction)[-k:])
    return len(best_truth & best_prediction) / k, k / n


def near_misses(
    truth: np.ndarray, prediction: np.ndarray, fraction: float
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """How far the predicted top *fraction* is from the true one, not just whether it hits.

    Top-k overlap is all-or-nothing: a neuron predicted in the top 10 % that is
    truly at the 11th percentile counts the same as one truly at the bottom.
    Two figures separate those cases:

    ``near``             share of the true top-k found in the predicted top-2k
                         (a relaxed recall), with its chance level ``2k / n``;
    ``true_percentile``  mean true percentile (0 = least, 1 = most suspicious,
                         ties averaged) of the neurons predicted in the top-k.
                         Chance is about 0.5; a perfect ranking reaches the
                         mean percentile of the true top-k itself,
                         ``1 - (k-1) / (2(n-1))`` -- about 0.95 for the top 10 %.

    Returns ``(near, near_chance, true_percentile)``, ``None`` for n < 2.
    """
    from scipy.stats import rankdata

    n = len(truth)
    if n < 2:
        return None, None, None
    k = max(1, int(round(fraction * n)))
    wide = min(n, 2 * k)
    best_truth = set(np.argsort(truth)[-k:])
    near = len(best_truth & set(np.argsort(prediction)[-wide:])) / k
    percentile = (rankdata(truth, method="average") - 1) / (n - 1)
    true_percentile = float(percentile[np.argsort(prediction)[-k:]].mean())
    return near, wide / n, true_percentile


#: Top fractions the ground-truth separation is reported for: column suffix -> fraction.
GT_FRACTIONS = {"05": 0.05, "10": 0.10}


def ground_truth_separation(
    raw: np.ndarray, prediction: np.ndarray, fraction: float
) -> Dict[str, Optional[float]]:
    """Mean MEASURED suspiciousness of the predicted top *fraction* vs the rest.

    The question behind it: do the neurons a predictor puts first actually carry
    high suspiciousness, even where they are not exactly the true top-k? Overlap
    is all-or-nothing; this reads the measured values instead.

    ``top``    mean raw suspiciousness of the neurons predicted in the top-k
    ``rest``   mean raw suspiciousness of all other neurons
    ``ideal``  mean raw suspiciousness of the TRUE top-k -- the best any ranking
               can reach
    ``gain``   ``(top - mean) / (ideal - mean)`` with ``mean`` over all neurons:
               0 = no better than a random pick, 1 = as good as the true top-k.
               Scale-free, so it is comparable between Tarantula, Ochiai and the
               unbounded D*.

    *raw* is the raw metric, whatever target the predictor was fitted on.
    Neurons whose raw value is not finite (D* of a neuron firing on failures
    only) are left out of all four, and ``None`` is returned for fewer than two
    finite neurons or when every neuron has the same value.
    """
    finite = np.isfinite(raw)
    raw, prediction = raw[finite], prediction[finite]
    n = len(raw)
    empty = {"top": None, "rest": None, "ideal": None, "gain": None}
    if n < 2:
        return empty
    k = max(1, int(round(fraction * n)))
    order = np.argsort(prediction, kind="stable")
    predicted_top = np.zeros(n, bool)
    predicted_top[order[-k:]] = True
    top = float(raw[predicted_top].mean())
    rest = float(raw[~predicted_top].mean()) if k < n else None
    ideal = float(np.sort(raw)[-k:].mean())
    mean = float(raw.mean())
    gain = (top - mean) / (ideal - mean) if ideal > mean else None
    return {"top": top, "rest": rest, "ideal": ideal, "gain": gain}


def _summarise(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float], int, int]:
    """``(mean, sd, n_defined, n_undefined)`` over values that may be ``None``."""
    defined = [v for v in values if v is not None]
    undefined = len(values) - len(defined)
    if not defined:
        return None, None, 0, undefined
    array = np.asarray(defined, dtype=float)
    return float(array.mean()), float(array.std()), len(defined), undefined


def evaluate_ranking(
    truth: np.ndarray,
    prediction: np.ndarray,
    model_keys: np.ndarray,
    layer_names: np.ndarray,
    raw: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """Ranking quality, aggregated over models with each model counted once.

    Every statistic is computed *within* a model and then averaged across
    models. For the per-layer figure the layers of a model are summarised first
    and only then are models averaged -- pooling every (model, layer) pair into
    one mean, as this function used to, gives a ten-layer model five times the
    weight of a two-layer one, and phase 2 spans depths 1 to 10.

    *raw* is the measured raw suspiciousness of the same neurons (the target
    itself when the target is raw); it feeds :func:`ground_truth_separation`.
    Without it those columns are ``None``.
    """
    per_model_rho, per_model_tau, per_top_k, per_chance, per_model_layer = [], [], [], [], []
    per_gt: Dict[str, List[Optional[float]]] = {
        f"gt_{part}_{suffix}": [] for suffix in GT_FRACTIONS for part in ("top", "rest", "ideal", "gain")
    }
    per_extra: Dict[str, List[Optional[float]]] = {
        name: [] for column in EXTRA_TOP_K for name in (column, f"{column}_chance")
    }
    per_near, per_near_chance, per_percentile = [], [], []

    for key in np.unique(model_keys):
        rows = model_keys == key
        per_model_rho.append(spearman(truth[rows], prediction[rows]))
        per_model_tau.append(weighted_tau(truth[rows], prediction[rows]))
        overlap, chance = top_k_overlap(truth[rows], prediction[rows], TOP_K_FRACTION)
        per_top_k.append(overlap)
        per_chance.append(chance)
        for column, fraction in EXTRA_TOP_K.items():
            extra, extra_chance = top_k_overlap(truth[rows], prediction[rows], fraction)
            per_extra[column].append(extra)
            per_extra[f"{column}_chance"].append(extra_chance)
        near, near_chance, percentile = near_misses(truth[rows], prediction[rows], TOP_K_FRACTION)
        per_near.append(near)
        per_near_chance.append(near_chance)
        per_percentile.append(percentile)
        if raw is not None:
            for suffix, fraction in GT_FRACTIONS.items():
                separation = ground_truth_separation(raw[rows], prediction[rows], fraction)
                for part, value in separation.items():
                    per_gt[f"gt_{part}_{suffix}"].append(value)

        within = [
            spearman(truth[cells], prediction[cells])
            for layer in np.unique(layer_names[rows])
            for cells in [rows & (layer_names == layer)]
            if cells.sum() >= 2
        ]
        defined = [v for v in within if v is not None]
        per_model_layer.append(float(np.mean(defined)) if defined else None)

    rho_mean, rho_sd, rho_n, rho_undef = _summarise(per_model_rho)
    tau_mean, tau_sd, _tau_n, _tau_undef = _summarise(per_model_tau)
    top_mean, _top_sd, _top_n, _top_undef = _summarise(per_top_k)
    chance_mean, _c_sd, _c_n, _c_undef = _summarise(per_chance)
    layer_mean, _l_sd, layer_n, layer_undef = _summarise(per_model_layer)

    ss_res = float(np.sum((truth - prediction) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "spearman_model": rho_mean,
        "spearman_model_sd": rho_sd,
        "spearman_model_n": rho_n,
        "spearman_undefined": rho_undef,
        "weightedtau_model": tau_mean,
        "weightedtau_model_sd": tau_sd,
        "top_k": top_mean,
        "top_k_chance": chance_mean,
        **{name: _summarise(values)[0] for name, values in per_extra.items()},
        "top_k_near": _summarise(per_near)[0],
        "top_k_near_chance": _summarise(per_near_chance)[0],
        "top_k_true_percentile": _summarise(per_percentile)[0],
        **{name: _summarise(values)[0] for name, values in per_gt.items()},
        "spearman_layer": layer_mean,
        "spearman_layer_models": layer_n,
        "spearman_layer_undefined": layer_undef,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
    }


def layer_mean_baseline(
    train: NeuronDataset, test: NeuronDataset, target: str
) -> np.ndarray:
    """Predict each neuron as the training mean of its (depth, layer index) slot.

    Keyed on the layer's position rather than its name so it transfers to models
    of a different depth. Falls back to the global mean for unseen slots.
    """
    index = FEATURE_NAMES.index("layer_index")
    depth = FEATURE_NAMES.index("depth")
    y = train.targets[target]

    means: Dict[tuple, float] = {}
    keys = list(zip(train.features[:, depth], train.features[:, index]))
    for key in set(keys):
        rows = np.array([k == key for k in keys])
        means[key] = float(y[rows].mean())

    fallback = float(y.mean())
    test_keys = zip(test.features[:, depth], test.features[:, index])
    return np.array([means.get(k, fallback) for k in test_keys])


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def _cell(value) -> str:
    """CSV cell: an undefined statistic is blank, never zero."""
    if value is None:
        return ""
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def write_results(
    phase: str, mode: str, dataset: str, metric: str, target: str, rows: List[dict]
) -> str:
    """One CSV per (phase, metric, target). The metric is in the filename.

    Not a column: a run fits one metric at a time, and a partial sweep in a
    shared file would be indistinguishable from a complete one.
    """
    path = variant_predictions_csv(phase, metric, target, mode, dataset)
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in RESULT_FIELDS})
    return str(path)


LOG_PREAMBLE = """\
Correlation report -- what these numbers are, before what they say
==================================================================

WHICH PAIRS ARE CORRELATED
    x = the predictor's output for one neuron, from static features only:
        weight statistics of that neuron plus the architecture around it. No
        forward pass, no data, no activation is involved in producing it.
    y = the measured suspiciousness of the same neuron, from the cached hit
        spectrum of the model it belongs to.
    One pair per neuron. The pairing is exact -- same model, same layer, same
    unit index -- so no alignment or matching step is involved.

HOW THEY ARE AGGREGATED
    Within one model first, then across models with every model weighted
    equally. A correlation pooled over all neurons of all models would be
    dominated by differences BETWEEN models, which is the easy part of the task
    and not what a localization result is judged on.
    'spearman_layer' goes one level further: correlations are computed within
    each (model, layer), averaged within the model, and only then across models.
    Averaging every (model, layer) pair into a single mean -- which this script
    used to do -- gives a ten-layer model five times the weight of a two-layer
    one.
    The reported 'sd' columns are the standard deviation ACROSS MODELS, not a
    standard error. With a small evaluation population they are wide, and that
    is information rather than noise to hide.

THE COEFFICIENTS
    spearman_model     Pearson on ranks. Every rank position counts equally, so
                       the number is dominated by the bulk of near-tied
                       middling neurons.
    weightedtau_model  Vigna's weighted Kendall tau: hyperbolic rank weighting,
                       tie-aware. On a 50-element ranking a swap of the top two
                       costs about 37x what a swap of the bottom two costs. A
                       suspiciousness ranking is read from the top down, so this
                       is the coefficient that matches the use.
    top_k              Share of the truly most suspicious 10% recovered in the
                       predicted top 10%, per model. Compared against
                       'top_k_chance' = k/n for that model, NOT against a flat
                       0.10: k is a whole number of neurons.
    top_k_05/_20       The same at 5% and 20% (CSV only), each with its chance.
    top_k_near         Share of the true top 10% found in the predicted top 20%:
                       a near miss counts. Chance = 2k/n.
    top_k_true_percentile
                       Mean true percentile of the neurons predicted in the top
                       10% (CSV only): ~0.5 by chance, ~0.95 for a perfect ranking.
    r2               Reported for continuity with a regression framing. It is
                       not the headline, and for D*, whose raw values span nine
                       orders of magnitude, it is not interpretable at all.

UNDEFINED VALUES
    A constant vector has no ranking, so a correlation against it does not
    exist. Those cases are counted in the '*_undefined' columns and excluded
    from the mean; they are never silently replaced by zero. The layer-mean
    baseline is constant within every layer by construction, so its per-layer
    correlation is undefined everywhere -- which is the honest answer, not 0.0.

THE TWO SCOPES
    scope = all      every neuron in the evaluation population.
    scope = active   only neurons that fired at least once, by the recorded
                     activity rate.
    A neuron that never fired scores exactly 0 under all three metrics, so under
    'all' a predictor is rewarded for recognising dead units -- a different and
    much easier task than ranking live ones, and one with no downstream value,
    since a unit that never fires is not a repair candidate. The gap between the
    two scopes is how much of the score came from that. It varies by population,
    which is why the phases cannot be compared under 'all' alone.

BASELINES
    baseline:layer_mean  every neuron predicted as its layer's training mean. A
                         predictor that does not beat it has learned only which
                         layer a neuron sits in, not anything about the neuron.
                         Read the MARGIN over this, not the raw correlation.
    baseline:random      chance.
"""


def write_correlation_log(
    phase: str, mode: str, dataset: str, sections: List[str]
) -> str:
    """The human-readable record: what the numbers mean, then the numbers."""
    path = variant_correlation_log(phase, mode, dataset)
    ensure_dir(path.parent)
    header = (
        f"phase: {phase}   mode: {mode}   dataset: {dataset}   written: "
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        f"{'=' * 66}\n\n"
    )
    path.write_text(header + LOG_PREAMBLE + "\n" + "\n".join(sections), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@click.command()
@click.option("--phase", type=click.Choice(PHASE_NAMES), required=True)
@click.option(
    "--mode", type=click.Choice(sorted(POPULATION_SIZES)), default="light",
    show_default=True,
)
@click.option(
    "--metric", "metrics", multiple=True, type=click.Choice(TARGET_METRICS),
    help="Suspiciousness metric(s) to predict; default is all three, each with "
         "its own estimators.",
)
@click.option(
    "--target", "target_kinds", multiple=True, type=click.Choice(TARGET_KINDS),
    help="Target flavour(s); default is rank_model and raw.",
)
@click.option(
    "--features", "feature_set", type=click.Choice(sorted(FEATURE_SETS)),
    default="all", show_default=True, help="Feature-group ablation.",
)
@click.option(
    "--save-estimators", is_flag=True,
    help="Also write the fitted estimators next to the results. Off by default: "
         "a 500-tree forest per metric per target per phase is gigabytes, and "
         "the results are what the paper quotes.",
)
@click.option(
    "--save-predictor", "save_predictors", multiple=True,
    type=click.Choice(PREDICTOR_NAMES),
    help="Write only these fitted estimators (repeatable); implies saving. The runtime "
         "comparison loads them to time the static prediction.",
)
@click.option(
    "--save-kind", "save_kinds", multiple=True, type=click.Choice(TARGET_KINDS),
    help="Save estimators only for these target kinds (repeatable; default: every fitted "
         "kind). The runtime comparison loads rank_model only.",
)
@click.option("--seed", type=int, default=DEFAULT_SEED, show_default=True)
@click.option(
    "--dataset", type=click.Choice(IMAGE_DATASET_NAMES), default="cifar10", show_default=True,
    help="Which population tree to read. Must match the dataset the models were "
         "trained on; every cache path is namespaced by it.",
)
@click.option("--verbose", is_flag=True, help="Debug-level logging.")
def main(phase, mode, metrics, target_kinds, feature_set, save_estimators, save_predictors, save_kinds, seed, dataset,
         verbose):
    """Fit and evaluate static predictors for one phase, for every metric."""
    configure_logging(verbose)
    spec = build_phase(phase, mode)
    path = variant_neurons(spec.name, mode, dataset)
    if not path.exists():
        raise click.ClickException(
            f"No neuron table at {path}. Run staticsbfl-neurons --phase "
            f"{phase} --mode {mode} first."
        )

    number, of = step_index("predict")
    with Step(
        logger, number, of, "predict",
        step_description("predict", spec),
        outputs=step_outputs("predict", phase, mode, dataset),
        context=step_context(phase, mode, dataset),
    ) as step:
        run(step, spec, mode, dataset, path, metrics, target_kinds, feature_set,
            save_estimators, seed, save_predictors, save_kinds)


def run(step, spec, mode, dataset, path, metrics, target_kinds, feature_set,
        save_estimators, seed, save_predictors=(), save_kinds=()):
    """The body of the predict step: one fit-and-evaluate block per target."""
    started = time.perf_counter()
    # Fit time per predictor, summed over targets: the one-time cost the runtime comparison reports.
    fit_seconds: Dict[str, float] = {}
    if mode == "debug":
        logger.warning(
            "DEBUG MODE: the populations are far too small to fit a predictor. "
            "This run checks that the pipeline produces the files it should; "
            "every correlation it reports is noise."
        )

    table = NeuronDataset.load(path)
    train = table.for_split(spec.train_split)
    if not len(train):
        raise click.ClickException(f"The training population {spec.train_split!r} is empty.")

    metrics = tuple(metrics) if metrics else TARGET_METRICS
    kinds = tuple(target_kinds) if target_kinds else ("rank_model", "raw")
    columns = FEATURE_SETS[feature_set]

    has_activity = ACTIVITY_TARGET in table.targets
    if not has_activity:
        logger.warning(
            "This neuron table predates the activity column, so the 'active' "
            "scope falls back to 'score > 0'. That is a close proxy -- a zero "
            "score is almost always a never-active unit -- but it also catches "
            "units that fired only on successes, which is evidence rather than "
            "its absence. Rebuild the table to separate them exactly."
        )

    logger.info(
        "metrics %s · features '%s' (%d columns) · fit on '%s' (%d models, %d rows)",
        ", ".join(metrics), feature_set, len(columns),
        spec.train_split, train.n_models, len(train),
    )
    logger.info("SCOPE: dense models only; the table holds no conv or recurrent rows.")
    plain(logger, style("  " + TABLE_LEGEND, "dim"))

    sections: List[str] = []
    written: List[str] = []
    headline: List[Tuple[str, str, str, str, Optional[float], Optional[float]]] = []
    blocks = [(metric, kind) for metric in metrics for kind in kinds]
    fmt = lambda v: "   n/a" if v is None else f"{v:.4f}"

    for block, (metric, kind) in enumerate(blocks, start=1):
        target = metric if kind == "raw" else f"{metric}_{kind}"
        step.section(f"[{block}/{len(blocks)}] metric {metric} · target {target}")
        if metric == "dstar" and kind == "raw":
            logger.warning(
                "raw D* spans many orders of magnitude; the fit is reported for "
                "continuity but its R^2 is not interpretable. The rank targets "
                "are the ones to read.",
            )
        y_train = train.targets[target]

        finite = np.isfinite(y_train)
        if not finite.all():
            logger.warning(
                "dropping %d non-finite training target(s).", int((~finite).sum()),
            )
        X_train = train.feature_subset(columns)[finite]
        y_fit = y_train[finite]

        fit_started = time.perf_counter()
        fitted = {}
        for name, estimator in make_predictors(seed).items():
            one_started = time.perf_counter()
            estimator.fit(X_train, y_fit)
            fit_seconds[name] = fit_seconds.get(name, 0.0) + time.perf_counter() - one_started
            fitted[name] = estimator
        logger.info(
            "fitted %s on %d rows in %s",
            ", ".join(fitted), len(y_fit), format_duration(time.perf_counter() - fit_started),
        )
        logger.info(
            "%-12s%-20s%-7s%11s%11s%9s%9s%11s",
            "population", "predictor", "scope", "rho/model", "wtau", "top-k",
            "chance", "rho/layer",
        )

        rng = np.random.default_rng(seed)
        rows: List[dict] = []
        lines: List[str] = [
            f"metric: {metric}   target: {target}   features: {feature_set}",
            f"fitted on '{spec.train_split}' ({train.n_models} models, "
            f"{len(train)} rows)",
            "-" * 66,
            f"{'population':<12}{'predictor':<20}{'scope':<8}{'rho/model':>11}"
            f"{'sd':>8}{'wtau':>9}{'top-k':>8}{'chance':>8}{'rho/layer':>10}"
            f"{'n_inact':>9}",
        ]

        for split in spec.evaluation_splits:
            full = table.for_split(split)
            if not len(full):
                continue
            keep = np.isfinite(full.targets[target])
            full = full.select(keep)

            if has_activity:
                active_mask = full.targets[ACTIVITY_TARGET] > 0
            else:
                active_mask = full.targets[target] > 0

            predictions = {
                name: estimator.predict(full.feature_subset(columns))
                for name, estimator in fitted.items()
            }
            predictions["baseline:layer_mean"] = layer_mean_baseline(
                train, full, target
            )
            predictions["baseline:random"] = rng.random(len(full))

            for scope in SCOPES:
                subset_mask = np.ones(len(full), bool) if scope == "all" else active_mask
                if subset_mask.sum() < 2:
                    continue
                subset = full.select(subset_mask)
                y_true = subset.targets[target]

                by_name: Dict[str, Optional[float]] = {}
                for name, values in predictions.items():
                    scores = evaluate_ranking(
                        y_true, values[subset_mask],
                        subset.model_keys, subset.layer_names,
                        raw=subset.targets[metric],
                    )
                    by_name[name] = scores["spearman_model"]
                    row = {
                        "phase": spec.name, "mode": mode, "metric": metric,
                        "target": target, "population": split, "predictor": name,
                        "scope": scope, "n_models": subset.n_models,
                        "n_rows": len(subset),
                        "n_inactive": int((~active_mask).sum()),
                        "inactive_fraction": float((~active_mask).mean()),
                        **scores,
                    }
                    rows.append(row)
                    logger.info(
                        "%-12s%-20s%-7s%11s%11s%9s%9s%11s",
                        split, name, scope, fmt(scores["spearman_model"]),
                        fmt(scores["weightedtau_model"]), fmt(scores["top_k"]),
                        fmt(scores["top_k_chance"]), fmt(scores["spearman_layer"]),
                    )
                    lines.append(
                        f"{split:<12}{name:<20}{scope:<8}"
                        f"{fmt(scores['spearman_model']):>11}"
                        f"{fmt(scores['spearman_model_sd']):>8}"
                        f"{fmt(scores['weightedtau_model']):>9}"
                        f"{fmt(scores['top_k']):>8}"
                        f"{fmt(scores['top_k_chance']):>8}"
                        f"{fmt(scores['spearman_layer']):>10}"
                        f"{row['n_inactive']:>9}"
                    )

                best, best_rho, margin = _best_over_baseline(by_name)
                headline.append((target, split, scope, best, best_rho, margin))
                if best:
                    plain(logger, style(
                        f"{'':>9}=> best {best} rho/model {best_rho:.4f}, "
                        f"margin over layer_mean "
                        + ("n/a" if margin is None else f"{margin:+.4f}"),
                        "yellow",
                    ))

        written.append(write_results(spec.name, mode, dataset, metric, target, rows))
        sections.append("\n".join(lines) + "\n")

        if (save_estimators or save_predictors) and (not save_kinds or kind in save_kinds):
            import joblib

            to_save = {
                name: estimator for name, estimator in fitted.items()
                if save_estimators or name in save_predictors
            }
            for name, estimator in to_save.items():
                destination = ensure_dir(
                    variant_predictions_dir(spec.name, mode, dataset) / "estimators"
                ) / f"{spec.name}__{metric}__{target}__{name}.joblib"
                joblib.dump(estimator, destination)
            logger.info("Wrote %d fitted estimator(s).", len(to_save))

    log_path = write_correlation_log(spec.name, mode, dataset, sections)

    step.section("Headline: best predictor per target, its margin over baseline:layer_mean")
    plain(logger, style(
        f"  {'target':<26}{'population':<12}{'scope':<8}{'best predictor':<20}"
        f"{'rho/model':>10}{'margin':>9}", "bold",
    ))
    for target, split, scope, best, best_rho, margin in headline:
        plain(logger,
              f"  {target:<26}{split:<12}{scope:<8}{best or '-':<20}"
              f"{fmt(best_rho):>10}{'   n/a' if margin is None else f'{margin:+.4f}':>9}")
    plain(logger, style(
        "  margin <= 0: the predictor knows no more than which layer a neuron is in. "
        "Compare 'all' with 'active' for the dead-unit share.", "dim",
    ))

    plain(logger, "")
    for destination in written:
        logger.info("Wrote %s", display_path(destination))
    logger.info("Wrote %s", display_path(log_path))

    record_timing(
        spec.name, mode, dataset, "predict", time.perf_counter() - started,
        len(table) * len(blocks), "neuron x target",
        fit_seconds=round(sum(fit_seconds.values()), 2),
        fit_seconds_by_predictor={k: round(v, 2) for k, v in fit_seconds.items()},
        fit_rows=len(train), fit_models=train.n_models, targets=len(blocks),
    )
    step.summary = f"{len(blocks)} target(s) x {len(spec.evaluation_splits)} evaluation population(s)"


def _best_over_baseline(
    by_name: Dict[str, Optional[float]]
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """The best fitted predictor by rho/model, and its margin over layer_mean."""
    fitted = {n: v for n, v in by_name.items() if not n.startswith("baseline:") and v is not None}
    if not fitted:
        return None, None, None
    best = max(fitted, key=fitted.get)
    baseline = by_name.get("baseline:layer_mean")
    return best, fitted[best], None if baseline is None else fitted[best] - baseline


if __name__ == "__main__":
    main()
