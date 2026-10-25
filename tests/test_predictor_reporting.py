"""The reporting contract of the predictor evaluation.

Every number this script writes is meant to be quoted in a paper, so what the
tests pin is not "does it run" but the claims the reporting makes about itself:
that an undefined correlation is reported as undefined rather than as zero, that
every model counts once regardless of its depth, that the top-weighted
coefficient really does weight the top, that chance is computed per group, and
that the active-only scope excludes exactly the units that never fired.
"""

import math

import numpy as np
import pytest

from staticsbfl.data import IMAGE_DATASET_NAMES
from staticsbfl.utils.paths import (
    variant_correlation_log, variant_design, variant_models_csv, variant_neurons,
    variant_predictions_csv, variants_root,
)
from staticsbfl.cli.phase_predict import (
    EXTRA_TOP_K, LOG_PREAMBLE, RESULT_FIELDS, SCOPES, TOP_K_FRACTION,
    GT_FRACTIONS, evaluate_ranking, ground_truth_separation, near_misses, spearman, top_k_overlap,
    weighted_tau, _cell, _summarise,
)


# --------------------------------------------------------------------------
# Undefined is not zero
# --------------------------------------------------------------------------


def test_a_constant_vector_has_no_correlation_and_says_so():
    """The layer-mean baseline is constant within a layer by construction.

    Turning that into 0.0 reports "no association measured" as "association
    measured, and it was none", which is the difference between an honest N/A
    column and a misleading zero.
    """
    constant = np.ones(10)
    varying = np.arange(10.0)
    assert spearman(constant, varying) is None
    assert spearman(varying, constant) is None
    assert weighted_tau(constant, varying) is None


def test_a_single_element_group_cannot_be_correlated():
    assert spearman(np.array([1.0]), np.array([2.0])) is None
    assert top_k_overlap(np.array([1.0]), np.array([2.0]), 0.1) == (None, None)


def test_undefined_values_are_counted_and_excluded_from_the_mean():
    mean, sd, n, undefined = _summarise([0.4, None, 0.6, None])
    assert math.isclose(mean, 0.5)
    assert n == 2 and undefined == 2
    assert _summarise([None, None]) == (None, None, 0, 2)


def test_an_undefined_statistic_writes_a_blank_cell_never_a_zero():
    assert _cell(None) == ""
    assert _cell(0.0) == "0.000000"


# --------------------------------------------------------------------------
# The coefficients say what the log claims they say
# --------------------------------------------------------------------------


def test_weighted_tau_charges_a_swap_at_the_top_more_than_one_at_the_bottom():
    """The whole reason for reporting it next to Spearman.

    A suspiciousness ranking is read from the top down, so a coefficient that
    prices rank 2 like rank 2000 answers a question nobody asked. Spearman is
    asserted to be indifferent here precisely to show the contrast.
    """
    truth = np.arange(50.0)[::-1]
    top, bottom = truth.copy(), truth.copy()
    top[0], top[1] = top[1], top[0]
    bottom[-1], bottom[-2] = bottom[-2], bottom[-1]

    assert math.isclose(spearman(truth, top), spearman(truth, bottom))
    assert weighted_tau(truth, top) < weighted_tau(truth, bottom)


def test_top_k_chance_is_k_over_n_not_the_nominal_fraction():
    """k is a whole number of neurons, so chance moves with the group size."""
    truth = np.arange(7.0)
    _overlap, chance = top_k_overlap(truth, truth, 0.10)
    assert math.isclose(chance, 1 / 7)   # not 0.10
    _overlap, chance = top_k_overlap(np.arange(100.0), np.arange(100.0), 0.10)
    assert math.isclose(chance, 0.10)


def test_a_perfect_and_a_reversed_ranking_bracket_the_scale():
    truth = np.arange(20.0)
    assert math.isclose(spearman(truth, truth), 1.0)
    assert math.isclose(spearman(truth, -truth), -1.0)
    overlap, _chance = top_k_overlap(truth, truth, TOP_K_FRACTION)
    assert overlap == 1.0
    overlap, _chance = top_k_overlap(truth, -truth, TOP_K_FRACTION)
    assert overlap == 0.0


# --------------------------------------------------------------------------
# Aggregation: one model, one vote
# --------------------------------------------------------------------------


def _keys(counts):
    """model_keys for models of the given row counts."""
    return np.concatenate([np.full(n, f"m{i}") for i, n in enumerate(counts)])


def test_every_model_counts_once_however_many_neurons_it_has():
    """A 1000-neuron model must not outvote a 10-neuron one."""
    truth = np.concatenate([np.arange(1000.0), np.arange(10.0)])
    prediction = np.concatenate([np.arange(1000.0), -np.arange(10.0)])
    keys = _keys([1000, 10])
    layers = np.full(len(truth), "fc0")

    scores = evaluate_ranking(truth, prediction, keys, layers)
    # One model perfect, one reversed: the mean is 0, not 1000/1010 of the way up.
    assert math.isclose(scores["spearman_model"], 0.0, abs_tol=1e-9)
    assert scores["spearman_model_n"] == 2


def test_layers_are_summarised_within_a_model_before_models_are_averaged():
    """A ten-layer model must not get five times the weight of a two-layer one.

    Pooling every (model, layer) pair into one mean -- which this script used to
    do -- makes the per-layer figure a depth-weighted average, and phase 2 spans
    depths 1 to 10.
    """
    # Model A: 4 layers, all perfectly ranked. Model B: 1 layer, reversed.
    truth = np.concatenate([np.tile(np.arange(5.0), 4), np.arange(5.0)])
    prediction = np.concatenate([np.tile(np.arange(5.0), 4), -np.arange(5.0)])
    keys = _keys([20, 5])
    layers = np.concatenate([
        np.repeat([f"fc{i}" for i in range(4)], 5), np.full(5, "fc0"),
    ])
    scores = evaluate_ranking(truth, prediction, keys, layers)
    # +1 averaged within A, -1 within B, then 0 across the two models.
    assert math.isclose(scores["spearman_layer"], 0.0, abs_tol=1e-9)
    assert scores["spearman_layer_models"] == 2


def test_a_model_whose_layers_are_all_constant_is_undefined_not_zero():
    truth = np.array([3.0, 3.0, 3.0, 3.0])
    prediction = np.array([1.0, 2.0, 3.0, 4.0])
    scores = evaluate_ranking(truth, prediction, _keys([4]), np.full(4, "fc0"))
    assert scores["spearman_model"] is None
    assert scores["spearman_undefined"] == 1
    assert scores["spearman_layer"] is None


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


def test_both_scopes_are_always_reported():
    """The gap between them is the dead-unit contribution; one alone hides it."""
    assert SCOPES == ("all", "active")
    for field in ("scope", "n_inactive", "inactive_fraction",
                  "weightedtau_model", "top_k_chance", "spearman_undefined"):
        assert field in RESULT_FIELDS


def test_the_log_preamble_explains_every_reported_quantity():
    """A number whose meaning lives only in the author's head is not a result."""
    for phrase in ("WHICH PAIRS ARE CORRELATED", "HOW THEY ARE AGGREGATED",
                   "UNDEFINED VALUES", "THE TWO SCOPES", "BASELINES",
                   "weightedtau_model", "top_k_chance", "spearman_layer"):
        assert phrase in LOG_PREAMBLE


@pytest.mark.parametrize("dataset", IMAGE_DATASET_NAMES)
def test_every_dataset_gets_its_own_cache_tree(dataset):
    """p1_train_000 names a model in both; they must not share a filename."""
    root = variants_root("light", dataset)
    assert root.name == dataset
    assert variants_root("debug", dataset) != root


def test_the_debug_tree_is_separate_from_every_real_one():
    assert "variants-debug" in str(variants_root("debug"))
    assert "variants-debug" not in str(variants_root("light"))
    assert "variants-debug" not in str(variants_root("heavy"))


@pytest.mark.parametrize("dataset", IMAGE_DATASET_NAMES)
def test_every_mode_has_its_own_tree(dataset):
    """A heavy run must never find, skip or overwrite a light checkpoint.

    Keys are shared across modes -- p2_train_000 is the same architecture and
    seed in both -- so a shared tree would let heavy keep 3-epoch light models
    as if they were its own 10-epoch ones.
    """
    roots = {mode: variants_root(mode, dataset) for mode in ("debug", "light", "heavy")}
    assert len(set(roots.values())) == 3
    assert "variants-heavy" in str(roots["heavy"])
    for mode in ("debug", "light", "heavy"):
        assert variant_models_csv("phase2", mode, dataset).is_relative_to(roots[mode])


def test_an_unknown_mode_has_no_tree():
    with pytest.raises(ValueError):
        variants_root("haevy")


def test_results_are_addressable_by_phase_metric_and_target():
    a = variant_predictions_csv("phase1", "ochiai", "ochiai_rank_model")
    b = variant_predictions_csv("phase1", "dstar", "dstar_rank_model")
    assert a != b
    assert "ochiai" in a.name and "phase1" in a.name


def test_the_four_per_phase_artifacts_live_where_the_docs_say():
    names = {
        variant_design("phase1").name,
        variant_models_csv("phase1").name,
        variant_neurons("phase1").name,
        variant_correlation_log("phase1").name,
    }
    assert names == {"design.json", "models.csv", "neurons.npz", "correlations.log"}


# --------------------------------------------------------------------------
# Top-k at several sizes, and near misses
# --------------------------------------------------------------------------


def test_every_top_k_size_and_near_miss_column_is_written():
    for column in list(EXTRA_TOP_K) + [f"{c}_chance" for c in EXTRA_TOP_K]:
        assert column in RESULT_FIELDS
    for column in ("top_k_near", "top_k_near_chance", "top_k_true_percentile"):
        assert column in RESULT_FIELDS


def test_a_near_miss_counts_in_the_relaxed_recall_but_not_in_the_overlap():
    """20 neurons, true top-2 = {19, 18}; the prediction puts 19 first and 18 fourth."""
    truth = np.arange(20.0)
    prediction = truth.copy()
    prediction[18] = 15.5                                   # 18 drops to rank 4 of the prediction
    overlap, chance = top_k_overlap(truth, prediction, 0.10)
    assert overlap == 0.5 and chance == pytest.approx(2 / 20)
    near, near_chance, true_percentile = near_misses(truth, prediction, 0.10)
    assert near == 1.0 and near_chance == pytest.approx(4 / 20)
    # predicted top-2 = {19, 17}: true percentiles 19/19 and 17/19
    assert true_percentile == pytest.approx((19 / 19 + 17 / 19) / 2)


def test_true_percentile_brackets_perfect_reversed_and_random_rankings():
    rng = np.random.default_rng(5)
    truth = rng.normal(size=500)
    k, n = 50, 500
    ideal = 1 - (k - 1) / (2 * (n - 1))                     # mean percentile of the true top 50: ~0.951
    assert near_misses(truth, truth, 0.10)[2] == pytest.approx(ideal)
    assert near_misses(truth, -truth, 0.10)[2] == pytest.approx(1 - ideal)
    randoms = [near_misses(truth, rng.normal(size=500), 0.10)[2] for _ in range(200)]
    assert np.mean(randoms) == pytest.approx(0.5, abs=0.02)


def test_near_misses_are_undefined_for_a_single_neuron():
    assert near_misses(np.ones(1), np.ones(1), 0.10) == (None, None, None)


def test_evaluate_ranking_reports_every_top_k_size_with_its_own_chance():
    truth = np.arange(40.0)
    keys = np.array(["a"] * 20 + ["b"] * 20)
    layers = np.array(["fc0"] * 40)
    scores = evaluate_ranking(truth, truth, keys, layers)
    for column, fraction in EXTRA_TOP_K.items():
        assert scores[column] == 1.0
        assert scores[f"{column}_chance"] == pytest.approx(max(1, round(fraction * 20)) / 20)
    assert scores["top_k_near"] == 1.0 and scores["top_k_true_percentile"] == pytest.approx(
        np.mean([(19 / 19 + 18 / 19) / 2]))


# --------------------------------------------------------------------------
# Measured suspiciousness of the predicted top vs the rest
# --------------------------------------------------------------------------


def test_ground_truth_separation_on_a_hand_computed_case():
    """20 neurons with raw values 0..19; the prediction swaps neuron 19 to the bottom."""
    raw = np.arange(20.0)
    prediction = raw.copy()
    prediction[19] = -1.0                                   # the truly best neuron is predicted last
    out = ground_truth_separation(raw, prediction, 0.10)    # k = 2: predicted top = {18, 17}
    assert out["top"] == pytest.approx(17.5)
    assert out["rest"] == pytest.approx((raw.sum() - 35.0) / 18)
    assert out["ideal"] == pytest.approx(18.5)              # the true top-2 = {19, 18}
    mean = raw.mean()                                        # 9.5
    assert out["gain"] == pytest.approx((17.5 - mean) / (18.5 - mean))


def test_gain_is_one_for_a_perfect_ranking_and_about_zero_for_a_random_one():
    rng = np.random.default_rng(9)
    raw = rng.exponential(size=1000)
    assert ground_truth_separation(raw, raw, 0.05)["gain"] == pytest.approx(1.0)
    gains = [ground_truth_separation(raw, rng.normal(size=1000), 0.10)["gain"] for _ in range(300)]
    assert np.mean(gains) == pytest.approx(0.0, abs=0.03)
    assert ground_truth_separation(raw, -raw, 0.10)["gain"] < 0          # reversed: worse than random


def test_non_finite_raw_values_are_left_out_and_constant_ones_are_undefined():
    raw = np.array([1.0, 2.0, np.inf, 3.0, 4.0])
    out = ground_truth_separation(raw, raw, 0.25)            # 4 finite neurons, k = 1
    assert out["top"] == 4.0 and out["ideal"] == 4.0 and out["rest"] == pytest.approx(2.0)
    assert ground_truth_separation(np.ones(10), np.arange(10.0), 0.10)["gain"] is None
    assert ground_truth_separation(np.ones(1), np.ones(1), 0.10) == {"top": None, "rest": None,
                                                                      "ideal": None, "gain": None}


def test_evaluate_ranking_averages_the_separation_over_models():
    raw = np.concatenate([np.arange(20.0), 10 * np.arange(20.0)])
    keys = np.array(["a"] * 20 + ["b"] * 20)
    scores = evaluate_ranking(raw, raw, keys, np.array(["fc0"] * 40), raw=raw)
    # model a: top-2 = {19, 18} -> 18.5, model b: 185; mean over models
    assert scores["gt_top_10"] == pytest.approx((18.5 + 185.0) / 2)
    assert scores["gt_gain_10"] == pytest.approx(1.0) and scores["gt_gain_05"] == pytest.approx(1.0)
    for suffix in GT_FRACTIONS:
        assert f"gt_top_{suffix}" in RESULT_FIELDS and f"gt_gain_{suffix}" in RESULT_FIELDS
    without = evaluate_ranking(raw, raw, keys, np.array(["fc0"] * 40))
    assert without["gt_top_10"] is None                     # no raw values given: undefined, not zero
