"""The accuracy half of the evaluation, against hand-made result tables.

Every expected value below is computed by hand from the mock rows, so a test
fails if the evaluation reads a cell wrongly, picks something other than the
maximum, lets a baseline or an undefined cell win, filters the wrong rows, or
highlights the wrong entries.
"""

import csv
import json
import math

import numpy as np
import pytest

from staticsbfl.evaluation import accuracy as acc
from staticsbfl.utils import paths
from staticsbfl.cli.phase_predict import RESULT_FIELDS


def mock(population, predictor, scope, rho, *, metric="ochiai", target=None, wtau=None, top_k=0.3,
         chance=0.1, sd=0.1, n_models=20):
    return {
        "phase": "phase3", "mode": "light", "metric": metric, "target": target or f"{metric}_rank_model",
        "population": population, "predictor": predictor, "scope": scope, "n_models": n_models,
        "inactive_fraction": 0.1, "spearman_model": rho, "spearman_model_sd": sd,
        "weightedtau_model": rho if wtau is None else wtau, "weightedtau_model_sd": 0.1, "top_k": top_k,
        "top_k_chance": chance, "spearman_layer": None, "model_type": acc.PREDICTOR_TYPES.get(predictor, "other"),
    }


@pytest.fixture
def rows():
    return [
        mock("deeper", "gradient_boosting", "active", 0.70, top_k=0.40),
        mock("deeper", "random_forest", "active", 0.72, top_k=0.35, wtau=0.50),
        mock("deeper", "mlp", "active", None, top_k=0.90),          # undefined rho must not win
        mock("deeper", "baseline:layer_mean", "active", 0.25, top_k=0.20),
        mock("deeper", "baseline:random", "active", 0.01, top_k=0.10),
        mock("deeper", "random_forest", "all", 0.80),
        mock("wider", "mlp", "active", 0.60, metric="tarantula"),
        mock("wider", "baseline:layer_mean", "active", 0.30, metric="tarantula"),
        mock("wider", "baseline:random", "active", 0.99, metric="tarantula"),  # a baseline beating everyone
    ]


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def test_a_blank_or_nan_cell_is_undefined_not_zero():
    assert acc.number("") is None and acc.number("nan") is None and acc.number(None) is None
    assert acc.number("0.000000") == 0.0 and acc.number("-0.25") == -0.25
    assert acc.number("abc") is None


def write_results(phase, mode, rows_by_file):
    directory = paths.variant_predictions_dir(phase, mode, "cifar10")
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in rows_by_file.items():
        with open(directory / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


@pytest.fixture
def trees(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "VARIANTS_DIR", tmp_path / "variants")
    monkeypatch.setattr(paths, "HEAVY_VARIANTS_DIR", tmp_path / "variants-heavy")
    return tmp_path


def test_results_are_read_with_numbers_parsed_and_types_assigned(trees):
    write_results("phase1", "light", {"phase1__ochiai__ochiai_rank_model.csv": [
        {"phase": "phase1", "metric": "ochiai", "target": "ochiai_rank_model", "population": "test",
         "predictor": "mlp", "scope": "active", "n_models": "20", "spearman_model": "0.5", "spearman_layer": "",
         "top_k": "0.25", "top_k_chance": "0.1"},
        {"phase": "phase1", "metric": "ochiai", "target": "ochiai_rank_model", "population": "test",
         "predictor": "baseline:random", "scope": "active", "n_models": "20", "spearman_model": "nan"},
    ]})
    found, files = acc.read_results("phase1", "light", "cifar10")
    assert files == ["phase1__ochiai__ochiai_rank_model.csv"]
    assert found[0]["spearman_model"] == 0.5 and found[0]["spearman_layer"] is None
    assert found[0]["top_k"] == 0.25 and found[0]["n_models"] == 20
    assert found[0]["model_type"] == "neural_network" and found[1]["model_type"] == "baseline"
    assert found[1]["spearman_model"] is None
    assert acc.read_results("phase4", "light", "cifar10") == ([], [])
    assert acc.load_phase("phase4", "light", "cifar10") is None


def test_load_phase_orders_populations_as_the_design_does_and_reads_accuracy(trees):
    rows = [dict(mock(p, "mlp", "active", 0.5), phase="phase3") for p in ("wider", "saturating", "deeper")]
    write_results("phase3", "light", {"phase3__ochiai__ochiai_rank_model.csv": rows})
    root = paths.variants_root("light", "cifar10") / "phase3"
    (root / "design.json").write_text(json.dumps({"evaluation_populations": ["deeper", "wider", "saturating"],
                                                  "fit_population": "base"}))
    with open(root / "models.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "split", "test_accuracy"])
        writer.writeheader()
        writer.writerows([{"key": "a", "split": "deeper", "test_accuracy": "0.40"},
                          {"key": "b", "split": "deeper", "test_accuracy": "0.50"},
                          {"key": "c", "split": "wider", "test_accuracy": ""}])
    result = acc.load_phase("phase3", "light", "cifar10")
    assert result.populations == ["deeper", "wider", "saturating"]
    assert result.accuracy == {"deeper": pytest.approx(0.45)}
    assert not result.complete and result.expected_files == 6


# --------------------------------------------------------------------------
# The maximum
# --------------------------------------------------------------------------


def test_the_best_case_is_the_maximum_over_fitted_predictors_on_the_requested_scope(rows):
    best = acc.best_of(rows, "spearman_model", population="deeper", scope="active")
    assert best.row["predictor"] == "random_forest" and best.value == 0.72
    # scope "all" has its own maximum
    assert acc.best_of(rows, "spearman_model", population="deeper", scope="all").value == 0.80
    # the baseline at 0.99 never wins; the fitted maximum is 0.60
    best = acc.best_of(rows, "spearman_model", population="wider", scope="active")
    assert best.row["predictor"] == "mlp" and best.value == 0.60


def test_an_undefined_cell_never_wins_the_maximum_but_its_other_columns_count(rows):
    # mlp has rho None but the highest top-k (0.90): it wins top-k, not rho
    assert acc.best_of(rows, "spearman_model", population="deeper", scope="active").row["predictor"] != "mlp"
    best_top = acc.best_of(rows, "top_k", population="deeper", scope="active")
    assert best_top.row["predictor"] == "mlp" and best_top.value == 0.90


def test_the_maximum_matches_a_brute_force_maximum_on_random_tables():
    rng = np.random.default_rng(7)
    for _ in range(50):
        table = [mock("p", rng.choice(["gradient_boosting", "random_forest", "mlp", "baseline:random"]),
                      "active", float(rng.uniform(-1, 1))) for _ in range(12)]
        fitted = [r["spearman_model"] for r in table if not r["predictor"].startswith("baseline:")]
        best = acc.best_of(table, "spearman_model", population=None, scope="active")
        assert (best is None and not fitted) or best.value == max(fitted)


def test_the_baselines_come_from_the_same_case(rows):
    best = acc.best_of(rows, "spearman_model", population="deeper", scope="active")
    assert best.layer_mean == 0.25 and best.random == 0.01
    assert best.margin == pytest.approx(0.47)
    assert best.case == "random_forest · ochiai_rank_model"


def test_the_phase_best_names_its_population(rows):
    best = acc.best_of(rows, "spearman_model", population=None, scope="active")
    assert best.row["population"] == "deeper" and best.value == 0.72


def test_best_per_metric_reports_each_metric_separately(rows):
    assert acc.best_per_metric(rows, "deeper", "active") == {"tarantula": None, "ochiai": 0.72, "dstar": None}
    assert acc.best_per_metric(rows, "wider", "active") == {"tarantula": 0.60, "ochiai": None, "dstar": None}


def test_candidates_count_fitted_rows_only(rows):
    assert acc.n_candidates(rows, "deeper", "active") == 3
    assert acc.n_candidates(rows, "wider", "active") == 1


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------


def test_filtering_by_model_type_keeps_the_baselines_and_changes_the_best(rows):
    trees_only = acc.ResultFilter(model_types=("tree_ensemble",)).apply(rows)
    assert {r["predictor"] for r in trees_only} == {"gradient_boosting", "random_forest",
                                                    "baseline:layer_mean", "baseline:random"}
    assert acc.best_of(trees_only, "top_k", population="deeper", scope="active").value == 0.40
    neural = acc.ResultFilter(model_types=("neural_network",)).apply(rows)
    assert acc.best_of(neural, "spearman_model", population="deeper", scope="active") is None
    assert acc.best_of(neural, "spearman_model", population="wider", scope="active").value == 0.60


def test_filtering_by_predictor_metric_and_population(rows):
    assert all(r["predictor"] in ("mlp",) or r["predictor"].startswith("baseline:")
               for r in acc.ResultFilter(predictors=("mlp",)).apply(rows))
    assert {r["metric"] for r in acc.ResultFilter(metrics=("tarantula",)).apply(rows)} == {"tarantula"}
    assert {r["population"] for r in acc.ResultFilter(populations=("wider",)).apply(rows)} == {"wider"}
    assert acc.NO_FILTER.apply(rows) == rows and not acc.NO_FILTER.active
    described = acc.ResultFilter(model_types=("tree_ensemble",), metrics=("ochiai",)).describe()
    assert "model type tree_ensemble" in described and "metric ochiai" in described


# --------------------------------------------------------------------------
# Overview and uncertainty
# --------------------------------------------------------------------------


def test_the_overview_row_holds_the_hand_computed_values(rows):
    result = acc.PhaseResult("phase3", "light", "cifar10", rows, files=[], populations=["deeper", "wider"])
    table = acc.overview_rows([result], "active")
    deeper = next(r for r in table if r["population"] == "deeper")
    assert deeper["best_rho_model"] == 0.72 and deeper["layer_mean_rho"] == 0.25
    assert deeper["margin_over_layer_mean"] == pytest.approx(0.47)
    assert deeper["best_rho_model_all"] == 0.80
    assert deeper["best_wtau"] == 0.70                       # gradient boosting: wtau = rho = 0.70 > 0.50
    assert deeper["best_top_k"] == 0.90 and deeper["top_k_x_chance"] == pytest.approx(9.0)
    assert deeper["n_candidates"] == 3 and deeper["mode"] == "light"
    wider = next(r for r in table if r["population"] == "wider")
    assert wider["margin_over_layer_mean"] == pytest.approx(0.30)


def test_typical_uncertainty_is_two_sd_over_root_n(rows):
    result = acc.PhaseResult("phase3", "light", "cifar10", rows, files=[], populations=["deeper", "wider"])
    band, sd, n = acc.typical_uncertainty([result], "active")
    assert sd == 0.1 and n == 20 and band == pytest.approx(2 * 0.1 / math.sqrt(20))


def test_the_summary_csv_round_trips(rows, trees):
    result = acc.PhaseResult("phase3", "light", "cifar10", rows, files=[], populations=["deeper", "wider"])
    paths.variants_root("light", "cifar10").mkdir(parents=True)
    path = acc.write_summary(acc.overview_rows([result], "active"), "light", "cifar10", "active")
    written = list(csv.DictReader(open(path)))
    assert float(written[0]["best_rho_model"]) == pytest.approx(0.72)
    assert acc.write_summary([], "light", "cifar10", "active") is None


# --------------------------------------------------------------------------
# Highlights
# --------------------------------------------------------------------------


def test_quantile_matches_numpy():
    rng = np.random.default_rng(3)
    for size in (1, 2, 7, 50):
        values = list(rng.normal(size=size))
        for q in (0.0, 0.5, 0.9, 1.0):
            assert acc.quantile(values, q) == pytest.approx(float(np.quantile(values, q)))


def test_the_best_and_the_top_decile_of_a_phase_are_marked():
    # eleven fitted rows with rho 0.0, 0.1, ..., 1.0: the 90 % quantile is 0.9
    table = [mock("p", "mlp", "active", round(i / 10, 1), wtau=0.0, top_k=0.0) for i in range(11)]
    table.append(mock("p", "baseline:random", "active", 5.0))    # never marked, even when larger
    table.append(mock("p", "mlp", "active", None))               # undefined: never marked
    marks = acc.phase_highlights(table, ["spearman_model"])
    assert marks[(10, "spearman_model")] == "best"
    assert marks[(9, "spearman_model")] == "high"
    assert (8, "spearman_model") not in marks
    assert (11, "spearman_model") not in marks and (12, "spearman_model") not in marks


def test_ties_at_the_maximum_are_all_best_and_scopes_are_marked_apart():
    table = [mock("p", "mlp", "active", 0.5), mock("p", "random_forest", "active", 0.5),
             mock("p", "mlp", "all", 0.2), mock("p", "random_forest", "all", 0.1)]
    marks = acc.phase_highlights(table, ["spearman_model"])
    assert marks[(0, "spearman_model")] == marks[(1, "spearman_model")] == "best"
    assert marks[(2, "spearman_model")] == "best"          # best of scope "all", although lower
    assert (3, "spearman_model") not in marks              # 0.1 < quantile(0.9) of [0.1, 0.2] = 0.19
