"""Descriptive statistics of the result rows: right values, nothing lost, nothing counted twice.

``describe_values`` is checked against numpy on random data; the grouping
functions are checked for the partition property -- every row lands in exactly
one group, so the groups' defined + undefined counts add up to the all-rows
line -- and for keeping the two scopes apart.
"""

import csv
import math

import numpy as np
import pytest

from staticsbfl.evaluation import accuracy as acc


def test_describe_values_matches_numpy_and_ignores_undefined():
    rng = np.random.default_rng(1)
    values = list(rng.normal(size=37))
    noisy = values + [None, float("nan"), None]
    out = acc.describe_values(noisy)
    assert out["rows"] == 37 and out["undefined"] == 3
    assert out["min"] == pytest.approx(min(values)) and out["max"] == pytest.approx(max(values))
    assert out["mean"] == pytest.approx(float(np.mean(values)))
    assert out["median"] == pytest.approx(float(np.median(values)))
    assert out["q25"] == pytest.approx(float(np.quantile(values, 0.25)))
    assert out["q75"] == pytest.approx(float(np.quantile(values, 0.75)))
    assert out["sd"] == pytest.approx(float(np.std(values)))
    assert out["range"] == pytest.approx(max(values) - min(values))


def test_an_all_undefined_group_has_counts_but_no_statistics():
    out = acc.describe_values([None, None])
    assert out["rows"] == 0 and out["undefined"] == 2
    assert all(out[k] is None for k in ("min", "median", "mean", "max", "sd", "range"))


def mock_rows(rng, n=60):
    predictors = ["gradient_boosting", "random_forest", "mlp", "baseline:layer_mean", "baseline:random"]
    rows = []
    for i in range(n):
        predictor = predictors[i % len(predictors)]
        rows.append({"predictor": predictor, "model_type": acc.PREDICTOR_TYPES[predictor],
                     "metric": ["tarantula", "ochiai", "dstar"][i % 3], "population": ["deeper", "wider"][i % 2],
                     "scope": ["active", "all"][(i // 2) % 2], "target": "ochiai_rank_model",
                     "spearman_model": None if i % 7 == 0 else float(rng.uniform(-1, 1)),
                     "top_k": float(rng.uniform(0, 1))})
    return rows


@pytest.mark.parametrize("group_by", ["model_type", "predictor", "metric", "population", "scope"])
def test_groups_partition_the_rows(group_by):
    rows = mock_rows(np.random.default_rng(2))
    stats = acc.group_statistics(rows, "spearman_model", group_by)
    groups = [s for s in stats if s["group"] != "*"]
    everything = next(s for s in stats if s["group"] == "*")
    assert {s["group"] for s in groups} == {str(r[group_by]) for r in rows}
    assert sum(s["rows"] + s["undefined"] for s in groups) == len(rows)
    assert everything["rows"] + everything["undefined"] == len(rows)
    assert sum(s["rows"] for s in groups) == everything["rows"]
    assert everything["undefined"] == sum(1 for r in rows if r["spearman_model"] is None)
    # the overall extremes are the extremes of the group extremes
    assert everything["max"] == max(s["max"] for s in groups if s["max"] is not None)
    assert everything["min"] == min(s["min"] for s in groups if s["min"] is not None)


def test_a_group_holds_exactly_its_rows():
    rows = mock_rows(np.random.default_rng(3))
    stats = {s["group"]: s for s in acc.group_statistics(rows, "top_k", "model_type")}
    trees = [r["top_k"] for r in rows if r["model_type"] == "tree_ensemble"]
    assert stats["tree_ensemble"]["rows"] == len(trees)
    assert stats["tree_ensemble"]["mean"] == pytest.approx(sum(trees) / len(trees))
    assert stats["tree_ensemble"]["max"] == max(trees)


def test_the_statistics_table_keeps_scopes_apart_and_groups_by_scope_only_over_both():
    rows = mock_rows(np.random.default_rng(4))
    result = acc.PhaseResult("phase3", "light", "cifar10", rows, files=[], populations=["deeper", "wider"])
    table = acc.statistics_table([result], columns=("spearman_model",))
    by_scope = [s for s in table if s["group_by"] == "scope"]
    assert by_scope and all(s["scope"] == "*" for s in by_scope)
    assert all(s["scope"] in ("active", "all") for s in table if s["group_by"] != "scope")
    for scope in ("active", "all"):
        line = next(s for s in table if s["scope"] == scope and s["group_by"] == "model_type" and s["group"] == "*")
        assert line["rows"] + line["undefined"] == sum(1 for r in rows if r["scope"] == scope)


def test_the_statistics_csv_round_trips(tmp_path):
    rows = mock_rows(np.random.default_rng(5))
    result = acc.PhaseResult("phase3", "light", "cifar10", rows, files=[], populations=["deeper", "wider"])
    table = acc.statistics_table([result], columns=("top_k",), groupings=("model_type",))
    path = acc.write_statistics(table, tmp_path / "s" / "statistics.csv")
    written = list(csv.DictReader(open(path)))
    assert len(written) == len(table)
    first = next(w for w in written if w["group"] == "tree_ensemble" and w["scope"] == "active")
    expected = next(t for t in table if t["group"] == "tree_ensemble" and t["scope"] == "active")
    assert float(first["mean"]) == pytest.approx(expected["mean"]) and int(first["rows"]) == expected["rows"]
