"""The runtime half of the evaluation, against hand-made run tables.

The mock runs below are small enough to compute every reported figure by hand:
per-model medians over repeats, medians and maxima over models (and which
model is the maximum), speed-ups as geometric means, the one-time cost and the
break-even point, and the fidelity summary. Failed runs must be ignored, and a
maximum must be the true maximum.
"""

import json
import math

import numpy as np
import pytest

from staticsbfl.evaluation import runtime as rt
from staticsbfl.utils import paths


def run(tool, model, repeat, seconds, *, memory=10.0, process=100.0, status="ok", setup=1.0, gpu=None, neurons=384):
    return {"tool": tool, "model": model, "repeat": str(repeat), "status": status, "per_model_seconds": str(seconds),
            "setup_seconds": str(setup), "per_model_peak_above_start_bytes": str(memory),
            "process_lifetime_max_bytes": str(process), "gpu_peak_above_baseline_bytes": "" if gpu is None else str(gpu),
            "n_neurons": str(neurons), "detail": ""}


M1, M2 = "p4_designed_small_mlp_s0", "p4_designed_wide_mlp_s0"


@pytest.fixture
def runs():
    return [
        # static MLP: m1 repeats 0.01, 0.03, 0.02 -> median 0.02; m2 single 0.05
        run("staticsbfl_mlp", M1, 0, 0.01, memory=5), run("staticsbfl_mlp", M1, 1, 0.03, memory=7),
        run("staticsbfl_mlp", M1, 2, 0.02, memory=6), run("staticsbfl_mlp", M2, 0, 0.05, memory=30, neurons=3584),
        # static GB: slower than HS@pt on m1
        run("staticsbfl_gradient_boosting", M1, 0, 0.5), run("staticsbfl_gradient_boosting", M2, 0, 1.2, neurons=3584),
        # HS@pt CPU: m1 0.2, 0.4, 0.3 -> 0.3; m2 1.0; a failed run with an absurd time must be ignored
        run("hs_pt_cpu", M1, 0, 0.2), run("hs_pt_cpu", M1, 1, 0.4), run("hs_pt_cpu", M1, 2, 0.3),
        run("hs_pt_cpu", M2, 0, 1.0, memory=80, neurons=3584), run("hs_pt_cpu", M2, 1, 999.0, status="failed"),
        # DeepFault: m1 100 s, m2 300 s, the maximum
        run("deepfault_tf1", M1, 0, 100.0, memory=200, process=600), run("deepfault_tf1", M2, 0, 300.0, memory=900,
                                                                       process=1500, neurons=3584),
    ]


def test_per_model_values_are_medians_over_repeats_of_successful_runs(runs):
    seconds = rt.per_model_medians(runs, "per_model_seconds")
    assert seconds["staticsbfl_mlp"] == {M1: 0.02, M2: 0.05}
    assert seconds["hs_pt_cpu"] == {M1: 0.3, M2: 1.0}          # 999 s (failed) ignored


def test_tool_rows_report_the_true_median_and_maximum_and_where_it_is(runs):
    rows = {r.tool: r for r in rt.tool_rows(runs)}
    df = rows["deepfault_tf1"]
    assert df.seconds_median == 200.0 and df.seconds_max == 300.0 and df.seconds_max_model == M2
    assert df.seconds_total == 400.0
    assert df.seconds_mean == 200.0 and df.seconds_min == 100.0
    assert df.memory_mean == pytest.approx((200 + 900) / 2) and df.memory_min == 200.0
    assert df.memory_max == 900.0 and df.memory_max_model == M2 and df.process_max == 1500.0
    mlp = rows["staticsbfl_mlp"]
    assert mlp.seconds_median == pytest.approx(0.035) and mlp.seconds_max == 0.05
    assert mlp.memory_median == pytest.approx((6 + 30) / 2) and mlp.memory_max == 30.0
    assert mlp.seconds_mean == pytest.approx(0.035) and mlp.seconds_min == 0.02
    assert mlp.memory_mean == pytest.approx(18.0) and mlp.memory_min == 6.0
    assert rows["hs_pt_cpu"].status == {"ok": 4, "failed": 1}
    # reporting order follows TOOL_LABELS, not the order of the runs
    assert [r.tool for r in rt.tool_rows(runs)] == [
        "staticsbfl_gradient_boosting", "staticsbfl_mlp", "hs_pt_cpu", "deepfault_tf1"]


def test_the_maximum_matches_a_brute_force_maximum_on_random_runs():
    rng = np.random.default_rng(11)
    for _ in range(30):
        table = [run("hs_pt_cpu", f"p4_designed_m{m}_s0", k, float(rng.uniform(0.01, 5)))
                 for m in range(6) for k in range(3)]
        medians = {m: np.median([float(r["per_model_seconds"]) for r in table if r["model"] == m])
                   for m in {r["model"] for r in table}}
        row = rt.tool_rows(table)[0]
        assert row.seconds_max == pytest.approx(max(medians.values()))
        assert row.seconds_max_model == max(medians, key=medians.get)
        assert row.seconds_median == pytest.approx(float(np.median(list(medians.values()))))
        assert row.seconds_mean == pytest.approx(float(np.mean(list(medians.values()))))
        assert row.seconds_min == pytest.approx(min(medians.values()))


def test_per_design_rows_group_seeds_and_sort_by_size(runs):
    designs = rt.by_design(runs)
    assert [d["design"] for d in designs] == ["small_mlp", "wide_mlp"]
    assert designs[0]["neurons"] == 384 and designs[1]["neurons"] == 3584
    assert designs[1]["deepfault_tf1"] == 300.0 and designs[0]["staticsbfl_mlp"] == 0.02


def test_speedups_are_geometric_means_over_the_common_models(runs):
    speedups = {(s.static, s.dynamic): s for s in rt.speedups(runs)}
    s = speedups[("staticsbfl_mlp", "hs_pt_cpu")]
    assert s.minimum == pytest.approx(15.0) and s.maximum == pytest.approx(20.0)   # 0.3/0.02, 1.0/0.05
    assert s.geometric_mean == pytest.approx(math.sqrt(15.0 * 20.0)) and s.models == 2
    slower = speedups[("staticsbfl_gradient_boosting", "hs_pt_cpu")]
    assert slower.geometric_mean < 1                                      # 0.3/0.5 and 1.0/1.2
    assert slower.geometric_mean == pytest.approx(math.sqrt(0.6 * (1.0 / 1.2)))


COST = {"train_seconds": 100.0, "score_seconds": 10.0, "neurons_seconds": 5.0, "predict_seconds": 400.0,
        "fit_by_predictor": {"mlp": 85.0}, "train_note": ""}


def test_amortization_is_the_one_time_cost_over_the_median_saving(runs):
    entries = {(a.static, a.dynamic): a for a in rt.amortization(runs, COST)}
    a = entries[("staticsbfl_mlp", "hs_pt_cpu")]
    assert a.one_time_seconds == 200.0 and not a.fit_is_upper_bound_of_predict
    assert a.saving_per_model == pytest.approx(statistics_median([0.3 - 0.02, 1.0 - 0.05]))
    assert a.models_to_pay_off == math.ceil(200.0 / a.saving_per_model)            # 326
    assert a.models_to_pay_off_without_training == math.ceil(100.0 / a.saving_per_model)
    # gradient boosting has no recorded fit: the whole predict step stands in, flagged
    gb = entries[("staticsbfl_gradient_boosting", "deepfault_tf1")]
    assert gb.fit_is_upper_bound_of_predict and gb.one_time_seconds == 515.0
    # slower than HS@pt per model: no break-even point at all
    never = entries[("staticsbfl_gradient_boosting", "hs_pt_cpu")]
    assert never.saving_per_model == pytest.approx(statistics_median([0.3 - 0.5, 1.0 - 1.2]))
    assert never.models_to_pay_off is None


def statistics_median(values):
    import statistics
    return statistics.median(values)


def test_amortization_without_recorded_costs_says_so(runs):
    entries = rt.amortization(runs, {"fit_by_predictor": {}})
    assert entries and all(e.dynamic is None and e.one_time_seconds is None for e in entries)


def test_one_time_cost_takes_the_fit_populations_share_of_the_score_step():
    cost_rows = [
        {"mode": "light", "dataset": "cifar10", "phase": "phase4", "population": "sampled", "step": "train",
         "n_models": 100, "total_seconds": 900.0, "note": ""},
        {"mode": "light", "dataset": "cifar10", "phase": "phase4", "population": "*", "step": "train",
         "n_models": 120, "total_seconds": 1000.0},
        {"mode": "light", "dataset": "cifar10", "phase": "phase4", "population": "*", "step": "score",
         "total_seconds": 120.0},
        {"mode": "light", "dataset": "cifar10", "phase": "phase4", "population": "*", "step": "neurons",
         "total_seconds": 20.0},
        {"mode": "light", "dataset": "cifar10", "phase": "phase4", "population": "*", "step": "predict",
         "total_seconds": 600.0, "fit_seconds_by_predictor": json.dumps({"mlp": 280.0})},
        {"mode": "heavy", "dataset": "cifar10", "phase": "phase4", "population": "sampled", "step": "train",
         "n_models": 500, "total_seconds": 99999.0},
    ]
    cost = rt.one_time_cost("light", "cifar10", cost_rows)
    assert cost["train_seconds"] == 900.0                      # the sampled population, light only
    assert cost["score_seconds"] == pytest.approx(100.0)       # 120 s x 100/120
    assert cost["neurons_seconds"] == 20.0 and cost["fit_by_predictor"] == {"mlp": 280.0}


def test_fidelity_summary_counts_and_takes_the_worst_case():
    rows = [
        {"tool": "deepfault_tf1", "model": M1, "reference": "hs_pt_cpu", "n_neurons": "384", "neurons_differing": "2",
         "max_count_difference": "1", "identical": "False", "spearman_tarantula": "0.9999996", "spearman_ochiai": "1.0",
         "spearman_dstar": ""},
        {"tool": "deepfault_tf1", "model": M2, "reference": "hs_pt_cpu", "n_neurons": "3584", "neurons_differing": "0",
         "max_count_difference": "0", "identical": "True", "spearman_tarantula": "1.0", "spearman_ochiai": "1.0",
         "spearman_dstar": "1.0"},
    ]
    summary = rt.fidelity_summary(rows)[0]
    assert summary.identical == 1 and summary.models == 2
    assert summary.neurons_differing == 2 and summary.neurons == 3968 and summary.max_count_difference == 1
    assert summary.min_rho == {"tarantula": 0.9999996, "ochiai": 1.0, "dstar": 1.0}


def test_compare_spectra_counts_differences_and_rank_agreement():
    ref = {"a_f": [np.array([1, 2, 3, 4])], "a_s": [np.array([5, 5, 5, 5])],
           "tarantula": [np.array([0.1, 0.2, 0.3, 0.4])], "ochiai": [np.zeros(4)]}
    same = rt.compare_spectra(ref, ref)
    assert same["identical"] and same["neurons_differing"] == 0 and same["spearman_tarantula"] == pytest.approx(1.0)
    assert same["spearman_ochiai"] is None                     # constant: rho undefined, not 0 or 1
    got = {"a_f": [np.array([1, 2, 3, 6])], "a_s": [np.array([5, 5, 4, 5])],
           "tarantula": [np.array([0.4, 0.3, 0.2, 0.1])]}
    diff = rt.compare_spectra(got, ref)
    assert diff["neurons_differing"] == 2 and diff["max_count_difference"] == 2
    assert diff["spearman_tarantula"] == pytest.approx(-1.0) and diff["spearman_ochiai"] is None


def test_a_record_and_its_status_become_one_flat_row(tmp_path):
    phases = [
        {"name": "import", "seconds": 1.5, "bytes_before": 0, "bytes_after": 100, "sampled_peak_bytes": 120,
         "sampled_peak_above_start_bytes": 120, "lifetime_max_after_bytes": 120, "lifetime_max_increase_bytes": 120},
        {"name": "spectrum", "seconds": 0.25, "bytes_before": 100, "bytes_after": 110, "sampled_peak_bytes": 300,
         "sampled_peak_above_start_bytes": 200, "lifetime_max_after_bytes": 300, "lifetime_max_increase_bytes": 180},
    ]
    (tmp_path / "k__r0.json").write_text(json.dumps({"n_neurons": 384, "phases": phases, "exact_malloc": True,
                                                     "process_lifetime_max_bytes": 300}))
    (tmp_path / "k__r0.status.json").write_text(json.dumps({"status": "ok", "process_wall_seconds": 2.0}))
    row = rt.flatten("hs_pt_cpu", "k", 0, tmp_path)
    assert row["status"] == "ok" and row["n_neurons"] == 384 and row["exact_malloc"] is True
    assert row["setup_seconds"] == 1.5 and row["per_model_seconds"] == 0.25
    assert row["per_model_peak_above_start_bytes"] == 200          # 300 - 100
    assert row["spectrum_seconds"] == 0.25 and row["spectrum_peak_bytes"] == 180
    (tmp_path / "x__r0.status.json").write_text(json.dumps({"status": "timeout", "detail": "exceeded 10 s"}))
    missing = rt.flatten("deepfault_tf1", "x", 0, tmp_path)
    assert missing["status"] == "timeout" and "per_model_seconds" not in missing


def test_summarise_reads_the_tables_from_disk_and_filters_tools(tmp_path, monkeypatch, runs):
    import csv as csv_module

    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(paths, "VARIANTS_DIR", tmp_path / "variants")
    root = paths.runtime_root("light", "cifar10")
    root.mkdir(parents=True)
    columns = sorted({k for r in runs for k in r})
    with open(root / "runs.csv", "w", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(runs)
    summary = rt.summarise("light", "cifar10")
    assert {t.tool for t in summary.tools} == {"staticsbfl_mlp", "staticsbfl_gradient_boosting", "hs_pt_cpu",
                                               "deepfault_tf1"}
    only = rt.summarise("light", "cifar10", tools=("staticsbfl_mlp", "hs_pt_cpu"))
    assert {t.tool for t in only.tools} == {"staticsbfl_mlp", "hs_pt_cpu"}
    assert rt.summarise("heavy", "cifar10") is None
    written = rt.write_summary(summary)
    assert {p.name for p in written} == {"summary.csv", "by_design.csv"}
