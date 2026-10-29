"""The evaluation end to end on a mock cache: printout, log, CSVs and HTML report.

A tiny tree is written by hand -- one phase in light and heavy, one runtime
run -- and ``staticsbfl.cli.evaluate`` is run on it. The test checks that the log
carries the headline with the right maximum and the light-vs-heavy table, that
the HTML lists every row, marks the phase maximum (and only fitted rows), keeps
the filterable attributes, explains the memory measurement, and that a filter
removes exactly what it should.
"""

import csv
import json
import re

import pytest
from click.testing import CliRunner

from staticsbfl.evaluation import costs
from staticsbfl.utils import paths
from staticsbfl.viz import html_report
from staticsbfl.cli import evaluate
from staticsbfl.cli.phase_predict import RESULT_FIELDS


def result_row(predictor, rho, *, population="test", scope="active", metric="ochiai", top_k=0.3):
    return {"phase": "phase1", "mode": "light", "metric": metric, "target": f"{metric}_rank_model",
            "population": population, "predictor": predictor, "scope": scope, "n_models": 20,
            "n_rows": 1000, "n_inactive": 10, "inactive_fraction": 0.01, "spearman_model": rho,
            "spearman_model_sd": 0.05, "weightedtau_model": rho, "weightedtau_model_sd": 0.05,
            "top_k": top_k, "top_k_chance": 0.1, "spearman_layer": "", "r2": ""}


PHASE1 = [
    result_row("gradient_boosting", 0.61), result_row("random_forest", 0.66), result_row("mlp", 0.70),
    result_row("baseline:layer_mean", 0.40), result_row("baseline:random", 0.95),
    result_row("mlp", 0.80, scope="all"),
]


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "VARIANTS_DIR", tmp_path / "variants")
    monkeypatch.setattr(paths, "HEAVY_VARIANTS_DIR", tmp_path / "variants-heavy")
    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(evaluate, "CACHE_DIR", tmp_path)
    for mode, shift in (("light", 0.0), ("heavy", 0.05)):
        directory = paths.variant_predictions_dir("phase1", mode, "cifar10")
        directory.mkdir(parents=True)
        with open(directory / "phase1__ochiai__ochiai_rank_model.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in PHASE1:
                writer.writerow({**row, "mode": mode, "spearman_model": row["spearman_model"] + shift})
        (directory.parent / "design.json").write_text(json.dumps({"fit_population": "train",
                                                                 "question": "Does it reseed?"}))
    root = paths.runtime_root("light", "cifar10")
    root.mkdir(parents=True)
    runs = [
        {"tool": "staticsbfl_mlp", "model": "p4_designed_small_mlp_s0", "repeat": 0, "status": "ok",
         "per_model_seconds": 0.01, "setup_seconds": 1.0, "per_model_peak_above_start_bytes": 2e6,
         "process_lifetime_max_bytes": 3e8, "n_neurons": 384},
        {"tool": "deepfault_tf1", "model": "p4_designed_small_mlp_s0", "repeat": 0, "status": "ok",
         "per_model_seconds": 140.0, "setup_seconds": 2.0, "per_model_peak_above_start_bytes": 1e8,
         "process_lifetime_max_bytes": 5e8, "n_neurons": 384},
        {"tool": "deepfault_tf2", "model": "p4_designed_small_mlp_s0", "repeat": 0, "status": "timeout",
         "detail": "exceeded 10 s"},
    ]
    with open(root / "runs.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in runs for k in r}))
        writer.writeheader()
        writer.writerows(runs)
    return tmp_path


def invoke(*args):
    result = CliRunner().invoke(evaluate.main, ["--no-legend", "--phase", "phase1", *args])
    assert result.exit_code == 0, result.output + repr(result.exception)
    return result.output


def test_the_log_carries_the_headline_maximum_and_the_mode_comparison(cache):
    output = invoke()
    assert re.search(r"best rho/model\s+0\.700", output)            # light: mlp 0.70, not the 0.95 baseline
    assert re.search(r"best rho/model\s+0\.750", output)            # heavy: shifted by 0.05
    assert "Light vs heavy" in output and "+0.050" in output
    log = (cache / "evaluation" / "cifar10" / "evaluation.log").read_text()
    assert "best rho/model" in log and "\033[" not in log            # same content, no colour codes
    assert "MallocLargeCache=0" in log                               # the memory note is logged
    assert "DF@tf1" in output and "timeout" in output


def test_the_overview_csv_has_one_file_per_mode(cache):
    invoke()
    light = list(csv.DictReader(open(paths.variants_root("light", "cifar10") / "summary_active.csv")))
    heavy = list(csv.DictReader(open(paths.variants_root("heavy", "cifar10") / "summary_active.csv")))
    assert float(light[0]["best_rho_model"]) == pytest.approx(0.70)
    assert float(heavy[0]["best_rho_model"]) == pytest.approx(0.75)


def test_the_html_lists_every_row_and_marks_the_phase_maximum(cache):
    invoke()
    page = (cache / "evaluation" / "cifar10" / "report.html").read_text()
    assert page.startswith("<!doctype html>") and "<title>StaticSBFL Evaluation Report</title>" in page
    result_rows = re.findall(r'<tr data-row="1" data-mode="(light|heavy)" data-predictor', page)
    assert len(result_rows) == 2 * len(PHASE1)                       # every result row, both modes
    # the light maximum (mlp, active, 0.700) is marked "best" ...
    assert re.search(r'data-predictor="mlp"[^>]*data-scope="active"[^>]*>(?:(?!</tr>).)*<td class="best">0\.700</td>',
                     page, re.S)
    # ... and the baseline, although larger (0.950), is never marked
    baseline = re.search(r'data-mode="light" data-predictor="baseline:random"[^>]*>(.*?)</tr>', page, re.S).group(1)
    assert 'class="best"' not in baseline and 'class="high"' not in baseline
    assert "MallocLargeCache=0" in page and 'id="filters"' in page
    assert "Every measured run" in page and "exceeded 10 s" in page


def test_a_model_type_filter_reaches_the_html_and_keeps_the_baselines(cache):
    invoke("--model-type", "neural_network")
    page = (cache / "evaluation" / "cifar10" / "report.html").read_text()
    assert 'data-predictor="gradient_boosting"' not in page and 'data-predictor="random_forest"' not in page
    assert 'data-predictor="mlp"' in page and 'data-predictor="baseline:layer_mean"' in page
    assert "model type neural_network" in page
    # the static tools of the runtime section follow the same filter (MLP stays)
    assert "StaticSBFL (MLP)" in page


def test_render_report_escapes_and_needs_no_network():
    page = html_report.render_report(dataset="cifar10", modes=["light"], scope="active", results=[],
                                     overview=[], runtime=[], filter_text="<script>")
    assert "&lt;script&gt;" in page
    assert "http://" not in page and "https://" not in page          # self-contained: no CDN, no fonts


def test_the_report_has_its_tabs_placeholders_and_explains_every_header(cache):
    invoke()
    page = (cache / "evaluation" / "cifar10" / "report.html").read_text()
    for tab in ("overview", "across", "phase1", "phase4", "statistics", "runtime", "costs", "glossary"):
        assert f'data-tab="{tab}"' in page and f'id="{tab}"' in page
    # only phase 1 exists: the other phases are placeholders naming their command, for both modes
    assert "phase2 (light) — not computed yet" in page and "make phase PHASE=phase2 VMODE=heavy" in page
    assert "Runtime comparison (heavy) — not computed yet" in page
    untitled = re.findall(r'<th class="[lr]">([^<]*)</th>', page)
    assert untitled == [], f"headers without a hover text: {sorted(set(untitled))}"
    assert 'class="tag population population-designed"' in page or "population-test" in page
    assert "only the activation moved" in page                      # the glossary explains 'saturating'


def test_the_log_explains_the_populations_and_scopes(cache):
    output = CliRunner().invoke(evaluate.main, ["--phase", "phase1"]).output  # with legend and glossary
    assert "designed" in output and "only neurons that fired at least once" in output
    assert "Spread of spearman_model by model type" in output


def test_the_cost_report_stays_inside_the_redirected_cache(cache):
    # costs.csv used to be bound to the real cache at import, so this suite
    # overwrote cache/runtime/costs.csv with rows pointing at pytest's tmp dirs.
    invoke()
    written = cache / "runtime" / "costs.csv"
    assert written.exists()
    assert costs.costs_csv() == written
