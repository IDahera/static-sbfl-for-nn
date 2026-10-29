"""The final evaluation as one self-contained HTML page with tabs.

    write_report(path, dataset=..., modes=..., results=..., overview=..., runtime=..., costs=...)

The console prints what matters most; this page lists **every** row, so a
number in the paper can be traced back to its case. Tabs:

``Overview``       best fitted predictor per mode, phase and population
``Across phases``  one row per (mode, scope, metric, target, predictor), one
                   column per phase/population -- how a case holds up from
                   phase to phase; switchable between rho, wtau, top-k, ...
``Phase 1..4``     every result row of the phase, both modes
``Statistics``     count, undefined, min, quartiles, mean, max, sd, range per
                   group (model type, predictor, metric, population, ...)
``Runtime``        per mode: tools, designs, speed-ups, amortization,
                   fidelity, every measured run
``Costs``          training / score / table / fit time of the whole run
``Glossary``       every population, scope, mode, predictor, metric, target,
                   tool and column, plus how time and memory were measured

Highlights are per phase, column and scope among fitted predictors only: the
maximum in a strong colour, the top 10 % in a light one; baselines are listed
but never marked. Parameter values are coloured tags; every header and tag
explains itself on hover (texts from :mod:`staticsbfl.evaluation.glossary`). Anything
not computed yet appears as a placeholder that names the command producing it.
Filters act in the browser; the page needs no network and no library, and
follows the system's light or dark theme. Nothing here reads files or computes
figures -- it renders :mod:`staticsbfl.evaluation` data.
"""

import html
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from staticsbfl.benchmark.footprint import MEASUREMENT_NOTE
from staticsbfl.evaluation import accuracy as acc
from staticsbfl.evaluation import costs as cost_data
from staticsbfl.evaluation.glossary import COLUMNS, TERMS, column_help, describe
from staticsbfl.evaluation.runtime import SHORT_LABELS, TOOL_LABELS, RuntimeSummary

PHASES = ("phase1", "phase2", "phase3", "phase4")
ALL_MODES = ("light", "heavy")

#: The compact per-phase table: one column group per quantity, one column per metric.
COMPACT_GROUPS: Tuple[Tuple[str, str, int], ...] = (
    ("spearman_model", "rho/model", 3), ("weightedtau_model", "wtau", 3),
    ("top_k_05", "top-5%", 3), ("top_k", "top-10%", 3),
    ("gt_gain_05", "gain@5%", 2), ("gt_gain_10", "gain@10%", 2), ("gt_lift_10", "lift@10%", 2),
)
METRIC_ORDER = (("tarantula", "T"), ("ochiai", "O"), ("dstar", "D*"))
#: The long, one-row-per-metric table kept (collapsed) under each compact one.
RESULT_COLUMNS: Tuple[Tuple[str, str, int], ...] = (
    ("spearman_model", "rho/model", 3), ("spearman_model_sd", "± sd", 3),
    ("weightedtau_model", "wtau", 3), ("top_k_05", "top-5%", 3), ("top_k", "top-10%", 3),
    ("top_k_chance", "chance", 3), ("gt_top_10", "GT top 10%", 3), ("gt_rest_10", "GT rest 90%", 3),
    ("gt_gain_10", "gain@10%", 2), ("spearman_layer", "rho/layer", 3), ("inactive_fraction", "dead units", 3),
)
MARKED_COLUMNS = tuple(c for c, _, _ in COMPACT_GROUPS) + ("gt_top_10", "spearman_layer")
#: Values the "Across phases" tab can show.
ACROSS_VALUES = (("spearman_model", "rho/model"), ("weightedtau_model", "wtau"), ("top_k_05", "top-5%"),
                 ("top_k", "top-10%"), ("gt_gain_10", "gain@10%"))
STAT_HEADERS = ("rows", "undefined", "min", "q25", "median", "mean", "q75", "max", "sd", "range")

CSS = """
:root { --bg:#ffffff; --fg:#1f2328; --muted:#656d76; --line:#d0d7de; --head:#f6f8fa; --card:#f6f8fa;
        --best:#1a7f37; --best-bg:#dafbe1; --high-bg:#fff8c5; --warn:#9a6700; --bad:#cf222e; --accent:#0969da;
        --t-gb:#ddf4ff; --t-gb-fg:#0550ae; --t-rf:#dafbe1; --t-rf-fg:#116329; --t-mlp:#fbefff; --t-mlp-fg:#8250df;
        --t-base:#eaeef2; --t-base-fg:#57606a; --t-metric:#fff1e5; --t-metric-fg:#953800;
        --t-scope:#ffebe9; --t-scope-fg:#a40e26; --t-pop:#e7f3ef; --t-pop-fg:#1b6e5a;
        --t-mode:#f1e9ff; --t-mode-fg:#5a32a3; --t-target:#f6f8fa; --t-target-fg:#24292f; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --line:#30363d; --head:#161b22; --card:#161b22;
          --best:#3fb950; --best-bg:#1f3d2a; --high-bg:#3b3215; --warn:#d29922; --bad:#f85149; --accent:#4493f8;
          --t-gb:#0c2d6b; --t-gb-fg:#a5d6ff; --t-rf:#0f3b20; --t-rf-fg:#aff5b4; --t-mlp:#3c1e70; --t-mlp-fg:#e2c5ff;
          --t-base:#2d333b; --t-base-fg:#adbac7; --t-metric:#4a2500; --t-metric-fg:#ffc680;
          --t-scope:#5d0f12; --t-scope-fg:#ffaba8; --t-pop:#0d3b30; --t-pop-fg:#a3e8d4;
          --t-mode:#2e1a5e; --t-mode-fg:#d8c8ff; --t-target:#22272e; --t-target-fg:#cdd9e5; } }
* { box-sizing: border-box; }
body { margin:0; padding:20px 16px 64px; background:var(--bg); color:var(--fg);
       font:14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 1440px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 24px 0 8px; }
h3 { font-size: 15px; margin: 20px 0 6px; }
.muted { color: var(--muted); } .note { color: var(--muted); font-size: 12.5px; max-width: 115ch; }
.scroll { overflow-x: auto; border:1px solid var(--line); border-radius:6px; margin-bottom: 8px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 8px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th { background: var(--head); position: sticky; top: 0; font-weight: 600; }
th[title], .tag[title] { cursor: help; } th[title] { text-decoration: underline dotted var(--muted); }
td.l, th.l { text-align: left; }
tr.baseline td { color: var(--muted); }
td.best { background: var(--best-bg); color: var(--best); font-weight: 700; }
td.high { background: var(--high-bg); }
td.bad { color: var(--bad); font-weight: 600; } .warn { color: var(--warn); }
.tag { display:inline-block; padding:0 7px; border-radius: 10px; font-size: 12px; line-height: 19px;
       background: var(--t-target); color: var(--t-target-fg); }
.tag.predictor-gradient_boosting { background: var(--t-gb); color: var(--t-gb-fg); }
.tag.predictor-random_forest { background: var(--t-rf); color: var(--t-rf-fg); }
.tag.predictor-mlp { background: var(--t-mlp); color: var(--t-mlp-fg); }
.tag.model_type-baseline, .tag.predictor-baseline { background: var(--t-base); color: var(--t-base-fg); }
.tag.model_type-tree_ensemble { background: var(--t-rf); color: var(--t-rf-fg); }
.tag.model_type-neural_network { background: var(--t-mlp); color: var(--t-mlp-fg); }
.tag.metric { background: var(--t-metric); color: var(--t-metric-fg); }
.tag.scope { background: var(--t-scope); color: var(--t-scope-fg); }
.tag.population { background: var(--t-pop); color: var(--t-pop-fg); }
.tag.mode { background: var(--t-mode); color: var(--t-mode-fg); }
.legend span { display:inline-block; padding:1px 8px; margin-right:8px; border-radius:4px; }
.legend .best { background: var(--best-bg); color: var(--best); font-weight:700; }
.legend .high { background: var(--high-bg); }
.tabs { display:flex; flex-wrap:wrap; gap:4px; border-bottom:1px solid var(--line); margin-top:12px; }
.tabs button { background:none; border:1px solid transparent; border-bottom:none; padding:6px 12px; font:inherit;
               color: var(--muted); cursor:pointer; border-radius:6px 6px 0 0; }
.tabs button.on { color: var(--fg); border-color: var(--line); background: var(--bg); margin-bottom:-1px; font-weight:600; }
section.tab { display:none; } section.tab.on { display:block; }
.filters { position: sticky; top: 0; z-index: 2; background: var(--bg); padding: 8px 0;
           border-bottom: 1px solid var(--line); display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; }
.filters label, .local label { font-size: 12.5px; color: var(--muted); }
details { margin: 4px 0 12px; } summary { cursor: pointer; }
select { margin-left: 4px; background: var(--bg); color: var(--fg); border:1px solid var(--line); border-radius:4px; padding:2px 4px; }
.count { font-size: 12.5px; color: var(--muted); }
.placeholder { border:1px dashed var(--line); border-radius:6px; padding:12px 14px; margin:8px 0; color:var(--muted); background:var(--card); }
.placeholder code { color: var(--fg); }
.local { margin: 6px 0; }
dl.glossary { display:grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 0; }
dl.glossary dt { font-weight:600; } dl.glossary dd { margin:0; color: var(--muted); }
"""

SCRIPT = """
(function () {
  // tabs
  const buttons = Array.from(document.querySelectorAll(".tabs button"));
  function show(id) {
    buttons.forEach(b => b.classList.toggle("on", b.dataset.tab === id));
    document.querySelectorAll("section.tab").forEach(s => s.classList.toggle("on", s.id === id));
    try { history.replaceState(null, "", "#" + id); } catch (e) {}
  }
  buttons.forEach(b => b.addEventListener("click", () => show(b.dataset.tab)));
  const start = location.hash.slice(1);
  show(buttons.some(b => b.dataset.tab === start) ? start : buttons[0].dataset.tab);

  // global filters
  const keys = ["mode","predictor","model_type","metric","target","population","scope","tool"];
  const rows = Array.from(document.querySelectorAll("tr[data-row]"));
  const bar = document.getElementById("filters");
  const selects = {};
  keys.forEach(function (key) {
    const values = Array.from(new Set(rows.map(r => r.dataset[key]).filter(v => v))).sort();
    if (values.length < 2) return;
    const label = document.createElement("label");
    label.textContent = key.replace("_", " ");
    const select = document.createElement("select");
    select.innerHTML = '<option value="">all</option>' + values.map(v => '<option>' + v + '</option>').join("");
    select.addEventListener("change", apply);
    label.appendChild(select); bar.appendChild(label); selects[key] = select;
  });
  const marked = document.createElement("label");
  marked.innerHTML = '<input type="checkbox" id="marked-only"> marked only';
  bar.appendChild(marked);
  document.getElementById("marked-only").addEventListener("change", apply);
  const count = document.createElement("span"); count.className = "count"; bar.appendChild(count);

  // local switches: "Across phases" value and "Statistics" grouping
  const locals = Array.from(document.querySelectorAll("select[data-local]"));
  locals.forEach(s => s.addEventListener("change", apply));

  function apply() {
    const only = document.getElementById("marked-only").checked;
    const local = {};
    locals.forEach(s => { local[s.dataset.local] = s.value; });
    let shown = 0;
    rows.forEach(function (r) {
      let ok = true;
      for (const key in selects) {
        const want = selects[key].value;
        // baselines stay visible under a predictor filter, as on the command line:
        // a best value cannot be read without its layer-mean baseline
        if ((key === "predictor" || key === "model_type") && r.dataset.model_type === "baseline") continue;
        if (want && r.dataset[key] !== undefined && r.dataset[key] !== want) { ok = false; break; }
      }
      for (const key in local) {
        if (ok && r.dataset[key] !== undefined && r.dataset[key] !== local[key]) ok = false;
      }
      if (ok && only && r.dataset.marked === "0") ok = false;
      r.style.display = ok ? "" : "none";
      if (ok) shown++;
    });
    document.querySelectorAll("table[data-local-table]").forEach(function (t) {
      const [key, value] = t.dataset.localTable.split("=");
      t.closest(".scroll").style.display = (local[key] === undefined || local[key] === value) ? "" : "none";
    });
    count.textContent = shown + " of " + rows.length + " rows shown";
  }
  apply();
})();
"""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def esc(value) -> str:
    """HTML-escape *value*; ``None`` becomes the empty string."""
    return html.escape("" if value is None else str(value))


def fmt(value, digits: int = 3) -> str:
    """A number with *digits* decimals, ``–`` for undefined."""
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_seconds(value) -> str:
    """Seconds as ms, s, m:s or h:m."""
    if value is None or value == "":
        return "–"
    value = float(value)
    if value < 1:
        return f"{value * 1000:.0f} ms"
    if value < 120:
        return f"{value:.1f} s"
    minutes, seconds = divmod(int(round(value)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


def fmt_bytes(value) -> str:
    """Bytes in B, KB, MB or GB (base 1024)."""
    if value is None or value == "":
        return "–"
    value = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024
    return f"{value:.1f} GB"


def tag(kind: str, value) -> str:
    """A coloured chip for a parameter value, explained on hover."""
    text = "" if value is None else str(value)
    if not text:
        return "–"
    variant = text.replace("baseline:", "baseline") if kind == "predictor" and text.startswith("baseline:") else text
    help_text = describe(kind, text)
    title = f' title="{esc(help_text)}"' if help_text else ""
    return f'<span class="tag {esc(kind)} {esc(kind)}-{esc(variant)}"{title}>{esc(text)}</span>'


def table(headers: Sequence[tuple], body: str, *, local: Optional[str] = None) -> str:
    """``headers`` = [(text, "l"|"r"[, hover text])]; *body* is the rendered ``<tr>`` rows.

    Without an explicit hover text a header explains itself from the glossary.
    """
    def th(text, align, help_text=None):
        """One header cell; its hover text comes from the glossary unless given."""
        help_text = column_help(text) if help_text is None else help_text
        title = f' title="{esc(help_text)}"' if help_text else ""
        return f'<th class="{align}"{title}>{esc(text)}</th>'
    head = "".join(th(*h) for h in headers)
    attr = f' data-local-table="{esc(local)}"' if local else ""
    return f'<div class="scroll"><table{attr}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def row(cells: Iterable[Tuple[str, str]], data: Optional[Dict[str, str]] = None, css: str = "") -> str:
    """One ``<tr>``: *cells* = [(html, class)], *data* becomes filterable ``data-*`` attributes."""
    attributes = ""
    if data is not None:
        attributes = ' data-row="1"' + "".join(f' data-{k}="{esc(v)}"' for k, v in data.items() if v not in (None, ""))
    css_attr = f' class="{css}"' if css else ""
    return f"<tr{attributes}{css_attr}>" + "".join(f'<td class="{c}">{h}</td>' for h, c in cells) + "</tr>"


def placeholder(what: str, command: str) -> str:
    """The card shown for something not computed yet, with its command."""
    return f'<div class="placeholder">{esc(what)} — not computed yet. Run <code>{esc(command)}</code>.</div>'


# --------------------------------------------------------------------------
# Accuracy tabs
# --------------------------------------------------------------------------


def overview_tab(table_rows: Sequence[dict], scope: str, results: Sequence[acc.PhaseResult], modes: Sequence[str]) -> str:
    """The Overview tab: best fitted predictor per mode, phase and population."""
    other = "all" if scope == "active" else "active"
    body = []
    for r in table_rows:
        margin = r.get("margin_over_layer_mean")
        body.append(row([
            (tag("mode", r["mode"]), "l"), (esc(r["phase"]), "l"), (tag("population", r["population"]), "l"),
            (fmt(r["best_rho_model"]), ""), (esc(r["best_rho_case"]), "l"), (fmt(r["layer_mean_rho"]), ""),
            (fmt(margin), "bad" if margin is not None and margin <= 0 else ""),
            (fmt(r.get(f"best_rho_model_{other}")), ""), (fmt(r["best_wtau"]), ""), (fmt(r["best_top_k"]), ""),
            (fmt(r.get("top_k_x_chance"), 1) + "×" if r.get("top_k_x_chance") else "–", ""), (esc(r["n_models"]), ""),
        ], data={"mode": r["mode"], "population": r["population"], "marked": "1"}))
    headers = [("mode", "l"), ("phase", "l"), ("population", "l"), ("best rho/model", "r"), ("case", "l"),
               ("layer mean", "r"), ("margin", "r"), (f"rho ({other})", "r"), ("best wtau", "r"),
               ("best top-k", "r"), ("× chance", "r"), ("models", "r")]
    present = {(r.phase, r.mode) for r in results}
    missing = [f"{m}/{p}" for m in modes for p in PHASES if (p, m) not in present]
    note = ("" if not missing else f'<p class="note">Not computed yet: {esc(", ".join(missing))} '
            "(<code>make light</code> / <code>make heavy</code>).</p>")
    content = table(headers, "".join(body)) if body else placeholder("No phase result", "make light")
    return ('<p class="note">Headline scope <b>' + esc(scope) + "</b>. A margin ≤ 0 (red) means the predictor knows no "
            "more than the layer a neuron is in. Each best is a maximum over many fitted cases and therefore optimistic.</p>"
            + content + note)


def across_tab(results: Sequence[acc.PhaseResult]) -> str:
    """One row per case, one column per (phase, population): the case across the ladder."""
    if not results:
        return placeholder("No phase result", "make light")
    columns = []
    for r in sorted(results, key=lambda x: (PHASES.index(x.phase), x.mode)):
        for p in r.populations:
            if (r.phase, p) not in columns:
                columns.append((r.phase, p))
    cases: Dict[tuple, Dict[tuple, dict]] = {}
    for r in results:
        for x in r.rows:
            key = (r.mode, x["scope"], x["metric"], x["target"], x["predictor"])
            cases.setdefault(key, {})[(r.phase, x["population"])] = x
    switch = ('<div class="local"><label>value<select data-local="value">'
              + "".join(f'<option value="{c}">{esc(h)}</option>' for c, h in ACROSS_VALUES) + "</select></label>"
              ' <span class="note">best per column (fitted predictors, per mode and scope) marked</span></div>')
    parts = [switch]
    for column, header in ACROSS_VALUES:
        best: Dict[tuple, float] = {}
        for key, cells in cases.items():
            if key[4].startswith("baseline:"):
                continue
            for col, x in cells.items():
                value = x.get(column)
                if value is not None:
                    slot = (key[0], key[1], col)
                    best[slot] = max(best.get(slot, value), value)
        body = []
        for key in sorted(cases):
            mode, scope, metric, target, predictor = key
            cells = cases[key]
            out = [(tag("mode", mode), "l"), (tag("scope", scope), "l"), (tag("metric", metric), "l"),
                   (tag("target", target), "l"), (tag("predictor", predictor), "l")]
            marked = False
            for col in columns:
                value = cells.get(col, {}).get(column)
                mark = "best" if (not predictor.startswith("baseline:") and value is not None
                                  and value == best.get((mode, scope, col))) else ""
                marked = marked or bool(mark)
                out.append((fmt(value), mark))
            body.append(row(out, data={"mode": mode, "scope": scope, "metric": metric, "target": target,
                                       "predictor": predictor, "model_type": acc.PREDICTOR_TYPES.get(predictor, "other"),
                                       "marked": "1" if marked else "0"},
                            css="baseline" if predictor.startswith("baseline:") else ""))
        headers = [("mode", "l"), ("scope", "l"), ("metric", "l"), ("target", "l"), ("predictor", "l")] + [
            (f"{phase.replace('phase', 'P')} {population}", "r",
             f"{header} on {phase}, population {population}: {describe('population', population)}")
            for phase, population in columns]
        parts.append(table(headers, "".join(body), local=f"value={column}"))
    return "".join(parts)


def phase_tab(phase: str, results: Sequence[acc.PhaseResult], modes: Sequence[str]) -> str:
    """One phase tab: its full table per mode, or a placeholder per missing mode."""
    parts = []
    for mode in modes:
        result = next((r for r in results if r.phase == phase and r.mode == mode), None)
        if result is None:
            parts.append(f"<h3>{tag('mode', mode)}</h3>" + placeholder(
                f"{phase} ({mode})", f"make phase PHASE={phase}" + ("" if mode == "light" else f" VMODE={mode}")))
            continue
        parts.append(phase_table(result))
    return "".join(parts)


def target_kind(target: str) -> str:
    """``ochiai_rank_model`` -> ``rank_model``; a raw target -> ``raw``."""
    for kind in ("rank_model", "rank_layer"):
        if target.endswith(kind):
            return kind
    return "raw"


def metric_marks(rows: Sequence[dict], columns: Sequence[str]) -> Dict[Tuple[int, str], str]:
    """:func:`accuracy.phase_highlights` computed per metric, so each metric column has its own best."""
    marks: Dict[Tuple[int, str], str] = {}
    for metric, _ in METRIC_ORDER:
        indexed = [(i, r) for i, r in enumerate(rows) if r["metric"] == metric]
        local = acc.phase_highlights([r for _, r in indexed], columns)
        for (j, column), mark in local.items():
            marks[(indexed[j][0], column)] = mark
    return marks


def compact_table(result: acc.PhaseResult) -> str:
    """One row per (population, scope, target kind, predictor); the three metrics side by side.

    The range of one predictor over Tarantula, Ochiai and D* reads along a row;
    highlights are per metric column (best and top 10 % among fitted rows of
    that metric and scope in this phase).
    """
    marks = metric_marks(result.rows, [c for c, _, _ in COMPACT_GROUPS])
    by_key: Dict[tuple, Dict[str, Tuple[int, dict]]] = {}
    for index, r in enumerate(result.rows):
        key = (r["population"], r["scope"], target_kind(r["target"]), not acc.is_fitted(r), r["predictor"])
        by_key.setdefault(key, {})[r["metric"]] = (index, r)
    head_top = ('<th class="l" rowspan="2" title="' + esc(column_help("population")) + '">population</th>'
                '<th class="l" rowspan="2" title="' + esc(column_help("scope")) + '">scope</th>'
                '<th class="l" rowspan="2" title="' + esc(column_help("target kind")) + '">target</th>'
                '<th class="l" rowspan="2" title="' + esc(column_help("predictor")) + '">predictor</th>'
                + "".join(f'<th colspan="3" title="{esc(column_help(h))}">{esc(h)}</th>' for _, h, _ in COMPACT_GROUPS))
    head_sub = "".join(f'<th title="{esc(describe("metric", m))}">{esc(short)}</th>'
                       for _ in COMPACT_GROUPS for m, short in METRIC_ORDER)
    body = []
    for key in sorted(by_key):
        population, scope, kind, is_baseline, predictor = key
        cells = [(tag("population", population), "l"), (tag("scope", scope), "l"), (tag("target", kind), "l"),
                 (tag("predictor", predictor), "l")]
        marked = False
        for column, _header, digits in COMPACT_GROUPS:
            for metric, _ in METRIC_ORDER:
                index, r = by_key[key].get(metric, (None, {}))
                mark = marks.get((index, column), "") if index is not None else ""
                marked = marked or bool(mark)
                cells.append((fmt(r.get(column), digits), mark))
        body.append(row(cells, data={
            "mode": result.mode, "predictor": predictor, "model_type": acc.PREDICTOR_TYPES.get(predictor, "other"),
            "kind": kind, "population": population, "scope": scope, "marked": "1" if marked else "0"},
            css="baseline" if is_baseline else ""))
    chances = [r.get("top_k_05_chance") for r in result.rows if r.get("top_k_05_chance") is not None]
    chance10 = [r.get("top_k_chance") for r in result.rows if r.get("top_k_chance") is not None]
    note = ('<p class="note">Chance: top-5% about ' + (f"{sum(chances) / len(chances):.3f}" if chances else "0.05")
            + ", top-10% about " + (f"{sum(chance10) / len(chance10):.3f}" if chance10 else "0.10")
            + "; gain 0 = random pick, 1 = as good as the true top; lift 1 = no separation. "
              "T / O / D* = Tarantula / Ochiai / D*.</p>")
    return (f'<div class="scroll"><table><thead><tr>{head_top}</tr><tr>{head_sub}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>' + note)


def long_table(result: acc.PhaseResult) -> str:
    """Every row of one phase, one per metric, with the extra columns -- for tracing a number."""
    rows_sorted = sorted(enumerate(result.rows), key=lambda item: (
        item[1]["population"], item[1]["scope"], item[1]["metric"], item[1]["target"],
        not acc.is_fitted(item[1]), item[1]["predictor"]))
    marks = metric_marks(result.rows, MARKED_COLUMNS)
    body = []
    for index, r in rows_sorted:
        cells = [(tag("population", r["population"]), "l"), (tag("scope", r["scope"]), "l"),
                 (tag("metric", r["metric"]), "l"), (tag("target", r["target"]), "l"),
                 (tag("predictor", r["predictor"]), "l")]
        for column, _header, digits in RESULT_COLUMNS:
            cells.append((fmt(r.get(column), digits), marks.get((index, column), "")))
        cells.append((esc(r["n_models"]), ""))
        body.append(row(cells, css="" if acc.is_fitted(r) else "baseline"))
    headers = [("population", "l"), ("scope", "l"), ("metric", "l"), ("target", "l"), ("predictor", "l")] + [
        (h, "r") for _, h, _ in RESULT_COLUMNS] + [("models", "r")]
    return table(headers, "".join(body))


def phase_table(result: acc.PhaseResult) -> str:
    """The compact table of one phase, with the long one collapsed underneath."""
    design = result.design
    incomplete = "" if result.complete else (
        f' <b class="warn">Only {len(result.files)} of {result.expected_files} result files.</b>')
    old = "" if any(r.get("gt_gain_10") is not None for r in result.rows) else (
        ' <span class="warn">top-5% and the gain / lift columns are blank: this phase was predicted before they '
        "existed; rerun its predict step.</span>")
    return (f"<h3>{tag('mode', result.mode)} {esc(result.phase)}</h3>"
            f'<p class="note">{esc(design.get("question", ""))} Fitted on '
            f'{tag("population", design.get("fit_population", "?"))}.{incomplete}{old}</p>'
            + compact_table(result)
            + f"<details><summary class=\"note\">every row, one per metric (sd, chance, measured top/rest, "
              f"rho/layer, dead units)</summary>{long_table(result)}</details>")


def statistics_tab(statistics: Sequence[dict]) -> str:
    """The Statistics tab, switchable between groupings."""
    if not statistics:
        return placeholder("No phase result", "make light")
    switch = ('<div class="local"><label>group by<select data-local="group_by">'
              + "".join(f'<option value="{g}"{" selected" if g == "model_type" else ""}>{esc(g.replace("_", " "))}</option>'
                        for g in acc.GROUPINGS)
              + "</select></label> <span class=\"note\">per mode, phase and scope ('*' = both scopes, for a grouping by "
                "scope); 'rows' + 'undefined' of the groups add up to the '*' line</span></div>")
    label = dict((c, h) for c, h, _ in RESULT_COLUMNS)
    body = []
    for s in statistics:
        kind = s["group_by"]
        group = tag(kind, s["group"]) if s["group"] != "*" and kind in TERMS else esc(s["group"])
        cells = [(tag("mode", s["mode"]), "l"), (esc(s["phase"]), "l"), (tag("scope", s["scope"]) if s["scope"] != "*" else "*", "l"),
                 (group, "l"), (esc(label.get(s["column"], s["column"])), "l")]
        for key in STAT_HEADERS:
            digits = 0 if key in ("rows", "undefined") else 3
            cells.append((fmt(s.get(key), digits) if key not in ("rows", "undefined") else esc(s.get(key)), ""))
        data = {"mode": s["mode"], "group_by": kind, "marked": "1"}
        if s["scope"] != "*":
            data["scope"] = s["scope"]
        body.append(row(cells, data=data, css="baseline" if s["group"] == "*" else ""))
    headers = [("mode", "l"), ("phase", "l"), ("scope", "l"), ("group", "l"), ("column", "l")] + [(h, "r") for h in STAT_HEADERS]
    return switch + table(headers, "".join(body))


# --------------------------------------------------------------------------
# Runtime and costs
# --------------------------------------------------------------------------


def runtime_tab(runtime: Sequence[RuntimeSummary], modes: Sequence[str]) -> str:
    """The Runtime tab: one section per mode, placeholders for missing runs."""
    parts = []
    by_mode = {s.mode: s for s in runtime}
    for mode in modes:
        summary = by_mode.get(mode)
        parts.append(f"<h2>{tag('mode', mode)} runtime and memory</h2>")
        if summary is None:
            parts.append(placeholder(f"Runtime comparison ({mode})", f"make runtime VMODE={mode}"))
            continue
        parts.append(runtime_section(summary))
    return "".join(parts)


def runtime_section(summary: RuntimeSummary) -> str:
    """One mode's runtime comparison: tools, designs, speed-ups, amortization, fidelity, runs."""
    parts = []
    env = summary.environment
    if env:
        versions = "".join(f"<br>{esc(k[len('versions_'):])}: {esc(v)}" for k, v in env.items() if k.startswith("versions_"))
        parts.append(f'<p class="note">{esc(env.get("cpu"))} · {esc(env.get("platform"))} · '
                     f'{esc(env.get("logical_cpus"))} logical CPUs · python {esc(env.get("python"))}{versions}</p>')
    data = {"mode": summary.mode}
    def lowest(attribute):
        return min((getattr(t, attribute) for t in summary.tools if getattr(t, attribute) is not None), default=None)

    def cell(t, attribute, fmt):
        value = getattr(t, attribute)
        return fmt(value), "best" if value is not None and value == best[attribute] else ""

    best = {a: lowest(a) for a in ("seconds_mean", "seconds_median", "seconds_min", "seconds_max",
                                   "memory_mean", "memory_median", "memory_min", "memory_max")}
    body = []
    for t in summary.tools:
        body.append(row([
            (tag("tool", t.tool).replace(f">{esc(t.tool)}<", f">{esc(t.label)}<"), "l"), (esc(t.models), ""),
            cell(t, "seconds_mean", fmt_seconds), cell(t, "seconds_median", fmt_seconds),
            cell(t, "seconds_min", fmt_seconds), cell(t, "seconds_max", fmt_seconds),
            (esc(t.seconds_max_model or "–"), "l"), (fmt_seconds(t.seconds_total), ""),
            (fmt_seconds(t.setup_median), ""),
            cell(t, "memory_mean", fmt_bytes), cell(t, "memory_median", fmt_bytes),
            cell(t, "memory_min", fmt_bytes), cell(t, "memory_max", fmt_bytes),
            (fmt_bytes(t.process_median), ""), (fmt_bytes(t.process_max), ""), (fmt_bytes(t.gpu_median), ""),
            (esc(t.as_dict()["status"]), "l"),
        ], data={**data, "tool": t.tool, "marked": "1"}))
    parts.append("<h3>Per tool</h3>" + table(
        [("tool", "l"), ("models", "r"), ("per model (mean)", "r"), ("median", "r"), ("min", "r"), ("max", "r"),
         ("slowest model", "l"), ("total", "r"), ("setup", "r"), ("mem/model (mean)", "r"), ("median", "r"),
         ("min", "r"), ("max", "r"), ("process max (median)", "r"),
         ("max", "r"), ("gpu", "r"), ("runs", "l")], "".join(body)))

    if summary.designs:
        tools = [t.tool for t in summary.tools]
        body = []
        for d in summary.designs:
            values = [d.get(t) for t in tools]
            best = min((v for v in values if v is not None), default=None)
            body.append(row([(esc(d["design"]), "l"), (esc(d["neurons"]), "")]
                            + [(fmt_seconds(v), "best" if v is not None and v == best else "") for v in values]))
        parts.append("<h3>Per design: median per-model time over its seeds</h3>"
                     + table([("design", "l"), ("neurons", "r")]
                           + [(SHORT_LABELS.get(t, t), "r", describe("tool", t)) for t in tools], "".join(body)))

    if summary.speedups:
        body = "".join(row([(esc(TOOL_LABELS[s.static]), "l"), (esc(TOOL_LABELS[s.dynamic]), "l"),
                            (f"{s.geometric_mean:.1f}×", "bad" if s.geometric_mean < 1 else ("best" if s.geometric_mean >= 10 else "")),
                            (f"{s.minimum:.1f}× – {s.maximum:.1f}×", ""), (esc(s.models), "")],
                           data={**data, "tool": s.static, "marked": "1"}) for s in summary.speedups)
        parts.append("<h3>Speed-up per model</h3>"
                     '<p class="note">Dynamic / static per-model time on the same model, geometric mean over models. '
                     "Red: the static predictor is slower; green: at least 10×.</p>"
                     + table([("static", "l"), ("vs dynamic", "l"), ("speed-up", "r"), ("range ×", "r"), ("models", "r")], body))

    cost = summary.cost
    body = []
    for a in summary.amortization:
        if a.dynamic is None:
            continue
        body.append(row([(esc(TOOL_LABELS[a.static]), "l"), (esc(TOOL_LABELS[a.dynamic]), "l"),
                         (fmt_seconds(a.one_time_seconds), ""),
                         (fmt_seconds(a.saving_per_model), "bad" if (a.saving_per_model or 0) <= 0 else ""),
                         (f"{a.models_to_pay_off:,}" if a.models_to_pay_off else "never", ""),
                         (f"{a.models_to_pay_off_without_training:,}" if a.models_to_pay_off else "–", "")],
                        data={**data, "tool": a.static, "marked": "1"}))
    parts.append("<h3>One-time cost and amortization</h3>"
                 f'<p class="note">Train fit population {fmt_seconds(cost.get("train_seconds"))} · score it (pro rata) '
                 f'{fmt_seconds(cost.get("score_seconds"))} · neuron table {fmt_seconds(cost.get("neurons_seconds"))} · '
                 "plus the predictor fit (an upper bound: make predict fits every target kind). "
                 f'{esc(cost.get("train_note") or "")}</p>'
                 + (table([("static", "l"), ("vs dynamic", "l"), ("one-time", "r"), ("saving / model", "r"),
                           ("pays off after", "r"), ("… if trained anyway", "r")], "".join(body)) if body
                    else placeholder("Amortization", "make runtime-estimators")))

    if summary.accuracy:
        body = "".join(row([(esc(TOOL_LABELS[t]), "l")] + [(fmt(v), "") for v in values.values()],
                           data={**data, "tool": t, "marked": "1"}) for t, values in summary.accuracy.items())
        parts.append("<h3>Accuracy next to speed (rho/model, phase 4 designed, rank_model)</h3>"
                     + table([("static", "l"), ("tarantula", "r"), ("ochiai", "r"), ("dstar", "r")], body))

    if summary.fidelity:
        body = "".join(row([(esc(TOOL_LABELS.get(f.tool, f.tool)), "l"), (esc(f.reference), "l"),
                            (f"{f.identical}/{f.models}", ""), (f"{f.neurons_differing}/{f.neurons}", ""),
                            (esc(f.max_count_difference), "")]
                           + [(fmt(v, 7), "") for v in f.min_rho.values()], data={**data, "tool": f.tool, "marked": "1"})
                       for f in summary.fidelity)
        parts.append("<h3>Fidelity: same model, same data, same spectrum?</h3>"
                     + table([("tool", "l"), ("reference", "l"), ("identical", "r"), ("neurons off", "r"),
                              ("max diff", "r"), ("min rho T", "r"), ("min rho O", "r"), ("min rho D*", "r")], body))

    body = []
    for r in summary.runs:
        status = r.get("status", "")
        body.append(row([
            (esc(TOOL_LABELS.get(r["tool"], r["tool"])), "l"), (esc(r["model"]), "l"), (esc(r["repeat"]), ""),
            (esc(status), "l" if status in ("ok", "cached") else "l bad"), (esc(r.get("n_neurons")), ""),
            (fmt_seconds(r.get("per_model_seconds")), ""), (fmt_seconds(r.get("setup_seconds")), ""),
            (fmt_bytes(r.get("per_model_peak_above_start_bytes")), ""), (fmt_bytes(r.get("process_lifetime_max_bytes")), ""),
            (fmt_bytes(r.get("gpu_peak_above_baseline_bytes")), ""), (esc((r.get("detail") or "")[:80]), "l"),
        ], data={**data, "tool": r["tool"], "marked": "1"}))
    parts.append("<h3>Every measured run</h3>"
                 + table([("tool", "l"), ("model", "l"), ("repeat", "r"), ("status", "l"), ("neurons", "r"),
                          ("per model", "r"), ("setup", "r"), ("mem/model", "r"), ("process max", "r"),
                          ("gpu", "r"), ("detail", "l")], "".join(body)))
    return "".join(parts)


def costs_tab(cost_rows: Sequence[dict], dataset: str, modes: Sequence[str]) -> str:
    """The Costs tab: training and step times per phase, per mode."""
    parts = []
    for mode in modes:
        summary = cost_data.summarise_mode(list(cost_rows), mode, dataset)
        parts.append(f"<h3>{tag('mode', mode)} {esc(dataset)}</h3>")
        if not summary["recorded"]:
            parts.append(placeholder(f"The {mode} run", f"make {mode}"))
            continue
        body = []
        for entry in summary["phases"]:
            train, steps = entry["train"], entry["steps"]
            if train.get("status") in ("missing", "unreadable", "error"):
                body.append(row([(esc(entry["phase"]), "l"), (esc(train.get("status")), "l warn")] + [("", "")] * 10))
                continue
            body.append(row([
                (esc(entry["phase"]), "l"), (esc(train.get("n_models")), ""), (esc(train.get("n_timed")), ""),
                (esc(train.get("epochs")), ""), (fmt_seconds(train.get("total_seconds")), ""),
                (fmt_seconds(train.get("mean_seconds")), ""), (fmt_seconds(train.get("sd_seconds")), ""),
                (fmt_seconds(train.get("seconds_per_model_epoch")), ""),
                (fmt_seconds(steps.get("score", {}).get("total_seconds")), ""),
                (fmt_seconds(steps.get("neurons", {}).get("total_seconds")), ""),
                (fmt_seconds(steps.get("predict", {}).get("total_seconds")), ""),
                (fmt_seconds(steps.get("predict", {}).get("fit_seconds")), ""),
            ]))
        totals = summary["totals"]
        body.append(row([("<b>total</b>", "l"), (esc(summary["n_models"]), ""), (esc(summary["n_timed"]), ""), ("", ""),
                         (fmt_seconds(totals["train"]), "best"), (fmt_seconds(summary["mean_train_seconds"]), ""),
                         ("", ""), ("", ""), (fmt_seconds(totals["score"]), ""), (fmt_seconds(totals["neurons"]), ""),
                         (fmt_seconds(totals["predict"]), ""), (fmt_seconds(totals["fit"] or None), "")]))
        parts.append(table([("phase", "l"), ("models", "r"), ("timed", "r"), ("epochs", "r"), ("train total", "r"),
                            ("per model", "r"), ("sd", "r"), ("per m-ep", "r"), ("score", "r"), ("neurons", "r"),
                            ("predict", "r"), ("(fit)", "r")], "".join(body)))
        flags = "".join(f"<li>{esc(f)}</li>" for f in summary["flags"])
        parts.append(f'<p class="note">Whole {esc(mode)} run: {fmt_seconds(summary["grand_total_seconds"])} measured '
                     "(train + score + neurons + predict).</p>" + (f'<ul class="note warn">{flags}</ul>' if flags else ""))
    return "".join(parts)


def glossary_tab() -> str:
    """The Glossary tab: every term and column, and how time and memory were measured."""
    parts = []
    for kind, terms in TERMS.items():
        items = "".join(f"<dt>{tag(kind, value) if not value.startswith('<') else esc(value)}</dt><dd>{esc(text)}</dd>"
                        for value, text in terms.items())
        parts.append(f"<h3>{esc(kind.replace('_', ' '))}</h3><dl class=\"glossary\">{items}</dl>")
    columns = "".join(f"<dt>{esc(name)}</dt><dd>{esc(text)}</dd>" for name, text in COLUMNS.items())
    parts.append(f"<h3>columns</h3><dl class=\"glossary\">{columns}</dl>")
    from staticsbfl.log.evaluation import TIME_NOTE
    parts.append(f'<h3>How time and memory were measured</h3><p class="note">{esc(TIME_NOTE)}</p>'
                 f'<p class="note">{esc(MEASUREMENT_NOTE)}</p>')
    return "".join(parts)


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------


def render_report(*, dataset: str, modes: Sequence[str], scope: str, results: Sequence[acc.PhaseResult],
                  overview: Sequence[dict], runtime: Sequence[RuntimeSummary], costs: Sequence[dict] = (),
                  statistics: Optional[Sequence[dict]] = None, filter_text: str = "none",
                  missing: Sequence[str] = ()) -> str:
    """The whole page as a string. *modes* are the modes shown (missing ones get placeholders)."""
    generated = time.strftime("%Y-%m-%d %H:%M")
    statistics = acc.statistics_table(results) if statistics is None else statistics
    tabs = [("overview", "Overview", overview_tab(overview, scope, results, modes)),
            ("across", "Across phases", across_tab(results))]
    tabs += [(p, f"Phase {p[-1]}", phase_tab(p, results, modes)) for p in PHASES]
    tabs += [("statistics", "Statistics", statistics_tab(statistics)),
             ("runtime", "Runtime", runtime_tab(runtime, modes)),
             ("costs", "Costs", costs_tab(costs, dataset, modes)),
             ("glossary", "Glossary", glossary_tab())]
    buttons = "".join(f'<button data-tab="{tid}">{esc(label)}</button>' for tid, label, _ in tabs)
    sections = "".join(f'<section class="tab" id="{tid}">{body}</section>' for tid, _, body in tabs)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StaticSBFL Evaluation Report</title>
<style>{CSS}</style></head>
<body><main>
<h1>StaticSBFL evaluation · {esc(dataset)} · {esc(" / ".join(modes))}</h1>
<p class="muted">Generated {esc(generated)} by <code>staticsbfl/cli/evaluate.py</code> · headline scope <b>{esc(scope)}</b> ·
filter applied when generating: <b>{esc(filter_text)}</b> · hover a header or a tag for its meaning</p>
<p class="legend"><span class="best">best of its phase</span><span class="high">top 10 % of its phase</span>
<span class="muted">per column and scope, fitted predictors only; baselines in grey are never marked</span></p>
<div class="filters" id="filters"></div>
<nav class="tabs">{buttons}</nav>
{sections}
</main>
<script>{SCRIPT}</script>
</body></html>
"""


def write_report(path: Path, **kwargs) -> Path:
    """Render the page and write it to *path*; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(**kwargs), encoding="utf-8")
    return path
