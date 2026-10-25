"""What a pipeline run is doing, where it writes, and how long it will take.

    python -m staticsbfl.cli.status overview --mode light --dataset cifar10
    python -m staticsbfl.cli.status start    --phase phase2 --mode light
    python -m staticsbfl.cli.status end      --phase phase2 --mode light --since 1760534400
    python -m staticsbfl.cli.status status   --mode light --dataset cifar10

The four step scripts each say what *they* do. What none of them can say is
where a step sits in its phase, what the phase as a whole asks, which of its
files is the one to read, and what the columns of the result table mean. That
is the job of this module: the Makefile calls it before and after every phase,
and the step scripts take their headers and their output lists from it, so the
description of a step exists exactly once.

The runtime estimate is computed from what has already been measured -- the
per-model training times in each ``models.csv`` and the per-step times in each
``timings.json`` -- and says what it is based on. Where nothing has been measured
yet it falls back to a reference measurement and says that instead.
"""

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


from staticsbfl.log.console import (
    configure_logging, display_path, format_duration, phase_banner, phase_done, plain, style,
)
from staticsbfl.utils.paths import (
    CACHE_DIR, VARIANTS_DIR, ensure_dir, variant_activation_rates, variant_correlation_log,
    variant_design, variant_model_path, variant_models_csv, variant_neurons,
    variant_predictions_csv, variant_predictions_dir, variant_spectrum_path,
    variant_timings, variants_root,
)
from staticsbfl.variation import EPOCHS_BY_MODE, PHASE_NAMES, PhaseSpec, build_phase
from staticsbfl.variation.phases import POPULATION_SIZES

logger = logging.getLogger("staticsbfl")

#: Datasets a population can be built on; mirrors ``staticsbfl.data.IMAGE_DATASET_NAMES``
#: without importing torchvision just to print a status table.
DATASETS = ("cifar10", "fashion_mnist")

#: The four steps of every phase, in order: (name, make target).
STEPS: Tuple[Tuple[str, str], ...] = (
    ("train", "make train"),
    ("score", "make score"),
    ("neurons", "make neurons"),
    ("predict", "make predict"),
)

#: One line per step, for the phase banner. The step's own header says more.
STEP_SUMMARY: Dict[str, str] = {
    "train": "train + cache every model (cached ones are skipped)",
    "score": "hit spectrum per model -> Tarantula / Ochiai / D* per neuron",
    "neurons": "neuron table: static features + measured targets",
    "predict": "fit predictors, rank held-out neurons, compare with measured",
}

#: Metric x target flavour combinations ``make predict`` writes by default.
DEFAULT_METRICS = ("tarantula", "ochiai", "dstar")
DEFAULT_KINDS = ("rank_model", "raw")


# --------------------------------------------------------------------------
# What each phase asks, in one line each
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseInfo:
    title: str      # short name, for banners and tables
    varies: str     # what changes between the models of this phase
    reading: str    # how its result is to be read


PHASES: Dict[str, PhaseInfo] = {
    "phase1": PhaseInfo(
        "reseeding — the noise floor",
        "only the initialisation seed; one fixed architecture",
        "the ceiling: no static predictor can be more consistent than the "
        "target itself is across seeds",
    ),
    "phase2": PhaseInfo(
        "held-out architectures — interpolation",
        "depth, width and activation, sampled from one region; split by model",
        "does a predictor work on unseen architectures from the region it "
        "was fitted on?",
    ),
    "phase3": PhaseInfo(
        "one axis at a time — extrapolation",
        "fit on a base region; each evaluation group moves exactly one axis "
        "(deeper / wider / saturating)",
        "a drop in one group is attributable to the one axis it moved",
    ),
    "phase4": PhaseInfo(
        "sampled → designed — transfer",
        "fit on sampled ReLU stacks; evaluate on hand-designed shapes "
        "(512-256, 300-100, 1024-512-256-128, …)",
        "does it transfer to architectures someone actually chose?",
    ),
}


def phase_title(phase: str) -> str:
    """``'PHASE 2/4 · held-out architectures — interpolation'``."""
    index = PHASE_NAMES.index(phase) + 1
    return f"PHASE {index}/{len(PHASE_NAMES)} · {PHASES[phase].title}"


# --------------------------------------------------------------------------
# What each step does and writes -- the one place this is described
# --------------------------------------------------------------------------


def step_description(step: str, spec: PhaseSpec, *, epochs: Optional[int] = None) -> List[str]:
    """What *step* does, in two or three short lines."""
    if step == "train":
        return [
            f"Build every model of the phase from its config and train it "
            f"({epochs} epochs each, same budget for all).",
            "Models already cached are skipped, so an interrupted run resumes.",
        ]
    if step == "score":
        return [
            "One pass over the test set per model: count, per neuron, a_f / a_s",
            "(active on failing / successful samples) -> Tarantula, Ochiai, D*.",
        ]
    if step == "neurons":
        return [
            "One table row per hidden neuron: static features (weights and",
            "architecture, no forward pass) beside the measured targets.",
        ]
    if step == "predict":
        evaluated = ", ".join(spec.evaluation_splits)
        return [
            f"Fit gradient boosting and an MLP on '{spec.train_split}';",
            f"rank the neurons of [{evaluated}] and compare with the measured ranking.",
        ]
    raise ValueError(step)


def step_outputs(step: str, phase: str, mode: str, dataset: str) -> List[Tuple[Path, str]]:
    """The files *step* writes, each with what it holds."""
    root = variants_root(mode, dataset) / phase
    if step == "train":
        return [
            (root / "<population>" / "<key>.pt", "model weights"),
            (variant_models_csv(phase, mode, dataset), "one row per model: config + accuracy"),
            (variant_design(phase, mode, dataset), "the phase specification + training cost"),
        ]
    if step == "score":
        return [
            (root / "<population>" / "spectra" / "<key>.pt", "hit spectrum + scores"),
            (variant_activation_rates(phase, mode, dataset), "per-layer activity: dead units show here"),
        ]
    if step == "neurons":
        return [(variant_neurons(phase, mode, dataset), "features + targets, one row per neuron")]
    if step == "predict":
        return central_outputs(phase, mode, dataset)[:2]
    raise ValueError(step)


def central_outputs(phase: str, mode: str, dataset: str) -> List[Tuple[Path, str]]:
    """The files a phase's result is read from, the first one first."""
    predictions = variant_predictions_dir(phase, mode, dataset)
    return [
        (variant_correlation_log(phase, mode, dataset), "READ THIS FIRST: meaning, then numbers"),
        (predictions / f"{phase}__<metric>__<target>.csv", "the same numbers, machine-readable"),
        (variant_activation_rates(phase, mode, dataset), "dead / saturated units per layer"),
        (variant_models_csv(phase, mode, dataset), "accuracy per model"),
    ]


def step_context(phase: str, mode: str, dataset: str) -> str:
    """``'phase2 · light · cifar10'`` -- the tail of every step header."""
    return f"{phase} · {mode} · {dataset}"


def step_index(step: str) -> Tuple[int, int]:
    """``(index, total)`` of *step*, 1-based."""
    names = [name for name, _ in STEPS]
    return names.index(step) + 1, len(names)


# --------------------------------------------------------------------------
# Glossary
# --------------------------------------------------------------------------

GLOSSARY = """\
SUSPICIOUSNESS -- the targets, measured from each model's hit spectrum
  a_f / a_s     samples on which a neuron was active (> 0) and the model was
                wrong / right.  N_f / N_s = all failing / successful samples.
  tarantula     (a_f/N_f) / (a_f/N_f + a_s/N_s)            in [0, 1]
  ochiai        a_f / sqrt(N_f * (a_f + a_s))              in [0, 1]
  dstar         a_f^3 / (a_s + N_f - a_f)                  D* with * = 3, unbounded
  target        raw         the value itself
                rank_model  its rank among all neurons of the model, in [0, 1]
                rank_layer  its rank within its own layer
  activity_rate share of samples a neuron fired on; 0 = dead unit. Never a
                feature, only used to separate dead from live neurons.

EVALUATION -- predicted ranking vs measured ranking, per model, then averaged
  rho/model     Spearman within each model, mean over models. 1 = same order,
                0 = unrelated, -1 = reversed.
  sd            spread of rho/model ACROSS models (not a standard error).
  wtau          weighted Kendall tau: like rho, but a swap near the top costs
                far more than one near the bottom. Matches how a ranking is used.
  top-k         share of the truly top-10% neurons found in the predicted
                top-10%. Compare with 'chance' (= k/n), not with 0.10.
  rho/layer     Spearman within each layer, averaged within a model, then over
                models. Removes the easy "which layer" part of the signal.
  r2            regression fit; context only, meaningless for raw dstar.
  n_inact       neurons that never fired in the evaluation population.

  scope all     every neuron.
  scope active  only neurons that fired at least once. A dead unit scores 0
                under every metric, so 'all' rewards spotting dead units;
                the gap between the two scopes is how much that was worth.
  layer_mean    baseline: every neuron = its layer's training mean. Read the
                MARGIN over it -- below it, a predictor knows only the layer.
  random        baseline: chance.

RUN VARIABLES
  PHASE    phase1..phase4          VMODE  debug | light | heavy
  DATASET  cifar10 | fashion_mnist  DEVICE auto | cpu | mps | cuda
  METRIC   '--metric ochiai' narrows make predict   FORCE=--force redoes cached work
  population  a group of models: the one predictors are fitted on ('fit') and
              the ones they are evaluated on ('eval')."""

#: The short version, printed directly above the predictor tables.
TABLE_LEGEND = (
    "columns: rho/model = Spearman per model (mean) · wtau = top-weighted tau · "
    "top-k = share of true top-10% found (vs chance) · rho/layer = Spearman within "
    "layers · scope active = dead units removed · read the margin over "
    "baseline:layer_mean"
)


def print_glossary(log: logging.Logger = logger) -> None:
    plain(log, style("── What the numbers mean " + "─" * 53, "bold"))
    plain(log, GLOSSARY)
    plain(log, "")


# --------------------------------------------------------------------------
# Timings: measured, stored, and turned into an estimate
# --------------------------------------------------------------------------

#: Used only where nothing has been measured yet. Measured on 2026-10-15,
#: light mode, CIFAR-10, Apple-silicon MPS: 518 models across the four phases.
REFERENCE = {
    "train_seconds_per_model_epoch": 2.5,     # ~7.5 s per model at 3 epochs
    "score_seconds_per_model": 0.7,           # 45-110 s per phase of 120-160 models
    "neurons_seconds_per_model": 0.1,
    "predict_seconds_per_row_target": 4.5e-4,  # 2-11 min per phase, 6 targets
}
REFERENCE_NOTE = "reference: 2026-10-15, light, CIFAR-10, MPS"


def record_timing(
    phase: str, mode: str, dataset: str, step: str, seconds: float,
    units: int, unit: str, **extra,
) -> None:
    """Store how long *step* took for *units* units of work, for later estimates.

    Only a run that did work is recorded -- a run that skipped everything would
    record a rate of zero and make every later estimate optimistic.
    """
    if units <= 0:
        return
    path = variant_timings(phase, mode, dataset)
    ensure_dir(path.parent)
    try:
        timings = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        timings = {}
    timings[step] = {
        "seconds": round(seconds, 2),
        "units": units,
        "unit": unit,
        "seconds_per_unit": seconds / units,
        "mode": mode,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra,
    }
    path.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")


def _read_timings(phase: str, mode: str, dataset: str) -> dict:
    path = variant_timings(phase, mode, dataset)
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def _read_models_rows(phase: str, mode: str, dataset: str) -> List[dict]:
    path = variant_models_csv(phase, mode, dataset)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


#: Modes whose measurements may stand in for one another. A model-epoch costs
#: the same in light and heavy -- same architectures, same data -- so a heavy
#: estimate can be based on light measurements until heavy has its own. Debug
#: runs are too short to time anything and never count.
REAL_MODES = ("light", "heavy")


def _sources(phase: str, mode: str):
    """Where to look for a measurement, best first: ``(phase, mode, label)``."""
    others = [m for m in REAL_MODES if m != mode]
    yield [(phase, mode)], "this phase"
    yield [(p, mode) for p in PHASE_NAMES if p != phase], "other phases"
    for other in others:
        yield [(phase, other)], f"this phase, {other} run"
        yield [(p, other) for p in PHASE_NAMES if p != phase], f"other phases, {other} run"


def _train_rate(phase: str, mode: str, dataset: str) -> Tuple[float, str]:
    """Seconds per model per epoch, and what that figure is based on."""
    for places, label in _sources(phase, mode):
        pairs = [
            (float(r["train_seconds"]), int(r["epochs"]))
            for p, m in places for r in _read_models_rows(p, m, dataset)
            if r.get("train_seconds") and r.get("epochs")
        ]
        if pairs:
            rate = sum(s for s, _ in pairs) / sum(e for _, e in pairs)
            return rate, f"{len(pairs)} models, {label}"
    return REFERENCE["train_seconds_per_model_epoch"], REFERENCE_NOTE


def _step_rate(step: str, phase: str, mode: str, dataset: str, reference_key: str) -> Tuple[float, str]:
    """Seconds per unit for *step*, from the closest measurement there is."""
    for places, label in _sources(phase, mode):
        rates = [
            t[step]["seconds_per_unit"]
            for p, m in places for t in [_read_timings(p, m, dataset)] if step in t
        ]
        if rates:
            return sum(rates) / len(rates), label
    return REFERENCE[reference_key], REFERENCE_NOTE


# --------------------------------------------------------------------------
# Disk: what a phase occupies now, and what it will when complete
# --------------------------------------------------------------------------

#: Flattened input size and class count per dataset, for the size projection.
INPUT_SIZE = {"cifar10": 3 * 32 * 32, "fashion_mnist": 28 * 28}
N_CLASSES = 10

#: Bytes per stored item, from the formats in ``staticsbfl.persistence``: float32
#: weights; per neuron two int64 counts plus three float32 scores; per neuron
#: table row 34 float64 columns plus its keys; and a fixed cost per file.
BYTES_PER_PARAMETER = 4
BYTES_PER_SPECTRUM_NEURON = 2 * 8 + 3 * 4
BYTES_PER_TABLE_ROW = 34 * 8 + 40
BYTES_PER_FILE = 4_000


def expected_phase_bytes(phase: str, mode: str, dataset: str) -> int:
    """Projected size of a complete phase: weights, spectra, neuron table.

    Exact in the parameter count, which dominates; approximate in the per-file
    overhead. Prediction CSVs and logs are kilobytes and left out.
    """
    total = 0
    for config in build_phase(phase, mode).configs():
        widths = (INPUT_SIZE[dataset], *config.hidden_widths, N_CLASSES)
        params = sum((a + 1) * b for a, b in zip(widths[:-1], widths[1:]))
        neurons = config.total_hidden_units
        total += (params * BYTES_PER_PARAMETER
                  + neurons * (BYTES_PER_SPECTRUM_NEURON + BYTES_PER_TABLE_ROW)
                  + 2 * BYTES_PER_FILE)
    return total


def phase_bytes(phase: str, mode: str, dataset: str) -> int:
    from staticsbfl.cli.cache_size import tree_size

    return tree_size(variants_root(mode, dataset) / phase)[0]


@dataclass
class PhaseStatus:
    phase: str
    n_models: int
    trained: int
    scored: int
    has_neurons: bool
    predictions: int          # result CSVs present, of the default sweep
    predictions_expected: int
    has_log: bool
    epochs_mismatch: int      # cached models trained under another budget
    unrecorded: int = 0       # checkpoints without a models.csv row: budget unknown

    @property
    def complete(self) -> bool:
        return (
            self.trained == self.n_models and self.scored == self.n_models
            and self.has_neurons and self.has_log
            and self.predictions >= self.predictions_expected
        )


def phase_status(phase: str, mode: str, dataset: str) -> PhaseStatus:
    spec = build_phase(phase, mode)
    configs = spec.configs()
    trained = sum(
        variant_model_path(phase, c.split, c.key, mode, dataset).exists() for c in configs
    )
    scored = sum(
        variant_spectrum_path(phase, c.split, c.key, mode, dataset).exists() for c in configs
    )
    expected = [
        variant_predictions_csv(phase, m, m if k == "raw" else f"{m}_{k}", mode, dataset)
        for m in DEFAULT_METRICS for k in DEFAULT_KINDS
    ]
    keys = {c.key for c in configs}
    epochs = str(EPOCHS_BY_MODE[mode])
    rows = _read_models_rows(phase, mode, dataset)
    mismatch = sum(
        1 for r in rows
        if r.get("key") in keys and r.get("epochs") and r["epochs"] != epochs
    )
    recorded = {r.get("key") for r in rows}
    unrecorded = sum(
        1 for c in configs
        if c.key not in recorded
        and variant_model_path(phase, c.split, c.key, mode, dataset).exists()
    )
    return PhaseStatus(
        phase=phase,
        n_models=len(configs),
        trained=trained,
        scored=scored,
        has_neurons=variant_neurons(phase, mode, dataset).exists(),
        predictions=sum(p.exists() for p in expected),
        predictions_expected=len(expected),
        has_log=variant_correlation_log(phase, mode, dataset).exists(),
        epochs_mismatch=mismatch,
        unrecorded=unrecorded,
    )


@dataclass
class Estimate:
    seconds: Dict[str, float]
    basis: Dict[str, str]

    @property
    def total(self) -> float:
        return sum(self.seconds.values())


def estimate_phase(phase: str, mode: str, dataset: str, status: Optional[PhaseStatus] = None) -> Estimate:
    """Seconds each step of ``make phase`` would take now, and why.

    Training and scoring skip what is cached, so only the missing models count.
    The neuron table and the predictors are rebuilt on every run, so they always
    count in full.
    """
    spec = build_phase(phase, mode)
    status = status or phase_status(phase, mode, dataset)
    rows = sum(c.total_hidden_units for c in spec.configs())
    n_targets = len(DEFAULT_METRICS) * len(DEFAULT_KINDS)

    train_rate, train_basis = _train_rate(phase, mode, dataset)
    score_rate, score_basis = _step_rate("score", phase, mode, dataset, "score_seconds_per_model")
    neuron_rate, neuron_basis = _step_rate("neurons", phase, mode, dataset, "neurons_seconds_per_model")
    predict_rate, predict_basis = _step_rate("predict", phase, mode, dataset, "predict_seconds_per_row_target")

    return Estimate(
        seconds={
            "train": (status.n_models - status.trained) * EPOCHS_BY_MODE[mode] * train_rate,
            "score": (status.n_models - status.scored) * score_rate,
            "neurons": status.n_models * neuron_rate,
            "predict": rows * n_targets * predict_rate,
        },
        basis={
            "train": f"{train_rate:.2f} s/model-epoch · {train_basis}",
            "score": f"{score_rate:.2f} s/model · {score_basis}",
            "neurons": f"{neuron_rate:.2f} s/model · {neuron_basis}",
            "predict": f"{predict_rate * 1e3:.2f} ms/neuron/target x {rows:,} x {n_targets} "
                       f"· {predict_basis}",
        },
    )


def _estimate_lines(status: PhaseStatus, estimate: Estimate) -> List[str]:
    missing_train = status.n_models - status.trained
    missing_score = status.n_models - status.scored
    counts = {
        "train": f"{missing_train}/{status.n_models} to train",
        "score": f"{missing_score}/{status.n_models} to score",
        "neurons": "always rebuilt",
        "predict": "always refitted",
    }
    lines = [style(f"≈ {format_duration(estimate.total)} in total", "bold")]
    for step, _target in STEPS:
        lines.append(
            f"{step:<8} {format_duration(estimate.seconds[step]):>8}   {counts[step]:<16} "
            + style(estimate.basis[step], "dim")
        )
    return lines


def _mismatch_lines(status: PhaseStatus, mode: str) -> List[str]:
    lines = []
    if status.epochs_mismatch:
        lines += [
            f"! {status.epochs_mismatch} cached model(s) were trained for another epoch "
            f"count than {mode} uses ({EPOCHS_BY_MODE[mode]}). make train stops rather than",
            "  mix two budgets; TRAIN_ARGS=--retrain-mismatched retrains just those.",
        ]
    if status.unrecorded:
        lines += [
            f"! {status.unrecorded} checkpoint(s) have no models.csv row, so their budget is "
            "unchecked; the next make train",
            "  recovers the rows from the checkpoints and then checks them.",
        ]
    return [style(line, "yellow") for line in lines]


# --------------------------------------------------------------------------
# The four events
# --------------------------------------------------------------------------


def announce_phase(phase: str, mode: str, dataset: str, glossary: bool) -> None:
    spec = build_phase(phase, mode)
    info = PHASES[phase]
    status = phase_status(phase, mode, dataset)
    estimate = estimate_phase(phase, mode, dataset, status)
    populations = spec.describe().split("\n")[1:]

    rows = [
        ("Question", [spec.question]),
        ("Varies", [info.varies]),
        ("Read as", [info.reading]),
        ("Models", [line.strip() for line in populations]),
        ("Run", [f"{mode} · {dataset} · {EPOCHS_BY_MODE[mode]} epochs per model"]),
        ("Steps", [
            f"{i}  {name:<8} {style(STEP_SUMMARY[name], 'dim')}"
            for i, (name, _t) in enumerate(STEPS, start=1)
        ]),
        ("Estimate", _estimate_lines(status, estimate) + _mismatch_lines(status, mode)),
    ]
    phase_banner(logger, f"StaticSBFL · {phase_title(phase)}", rows, central_outputs(phase, mode, dataset))
    if glossary:
        print_glossary()


def close_phase(phase: str, mode: str, dataset: str, since: Optional[float]) -> None:
    status = phase_status(phase, mode, dataset)
    tick = lambda ok: style("✔", "green") if ok else style("✘", "red")
    rows = [("Status", [
        f"{tick(status.trained == status.n_models)} models trained   {status.trained}/{status.n_models}",
        f"{tick(status.scored == status.n_models)} spectra scored   {status.scored}/{status.n_models}",
        f"{tick(status.has_neurons)} neuron table",
        f"{tick(status.predictions >= status.predictions_expected)} result CSVs      "
        f"{status.predictions}/{status.predictions_expected}",
        f"{tick(status.has_log)} correlations.log",
    ])]
    elapsed = time.time() - since if since else None
    phase_done(
        logger, f"StaticSBFL · {phase_title(phase)} — {'complete' if status.complete else 'INCOMPLETE'}",
        elapsed, rows, central_outputs(phase, mode, dataset), ok=status.complete,
    )


def status_table(mode: str, dataset: str) -> Tuple[List[str], float, float]:
    """One line per phase, plus two totals: what is missing, and a full rerun.

    The two differ because the neuron table and the predictors are rebuilt on
    every ``make phase``, so ``make light`` costs more than the missing work.
    """
    from staticsbfl.cli.cache_size import human

    lines = [
        style(f"{'phase':<8}{'':<23}{'trained':>9}{'scored':>9}{'table':>6}"
              f"{'results':>10}  {'disk now / complete':>21}  {'remaining':>9}", "bold"),
    ]
    remaining, rerun = 0.0, 0.0
    for phase in PHASE_NAMES:
        status = phase_status(phase, mode, dataset)
        estimate = estimate_phase(phase, mode, dataset, status)
        left = 0.0 if status.complete else estimate.total
        remaining += left
        rerun += estimate.total
        mark = (style("✔ done".rjust(9), "green") if status.complete
                else style(format_duration(left).rjust(9), "yellow"))
        now, full = phase_bytes(phase, mode, dataset), expected_phase_bytes(phase, mode, dataset)
        disk = f"{human(now) if now else '–':>9} / ≈{human(full):>9}"
        lines.append(
            f"{phase:<8}{PHASES[phase].title.split(' — ')[0][:22]:<23}"
            f"{status.trained:>4}/{status.n_models:<4}{status.scored:>4}/{status.n_models:<4}"
            f"{'✔' if status.has_neurons else '–':>5}  "
            f"{status.predictions:>4}/{status.predictions_expected}"
            f"{'+log' if status.has_log else '    '}  {disk}  {mark}"
        )
        lines += ["        " + l for l in _mismatch_lines(status, mode)]
    return lines, remaining, rerun


def announce_experiment(mode: str, dataset: str) -> None:
    lines, remaining, rerun = status_table(mode, dataset)
    fit, evaluate = POPULATION_SIZES[mode]["fit"], POPULATION_SIZES[mode]["eval"]
    rows = [
        ("Question", ["Can a neuron's suspiciousness be predicted from its weights and",
                      "its architecture alone, without running the model?"]),
        ("Steps", ["each phase: " + " -> ".join(name for name, _ in STEPS)
                   + style("   (details in each phase banner)", "dim")]),
        ("Run", [f"{mode} · {dataset} · {fit} fit models, {evaluate} per evaluation group"]),
        ("Status", lines),
        ("Estimate", [style(f"≈ {format_duration(rerun)} for this run", "bold")
                      + f"   ({format_duration(remaining)} of it for incomplete phases)",
                      style("complete phases still rebuild their neuron table and refit "
                            "the predictors; run one phase with make phase PHASE=...", "dim")]),
    ]
    phase_banner(
        logger, f"CONTRIBUTION III · static prediction of suspiciousness · {mode} · {dataset}",
        rows, [(variants_root(mode, dataset) / "<phase>" / "predictions" / "correlations.log",
                "one per phase: read these first")],
    )
    print_glossary()


def print_status(mode: str, dataset: str) -> Tuple[int, int]:
    """One mode x dataset block; returns ``(bytes now, bytes still to come)``."""
    from staticsbfl.cli.cache_size import human

    lines, remaining, rerun = status_table(mode, dataset)
    now = sum(phase_bytes(p, mode, dataset) for p in PHASE_NAMES)
    full = sum(expected_phase_bytes(p, mode, dataset) for p in PHASE_NAMES)
    fit, evaluate = POPULATION_SIZES[mode]["fit"], POPULATION_SIZES[mode]["eval"]
    plain(logger, "")
    plain(logger, style(f"StaticSBFL · {mode} · {dataset}", "bold", "cyan") + style(
        f"   {fit} fit / {evaluate} eval models per group, {EPOCHS_BY_MODE[mode]} epochs · "
        f"{display_path(variants_root(mode, dataset))}/", "dim"))
    for line in lines:
        plain(logger, "  " + line)
    plain(logger, f"  {'incomplete phases only (make phase PHASE=...)':<52}≈ {format_duration(remaining)}")
    plain(logger, f"  {f'full make {mode} DATASET={dataset}':<52}≈ {format_duration(rerun)}")
    plain(logger, f"  {'disk':<52}{human(now)} now, ≈ {human(full)} when complete")
    return now, max(0, full - now)
