"""What every name in the evaluation means -- one source for the log, the HTML and its hover texts.

``TERMS`` explains the values that appear *in* the tables (populations, scopes,
modes, predictors, metrics, target kinds, tools); ``COLUMNS`` explains the
columns themselves. :mod:`staticsbfl.log.evaluation` prints them once,
:mod:`staticsbfl.viz.html_report` shows them as a glossary tab and as the hover text
of every header and tag. A test asserts that every column the report shows has
an entry here, so no column goes unexplained.
"""

from typing import Dict, Tuple

#: kind -> {value: explanation}
TERMS: Dict[str, Dict[str, str]] = {
    "population": {
        "train": "phase 1/2 fit population: the models the predictor is fitted on (200 light, 400 heavy).",
        "test": "phase 1/2 evaluation population: held-out models (40 light, 80 heavy); "
                "phase 1: same architecture 256-128-64 ReLU, new seeds; phase 2: unseen architectures "
                "from the same region (depth 1-10, width 32-512, five activations).",
        "base": "phase 3 fit population: depth 2-4, width 64-128, relu / leaky_relu / elu.",
        "deeper": "phase 3 evaluation: like base, but depth 5-10 -- only depth moved.",
        "wider": "phase 3 evaluation: like base, but width 129-512 -- only width moved.",
        "saturating": "phase 3 evaluation: like base, but tanh / sigmoid -- only the activation moved.",
        "sampled": "phase 4 fit population: sampled ReLU stacks, depth 2-4, width 64-2048.",
        "designed": "phase 4 evaluation: five hand-designed shapes (256-128, 512-256, 300-100, "
                    "2048-1024-512, 1024-512-256-128) x 8 seeds (light) / 16 (heavy) -- established designs.",
    },
    "scope": {
        "all": "every hidden neuron of the evaluation models.",
        "active": "only neurons that fired at least once on the test set. Dead units score 0 under "
                  "every metric, so 'all' rewards recognising them; 'active' is the honest headline.",
    },
    "mode": {
        "light": "200 fit / 40 evaluation models per phase (per group in phase 3), 3 epochs each (cache/variants/).",
        "heavy": "400 fit / 80 evaluation models per phase (per group in phase 3), 10 epochs each (cache/variants-heavy/).",
        "debug": "20 fit / 4 evaluation models, 1 epoch -- a smoke test, its numbers are noise.",
    },
    "predictor": {
        "gradient_boosting": "HistGradientBoostingRegressor (tree ensemble, boosted), early stopping.",
        "random_forest": "RandomForestRegressor, 500 trees -- only in results fitted before 2026-10-22; dropped "
                         "(no better than gradient boosting, ~5 GB per fitted target).",
        "mlp": "MLPRegressor 256-128-64 behind a StandardScaler (neural network), early stopping.",
        "baseline:layer_mean": "predicts every neuron as its layer slot's training mean: what knowing "
                               "only the layer is worth. Read the margin over it.",
        "baseline:random": "random scores: the chance level.",
    },
    "model_type": {
        "tree_ensemble": "gradient boosting (and the random forest of results fitted before 2026-10-22).",
        "neural_network": "the MLP regressor.",
        "baseline": "layer mean and random; listed for reading, never 'best'.",
    },
    "metric": {
        "tarantula": "(a_f/N_f) / (a_f/N_f + a_s/N_s): concentration of a neuron's firing on failures, [0, 1].",
        "ochiai": "a_f / sqrt(N_f (a_f + a_s)): cosine of 'fired' and 'failed', [0, 1].",
        "dstar": "a_f^3 / (a_s + N_f - a_f): unbounded, sharpens towards neurons firing on many failures.",
    },
    "target": {
        "<metric>": "the raw suspiciousness value.",
        "<metric>_rank_model": "its rank among all neurons of the model, in [0, 1] -- what a ranking is judged on.",
        "<metric>_rank_layer": "its rank within the neuron's own layer.",
    },
    "tool": {
        "staticsbfl_gradient_boosting": "StaticSBFL with the phase-4 gradient-boosting predictors: weights in, "
                                        "ranking out; no data, no forward pass.",
        "staticsbfl_mlp": "StaticSBFL with the phase-4 MLP predictors.",
        "hs_pt_cpu": "HS@pt: this repository's streaming PyTorch spectrum (one pass, all layers, "
                       "counts per batch), on the CPU.",
        "hs_pt_mps": "HS@pt on the Apple-silicon GPU (MPS).",
        "icsoft_pt_cpu": "ICSoft@pt: the published ICSOFT 2025 code (stores every activation, one pass per "
                         "layer), on the CPU. Opt-in.",
        "icsoft_pt_mps": "ICSoft@pt on MPS. Opt-in.",
        "deepfault_tf1": "DeepFault on TensorFlow's graph mode (tf.compat.v1, eager off) with the Keras 2 API.",
        "deepfault_tf2": "DeepFault on TensorFlow 2 eager with Keras 3.",
    },
}

#: column -> explanation (hover text of the column header)
COLUMNS: Dict[str, str] = {
    # identity
    "mode": "light or heavy population (size and epochs, see the glossary).",
    "phase": "which generalisation question: 1 reseeding, 2 interpolation, 3 one axis, 4 designed shapes.",
    "population": "the group of models the row is evaluated on (see the glossary).",
    "scope": "all neurons, or only those that fired at least once ('active').",
    "metric": "the suspiciousness metric predicted: Tarantula, Ochiai or D*.",
    "target": "raw value, or its rank within the model / within the layer.",
    "predictor": "the fitted regressor, or a baseline.",
    "model type": "tree ensemble, neural network or baseline.",
    "case": "predictor · target of the best row.",
    "models": "number of evaluation models the value is averaged over.",
    # ranking quality
    "rho/model": "Spearman rank correlation of predicted vs measured suspiciousness, within each model, "
                 "then the mean over models. 1 = same order, 0 = unrelated.",
    "± sd": "spread of rho/model across models (not a standard error; that is sd/sqrt(models)).",
    "wtau": "weighted Kendall tau: like rho, but a swap near the top costs far more than one near the bottom.",
    "top-5%": "share of the true top 5% found in the predicted top 5%, per model, averaged. Chance about 0.05.",
    "top-10%": "share of the true top 10% found in the predicted top 10%, per model, averaged. Chance about 0.10.",
    "gain@5%": "(measured suspiciousness of the predicted top 5% - mean of all) / (that of the true top 5% - mean of "
               "all): 0 = random pick, 1 = as good as the true top. Scale-free, comparable across metrics.",
    "gain@10%": "the same for the top 10%. High gain with modest overlap: the candidates are not exactly the true "
                "top, but highly suspicious ones.",
    "lift@10%": "mean measured suspiciousness of the predicted top 10% divided by that of the remaining 90%.",
    "GT top 10%": "mean measured (raw) suspiciousness of the neurons predicted in the top 10%.",
    "GT rest 90%": "mean measured (raw) suspiciousness of all other neurons.",
    "target kind": "what the predictor was fitted on: the raw value, or its rank within the model.",
    "top-k overlap": "share of the true top 10% found in the predicted top 10%, per model, averaged.",
    "top-20%": "share of the true top 20% found in the predicted top 20%.",
    "chance": "what a random ranking reaches on top-10%: k/n, about 0.10.",
    "× chance": "top-10% overlap divided by its chance level.",
    "near (2k)": "share of the true top 10% found in the predicted top 20%: a near miss counts. Chance 2k/n.",
    "true pct": "mean true percentile of the neurons predicted in the top 10%: ~0.5 by chance, "
                "~0.95 for a perfect ranking. Shows how close the misses are.",
    "rho/layer": "Spearman within each layer, averaged: what is left once 'which layer' is known.",
    "dead units": "share of the population's neurons that never fired.",
    # overview
    "best rho/model": "highest rho/model of a fitted predictor on this population and scope.",
    "layer mean": "rho/model of the layer-mean baseline on the same case.",
    "margin": "best minus layer mean. <= 0: the predictor knows no more than the layer.",
    "rho (all)": "best rho/model on the other scope.",
    "rho (active)": "best rho/model on the other scope.",
    "best wtau": "highest wtau of a fitted predictor.",
    "best top-k": "highest top-10% overlap of a fitted predictor.",
    # statistics
    "group": "the value the rows are grouped by.",
    "rows": "rows in the group with a defined value of the column.",
    "undefined": "rows in the group whose value is undefined (blank / nan) -- excluded, not zero.",
    "min": "smallest defined value.", "q25": "first quartile.", "median": "median.",
    "mean": "arithmetic mean.", "q75": "third quartile.", "max": "largest defined value.",
    "sd": "standard deviation (population, ddof 0).", "range": "max - min.",
    "column": "which result column the statistics describe.",
    # runtime
    "tool": "the measured implementation (see the glossary).",
    "per model (median)": "load + analyse one model, median over models (each model: median over repeats).",
    "max": "largest value over the models.",
    "slowest model": "the model with the largest per-model time.",
    "total": "sum over the models.",
    "setup": "framework import + test data or predictor load, paid once per session.",
    "mem/model (median)": "footprint above the set-up process while one model is analysed.",
    "process max (median)": "high-water mark of the whole process footprint, incl. framework import.",
    "gpu": "MPS allocator share (sampled), MPS runs only.",
    "runs": "measured processes by status.",
    "design": "the designed architecture (all its seeds).",
    "neurons": "hidden neurons of the design.",
    "static": "the static predictor.", "vs dynamic": "the dynamic tool it is compared with.",
    "speed-up": "dynamic / static per-model time, geometric mean over models.",
    "range ×": "smallest and largest per-model speed-up.",
    "one-time": "train + score the fit population, build the table, fit the predictor (upper bound).",
    "saving / model": "median of (dynamic - static) per-model time.",
    "pays off after": "number of analysed models after which the one-time cost is recovered.",
    "… if trained anyway": "the same, if the fit population had been trained for other reasons.",
    "reference": "what the spectrum is compared with: the score-step cache or HS@pt on the CPU.",
    "identical": "models whose counts match exactly.",
    "neurons off": "neurons with any count difference (float rounding at the threshold).",
    "max diff": "largest count difference of any neuron.",
    "min rho T": "worst Spearman rho of Tarantula over the models.",
    "min rho O": "worst Spearman rho of Ochiai over the models.",
    "min rho D*": "worst Spearman rho of D* over the models.",
    "model": "the evaluated model.", "repeat": "index of the fresh process.", "status": "ok, failed, timeout.",
    "per model": "load + analyse this model in this process.", "mem/model": "see mem/model (median).",
    "process max": "see process max (median).", "detail": "error tail for a failed run.",
    # costs
    "timed": "models with a recorded training time (rows rebuilt from a checkpoint have none).",
    "epochs": "training epochs per model; two values mean the phase mixes budgets.",
    "train total": "sum of the measured training times.",
    "per m-ep": "training seconds per model and epoch -- comparable between light and heavy.",
    "score": "the spectrum step (make score) of the whole phase.",
    "predict": "the predict step: fitting every predictor for every target, then evaluating.",
    "(fit)": "the fitting alone, part of predict.",
    "tarantula": "rho/model of the static Tarantula ranking on the designed models.",
    "ochiai": "rho/model of the static Ochiai ranking.", "dstar": "rho/model of the static D* ranking.",
}


def describe(kind: str, value: str) -> str:
    """The explanation of *value* of *kind*, or ``""``. Targets match by suffix."""
    terms = TERMS.get(kind, {})
    if value in terms:
        return terms[value]
    if kind == "target":
        for suffix in ("_rank_model", "_rank_layer"):
            if value.endswith(suffix):
                return terms[f"<metric>{suffix}"]
        return terms["<metric>"]
    return ""


def column_help(header: str) -> str:
    """The hover text of a column header, or ``""``."""
    return COLUMNS.get(header, "")


#: The kinds the log prints, in order, with a heading.
LOG_KINDS: Tuple[Tuple[str, str], ...] = (
    ("population", "populations"), ("scope", "scopes"), ("mode", "modes"),
    ("predictor", "predictors"), ("metric", "metrics"), ("target", "targets"),
)
