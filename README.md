# StaticSBFL — static suspiciousness prediction for neural networks

Spectrum-based fault localization (SBFL) ranks the neurons of a network by how
strongly their activity is associated with misclassified inputs. Computing that
ranking needs a *hit spectrum*: a forward pass over a labelled dataset, with
per-neuron counts of how often each neuron fired on failing and on successful
inputs. StaticSBFL predicts the suspiciousness of every neuron — Tarantula,
Ochiai and D\* — **without executing the model**, from per-neuron weight
statistics and features of the architecture around it.

This repository is the replication package of the study. It contains the
complete pipeline (generating and training the model populations, measuring
their hit spectra, building the neuron table, fitting and evaluating the
predictors), the runtime and memory comparison with dynamic hit-spectrum tools,
and the final evaluation report. Version 2.0.0 replaces the pipeline of
version 1; the earlier versions remain in the git history.

## Contents

- [Quick start](#quick-start)
- [Pipeline overview](#pipeline-overview)
- [The four phases](#the-four-phases)
- [Run modes: debug, light, heavy](#run-modes-debug-light-heavy)
- [Running the study](#running-the-study)
- [Runtime and memory comparison](#runtime-and-memory-comparison)
- [Environments](#environments)
- [What the results mean](#what-the-results-mean)
- [Where the results are](#where-the-results-are)
- [Repository layout](#repository-layout)
- [Licensing](#licensing)

## Quick start

```bash
make install        # .venv + the package, editable, with pytest
make test           # the unit test suite
make debug          # all four phases, tiny populations, 1 epoch: checks that everything runs
make light          # all four phases, 200 fit / 40 eval models per population, 3 epochs
make evaluate       # the final evaluation: console, log and cache/evaluation/cifar10/report.html
```

`make help` lists every target and variable. A bare `make` only prints the help.
Every step skips work that is already cached, so an interrupted run resumes
where it stopped; `FORCE=--force` redoes cached work. CIFAR-10 and
Fashion-MNIST are downloaded into `cache/data/` on first use.

## Pipeline overview

Each phase builds *populations* of dense MLPs (CIFAR-10 by default) whose
architectures differ in controlled ways, and runs four steps:

| step | make target | module (`staticsbfl/cli/`) | what it does | writes |
|---|---|---|---|---|
| 1 train | `make train` | `phase_train.py` | build every model from its configuration and train it under the mode's budget; cached models are skipped | `<population>/<key>.pt`, `models.csv`, `design.json` |
| 2 score | `make score` | `phase_score.py` | one pass over the test set per model: count `a_f` / `a_s` per neuron → Tarantula, Ochiai, D\* | `<population>/spectra/<key>.pt`, `activation_rates.csv` |
| 3 neurons | `make neurons` | `phase_neurons.py` | one row per hidden neuron: 24 static features (14 of the neuron's own incoming and outgoing weights, 10 of the architecture) plus the measured targets | `neurons.npz` |
| 4 predict | `make predict` | `phase_predict.py` | fit a histogram gradient-boosting regressor and an MLP regressor per metric and target; rank the neurons of held-out models; compare with the measured ranking | `predictions/correlations.log`, `predictions/*.csv` |

`make phase PHASE=phaseN` runs the four steps of one phase with a banner
before (what the phase asks, which files hold its result, a time estimate) and
a check of every written artifact after. Steps 1 and 2 skip cached work; steps
3 and 4 rerun on every `make phase`.

The features are statistics of the weights (mean, standard deviation, L1/L2
norm, largest absolute value, fraction of near-zero weights, bias, fan-in and
fan-out), not the weights themselves, so rows of layers and models of different
widths are commensurable. The full schema, and the features deliberately left
out, is declared in [`staticsbfl/dataset/schema.py`](staticsbfl/dataset/schema.py).

## The four phases

Each phase fits the predictors on one population and evaluates them on others.
Splits are always **by model, never by neuron**: neurons of one model share its
architecture, initialization and training run.

| phase | question | varies | fitted on | evaluated on |
|---|---|---|---|---|
| **1** reseeding — *noise floor* | How far does suspiciousness move under reseeding alone? | **only the seed**; architecture fixed at 256-128-64, ReLU | `train` | `test` |
| **2** held-out architectures — *interpolation* | Does a predictor transfer to unseen architectures from the same region? | depth 1–10, width 32–512, five activations | `train` | `test` |
| **3** one axis at a time — *extrapolation* | Does it extrapolate along a single architecture axis? | base: depth 2–4, width 64–128, relu / leaky_relu / elu; each evaluation group moves **one** axis | `base` | `deeper` (depth 5–10), `wider` (width 129–512), `saturating` (tanh / sigmoid) |
| **4** sampled → designed — *transfer* | Does it transfer from sampled to designed architectures? | sampled ReLU, depth 2–4, width 64–2048 | `sampled` | `designed`: 5 hand-picked shapes (`256-128`, `2048-1024-512`, `512-256`, `300-100`, `1024-512-256-128`) × seeds |

Phase 1 is the ceiling every later number is read against: no static predictor
can be more consistent than its target is across seeds. The ranges of every
phase are defined in [`staticsbfl/variation/phases.py`](staticsbfl/variation/phases.py);
`make preview PHASE=phase2` lists a population without building anything.

`DATASET=fashion_mnist` runs the same four phases on Fashion-MNIST, a
*control*: dense models plateau near 50 % accuracy on CIFAR-10 but reach about
88 % here, which separates "units die because the task is too hard for this
capacity" from "deep narrow ReLU stacks lose units anyway".

## Run modes: debug, light, heavy

| mode | fit / eval models per population | epochs | cache tree | purpose |
|---|---|---|---|---|
| `debug` | 20 / 4 | 1 | `cache/variants-debug/` | pipeline check only — do not read its numbers |
| `light` | 200 / 40 | 3 | `cache/variants/` | the main configuration |
| `heavy` | 400 / 80 | 10 | `cache/variants-heavy/` | more models, longer training |

Fixed in every mode: Adam (learning rate 1e-3), batch size 256, D\* exponent 3,
a neuron is active iff its pre-activation is > 0, seed-controlled
initialization and data order. Every model of a population is trained under
the same budget; `make train` refuses to mix cached models trained for another
epoch count (`TRAIN_ARGS=--retrain-mismatched` retrains just those).

Every population is a prefix of one fixed pool of configurations, so a
smaller mode's models are the first models of a larger one, with the same
architecture and seed. Each mode still keeps and trains its own checkpoints in its own tree. If
a population is ever shrunk, `make trim VMODE=heavy` lists the surplus models
(dry run) and `APPLY=--apply` removes them.

Approximate cost on an Apple-silicon laptop (MPS), CIFAR-10: `light` about
3–4 hours for all four phases, `heavy` most of a day; disk about 8 GB (light)
and 16 GB (heavy), most of it the wide models of phase 4. `make status`
recomputes both from what has already been measured and says what each figure
is based on.

## Running the study

```bash
make status                          # done / missing, time and disk, light AND heavy
make status VMODE=heavy              # one mode
make preview PHASE=phase2            # list the population, build nothing
make phase   PHASE=phase1            # one phase end to end (VMODE=light by default)
make light                           # all four phases, CIFAR-10
make light   DATASET=fashion_mnist   # the control population
make heavy                           # all four phases, heavy budget
make runtime-setup                   # once: the two DeepFault conda environments
make runtime                         # runtime + memory comparison on the phase-4 models
make evaluate                        # everything that exists, read only
make all                             # light -> heavy -> runtime -> evaluate
```

`METRIC='--metric ochiai'` narrows the predictor sweep of `make predict`;
`DEVICE=cpu|mps|cuda` overrides the automatic device choice.

## Runtime and memory comparison

`make runtime` applies every tool to 20 *designed* models of phase 4 (the 5
shapes × their first 4 seeds; `RUNTIME_ARGS='--seeds-per-design 0'` takes all),
one fresh process per (tool, model, repeat), in light and heavy mode, whichever
has phase-4 models:

| tool | what runs | where |
|---|---|---|
| **StaticSBFL** (gradient boosting, MLP) | the phase-4 predictors on the model's weights: features + prediction; no data, no forward pass | `staticsbfl/cli/runtime_static.py` |
| **HS@pt**, CPU and MPS | this repository's streaming hit spectrum, exactly as `make score` computes it: one pass over the data for all layers, counts accumulated per batch | `staticsbfl/cli/runtime_hs.py` |
| **DF@tf1** | DeepFault, TensorFlow 2.22 in graph mode (`tf.compat.v1`, eager off) with the Keras 2 API (`tf_keras`) — the TF1 execution model on a TF2 binary | `baselines/deepfault_tf1/` |
| **DF@tf2** | DeepFault, TensorFlow 2.22 eager with Keras 3 | `baselines/deepfault_tf2/` |
| **ICSoft@pt** (opt-in: `RUNTIME_ARGS='--tool icsoft_pt_cpu'`) | the published ICSOFT 2025 spectrum code: one forward pass per layer, every activation stored | `baselines/icsoft_pt/` |

`make runtime` refits and saves the phase-4 estimators first if they were
never saved (`make runtime-estimators`). Time is measured per phase inside
each process (import, data, model load, spectrum or prediction, scoring).
Memory is the process **physical footprint** (macOS `phys_footprint`, which
includes Metal/MPS buffers; `VmRSS` on Linux), with every process started
under `MallocLargeCache=0` so that macOS charges re-used memory correctly;
`tests/test_footprint.py` checks the probe against allocations of known size.
Every run also checks that each dynamic tool reproduced the spectrum of
`make score` (`fidelity.csv`; `tests/test_spectrum_agreement.py`). DeepFault
dominates the wall clock — its spectrum loop is quadratic in the number of
samples — at about 2.7 h per port and mode for the 20 models. Smoke run:
`make runtime VMODE=light RUNTIME_ARGS='--max-samples 500 --repeats 1'`.

What was ported, every change and its possible effect on the measured time,
and every library version: [`baselines/README.md`](baselines/README.md).
`make costs` reports the one-time cost of the whole run (training, scoring,
table building, fitting), and `make evaluate` combines both into the
amortization point: the number of models after which the static prediction
has paid for its training data.

## Environments

| purpose | environment | set up by |
|---|---|---|
| the four phases and the evaluation | Python 3.13 virtual environment `.venv` (any Python ≥ 3.11; the test suite passes on 3.13 and 3.14) | `make install` |
| runtime comparison: StaticSBFL, HS@pt, ICSoft@pt | the main environment — whatever `PYTHON` is (`.venv`, an active conda/virtualenv, or `PYTHON=...`). For a like-for-like comparison, use the same interpreter build as the DeepFault environments (below), e.g. a conda env created with `python=3.14.2=h1858270_101_cp314` and `pip install -r requirements.txt -e .` | — |
| DF@tf1, DF@tf2 | conda environments `deepfault-tf1` and `deepfault-tf2`: Python **3.14.2** (`h1858270_101_cp314`), TensorFlow **2.22.0rc0** (the only TensorFlow with a Python 3.14 wheel), `tf_keras` 2.21.0 (tf1) / Keras 3 nightly (tf2), all pinned in `baselines/deepfault_tf*/requirements*.txt` | `make runtime-setup` (needs `conda`) |

[`requirements.txt`](requirements.txt) pins the main environment used for the
measurements; `pyproject.toml` states the minimum versions. Install PyTorch
first with the build that matches the machine if the default wheel does not
(see pytorch.org). Every runtime run records the versions it actually used in
`cache/runtime/<dataset>/<mode>/environment.json`, and `make evaluate` prints
them. The runtime comparison measures on CPU and Apple-silicon MPS;
TensorFlow has no GPU build for Python 3.14 on macOS, so a CPU-vs-CPU
comparison always exists. A local, uncommitted `config.mk` can set `PYTHON=...`
for every `make` call.

## What the results mean

**Targets**, measured per neuron from the hit spectrum. `a_f` / `a_s`: number
of failing / successful test inputs on which the neuron was active; `N_f` /
`N_s`: all failing / successful inputs.

| target | definition | range |
|---|---|---|
| `tarantula` | (a_f/N_f) / (a_f/N_f + a_s/N_s) | [0, 1] |
| `ochiai` | a_f / √(N_f · (a_f + a_s)) | [0, 1] |
| `dstar` | a_f³ / (a_s + N_f − a_f) | unbounded |
| `<metric>_rank_model` | the neuron's rank among all neurons of its model | [0, 1] |
| `<metric>_rank_layer` | the neuron's rank within its own layer | [0, 1] |
| `activity_rate` | share of inputs the neuron fired on; 0 = dead | never a feature |

Every metric is fitted **separately**: a predictor of Tarantula is not a
predictor of D\*.

**Evaluation**: predicted vs. measured ranking, always *within* a model first,
then averaged with every model weighted equally.

| column | meaning |
|---|---|
| `rho/model` | Spearman rank correlation per model, mean over models (1 same order, 0 unrelated, −1 reversed) |
| `sd` | spread of `rho/model` across models (not a standard error) |
| `wtau` | weighted Kendall τ: a swap near the top costs more than one near the bottom |
| `top-k` | share of the true top-10 % neurons found in the predicted top-10 %; compare with `chance` = k/n |
| `top_k_05`, `top_k_20`, `top_k_near`, `top_k_true_percentile` | the same at 5 % / 20 %, near misses, and how close the misses are |
| `gt_top_05/10`, `gt_rest_05/10`, `gt_gain_05/10` | mean **measured** suspiciousness of the predicted top 5 / 10 % vs. the rest, and the share of the achievable gain (0 random pick, 1 as good as the true top) |
| `rho/layer` | Spearman within each layer: removes the "which layer" signal |
| `n_inact` | neurons that never fired |
| scope `all` / `active` | every neuron / only neurons that fired at least once |
| `baseline:layer_mean`, `baseline:random` | each neuron predicted as its layer's training mean; chance |

Read the **margin over `baseline:layer_mean`**, not the raw correlation: a
predictor that does not beat it has learned only which layer a neuron sits in.
Every result is reported over all neurons and over the active ones only — a
neuron that never fired scores 0 under all three metrics, so a predictor
rewarded under the first scope may only have learned to recognise dead units.
Undefined correlations (a constant vector has no ranking) are counted and
excluded, never replaced by zero. `make light` prints the glossary of every
metric and column once at the top, and `correlations.log` opens with it.

**The final evaluation** (`make evaluate`, read only) goes through everything
the earlier steps wrote — result CSVs, `design.json`, `models.csv`,
`timings.json`, the runtime records — and trains, scores and fits nothing.
Anything not computed yet appears as a placeholder naming the command that
produces it. Variables: `VMODE=light` (one mode only), `DATASET=fashion_mnist`,
`SCOPE=all`, `EVAL_ARGS='--model-type tree_ensemble --metric ochiai'` (filters
apply to the log, the CSVs and the HTML alike).

## Where the results are

```
cache/variants/<dataset>/<phase>/          light   (heavy: cache/variants-heavy/…, debug: cache/variants-debug/…)
├── predictions/
│   ├── correlations.log         ◀ read this first: what each number means, then the numbers
│   ├── <phase>__<metric>__<target>.csv
│   └── estimators/              phase 4 only: the fitted predictors the runtime comparison loads
├── activation_rates.csv         per-layer activity: the file that shows dead units
├── models.csv                   one row per model: configuration, accuracy, training time
├── design.json                  the specification: question, regions, roles, budget, cost
├── neurons.npz                  the neuron table: features plus targets
├── timings.json                 measured time per step, used by the estimates
└── <population>/  <key>.pt  +  spectra/<key>.pt

cache/variants/<dataset>/summary_<scope>.csv    the overview across phases, per mode
cache/runtime/<dataset>/<mode>/                 runtime records, summary.csv, by_design.csv, fidelity.csv, environment.json
cache/runtime/costs.csv                         the cost report (make costs)
cache/evaluation/<dataset>/report.html          every row, in tabs, highlighted, filterable in the browser
cache/evaluation/<dataset>/evaluation.log       the console output of make evaluate
cache/evaluation/<dataset>/statistics.csv       per group: count, undefined, quartiles, mean, sd, …
```

Everything the pipeline produces lives under `cache/` (git-ignored);
`make cache-size` shows what each part occupies and which target deletes it.

## Repository layout

```
staticsbfl/
├── cli/              the entry points the Makefile calls: phase_{train,score,neurons,predict},
│                     status, evaluate, costs, trim, cache_size, runtime_{benchmark,static,hs,icsoft}
├── variation/        architecture configurations, the four phases, the model builder
├── training/         train / evaluate loops
├── suspiciousness/   hit-spectrum accumulation (spectrum.py) and the SBFL metrics (metrics.py)
├── dataset/          the neuron table: feature schema and extraction
├── evaluation/       the final evaluation and the runtime comparison, as data
├── benchmark/        memory and time probes of the runtime comparison
├── log/, viz/        console / log output and the HTML report
├── data/             CIFAR-10 and Fashion-MNIST loaders
├── models/           two fixed dense reference models (test fixtures), parameter counting
├── persistence.py    reading and writing checkpoints and spectra
└── utils/            device selection, seeding, cache paths
baselines/            the dynamic tools of the runtime comparison (see baselines/README.md)
tests/                the unit test suite (make test)
```

The package installs as `staticsbfl`; after `make install` the steps are also
available as commands (`staticsbfl-train`, `staticsbfl-score`,
`staticsbfl-neurons`, `staticsbfl-predict`, `staticsbfl-evaluate`, …).

## Licensing

- Everything outside `baselines/deepfault_tf1/`, `baselines/deepfault_tf2/`,
  `baselines/deepfault_upstream/` and `baselines/common/` is under the MIT
  licence ([`LICENSE`](LICENSE)).
- Those four directories are a modified version of DeepFault (Eniser,
  Gerasimou, Sen, FASE 2019; © 2019 Bogazici University) and are distributed
  under the **GPL-3.0-or-later**, independently of the rest of the repository.
  Each carries its own `LICENSE` and `NOTICE`; every modified file starts with
  an SPDX line, the original copyright and a dated modification notice, and
  every change is marked `# PORT:` and listed in the port's README. The
  unmodified upstream sources are in `baselines/deepfault_upstream/`.
- `baselines/icsoft_pt/` and `baselines/icsoft_upstream/` contain the ICSOFT
  2025 spectrum code (see [`baselines/README.md`](baselines/README.md#licence)).

If you use this software, please cite it as described in
[`CITATION.cff`](CITATION.cff).
