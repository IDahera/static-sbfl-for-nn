# StaticSBFL: static suspiciousness prediction for neural networks

Spectrum-based fault localization (SBFL) ranks the neurons of a network by how strongly their activity is associated with misclassified inputs. Computing it needs a forward pass over a labelled dataset. StaticSBFL predicts the Tarantula, Ochiai and D\* suspiciousness of every neuron of a dense network **without executing the model**, from statistics of its weights and features of its architecture.

This repository is the replication package of the paper. It contains everything needed to rebuild the model populations, measure the ground truth, fit and evaluate the predictors, and compare runtime and memory against DeepFault.

## TL;DR: reproduce the paper's tables with Docker

Requirements: Docker with Compose. CIFAR-10 is downloaded on first use; all results are written to `./cache` on the host.

```bash
docker compose build staticsbfl                                # build the CPU image once
docker compose run --rm staticsbfl make smoke                  # 1. quick check (minutes): every step on tiny populations
docker compose run --rm staticsbfl make paper-phase1           # 2. RQ1, Table II:  phase 1, light and heavy
docker compose run --rm staticsbfl make paper-phase2           # 3. RQ1, Table III: phase 2, heavy
docker compose run --rm staticsbfl make paper-phase3           # 4. RQ2, Table IV:  phase 3, heavy
docker compose run --rm staticsbfl make paper-runtime          # 5. RQ3, Table VI:  runtime and memory (without DeepFault, see below)
```

- Each target trains what it needs, prints its table to the console, and writes the complete report to `cache/evaluation/cifar10/report.html`.
- Every step skips cached work, so an interrupted run resumes where it stopped.
- `make smoke` numbers only show that everything runs; do not read them.
- Training the heavy populations takes many hours on a CPU. With an NVIDIA GPU, use the GPU image instead:

  ```bash
  docker compose --profile gpu run --rm staticsbfl-gpu make paper-phase2
  ```

- On an Apple-silicon Mac, run natively to use the GPU (MPS); containers cannot reach it. See [Native setup](#native-setup).

| host | image | device |
|---|---|---|
| Linux, Windows (WSL2), Intel or Apple-silicon Mac | `staticsbfl` | CPU |
| Linux or Windows (WSL2) with an NVIDIA GPU; needs the NVIDIA driver and the NVIDIA Container Toolkit | `staticsbfl-gpu` (CUDA 12.8; other versions via `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126`) | CUDA |
| Apple-silicon Mac, natively | — | MPS |

**DeepFault in RQ3.** The two DeepFault ports need their own conda environments (TensorFlow 2.22 on Python 3.14) and do not run in the containers. `paper-runtime` in Docker therefore measures StaticSBFL and HS@pt only. The complete Table VI needs the native setup: `make runtime-setup` once, then `make paper-runtime`.

## What the paper shows, and where it comes from

The paper reports a **compressed, filtered view** of the result files:
- the scope `all` (the files also contain `active`, without dead neurons)
- the two predictors and the layer-mean baseline
- three columns:
  - `spearman_model` → ρ
  - `top_k_05` → top-5 % overlap
  - `gt_gain_05` → gain@5 %
- values rounded to two decimals; memory in MiB

`report.html` shows every value, including those the paper leaves out.

| paper | target | result file |
|---|---|---|
| RQ1, Table II (phase 1, light and heavy) | `paper-phase1` | `cache/variants[-heavy]/cifar10/phase1/predictions/*.csv` |
| RQ1, Table III (phase 2, heavy) | `paper-phase2` | `cache/variants-heavy/cifar10/phase2/predictions/*.csv` |
| RQ1, top-10 % percentile in the text | `paper-phase1`, `paper-phase2` | the same files, column `top_k_true_percentile` |
| RQ2, Table IV (phase 3, heavy, per evaluation set) | `paper-phase3` | `cache/variants-heavy/cifar10/phase3/predictions/*.csv`, column `population` |
| RQ2, light results in the text | `make phase PHASE=phase3 VMODE=light` | `cache/variants/cifar10/phase3/predictions/*.csv` |
| RQ2, Table V (peak per phase) | — | the column maxima of Tables II–IV |
| RQ3, prediction quality on the designed models | `paper-runtime` | `cache/variants-heavy/cifar10/phase4/predictions/*.csv`, `population` = `designed` |
| RQ3, Table VI (runtime and memory) | `paper-runtime` | `cache/runtime/cifar10/heavy/summary.csv` (mean, min, max; median and per-design values not in the paper) |
| RQ3, one-time preparation in the text | `paper-runtime` | `cache/runtime/costs.csv` |

Table numbers refer to the submitted version.

Names in the code vs. the paper:
- `deepfault_tf1` = **DF@graph**
- `deepfault_tf2` = **DF@eager**
- `hs_pt_cpu` / `hs_pt_mps` = **HS@pt**

## Native setup

```bash
make install          # .venv with the package (Python >= 3.11; measured with 3.13 and 3.14)
make test             # unit tests
make smoke            # quick check of every step
make paper-phase1     # ... the same targets as above
```

- **Device:** chosen automatically (CUDA, then MPS, then CPU); `DEVICE=cpu|mps|cuda` forces one.
- **Pinned versions:** `requirements.txt` pins the measured environment.
- **DeepFault:** `make runtime-setup` creates the two conda environments for the DeepFault ports.
- **Help:** `make help` lists every target and variable.

## The study in brief

**Pipeline.** Each phase runs four steps (`make train`, `score`, `neurons`, `predict`; together `make phase PHASE=phaseN`):

1. Train the models of each population.
2. Compute the hit spectra on the CIFAR-10 test set and from them Tarantula, Ochiai and D\* per neuron.
3. Build one row per neuron with 24 static features.
4. Fit gradient boosting and an MLP, and compare the predicted with the measured ranking of held-out models.

**Phases.** Fitting and evaluation populations are always split by model, never by neuron.

| phase | fitted on | evaluated on | paper |
|---|---|---|---|
| 1 reseeding | one architecture, (256, 128, 64), ReLU | the same architecture, new seeds | RQ1 |
| 2 held-out architectures | depth 1–10, width 32–512, five activations | unseen architectures from the same region | RQ1 |
| 3 one axis beyond the fit | depth 2–4, width 64–128, ReLU / LeakyReLU / ELU | `deeper` (5–10), `wider` (129–512), `saturating` (tanh, sigmoid) | RQ2 |
| 4 sampled → designed | sampled ReLU networks, depth 2–4, width 64–2048 | five designed shapes, e.g. (300, 100) | RQ3 (runtime models) |

**Configurations.**

| mode | fit / eval models per population | epochs |
|---|---|---|
| `debug` | 20 / 4 | 1 |
| `light` | 200 / 40 | 3 |
| `heavy` | 400 / 80 | 10 |

All models use Adam (learning rate 10⁻³) with batch size 256 and fixed seeds. A neuron counts as active when its weighted input is positive. The ranges are defined in [`staticsbfl/variation/phases.py`](staticsbfl/variation/phases.py) and the features in [`staticsbfl/dataset/schema.py`](staticsbfl/dataset/schema.py).

**Runtime comparison.** `make runtime` applies every tool to 20 designed models (5 shapes × 4 seeds), one fresh process per tool and model:

| tool | what runs |
|---|---|
| StaticSBFL | feature extraction and prediction, no data, no forward pass |
| HS@pt | this repository's vectorized hit spectrum, CPU and MPS |
| DF@graph, DF@eager | DeepFault, ported to TensorFlow 2.22 in two variants, graph and eager mode |

Time is wall clock per model. Memory is the process's physical footprint. Every dynamic tool is checked to reproduce the same spectrum. The ports and every change to DeepFault are documented in [`baselines/README.md`](baselines/README.md).

## Where the results are

```
cache/variants[-heavy|-debug]/cifar10/<phase>/predictions/   result CSVs and correlations.log
cache/runtime/cifar10/<mode>/summary.csv                     runtime and memory per tool
cache/runtime/costs.csv                                      one-time cost of the whole run
cache/evaluation/cifar10/report.html                         everything, filterable, in the browser
```

## Repository layout

```
staticsbfl/   the package: cli/ (the steps the Makefile calls), variation/, training/,
              suspiciousness/, dataset/, evaluation/, benchmark/, log/, viz/, data/, utils/
baselines/    the dynamic tools of the runtime comparison (DeepFault ports, upstream sources)
tests/        unit tests (make test)
```

## Licensing

- MIT ([`LICENSE`](LICENSE)), except the DeepFault ports.
- `baselines/deepfault_tf1/`, `deepfault_tf2/`, `deepfault_upstream/` and `common/` are a modified version of DeepFault (Eniser, Gerasimou, Sen, FASE 2019) under GPL-3.0-or-later, each with its own `LICENSE` and `NOTICE`.
- `baselines/icsoft_pt/` and `icsoft_upstream/` are described in [`baselines/README.md`](baselines/README.md#licence).

If you use this software, please cite it as described in [`CITATION.cff`](CITATION.cff).
