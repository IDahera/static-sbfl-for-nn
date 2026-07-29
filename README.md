# Static SBFL for Neural Networks

## Overview

This repository contains the implementation and experiments for the SEFM 2026
paper *StaticSBFL: Static Suspiciousness Prediction for Neural Networks Without
Model Execution*. It generates dense neural networks with randomized
architectures, computes Ochiai and Tarantula suspiciousness, trains classifiers
to predict suspicious neurons, and ranks the classifiers' input features.

```text
generate_models.py
        |
evaluate_models.py
        |
compute_suspiciousness.py
        |
transform_datasets.py
        |
train_classifiers.py --------> RQ1 metrics and RQ2 timings
        |
interpret_features.py -------> RQ3 feature rankings
```

| Experiment | Models | Dataset |
|---|---:|---|
| RQ1--RQ3 baseline | 1,000 | MNIST and Fashion-MNIST |
| RQ4 small sample | 25 | MNIST |
| RQ4 larger sample | 50 | MNIST |

RQ4 uses two separate runs. The relevant configurations are in
`configs/artifact/`; the similarly named files in `configs/model_gen/` are
earlier exploratory configurations.

## Run natively with Python

Python 3.13 is recommended. Create an environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the two-model smoke test on the CPU:

```bash
SBFL_DEVICE=cpu bash artifact/run_pipeline.sh smoke
```

On an Apple Silicon Mac, use `SBFL_DEVICE=mps` instead of `cpu` to run natively
on Apple's MPS backend; `SBFL_DEVICE=mps bash artifact/run_pipeline.sh all`
runs `rq1-rq2`, `rq3`, and `rq4` in order.

On a CUDA-capable Linux or Windows/WSL2 machine, run:

```bash
SBFL_DEVICE=cuda bash artifact/run_pipeline.sh full
```

The wrapper is a Bash script. On native Windows, use WSL or Git Bash, or use
the Docker workflow below.

## Run with Docker on CPU

This is the portable option for Linux, Windows, Intel macOS, and Apple Silicon
macOS. Install Docker Desktop or Docker Engine with Compose, then run:

```bash
mkdir -p data outputs
docker compose build artifact
docker compose run --rm artifact
docker compose run --rm artifact \
  bash artifact/run_pipeline.sh smoke
```

The first `run` checks the environment. The second runs every pipeline stage
with two MNIST models; it verifies functionality but does not reproduce the
paper's numerical results.

On Apple Silicon, Docker automatically selects the Linux ARM64 image. Docker
cannot access macOS MPS, so this remains a CPU workflow.

Run the paper experiments separately as follows:

```bash
# RQ1 metrics and RQ2 timings; also creates the trained baseline classifiers
docker compose run --rm artifact \
  bash artifact/run_pipeline.sh rq1-rq2

# RQ3 feature rankings; run this after RQ1/RQ2
docker compose run --rm artifact \
  bash artifact/run_pipeline.sh rq3

# RQ4 results for separate sets of 25 and 50 MNIST models
docker compose run --rm artifact \
  bash artifact/run_pipeline.sh rq4
```

These commands use the CPU image and can therefore be very slow for the full
paper configurations. They are useful when no NVIDIA GPU is available or when
the corresponding generated models and intermediate datasets already exist.

## Run with Docker on an NVIDIA GPU

The complete experiment requires an x86-64 Linux NVIDIA host or Windows with
Docker Desktop's WSL2 GPU support. Linux additionally requires the NVIDIA
Container Toolkit. Docker on macOS cannot expose CUDA or Apple's MPS backend to
this Linux container.

Check GPU access, then build and run:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

docker compose --profile gpu build artifact-gpu
docker compose --profile gpu run --rm artifact-gpu
```

The GPU service runs `rq1-rq2`, `rq3`, and `rq4` in that order. It generates
the 1,000-model MNIST and Fashion-MNIST baselines, the separate 25- and
50-model MNIST experiments, and all downstream results. It can take several
hours. Generation is resumable, but use an empty `outputs/` directory for clean
timing measurements.

If transformed datasets have already been supplied, rerun only classification
and feature interpretation:

```bash
docker compose run --rm artifact \
  bash artifact/run_pipeline.sh classify
```

## Outputs and defaults

Downloads are stored in `data/`; generated files are stored in `outputs/`.
The main result files are:

```text
outputs/results/<configuration>/<dataset>/classification_results.csv
outputs/results/<configuration>/<dataset>/feature_rankings.csv
outputs/model_generation_times.csv
outputs/suspiciousness_computation_times.csv
outputs/dataset_transformation_times.csv
```

The artifact configurations use random seed 42, one training epoch, 1,000
suspiciousness samples, threshold 0.0, and an 80/20 classifier split. Runtime
measurements depend on the host hardware.
