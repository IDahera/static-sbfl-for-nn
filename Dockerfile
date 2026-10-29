# syntax=docker/dockerfile:1
#
# Containers for the StaticSBFL pipeline (the four phases, the predictors and the
# evaluation). Two targets:
#
#   cpu   (default) portable CPU image, linux/amd64 and linux/arm64
#   cuda  NVIDIA GPU image, linux/amd64; needs the host's NVIDIA driver and the
#         NVIDIA Container Toolkit
#
# Apple-silicon GPUs (MPS) are not reachable from a container: on a Mac, run the
# pipeline natively (make install) to use MPS, or use the cpu image.
# The DeepFault baselines of the runtime comparison need their own conda
# environments (make runtime-setup) and are not part of these images.

ARG PYTHON_IMAGE=python:3.14-slim-bookworm
FROM ${PYTHON_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    PYTHON=python \
    PAPER_RUNTIME_TOOLS="--tool staticsbfl_gradient_boosting --tool staticsbfl_mlp --tool hs_pt_cpu"

WORKDIR /workspace

# make drives the pipeline; libgomp1 is needed by scikit-learn's OpenMP code;
# tini forwards signals, so Ctrl-C stops a long run cleanly.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends make libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["make", "help"]

# --- CPU -------------------------------------------------------------------------
FROM base AS cpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url ${TORCH_INDEX}
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md LICENSE Makefile ./
COPY staticsbfl ./staticsbfl
COPY tests ./tests
COPY baselines ./baselines
RUN python -m pip install --no-cache-dir --no-deps -e .
ENV DEVICE=cpu

# --- NVIDIA GPU --------------------------------------------------------------------
FROM base AS cuda
# The CUDA runtime comes with the PyTorch wheel; override the index for another
# CUDA version, e.g. --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN python -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url ${TORCH_INDEX}
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md LICENSE Makefile ./
COPY staticsbfl ./staticsbfl
COPY tests ./tests
COPY baselines ./baselines
RUN python -m pip install --no-cache-dir --no-deps -e .
ENV DEVICE=cuda

# Building without --target gives the CPU image.
FROM cpu AS final
