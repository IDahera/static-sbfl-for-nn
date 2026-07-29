# syntax=docker/dockerfile:1

# This official Python image provides Linux/amd64 and Linux/arm64 variants.
ARG PYTHON_IMAGE=python:3.13.2-slim-bookworm
FROM ${PYTHON_IMAGE} AS base

# Avoid interactive package prompts, make logs appear immediately, and disable
# GUI plotting inside the headless container.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/workspace

WORKDIR /workspace

# libgomp1 is required by the pre-built LightGBM/XGBoost wheels. tini forwards
# signals correctly when a long experiment is stopped with Ctrl-C.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-artifact.txt .
RUN python -m pip install --no-cache-dir -r requirements-artifact.txt

# Install the project without resolving setup.py dependencies: every runtime
# dependency has already been installed explicitly and pinned above.
COPY setup.py README.md ./
COPY sbfl_nn ./sbfl_nn
COPY scripts ./scripts
COPY configs ./configs
COPY artifact ./artifact
RUN python -m pip install --no-cache-dir --no-deps .

# The data and outputs directories are mount points in compose.yaml.
RUN mkdir -p /workspace/data /workspace/outputs

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

# Portable CPU image. Docker automatically selects amd64 or arm64 to match the
# host, provided PyTorch publishes a wheel for that architecture.
FROM base AS cpu
RUN python -m pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cpu
ENV SBFL_DEVICE=cpu

# Verify the completed CPU environment after PyTorch has been installed.
RUN python artifact/check_environment.py --expect-device cpu

# CUDA 12.4 image used for the paper-compatible GPU workflow. The CUDA runtime
# libraries are supplied by the PyTorch wheels; the host supplies its NVIDIA
# driver and exposes the GPU using the NVIDIA Container Toolkit.
FROM base AS cuda
RUN python -m pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
ENV SBFL_DEVICE=cuda

# Do not require a GPU while building: GPUs are normally attached only at run
# time. The run script performs the strict CUDA check.
RUN SBFL_DEVICE=cpu python artifact/check_environment.py --expect-device cpu

# Building without --target produces the portable CPU image.
FROM cpu AS final
