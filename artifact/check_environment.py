"""Print and validate the execution environment used by the artefact."""

import argparse
import os
import platform
import sys

import lightgbm
import numpy
import pandas
import sklearn
import torch
import torchvision
import xgboost

from sbfl_nn.utils.gpu import DEVICE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expect-device",
        choices=("cpu", "cuda", "mps"),
        help="Fail unless the selected PyTorch device has this type.",
    )
    args = parser.parse_args()

    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"torch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"SBFL_DEVICE: {os.environ.get('SBFL_DEVICE', 'auto')}")
    print(f"selected device: {DEVICE}")
    print(f"numpy: {numpy.__version__}")
    print(f"pandas: {pandas.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"xgboost: {xgboost.__version__}")
    print(f"lightgbm: {lightgbm.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            memory_gib = properties.total_memory / 1024**3
            print(f"CUDA device {index}: {properties.name} ({memory_gib:.1f} GiB)")

    if args.expect_device and DEVICE.type != args.expect_device:
        raise SystemExit(
            f"Expected device {args.expect_device!r}, selected {DEVICE.type!r}"
        )


if __name__ == "__main__":
    main()
