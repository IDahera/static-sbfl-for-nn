import torch

# Use MPS if available, otherwise use CUDA on server
DEVICE_TYPE = "mps" if torch.backends.mps.is_available() else "cuda"
# PORT: ICSOFT_DEVICE=cpu|mps|cuda overrides the choice above. Upstream knows only
# MPS-else-CUDA, so a CPU run (the CPU-vs-CPU comparison with the TensorFlow
# ports) was impossible on a Mac. Unset, the behaviour is upstream's.
import os  # PORT
DEVICE_TYPE = os.environ.get("ICSOFT_DEVICE", DEVICE_TYPE)  # PORT
DEVICE = torch.device(DEVICE_TYPE)

BATCH_SIZE = 64

# Use 4 workers if MPS is available, otherwise use 10 on server
WORKERS = 4 if torch.backends.mps.is_available() else 16

# Path to the configuration file
TRAIN_YAML_PATH = "config/train-config.yaml"
SUS_YAML_PATH = "config/suspiciousness-config.yaml"
EXPERIMENT_YAML_PATH = "config/experiment-config.yaml"
