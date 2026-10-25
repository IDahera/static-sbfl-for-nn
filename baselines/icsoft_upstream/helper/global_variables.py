import torch

# Use MPS if available, otherwise use CUDA on server
DEVICE_TYPE = "mps" if torch.backends.mps.is_available() else "cuda"
DEVICE = torch.device(DEVICE_TYPE)

BATCH_SIZE = 64

# Use 4 workers if MPS is available, otherwise use 10 on server
WORKERS = 4 if torch.backends.mps.is_available() else 16

# Path to the configuration file
TRAIN_YAML_PATH = "config/train-config.yaml"
SUS_YAML_PATH = "config/suspiciousness-config.yaml"
EXPERIMENT_YAML_PATH = "config/experiment-config.yaml"
