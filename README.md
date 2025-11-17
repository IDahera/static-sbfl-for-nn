# Static SBFL for Neural Networks

This project implements static spectrum-based fault localization (SBFL) techniques for neural networks.

## Scripts

### 01_model_gen.py

Generates and trains multiple neural network models with random architectures on MNIST and Fashion MNIST datasets. 

**Features:**
- Automatically trains on both MNIST and Fashion MNIST
- Generates models with random layer configurations
- Stores configuration and detects changes
- Prompts user before deleting models when configuration changes (default: No)
- Supports multiple configuration profiles via `--config-name`

**Usage:**

```bash
# Generate 100 models with default settings (config1)
python 01_model_gen.py --num-models 100

# Use a different configuration profile
python 01_model_gen.py --num-models 100 --config-name config2

# Custom configuration
python 01_model_gen.py \
  --num-models 500 \
  --epochs 2 \
  --min-layers 5 \
  --max-layers 25 \
  --min-neurons 10 \
  --max-neurons 50 \
  --learning-rate 0.001 \
  --config-name my_experiment \
  --output-dir models/
```

**Output:**
```
models__config1/
├── mnist/
│   ├── config.json          # Training configuration
│   ├── model_0.pth          # Trained models
│   ├── model_1.pth
│   └── ...
└── fmnist/
    ├── config.json
    ├── model_0.pth
    ├── model_1.pth
    └── ...

models__config2/           # Different configuration
├── mnist/
│   └── ...
└── fmnist/
    └── ...
```

**Configuration Tracking:**
The script saves configuration parameters (min/max layers, min/max neurons, learning rate, epochs) in `config.json`. If you run the script again with different parameters, it will:
1. Detect the configuration change
2. Show which parameters changed
3. Ask if you want to delete existing models (default: No)
4. Continue training from where it left off or start fresh

Each configuration name creates a separate directory (e.g., `models__config1`, `models__config2`), allowing you to run multiple experiments with different hyperparameters without conflicts.

---

## Requirements

- Python 3.x
- PyTorch
- torchvision
- click
- tqdm
