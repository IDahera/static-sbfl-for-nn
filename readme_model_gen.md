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

**Usage:**

```bash
# Generate 100 models with default settings
python 01_model_gen.py --num-models 100

# Custom configuration
python 01_model_gen.py \
  --num-models 500 \
  --epochs 2 \
  --min-layers 5 \
  --max-layers 25 \
  --min-neurons 10 \
  --max-neurons 50 \
  --learning-rate 0.001 \
  --output-dir models/
```

**Output:**
```
models/
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
```

**Configuration Tracking:**
The script saves configuration parameters (min/max layers, min/max neurons, learning rate, epochs) in `config.json`. If you run the script again with different parameters, it will:
1. Detect the configuration change
2. Show which parameters changed
3. Ask if you want to delete existing models (default: No)
4. Continue training from where it left off or start fresh

---

## Requirements

- Python 3.x
- PyTorch
- torchvision
- click
- tqdm
