# Static SBFL for Neural Networks

## Overview
This project implements static Spectrum-Based Fault Localization (SBFL) techniques for neural networks. It generates models with random architectures, computes suspiciousness values, and trains classifiers to predict neuron suspiciousness.

## Project Structure

```
static-sbfl-for-nn/
├── sbfl_nn/                    # Core package
│   ├── data/                   # Dataset generation utilities
│   ├── models/                 # Neural network architectures
│   ├── prediction/             # Classifiers and regressors
│   ├── sbfl/                   # SBFL metrics (Ochiai, Tarantula)
│   └── utils/                  # General utilities (files, GPU, logging)
│
├── scripts/                    # Executable scripts
│   ├── 01_preprocessing/       # Model generation and suspiciousness computation
│   ├── 02_classification/      # Train classifiers on suspiciousness prediction
│   ├── 03_validation/          # End-to-end validation
│   └── 04_regression/          # Regression analysis
│
├── outputs/                    # Generated outputs
│   ├── models/                 # Trained neural network models
│   ├── suspiciousness/         # Computed suspiciousness values
│   ├── datasets/               # Transformed datasets for ML
│   └── results/                # Classification/regression results
│
├── configs/                    # Configuration files
│   └── model_gen/              # Model generation configs
│
├── data/                       # Raw datasets (MNIST, Fashion-MNIST)
├── docs/                       # Documentation
└── tests/                      # Unit tests

```

## Installation

1. Clone the repository
2. Install dependencies and package:
```bash
pip install -r requirements.txt
pip install -e .  # Install sbfl_nn package in editable mode
```

## Usage

### 1. Preprocessing Pipeline

Generate models with random architectures:
```bash
python scripts/01_preprocessing/generate_models.py --config configs/model_gen/config1.yaml
```

Evaluate generated models:
```bash
python scripts/01_preprocessing/evaluate_models.py
```

Compute suspiciousness values:
```bash
python scripts/01_preprocessing/compute_suspiciousness.py
```

Transform to ML datasets:
```bash
python scripts/01_preprocessing/transform_datasets.py
```

### 2. Classification

Train classifiers to predict neuron suspiciousness:
```bash
python scripts/02_classification/train_classifiers.py
```

Interpret feature importance:
```bash
python scripts/02_classification/interpret_features.py
```

### 3. Validation

Run end-to-end evaluation:
```bash
python scripts/03_validation/end_to_end_eval.py
```

## Configuration

Model generation parameters are defined in YAML files under `configs/model_gen/`:
- `num_models`: Number of models to generate
- `min_layers`/`max_layers`: Layer count range
- `min_neurons`/`max_neurons`: Neuron count range
- `learning_rate`: Training learning rate
- `epochs`: Training epochs

## Output Structure

All outputs are organized under `outputs/`:
- `outputs/models/config_name/dataset_name/`: Trained models (.pth files)
- `outputs/suspiciousness/config_name/dataset_name/`: Suspiciousness values (.pkl files)
- `outputs/datasets/config_name/dataset_name/`: ML datasets (.pkl files)
- `outputs/results/config_name/dataset_name/`: Classification results (.csv, .pkl files)

## Key Features

- **Multi-GPU Support**: Parallel model generation across multiple GPUs
- **SBFL Metrics**: Ochiai and Tarantula suspiciousness computation
- **ML Classifiers**: Decision Tree, Random Forest, Gradient Boosting
- **Feature Rankings**: Interpretable feature importance analysis
- **Configurable Pipeline**: YAML-based configuration system