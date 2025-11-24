# Quick Start Guide

## Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install sbfl_nn package in editable mode (required!)
pip install -e .
```

## Complete Pipeline

### Step 1: Generate Models (takes longest)
```bash
# Generate 50 models (quickest for testing)
python scripts/01_preprocessing/generate_models.py \
  --config configs/model_gen/config_50.yaml

# Or generate more models
python scripts/01_preprocessing/generate_models.py \
  --config configs/model_gen/config_250.yaml
```

### Step 2: Evaluate Models
```bash
python scripts/01_preprocessing/evaluate_models.py
```

### Step 3: Compute Suspiciousness
```bash
python scripts/01_preprocessing/compute_suspiciousness.py
```

### Step 4: Transform to ML Datasets
```bash
python scripts/01_preprocessing/transform_datasets.py
```

### Step 5: Train Classifiers
```bash
python scripts/02_classification/train_classifiers.py
```

### Step 6: Interpret Features
```bash
python scripts/02_classification/interpret_features.py
```

### Step 7: End-to-End Validation
```bash
python scripts/03_validation/end_to_end_eval.py \
  --model-path outputs/models/config_50/mnist/model_0.pth \
  --predictor-model outputs/results/config_50/mnist/classifier_model_random_forest.pkl \
  --metric tarantula \
  --visualize
```

## Quick Commands

### Generate only 10 models for quick testing
```bash
python scripts/01_preprocessing/generate_models.py \
  --config configs/model_gen/config_50.yaml \
  --max-models 10
```

### Run classification with specific train/val split
```bash
python scripts/02_classification/train_classifiers.py \
  --train-split 0.7 \
  --random-state 42
```

### Check outputs
```bash
# List generated models
ls -lh outputs/models/*/mnist/

# Check classification results
cat outputs/results/*/mnist/classification_results.csv

# View feature rankings
cat outputs/results/*/mnist/feature_rankings.csv
```

## Configuration Files

Configs are in `configs/model_gen/`:
- `config_50.yaml` - 50 models (fast, for testing)
- `config_250.yaml` - 250 models
- `config_500.yaml` - 500 models
- `config_all.yaml` - 1000 models (slow, for production)

## Output Locations

- **Models**: `outputs/models/<config>/<dataset>/model_*.pth`
- **Suspiciousness**: `outputs/suspiciousness/<config>/<dataset>/*_sus.pkl`
- **Datasets**: `outputs/datasets/<config>/<dataset>/*_dataset_numpy.pkl`
- **Results**: `outputs/results/<config>/<dataset>/classification_results.csv`
- **Classifiers**: `outputs/results/<config>/<dataset>/classifier_model_*.pkl`
- **Rankings**: `outputs/results/<config>/<dataset>/feature_rankings.csv`

## Troubleshooting

### NaN values in features
Already handled - script replaces NaN with 0.0 automatically

### CUDA out of memory
Reduce batch size or number of parallel GPU processes in generate_models.py

### Import errors
Make sure you're in the project root directory when running scripts

### Old paths not working
See `docs/MIGRATION.md` for path translations

## Performance Tips

1. **Multi-GPU**: Script automatically uses all available GPUs
2. **Parallel configs**: Run different configs simultaneously on different GPUs
3. **Incremental**: You can process results as models are generated (no need to wait for all)

## Current Active Models

Only 3 high-performing classifiers are trained:
- Decision Tree
- Random Forest
- Gradient Boosting

(Removed: XGBoost, LightGBM, Extra Trees, Logistic Regression, Naive Bayes, KNN due to poor performance)
