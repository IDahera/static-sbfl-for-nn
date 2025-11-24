# Migration Guide

## From Old Structure to New Structure

The project has been reorganized for better maintainability and usability.

### Directory Changes

| Old Path                            | New Path                                             |
| ----------------------------------- | ---------------------------------------------------- |
| `helpers_general/`                  | `sbfl_nn/utils/`                                     |
| `helpers_models/`                   | `sbfl_nn/models/`                                    |
| `helpers_prediction/`               | `sbfl_nn/prediction/`                                |
| `helpers_preprocess/`               | `sbfl_nn/data/`                                      |
| `helpers_susgen/`                   | `sbfl_nn/sbfl/`                                      |
| `pre01_model_gen.py`                | `scripts/01_preprocessing/generate_models.py`        |
| `pre02_eval_models.py`              | `scripts/01_preprocessing/evaluate_models.py`        |
| `pre03_sus_computation.py`          | `scripts/01_preprocessing/compute_suspiciousness.py` |
| `pre04_transform_sus_values.py`     | `scripts/01_preprocessing/transform_datasets.py`     |
| `mainA_classification.py`           | `scripts/02_classification/train_classifiers.py`     |
| `mainB_classification_interpret.py` | `scripts/02_classification/interpret_features.py`    |
| `mainC_end_2_end.py`                | `scripts/03_validation/end_to_end_eval.py`           |
| `main_regression.py`                | `scripts/04_regression/regression_analysis.py`       |
| `output_models/`                    | `outputs/models/`                                    |
| `output_sus_values/`                | `outputs/suspiciousness/`                            |
| `output_sus_datasets/`              | `outputs/datasets/`                                  |
| `output_results/`                   | `outputs/results/`                                   |

### Module/File Renames

| Old Module                                 | New Module                       |
| ------------------------------------------ | -------------------------------- |
| `helpers_general.file_handling`            | `sbfl_nn.utils.files`            |
| `helpers_general.logging`                  | `sbfl_nn.utils.logger`           |
| `helpers_models.models`                    | `sbfl_nn.models.architectures`   |
| `helpers_models.logging`                   | `sbfl_nn.models.training_logger` |
| `helpers_prediction.models_classification` | `sbfl_nn.prediction.classifiers` |
| `helpers_prediction.models_regression`     | `sbfl_nn.prediction.regressors`  |
| `helpers_susgen.sbfl`                      | `sbfl_nn.sbfl.metrics`           |
| `helpers_susgen.logging`                   | `sbfl_nn.sbfl.sbfl_logger`       |

### Import Changes

**Old:**
```python
from helpers_general.file_handling import get_pkl_files
from helpers_models.models import DenseModel
from helpers_prediction.models_classification import MODELS
```

**New:**
```python
from sbfl_nn.utils.files import get_pkl_files
from sbfl_nn.models.architectures import DenseModel
from sbfl_nn.prediction.classifiers import MODELS
```

### Running Scripts

**Old:**
```bash
python pre01_model_gen.py --num-models 100
python mainA_classification.py
```

**New:**
```bash
python scripts/01_preprocessing/generate_models.py --config configs/model_gen/config1.yaml
python scripts/02_classification/train_classifiers.py
```

### Output Paths

All scripts now default to the `outputs/` directory structure:
- Models: `outputs/models/`
- Suspiciousness: `outputs/suspiciousness/`
- Datasets: `outputs/datasets/`
- Results: `outputs/results/`

### Backward Compatibility

The old `helpers_*` directories and output folders remain in place temporarily. You can:
1. Continue using old scripts (they still reference old paths)
2. Gradually migrate to new structure
3. Remove old directories once fully migrated

### Key Benefits

1. **Clearer Organization**: Pipeline stages are numbered and grouped
2. **Better Naming**: More intuitive module and file names
3. **Proper Package**: `sbfl_nn` is now a proper Python package
4. **Centralized Outputs**: All outputs in one place
5. **No Namespace Conflicts**: No more multiple `logging.py` files

### Next Steps

1. Test scripts with new structure
2. Update any custom scripts/notebooks you have
3. Once verified, remove old `helpers_*` directories
4. Update any documentation/notes you maintain
