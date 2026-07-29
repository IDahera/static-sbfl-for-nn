# Model generation

This document previously described the removed `01_model_gen.py` interface.
Current model generation is configuration-driven:

```bash
python scripts/01_preprocessing/generate_models.py \
  --config configs/artifact/smoke.yaml \
  --dataset mnist
```

Use `--config` more than once to select multiple YAML files. Use `--dataset`
more than once to select both `mnist` and `fmnist`. If `--config` is omitted,
all YAML files in `--config-dir` are processed.

The `num_models` YAML field is the desired total in the corresponding output
directory. An interrupted run can therefore be resumed without creating a
second complete batch.

See [`README.md`](README.md) for the paper configurations and Docker workflow.
