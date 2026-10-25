# DF@tf1 — DeepFault with the TensorFlow 1 execution model, on Python 3.14

Port of DeepFault (upstream `7a068e7`, Keras 2.3.1 / TensorFlow 1.13) to
Python 3.14.2 + TensorFlow 2.22.0rc0, **with eager execution disabled** and the
legacy Keras 2 package `tf_keras` 2.21, in the conda environment `deepfault-tf1`. Every Keras call builds a static graph
and `K.function` runs it in a `tf.compat.v1.Session`, as under TensorFlow 1.
Why not TF 1.x itself: see [`../README.md`](../README.md).

```bash
make runtime-setup                                   # creates the conda env deepfault-tf1
conda run -n deepfault-tf1 python run.py --model mnist_test_model_2_30 --approach ochiai   # upstream CLI
```

## Files

| file | origin |
|---|---|
| `run.py`, `test_nn.py` | upstream, **unchanged** |
| `utils.py`, `spectrum_analysis.py`, `input_synthesis.py`, `train_mnist_nn.py`, `train_cifar_nn.py` | upstream, minimal port (table below) |
| `_tf1_runtime.py` | **new**: disables eager execution before any layer exists |
| `torch_bridge.py` | **new**: converts a StaticSBFL model into this port's model format |
| `runtime_run.py` | **new**: runtime entry point → `../common/deepfault_runtime.py` |
| `requirements.txt`, `requirements-nodeps.txt` | **new**: the pinned environment (`tf-keras` without its deps); versions in [`../README.md`](../README.md#versions) |
| `LICENSE` | upstream (GPL-3.0) |

## Every change to upstream code, and whether it can affect the measurement

"Measured path" = what the runtime comparison times: `load_model`, `test_model` (→ `get_layer_outs`,
`evaluate`, `predict`, `calculate_prediction_metrics`), `get_trainable_layers`,
`construct_spectrum_matrices`, `tarantula/ochiai/dstar_analysis`.

| # | where | change | why | effect on the measured time / memory |
|---|---|---|---|---|
| 1 | all modules (via `_tf1_runtime.py`) | `tf.compat.v1.disable_eager_execution()` before any Keras import | restores TF1 graph mode on TF 2 | **Intended**: keeps the TF1 mechanism (graph + `Session.run`). The kernels are TF 2.22's (arm64, current Eigen/oneDNN), not TF 1.13's, so the *forward-pass* time in `test_model` is not that of 2019 binaries. The spectrum loop is pure Python/NumPy and unaffected. |
| 2 | `utils.py`, `spectrum_analysis.py`, `input_synthesis.py`, `train_*.py` | `keras.*` → `tf_keras.*` | Keras 2.3.1 does not install on Python 3.14; `tf_keras` is the maintained Keras 2 API | **Small**: same API generation; `K.function` is still a graph function executed by a session (`GraphExecutionFunction`). Session-call overhead may differ slightly from Keras 2.3.1. |
| 3 | `utils.py` | `from keras.utils import np_utils` → `from tf_keras import utils as np_utils` | `np_utils` was removed in Keras 2.4; `to_categorical` is unchanged | None (data preparation, before the measured analysis). |
| 4 | `utils.py` (6×) | h5py `Dataset.value` → `Dataset[()]`; `except AttributeError` → `except (AttributeError, TypeError)` | `.value` was removed in h5py 3. Upstream ends its read loops on the `AttributeError` of `None.value`; `None[()]` raises `TypeError`, so both are caught | None: only the h5 cache readers, which the runtime runner does not use (see below). |
| 5 | `spectrum_analysis.py`, `ochiai()` | added the missing `return` | **upstream bug**: the inner function returns `None`, and `np.isnan(None)` raises `TypeError`, so Ochiai could never run | Makes Ochiai computable at all. Its cost per neuron is the same kind as Tarantula's (a few float operations). |
| 6 | `train_cifar_nn.py` | `keras.optimizers.rmsprop(lr, decay)` → `keras.optimizers.legacy.RMSprop(lr, decay)` | the lowercase alias no longer exists; the legacy class has the same update rule incl. `decay` | None (training script, not used by the runtime runner). |

Not changed, although it is slow: `construct_spectrum_matrices` visits every
(sample, neuron) pair in Python and tests `test_idx in correct_classifications`
against a **list**, so its cost grows with *samples² × neurons*. That is how
DeepFault traverses a model, so it is measured as is. (Measured on the M4 Pro:
≈ 2.2 µs + 3.3 ns × N per sample-neuron pair; 384 neurons × 10 000 samples ≈
2.4 min.) The interpreter is Python 3.14 instead of ≤ 3.7, whose faster bytecode
loop can only make this part *faster* than in 2019 — a bias in DeepFault's
favour, worth one sentence in the threats to validity.

## The model conversion (`torch_bridge.py`)

The phase-4 models are PyTorch `VariableMLP`s. The bridge rebuilds each one the
way upstream builds its own dense MNIST models (`train_mnist_nn.py`):
`Flatten(input_shape=(3, 32, 32))` → per hidden layer `Dense(width)` + a
separate `Activation('relu')` → `Dense(10, activation='softmax')`, sets the
weights (PyTorch `(out, in)` → Keras kernel `(in, out)`), and saves them as
upstream does (`to_json()` + `save_weights(<name>.h5)`), so the timed run loads
them with upstream `utils.load_model`.

* With a separate `Activation` layer, DeepFault counts the `Dense` output, the
  pre-activation — the quantity StaticSBFL hooks on `fc*`. For ReLU, `> 0` is the
  same decision before and after the activation.
* The softmax does not change the argmax, hence not the pass/fail oracle.
* Inputs keep the PyTorch layout `(C, H, W)` and the PyTorch normalisation;
  `Flatten` then orders the 3 072 inputs exactly as `torch.flatten`.
* Each conversion is checked: the reloaded Keras model must reproduce the
  PyTorch argmax on the whole test set (`<key>.export.json`).

Not timed: it plays the part of "a trained Keras model exists", which upstream
takes as input.

## The runtime runner (`runtime_run.py` → `../common/deepfault_runtime.py`)

Steps 0–3 of upstream `run.py`, with upstream functions and in upstream order,
one fresh process per model, each step timed and its memory recorded. Where it
departs from `run.py`, and the direction of the effect:

| departure from `run.py` | why | effect |
|---|---|---|
| test set read from one `.npz` (StaticSBFL pre-processing) instead of `keras.datasets` | the models were trained on normalised `(C, H, W)` inputs; every runtime tool reads the same file | equal for all tools |
| `--selected-class all` (default): the whole test set, no `filter_val_set` | the static predictor and the cached spectra describe the whole test set; `--selected-class 3` runs upstream's per-class protocol | same work per sample; the per-class mode is ~10× fewer samples |
| training set not loaded, no `train_test_split` | upstream loads and splits it, then overwrites `X_val` with the test set and never uses it | favours DeepFault (less time and memory) |
| no h5 caches (`save/load_classifications`, `save/load_layer_outs`, `save/load_suspicious_neurons`) | disk I/O, and a warm cache would skip the measured work | favours DeepFault |
| spectrum matrices built once, all three metrics scored from them | upstream computes one metric per invocation and rebuilds the matrices every time | favours DeepFault (one build instead of three) |
| steps 4–5 (input synthesis, adversarial test) not run | not fault localization | — |
| `suspicious_num = 10` | upstream default 1; only the size of the returned top list | negligible |
