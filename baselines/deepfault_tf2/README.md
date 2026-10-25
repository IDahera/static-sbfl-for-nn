# DF@tf2 — DeepFault on TensorFlow 2 (eager) and Keras 3, on Python 3.14

Port of DeepFault (upstream `7a068e7`, Keras 2.3.1 / TensorFlow 1.13) to
Python 3.14.2 + TensorFlow 2.22.0rc0 with **eager execution** and **Keras 3**
(`keras-nightly==3.16.0.dev2026092204`, the build TF 2.22.0rc0 requires), in the conda environment `deepfault-tf2`. This
is DeepFault as it would be written for current TensorFlow: no graph, no
session; every layer functor is a Keras model called eagerly.

```bash
make runtime-setup                                   # creates the conda env deepfault-tf2
conda run -n deepfault-tf2 python run.py --model mnist_test_model_2_30 --approach ochiai   # upstream CLI
```

## Files

| file | origin |
|---|---|
| `test_nn.py` | upstream, **unchanged** |
| `utils.py`, `run.py`, `spectrum_analysis.py`, `input_synthesis.py`, `train_mnist_nn.py`, `train_cifar_nn.py` | upstream, minimal port (table below) |
| `torch_bridge.py` | **new**: converts a StaticSBFL model into this port's model format |
| `runtime_run.py` | **new**: runtime entry point → `../common/deepfault_runtime.py` |
| `requirements.txt` | **new**: the pinned environment; versions in [`../README.md`](../README.md#versions) |
| `LICENSE` | upstream (GPL-3.0) |

`spectrum_analysis.py` still imports `keras.backend as K`; the module exists in
Keras 3 and the name is unused there, so the line is unchanged.

## Every change to upstream code, and whether it can affect the measurement

"Measured path" = what the runtime comparison times: `load_model`, `test_model` (→ `get_layer_outs`,
`evaluate`, `predict`, `calculate_prediction_metrics`), `get_trainable_layers`,
`construct_spectrum_matrices`, `tarantula/ochiai/dstar_analysis`.

| # | where | change | why | effect on the measured time / memory |
|---|---|---|---|---|
| 1 | `utils.get_layer_outs` | `K.function([inp], [out])` → `keras.Model(inputs=inp, outputs=out)`, called on the whole input, result converted with `ops.convert_to_numpy` and wrapped in a list | Keras 3 has no `K.function`. `K.function` returned a list of NumPy arrays; the wrapper returns exactly that, so `layer_outs[i][0][sample]` downstream is unchanged | **Yes, by design — this is the TF1/TF2 difference being measured.** Still one functor per layer, each recomputing the forward pass from the input on the whole test set at once, as upstream. Eager dispatch replaces session execution. Affects `test_model` only; the spectrum loop dominates. |
| 2 | `utils.get_layer_outs_old` | same as 1, `K.learning_phase()` → `training=True` | Keras 3 has no learning phase. Unused upstream | None (never called). |
| 3 | `utils.py`, `input_synthesis.py` | `model.input` → `model.inputs` | a Keras 3 `Sequential` has no `.input` until it has been called once; `.inputs` is the same input tensor(s) | None. |
| 4 | `utils.py`, `run.py` (all occurrences) | `layer.output_shape[k]` → `layer.output.shape[k]` | `Layer.output_shape` was removed in Keras 3 | None (an attribute lookup per layer/neuron loop header; the loop body is unchanged). |
| 5 | `utils.load_model` | weights file `<name>.h5` → `<name>.weights.h5` | Keras 3 requires the suffix | None: the same HDF5 weights, a different file name. |
| 6 | `utils.py` | `np_utils` → `keras.utils` (`to_categorical` unchanged) | `np_utils` was removed in Keras 2.4 | None (data preparation). |
| 7 | `utils.normalize` | `K.sqrt/K.mean/K.square` → `keras.ops.*` | moved in Keras 3. Used only by input synthesis | None on the measured path. |
| 8 | `utils.py` (6×) | h5py `.value` → `[()]`; also catch `TypeError` | as in DF@tf1 #4 | None: cache readers, unused by the runtime runner. |
| 9 | `spectrum_analysis.ochiai()` | added the missing `return` | **upstream bug** (see DF@tf1 #5) | Makes Ochiai computable at all. |
| 10 | `input_synthesis.synthesize` | `K.gradients` + `K.function` → `tf.GradientTape` on a sub-model up to the suspicious layer | no symbolic gradients in eager mode. Same loss (mean activation of the neuron), same gradient w.r.t. the input | Not on the measured path. (Faster than DF@tf1, whose graph grows with every `K.gradients` call.) |
| 11 | `run.py` | `model.evaluate([x], [y])` → `model.evaluate(np.array(x), np.array(y))` | Keras 3 reads a list as several model inputs | Not on the measured path (step 5). |
| 12 | `train_mnist_nn.py` | `Flatten(input_shape=…)` → `Input(shape=…)` + `Flatten()`; weights to `.weights.h5` | Keras 3 serialises the input shape only for an explicit `Input`; suffix as #5 | None (training script). |
| 13 | `train_cifar_nn.py` | `rmsprop(lr, decay)` → `RMSprop(InverseTimeDecay(lr, 1, decay))`; `ImageDataGenerator` import removed (augmentation is off upstream); `fit_generator` → `fit`, `workers` dropped; also `save_weights(<name>.weights.h5)` so `load_model` finds it | removed in Keras 3; `InverseTimeDecay` is the same `lr / (1 + decay · t)` | None (training script, not used by the runtime runner). |

Not changed, although it is slow: `construct_spectrum_matrices` (pure Python,
`in` on a list, cost ∝ samples² × neurons) — see the DF@tf1 README. It is
identical in both ports, so the DF@tf1 / DF@tf2 difference isolates the
framework part (`test_model`).

## The model conversion and the runtime runner

As in DF@tf1 (see [`../deepfault_tf1/README.md`](../deepfault_tf1/README.md)),
with two differences forced by Keras 3: the model starts with an explicit
`Input(shape=(3, 32, 32))` before `Flatten()` (otherwise the JSON carries no
input and `model.inputs` is undefined after loading), and the weights are
written as `<key>.weights.h5`. The checks are the same: argmax agreement with
PyTorch on the whole test set, and the spectrum compared with the `make score`
cache.
