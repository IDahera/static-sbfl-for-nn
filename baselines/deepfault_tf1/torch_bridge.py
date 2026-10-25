# SPDX-License-Identifier: GPL-3.0-or-later
# Part of a modified version of DeepFault (c) 2019 Bogazici University; written 2026-10-20 by
# Daniel Rashedi for the diss-code runtime comparison. Distributed under the GPL-3.0-or-later.
"""Bring a StaticSBFL (PyTorch) dense model into DF@tf1's own model format.

New file, not upstream. Not timed: it plays the role of "a trained Keras model
exists", which upstream assumes as input.

    python torch_bridge.py --weights-dir cache/runtime/<dataset>/<mode>/weights \
                          --out-dir cache/runtime/<dataset>/<mode>/keras_tf1 \
                          --test-set cache/runtime/<dataset>/test_set.npz

For every ``<key>.npz`` (written by ``staticsbfl/cli/runtime_benchmark.py``) the model
is rebuilt the way upstream builds its dense MNIST models
(``train_mnist_nn.py``): ``Flatten`` -> per hidden layer ``Dense`` (linear) +
a separate ``Activation`` layer -> ``Dense(softmax)``. The PyTorch weight
``(out, in)`` becomes the Keras kernel ``(in, out)``; the softmax does not
change the argmax, hence not the pass/fail oracle. Because the activation is a
separate layer, DeepFault counts the Dense output, i.e. the pre-activation --
the same quantity StaticSBFL hooks on ``fc*`` (``> 0`` either way). Inputs keep
the PyTorch layout ``(C, H, W)``; ``Flatten`` then orders them exactly as
``torch.flatten`` does.

Saved like upstream ``__save_trained_model``: ``model.to_json()`` to
``<key>.json`` and ``model.save_weights`` to ``<key>.h5``, so the timed
run loads it with upstream ``utils.load_model``. Each export is checked: the
reloaded model must reproduce the PyTorch predictions on the test set.
"""

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _tf1_runtime  # graph mode, as in every module of this port
from tf_keras import backend as K
from tf_keras.layers import Activation, Dense, Flatten, LeakyReLU
from tf_keras.models import Sequential
import numpy as np
import utils


def activation_layer(name):
    if name == "leaky_relu":
        return LeakyReLU(alpha=0.01)  # torch.nn.LeakyReLU default slope; upstream also uses 0.01
    return Activation(name)  # relu, elu, tanh, sigmoid: identical definitions in both frameworks


def build(meta, arrays):
    model = Sequential()
    model.add(Flatten(input_shape=tuple(meta["input_shape"])))  # as upstream train_mnist_nn.py
    for index, width in enumerate(meta["hidden_widths"]):
        model.add(Dense(int(width)))
        model.add(activation_layer(meta["activation"]))
    model.add(Dense(int(meta["num_classes"]), activation="softmax"))
    dense = [layer for layer in model.layers if isinstance(layer, Dense)]
    names = [f"fc{i}" for i in range(len(meta["hidden_widths"]))] + ["head"]
    for layer, name in zip(dense, names):
        layer.set_weights([arrays[f"{name}.weight"].T, arrays[f"{name}.bias"]])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test = np.load(args.test_set)
    x = test["x"].astype("float32")
    failures = 0
    for path in sorted(glob.glob(os.path.join(args.weights_dir, "*.npz"))):
        key = os.path.basename(path)[: -len(".npz")]
        stem = os.path.join(args.out_dir, key)
        report_path = stem + ".export.json"
        if os.path.exists(report_path) and not args.force:
            print(f"skip {key} (exported)")
            continue
        with open(os.path.join(args.weights_dir, key + ".json"), encoding="utf-8") as handle:
            meta = json.load(handle)
        arrays = np.load(path)
        model = build(meta, arrays)
        with open(stem + ".json", "w") as json_file:
            json_file.write(model.to_json())
        model.save_weights(stem + ".h5")
        K.clear_session()  # graph mode: start the reload in a fresh graph
        reloaded = utils.load_model(stem)
        predicted = np.argmax(reloaded.predict(x, batch_size=1000, verbose=0), axis=1)
        reference = arrays["reference_argmax"]
        agreement = float((predicted == reference).mean())
        report = {"key": key, "prediction_agreement_with_pytorch": agreement,
                  "n_samples": int(len(reference))}
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        flag = "" if agreement >= 0.999 else "   <-- CHECK"
        failures += agreement < 0.999
        print(f"exported {key}: argmax agreement with PyTorch {agreement:.5f}{flag}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
