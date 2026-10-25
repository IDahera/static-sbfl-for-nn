# SPDX-License-Identifier: GPL-3.0-or-later
# Part of a modified version of DeepFault (c) 2019 Bogazici University; written 2026-10-20 by
# Daniel Rashedi for the diss-code runtime comparison. Distributed under the GPL-3.0-or-later.
"""Runtime runner for one DeepFault port: the hit-spectrum part of ``run.py``, timed.

Started by ``<port>/runtime_run.py``, whose directory is first on ``sys.path``, so
``utils``, ``test_nn`` and ``spectrum_analysis`` below are the *port's* modules.
One process analyses one model, so its memory is not shared with another run.

What runs is steps 0-3 of upstream ``run.py`` with upstream functions, in the
upstream order:

    data         the test set (pre-processed exactly as for the PyTorch models),
                 optionally narrowed with upstream ``filter_val_set``
    model_load   upstream ``utils.load_model``: JSON + weights, then compile
    test_model   upstream ``test_nn.test_model``: one functor per layer over the
                 whole input, evaluate, predict, classification report
    spectrum     upstream ``get_trainable_layers`` + ``construct_spectrum_matrices``
    score_*      upstream ``tarantula_analysis`` / ``ochiai_analysis`` /
                 ``dstar_analysis`` on the same matrices

Deliberately not run: the training-set load and split (upstream loads and
splits the training set, then never uses it -- X_val is overwritten by
``filter_val_set`` on the test set), the h5 caches of classifications / layer
outputs / suspicious neurons (disk I/O, and a warm cache would skip the
measured work), and steps 4-5 (input synthesis), which are not fault
localization. Upstream computes one metric per invocation and rebuilds the
spectrum matrices each time; here the matrices are built once and all three
metrics are scored from them, which can only favour DeepFault.
"""

import argparse
import json
import os
import sys
import time

_STARTED = time.perf_counter()
# The memory probe lives in staticsbfl/benchmark/footprint.py (standard library
# only). This environment has no torch, so the file is loaded directly rather
# than through the staticsbfl package, whose benchmark package imports torch.
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "staticsbfl", "benchmark"))

from footprint import MEASUREMENT_NOTE, Recorder  # noqa: E402

METRICS = ("tarantula", "ochiai", "dstar")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True,
                        help="Model stem: <stem>.json plus the weights file the port expects.")
    parser.add_argument("--test-set", required=True, help="npz with x (N,C,H,W) float32 and y (N,) int.")
    parser.add_argument("--selected-class", default="all",
                        help="'all' (whole test set, matches the PyTorch spectra) or a class index "
                             "(upstream protocol: filter_val_set).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Use only the first N test samples (smoke runs).")
    parser.add_argument("--suspicious-num", type=int, default=10)
    parser.add_argument("--star", type=int, default=3)
    parser.add_argument("--record", required=True, help="Where to write the JSON measurement record.")
    parser.add_argument("--spectrum-out", default=None, help="Optional npz with the counts and scores.")
    parser.add_argument("--port", required=True, choices=("tf1", "tf2"))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_arguments(argv)
    recorder = Recorder()
    recorder.extra.update({
        "tool": f"deepfault_{args.port}",
        "model": os.path.basename(args.model),
        "selected_class": args.selected_class,
        "memory_note": MEASUREMENT_NOTE,
        "python": sys.version.split()[0],
        "seconds_before_runner": _STARTED,
    })
    print(MEASUREMENT_NOTE, flush=True)

    with recorder.phase("import"):
        import numpy as np
        import utils
        import test_nn
        import spectrum_analysis
        import tensorflow as tf
    recorder.extra["tensorflow"] = tf.__version__
    recorder.extra["executing_eagerly"] = bool(tf.executing_eagerly())

    with recorder.phase("data"):
        blob = np.load(args.test_set)
        X_test = blob["x"].astype("float32")
        y_test = blob["y"]
        if args.max_samples is not None:
            X_test, y_test = X_test[: args.max_samples], y_test[: args.max_samples]
        Y_test = utils.np_utils.to_categorical(y_test, num_classes=int(blob["num_classes"]))
        if args.selected_class == "all":
            X_val, Y_val = X_test, Y_test
        else:
            X_val, Y_val = utils.filter_val_set(int(args.selected_class), X_test, Y_test)
    recorder.extra["n_samples"] = int(len(X_val))

    with recorder.phase("model_load"):
        model = utils.load_model(args.model)

    with recorder.phase("test_model"):
        correct_classifications, misclassifications, layer_outs, predictions = \
            test_nn.test_model(model, X_val, Y_val)

    with recorder.phase("spectrum"):
        trainable_layers = utils.get_trainable_layers(model)
        scores, num_cf, num_uf, num_cs, num_us = utils.construct_spectrum_matrices(
            model, trainable_layers, correct_classifications, misclassifications, layer_outs,
        )

    per_metric = {}
    for metric in METRICS:
        with recorder.phase(f"score_{metric}"):
            if metric == "tarantula":
                spectrum_analysis.tarantula_analysis(
                    trainable_layers, scores, num_cf, num_uf, num_cs, num_us, args.suspicious_num)
            elif metric == "ochiai":
                spectrum_analysis.ochiai_analysis(
                    trainable_layers, scores, num_cf, num_uf, num_cs, num_us, args.suspicious_num)
            else:
                spectrum_analysis.dstar_analysis(
                    trainable_layers, scores, num_cf, num_uf, num_cs, num_us,
                    args.suspicious_num, args.star)
        # scores_with_foo overwrites `scores` in place; keep this metric's copy.
        per_metric[metric] = [np.array(layer, dtype=np.float64) for layer in scores]

    recorder.extra.update({
        "n_neurons": int(sum(len(layer) for layer in num_cf)),
        "n_layers": len(trainable_layers),
        "n_failing": len(misclassifications),
        "n_successful": len(correct_classifications),
        "accuracy": len(correct_classifications) / max(1, len(X_val)),
        "wall_seconds_in_runner": time.perf_counter() - _STARTED,
    })

    if args.spectrum_out:
        arrays = {}
        for index in range(len(trainable_layers)):
            arrays[f"a_f_{index}"] = np.asarray(num_cf[index], dtype=np.int64)
            arrays[f"a_s_{index}"] = np.asarray(num_cs[index], dtype=np.int64)
            for metric in METRICS:
                arrays[f"{metric}_{index}"] = per_metric[metric][index]
        np.savez_compressed(args.spectrum_out, **arrays)

    recorder.write(args.record)


if __name__ == "__main__":
    main()
