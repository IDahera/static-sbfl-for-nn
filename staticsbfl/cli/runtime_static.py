"""Runtime runner for StaticSBFL: predict one model's suspiciousness from its weights.

    python -m staticsbfl.cli.runtime_static --meta <key>.json --predictor gradient_boosting \
        --estimator-dir cache/variants/cifar10/phase4/predictions/estimators --record run.json

The static counterpart of ``runtime_hs.py``, ``runtime_icsoft.py`` and the DeepFault runners, in its
own fresh process per model. It reads no test data and runs no forward pass:

    predictor_load  the phase-4 estimators fitted by ``make predict
                    --save-predictor``, one per suspiciousness metric
    model_load      the checkpoint, rebuilt and loaded -- the state dict is
                    what the features are read from; the model is never called
    features        :func:`staticsbfl.dataset.build.model_feature_rows`, the same
                    code that built the training table
    predict         one ``predict`` per metric on those rows

``predictor_load`` is a one-off per session (a deployed predictor is loaded
once and applied to many models), so the evaluation reports it apart from the
per-model cost ``model_load + features + predict``.
"""

import argparse
import os
import sys
import time

_STARTED = time.perf_counter()
from staticsbfl.benchmark.footprint import MEASUREMENT_NOTE, Recorder  # noqa: E402  (standard library only)

METRICS = ("tarantula", "ochiai", "dstar")


def estimator_path(directory: str, phase: str, metric: str, target_kind: str, predictor: str) -> str:
    """The file ``phase_predict`` writes for (metric, target, predictor)."""
    target = metric if target_kind == "raw" else f"{metric}_{target_kind}"
    return os.path.join(directory, f"{phase}__{metric}__{target}__{predictor}.joblib")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--meta", required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--estimator-dir", required=True)
    parser.add_argument("--phase", default="phase4")
    parser.add_argument("--target-kind", default="rank_model", choices=("rank_model", "rank_layer", "raw"))
    parser.add_argument("--record", required=True)
    parser.add_argument("--predictions-out", default=None)
    args = parser.parse_args(argv)

    recorder = Recorder()
    recorder.extra.update({
        "tool": f"staticsbfl_{args.predictor}", "target_kind": args.target_kind,
        "memory_note": MEASUREMENT_NOTE, "python": sys.version.split()[0],
    })
    print(MEASUREMENT_NOTE, flush=True)

    with recorder.phase("import"):
        import json

        import joblib
        import numpy as np
        import sklearn
        import torch

        from staticsbfl.dataset.build import model_feature_rows
        from staticsbfl.persistence import load_model
        from staticsbfl.variation import build_model
        from staticsbfl.variation.config import ArchitectureConfig
    recorder.extra.update({"sklearn": sklearn.__version__, "torch": torch.__version__})

    with open(args.meta, encoding="utf-8") as handle:
        meta = json.load(handle)
    recorder.extra["model"] = meta["config"]["key"]

    with recorder.phase("predictor_load"):
        estimators = {
            metric: joblib.load(estimator_path(
                args.estimator_dir, args.phase, metric, args.target_kind, args.predictor))
            for metric in METRICS
        }
    recorder.extra["estimator_bytes_on_disk"] = int(sum(
        os.path.getsize(estimator_path(args.estimator_dir, args.phase, m, args.target_kind, args.predictor))
        for m in METRICS))

    with recorder.phase("model_load"):
        checkpoint = load_model(meta["checkpoint"])
        config = ArchitectureConfig(**{**meta["config"], "hidden_widths": tuple(meta["config"]["hidden_widths"])})
        model = build_model(config, num_classes=checkpoint.num_classes, input_shape=checkpoint.input_shape)
        model.load_state_dict(checkpoint.state_dict)

    with recorder.phase("features"):
        features, layers = model_feature_rows(
            model, config,
            input_size=int(np.prod(checkpoint.input_shape)),
            num_classes=checkpoint.num_classes,
        )

    predictions = {}
    with recorder.phase("predict"):
        for metric, estimator in estimators.items():
            predictions[metric] = estimator.predict(features)

    recorder.extra.update({
        "n_neurons": int(features.shape[0]), "n_layers": len(set(layers)),
        "wall_seconds_in_runner": time.perf_counter() - _STARTED,
    })
    if args.predictions_out:
        np.savez_compressed(args.predictions_out, layers=np.asarray(layers), **predictions)
    recorder.write(args.record)


if __name__ == "__main__":
    main()
