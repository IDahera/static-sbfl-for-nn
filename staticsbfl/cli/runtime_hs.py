"""Runtime runner for HS@pt: this repository's own streaming PyTorch spectrum, one model, timed.

    python -m staticsbfl.cli.runtime_hs --meta <key>.json --test-set test_set.npz \
        --device cpu --record run.json

Started by ``runtime_benchmark.py`` in a fresh process per model, like every
other runner, so the memory figures of two tools never mix. It runs what
``make score`` runs for one model -- :func:`staticsbfl.suspiciousness.compute_hit_spectrum`
over the test set (one pass, all hidden layers at once, counts accumulated per
batch, no activation stored), then :func:`score_spectrum` -- with the same
batch size (512) and the same threshold (``> 0``). The one difference from
``make score`` is the input: every runtime tool reads the same pre-processed test
set from one ``.npz`` held in memory, so torchvision decoding is excluded for
all of them alike.

Not to be confused with ICSoft@pt (``runtime_icsoft.py``), the published ICSOFT
2025 code, which stores every activation and runs one pass per layer.
"""

import argparse
import os
import sys
import time

_STARTED = time.perf_counter()
from staticsbfl.benchmark.footprint import MEASUREMENT_NOTE, Recorder  # noqa: E402  (standard library only)

#: Batch size of ``make score`` (``staticsbfl/cli/phase_score.py --batch-size``).
SCORE_BATCH_SIZE = 512


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--meta", required=True, help="<key>.json written by the runtime_benchmark export.")
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=SCORE_BATCH_SIZE)
    parser.add_argument("--selected-class", default="all")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--star", type=int, default=3)
    parser.add_argument("--record", required=True)
    parser.add_argument("--spectrum-out", default=None)
    args = parser.parse_args(argv)

    recorder = Recorder()
    recorder.extra.update({
        "tool": f"hs_pt_{args.device}", "selected_class": args.selected_class,
        "memory_note": MEASUREMENT_NOTE, "python": sys.version.split()[0],
    })
    print(MEASUREMENT_NOTE, flush=True)

    with recorder.phase("import"):
        import json

        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from staticsbfl.benchmark.resources import MemoryProbe
        from staticsbfl.persistence import load_model
        from staticsbfl.suspiciousness import compute_hit_spectrum, score_spectrum
        from staticsbfl.variation import build_model
        from staticsbfl.variation.config import ArchitectureConfig
    device = torch.device(args.device)
    if device.type == "mps":
        recorder.sync = torch.mps.synchronize
    elif device.type == "cuda":
        recorder.sync = torch.cuda.synchronize
    recorder.extra.update({"torch": torch.__version__, "torch_threads": torch.get_num_threads()})

    with open(args.meta, encoding="utf-8") as handle:
        meta = json.load(handle)
    recorder.extra["model"] = meta["config"]["key"]

    with recorder.phase("data"):
        blob = np.load(args.test_set)
        x, y = blob["x"], blob["y"]
        if args.max_samples is not None:
            x, y = x[: args.max_samples], y[: args.max_samples]
        if args.selected_class != "all":
            keep = y == int(args.selected_class)
            x, y = x[keep], y[keep]
        loader = DataLoader(
            TensorDataset(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                          torch.from_numpy(y.astype(np.int64))),
            batch_size=args.batch_size, shuffle=False,
        )
    recorder.extra["n_samples"] = int(len(y))

    with recorder.phase("model_load"):
        checkpoint = load_model(meta["checkpoint"])
        config = ArchitectureConfig(**{**meta["config"], "hidden_widths": tuple(meta["config"]["hidden_widths"])})
        model = build_model(config, num_classes=checkpoint.num_classes, input_shape=checkpoint.input_shape)
        model.load_state_dict(checkpoint.state_dict)

    probe = MemoryProbe(device, empty_cache=False)
    with recorder.phase("spectrum"):
        with probe:
            spectrum = compute_hit_spectrum(model, loader, device=device, threshold=0.0)

    with recorder.phase("score"):
        scores = score_spectrum(spectrum, star=args.star)
        spectrum = spectrum.to(torch.device("cpu"))
        scores = {m: {k: v.cpu() for k, v in layers.items()} for m, layers in scores.items()}

    gpu = probe.reading.as_dict() if probe.reading else {}
    recorder.extra.update({
        "gpu_reading": {k: gpu.get(k) for k in (
            "fidelity", "peak_device_bytes", "baseline_device_bytes",
            "peak_above_baseline_bytes", "driver_delta_bytes")},
        "n_neurons": int(spectrum.n_neurons), "n_layers": len(spectrum.layer_names),
        "n_failing": int(spectrum.n_failing), "n_successful": int(spectrum.n_successful),
        "accuracy": spectrum.accuracy,
        "wall_seconds_in_runner": time.perf_counter() - _STARTED,
    })

    if args.spectrum_out:
        arrays = {}
        for index, name in enumerate(spectrum.layer_names):
            arrays[f"a_f_{index}"] = spectrum.layers[name].a_f.numpy().astype(np.int64)
            arrays[f"a_s_{index}"] = spectrum.layers[name].a_s.numpy().astype(np.int64)
            for metric, layers in scores.items():
                arrays[f"{metric}_{index}"] = layers[name].numpy().astype(np.float64)
        np.savez_compressed(args.spectrum_out, **arrays)

    recorder.write(args.record)


if __name__ == "__main__":
    main()
