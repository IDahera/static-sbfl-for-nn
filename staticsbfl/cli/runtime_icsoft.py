"""Runtime runner for ICSoft@pt: the published ICSOFT 2025 spectrum code, one model, timed.

    python -m staticsbfl.cli.runtime_icsoft --meta <key>.json --test-set test_set.npz \
        --device cpu --record run.json

The code under test is ``baselines/icsoft_pt`` -- a minimal port of
``fast-hs-adversarial-input-gen`` (commit ``6e906ed``), the implementation
the ICSOFT paper links (repository owner withheld for double-blind review). It is run the way its notebook
``3_models_get_sus.ipynb`` runs it, once per hidden layer:

    hs_analysis(model, layer, loader)   one full forward pass over the test set
                                        with a hook on that layer; every
                                        activation is kept on the CPU and
                                        concatenated, then the four counts are
                                        summed over the stored tensor
    get_tarantula / get_ochiai          the two metrics ICSOFT defines

with the notebook's loader (batch size 64, ``shuffle=True``) over the same
pre-processed test set every runtime tool reads. Two settings differ from the
notebook, both for comparability, neither touches how the model is traversed:
the oracle is pass/fail (``oracle="correct"``, as DeepFault and StaticSBFL)
instead of a target class, and the whole test set is used instead of the
notebook's ``samples`` subset. ICSOFT has no D*; this tool reports Tarantula
and Ochiai, and its Ochiai keeps ICSOFT's own (targeted) numerator.
"""

import argparse
import os
import sys
import time

_STARTED = time.perf_counter()

from staticsbfl.benchmark.footprint import MEASUREMENT_NOTE, Recorder  # noqa: E402  (standard library only)

ICSOFT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baselines", "icsoft_pt")

#: The loader of the upstream notebook: ``DataLoader(..., batch_size=64, shuffle=True)``.
UPSTREAM_BATCH_SIZE = 64


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--meta", required=True, help="<key>.json written by the runtime export.")
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--selected-class", default="all")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Seeds the shuffle of the upstream loader.")
    parser.add_argument("--record", required=True)
    parser.add_argument("--spectrum-out", default=None)
    args = parser.parse_args(argv)

    # Upstream picks its device at import time from helper.global_variables.
    os.environ["ICSOFT_DEVICE"] = args.device
    sys.path.insert(0, ICSOFT_DIR)

    recorder = Recorder()
    recorder.extra.update({
        "tool": f"icsoft_pt_{args.device}", "selected_class": args.selected_class,
        "memory_note": MEASUREMENT_NOTE, "python": sys.version.split()[0],
        "upstream": "fast-hs-adversarial-input-gen@6e906ed",
    })
    print(MEASUREMENT_NOTE, flush=True)

    with recorder.phase("import"):
        import json

        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from helper.global_variables import DEVICE
        from helper.suspiciousness import get_ochiai, get_tarantula
        from hit_spectrum import hs_analysis

        from staticsbfl.persistence import load_model
        from staticsbfl.variation import build_model
        from staticsbfl.variation.config import ArchitectureConfig
    if DEVICE.type == "mps":
        recorder.sync = torch.mps.synchronize
    elif DEVICE.type == "cuda":
        recorder.sync = torch.cuda.synchronize
    recorder.extra.update({"torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                           "device": str(DEVICE)})

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
        torch.manual_seed(args.seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                          torch.from_numpy(y.astype(np.int64))),
            batch_size=UPSTREAM_BATCH_SIZE, shuffle=True,
        )
    recorder.extra["n_samples"] = int(len(y))

    with recorder.phase("model_load"):
        checkpoint = load_model(meta["checkpoint"])
        config = ArchitectureConfig(**{**meta["config"], "hidden_widths": tuple(meta["config"]["hidden_widths"])})
        model = build_model(config, num_classes=checkpoint.num_classes, input_shape=checkpoint.input_shape)
        model.load_state_dict(checkpoint.state_dict)
        model.to(DEVICE)  # upstream: load_model_with_weights(...) places the model on DEVICE

    spectra = {}
    with recorder.phase("spectrum"):
        for name in model.hidden_layer_names:
            spectra[name] = hs_analysis(model, getattr(model, name), loader, target_class=0, oracle="correct")

    scores = {}
    with recorder.phase("score"):
        for name, hs in spectra.items():
            scores[name] = {"tarantula": get_tarantula(hs), "ochiai": get_ochiai(hs)}

    any_layer = next(iter(spectra.values()))
    n_successful = int(any_layer.a_s[0] + any_layer.n_s[0])
    recorder.extra.update({
        "n_neurons": int(sum(hs.a_s.numel() for hs in spectra.values())), "n_layers": len(spectra),
        "n_successful": n_successful, "n_failing": int(len(y)) - n_successful,
        "accuracy": n_successful / max(1, len(y)),
        "wall_seconds_in_runner": time.perf_counter() - _STARTED,
    })

    if args.spectrum_out:
        arrays = {}
        for index, name in enumerate(model.hidden_layer_names):
            arrays[f"a_f_{index}"] = spectra[name].a_f.cpu().numpy().astype(np.int64)
            arrays[f"a_s_{index}"] = spectra[name].a_s.cpu().numpy().astype(np.int64)
            arrays[f"tarantula_{index}"] = scores[name]["tarantula"].cpu().numpy().astype(np.float64)
        np.savez_compressed(args.spectrum_out, **arrays)

    recorder.write(args.record)


if __name__ == "__main__":
    main()
