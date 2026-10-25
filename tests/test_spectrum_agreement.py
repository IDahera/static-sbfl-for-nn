"""Do the dynamic tools compute the same hit spectrum? HS@pt vs ICSoft@pt vs DF@tf1/DF@tf2.

Two kinds of model, because the answer has two parts:

* **exactly representable** -- weights, biases and inputs are small multiples
  of 1/8, so every product and sum is exact in float32 and no rounding can
  push a pre-activation across the threshold. Here every implementation must
  produce *bit-identical* counts: any difference is a bug in a port or an
  adapter, not arithmetic.
* **random** -- ordinary float weights, as in the trained models. Different
  frameworks, batch sizes and devices round differently, so a pre-activation
  within rounding of 0 may land on either side. Allowed: at most 2 % of the
  neurons off, each by at most 1 count, and Spearman rho >= 0.999 on Tarantula.
  This is the tolerance the paper can state (see the fidelity check of every
  runtime run, ``fidelity.csv``).

The DeepFault ports run in their own conda environments
(``make runtime-setup``); the tests that need them are skipped without them.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from staticsbfl.evaluation.runtime import compare_spectra
from staticsbfl.suspiciousness import compute_hit_spectrum, score_spectrum
from staticsbfl.utils.paths import PROJECT_ROOT
from staticsbfl.variation.builder import VariableMLP

sys.path.insert(0, str(PROJECT_ROOT / "baselines" / "icsoft_pt"))

INPUT_SHAPE = (3, 4, 4)
WIDTHS = (24, 12)


def make_model(exact: bool, seed: int = 0) -> VariableMLP:
    torch.manual_seed(seed)
    model = VariableMLP(WIDTHS, num_classes=10, input_shape=INPUT_SHAPE)
    if exact:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randint(-4, 5, parameter.shape, generator=generator).float() / 8)
    return model.eval()


def make_data(exact: bool, n: int = 300, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    if exact:
        x = torch.randint(-4, 5, (n, *INPUT_SHAPE), generator=generator).float() / 4
    else:
        x = torch.randn((n, *INPUT_SHAPE), generator=generator)
    y = torch.randint(0, 10, (n,), generator=generator)
    return x, y


def nnfl_spectrum(model, x, y, batch_size=512):
    spectrum = compute_hit_spectrum(model, DataLoader(TensorDataset(x, y), batch_size=batch_size),
                                    device=torch.device("cpu"))
    scores = score_spectrum(spectrum)
    names = spectrum.layer_names
    return {"a_f": [spectrum.layers[n].a_f.numpy() for n in names],
            "a_s": [spectrum.layers[n].a_s.numpy() for n in names],
            "tarantula": [scores["tarantula"][n].double().numpy() for n in names]}


def icsoft_spectrum(model, x, y):
    os.environ["ICSOFT_DEVICE"] = "cpu"
    from helper.suspiciousness import get_tarantula
    from hit_spectrum import hs_analysis

    torch.manual_seed(0)
    loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)   # the upstream notebook's loader
    out = {"a_f": [], "a_s": [], "tarantula": []}
    for name in model.hidden_layer_names:
        hs = hs_analysis(model, getattr(model, name), loader, target_class=0, oracle="correct")
        out["a_f"].append(hs.a_f.numpy())
        out["a_s"].append(hs.a_s.numpy())
        out["tarantula"].append(get_tarantula(hs).double().numpy())
    return out


def assert_identical(got, ref):
    result = compare_spectra(got, ref)
    assert result["identical"], result


def assert_within_rounding(got, ref):
    result = compare_spectra(got, ref)
    assert result["neurons_differing"] <= 0.02 * result["n_neurons"], result
    assert result["max_count_difference"] <= 1, result
    assert result["spearman_tarantula"] is None or result["spearman_tarantula"] >= 0.999, result


# --------------------------------------------------------------------------
# HS@pt with itself, and ICSoft@pt
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 7, 64, 512])
def test_nnfl_counts_do_not_depend_on_the_batch_size_when_arithmetic_is_exact(batch_size):
    model = make_model(exact=True)
    x, y = make_data(exact=True)
    assert_identical(nnfl_spectrum(model, x, y, batch_size), nnfl_spectrum(model, x, y, 512))


def test_icsoft_and_nnfl_count_identically_when_arithmetic_is_exact():
    model = make_model(exact=True)
    x, y = make_data(exact=True)
    assert_identical(icsoft_spectrum(model, x, y), nnfl_spectrum(model, x, y))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_icsoft_and_nnfl_agree_within_rounding_on_random_weights(seed):
    model = make_model(exact=False, seed=seed)
    x, y = make_data(exact=False, seed=seed + 10)
    assert_within_rounding(icsoft_spectrum(model, x, y), nnfl_spectrum(model, x, y))


# --------------------------------------------------------------------------
# The DeepFault ports, in their own environments
# --------------------------------------------------------------------------


def port_python(port: str) -> Path:
    from staticsbfl.cli.runtime_benchmark import port_python as resolve
    return resolve(port)


def deepfault_spectrum(port, model, x, y, tmp_path):
    """Export *model* like the benchmark does, convert it in the port's env, run the port's runner."""
    python = port_python(port)
    if not python.exists():
        pytest.skip(f"no conda env for DF@{port} (make runtime-setup)")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    with torch.no_grad():
        reference = model(x).argmax(dim=1).numpy()
    key = "m"
    np.savez(weights / f"{key}.npz", reference_argmax=reference, **{k: v.numpy() for k, v in model.state_dict().items()})
    (weights / f"{key}.json").write_text(json.dumps({
        "hidden_widths": list(WIDTHS), "activation": "relu", "input_shape": list(INPUT_SHAPE), "num_classes": 10}))
    test_set = tmp_path / "test_set.npz"
    np.savez(test_set, x=x.numpy(), y=y.numpy(), num_classes=10)
    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}
    keras_dir = tmp_path / f"keras_{port}"
    bridge = subprocess.run([str(python), str(PROJECT_ROOT / "baselines" / f"deepfault_{port}" / "torch_bridge.py"),
                             "--weights-dir", str(weights), "--out-dir", str(keras_dir), "--test-set", str(test_set)],
                            capture_output=True, text=True, env=env, timeout=300)
    assert bridge.returncode == 0, bridge.stdout[-1500:] + bridge.stderr[-1500:]
    spectrum_out = tmp_path / f"{port}.npz"
    run = subprocess.run([str(python), str(PROJECT_ROOT / "baselines" / f"deepfault_{port}" / "runtime_run.py"),
                          "--model", str(keras_dir / key), "--test-set", str(test_set),
                          "--record", str(tmp_path / f"{port}.json"), "--spectrum-out", str(spectrum_out)],
                         capture_output=True, text=True, env=env, timeout=600)
    assert run.returncode == 0, run.stdout[-1500:] + run.stderr[-1500:]
    blob = np.load(spectrum_out)
    layers = len(WIDTHS)
    return {"a_f": [blob[f"a_f_{i}"] for i in range(layers)], "a_s": [blob[f"a_s_{i}"] for i in range(layers)],
            "tarantula": [blob[f"tarantula_{i}"] for i in range(layers)]}


@pytest.mark.parametrize("port", ["tf1", "tf2"])
def test_deepfault_counts_identically_to_nnfl_when_arithmetic_is_exact(port, tmp_path):
    model = make_model(exact=True)
    x, y = make_data(exact=True)
    assert_identical(deepfault_spectrum(port, model, x, y, tmp_path), nnfl_spectrum(model, x, y))


@pytest.mark.parametrize("port", ["tf1", "tf2"])
def test_deepfault_agrees_with_nnfl_within_rounding_on_random_weights(port, tmp_path):
    model = make_model(exact=False, seed=3)
    x, y = make_data(exact=False, seed=13)
    assert_within_rounding(deepfault_spectrum(port, model, x, y, tmp_path), nnfl_spectrum(model, x, y))
