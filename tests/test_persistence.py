"""Cache round-trip tests.

The cache is the interface between the pipeline steps: training writes it,
scoring reads and extends it, the neuron table reads it again. A round trip
that loses the layer shapes or the outcome totals would only surface as a
confusing table much later.
"""

import pytest
import torch

from staticsbfl.persistence import (
    ModelCheckpoint,
    SuspiciousnessBundle,
    load_model,
    load_suspiciousness,
    save_model,
    save_suspiciousness,
)
from staticsbfl.suspiciousness import compute_hit_spectrum, score_spectrum
from staticsbfl.suspiciousness.spectrum import HitSpectrum, LayerSpectrum


def test_a_model_survives_the_round_trip(tmp_path):
    from staticsbfl.models import SmallMLP

    torch.manual_seed(0)
    model = SmallMLP()
    path = tmp_path / "mlp_small.pt"

    save_model(
        path,
        ModelCheckpoint(
            key="mlp_small",
            state_dict=model.state_dict(),
            dataset="cifar10",
            input_shape=(3, 32, 32),
            num_classes=10,
            metadata={"test_accuracy": 0.41},
        ),
    )
    loaded = load_model(path)

    assert loaded.key == "mlp_small"
    assert loaded.input_shape == (3, 32, 32)
    assert loaded.num_classes == 10
    assert loaded.metadata["test_accuracy"] == pytest.approx(0.41)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict[name])


def test_suspiciousness_survives_the_round_trip(tmp_path, scripted_mlp, loader_factory):
    """Counts, totals, threshold and every score tensor come back unchanged."""
    model = scripted_mlp()
    spectrum = compute_hit_spectrum(
        model,
        loader_factory(torch.randn(10, 4), torch.randint(0, 2, (10,)), batch_size=3),
        device=torch.device("cpu"),
        threshold=0.25,
    )
    scores = score_spectrum(spectrum, star=2)
    path = tmp_path / "scored.pt"

    save_suspiciousness(
        path,
        SuspiciousnessBundle(
            key="mlp_small", hit_spectrum=spectrum, scores=scores, metadata={"star": 2}
        ),
    )
    loaded = load_suspiciousness(path)

    assert loaded.key == "mlp_small"
    assert loaded.hit_spectrum.n_failing == spectrum.n_failing
    assert loaded.hit_spectrum.n_successful == spectrum.n_successful
    assert loaded.hit_spectrum.threshold == pytest.approx(0.25)
    assert loaded.layer_names == spectrum.layer_names
    assert loaded.metric_names == tuple(scores)

    for name, layer in spectrum.layers.items():
        restored = loaded.hit_spectrum.layers[name]
        assert torch.equal(layer.a_f, restored.a_f)
        assert torch.equal(layer.a_s, restored.a_s)
        # The derived counts must reconstruct too, not just the stored ones.
        assert torch.equal(layer.n_f, restored.n_f)
        assert torch.equal(layer.n_s, restored.n_s)

    for metric, per_layer in scores.items():
        for name, values in per_layer.items():
            assert torch.equal(values, loaded.scores[metric][name])


def test_score_tensors_keep_their_layer_shape_through_the_cache(tmp_path):
    """A layer's (hidden,) scores must keep their shape through storage."""
    spectrum = HitSpectrum(
        layers={
            "fc1": LayerSpectrum(
                name="fc1",
                a_f=torch.randint(0, 4, (6,), dtype=torch.int32),
                a_s=torch.randint(0, 9, (6,), dtype=torch.int32),
                n_failing=4,
                n_successful=9,
            )
        },
        n_failing=4,
        n_successful=9,
    )
    scores = score_spectrum(spectrum)
    path = tmp_path / "dense.pt"
    save_suspiciousness(path, SuspiciousnessBundle("mlp_small", spectrum, scores))

    loaded = load_suspiciousness(path)
    for per_layer in loaded.scores.values():
        assert per_layer["fc1"].shape == (6,)
    assert loaded.hit_spectrum.layers["fc1"].neuron_shape == (6,)


def test_everything_is_stored_on_the_cpu(tmp_path, scripted_mlp, loader_factory):
    """A checkpoint written on one backend has to load on another."""
    spectrum = compute_hit_spectrum(
        scripted_mlp(),
        loader_factory(torch.randn(6, 4), torch.randint(0, 2, (6,)), batch_size=2),
        device=torch.device("cpu"),
    )
    path = tmp_path / "cpu.pt"
    save_suspiciousness(
        path, SuspiciousnessBundle("mlp_small", spectrum, score_spectrum(spectrum))
    )

    loaded = load_suspiciousness(path)
    for layer in loaded.hit_spectrum.layers.values():
        assert layer.a_f.device.type == "cpu"


def test_a_missing_cache_entry_says_which_step_creates_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="make train"):
        load_model(tmp_path / "absent.pt")
    with pytest.raises(FileNotFoundError, match="make score"):
        load_suspiciousness(tmp_path / "absent.pt")
