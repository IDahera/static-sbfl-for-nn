"""Device selection and placement.

Two questions are asked separately:

    * is the ``--device`` string resolved to the right :class:`torch.device`?
      Checked with mocked availability, so the same assertions run on a CPU-only
      CI box and on this laptop.
    * do the tensors actually end up there? Checked by inspecting the counters a
      real counting pass produced, and -- for accelerators that may not exist --
      by mocking ``Tensor.to`` and recording what it was asked for.
"""

import pytest
import torch

from staticsbfl.utils.devices import (
    DEVICE_CHOICES,
    best_available_device,
    resolve_device,
    same_device,
)


# --------------------------------------------------------------------------
# Resolution, with availability mocked
# --------------------------------------------------------------------------


def _mock_availability(monkeypatch, *, mps: bool, cuda: bool) -> None:
    """Pretend this machine has (or lacks) each accelerator."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)


def test_cpu_always_resolves():
    assert resolve_device("cpu").type == "cpu"


@pytest.mark.parametrize(
    "mps, cuda, expected",
    [
        (True, True, "mps"),    # MPS wins when both are present
        (True, False, "mps"),
        (False, True, "cuda"),
        (False, False, "cpu"),
    ],
)
def test_auto_follows_the_documented_preference_order(monkeypatch, mps, cuda, expected):
    """auto = MPS, then CUDA, then CPU."""
    _mock_availability(monkeypatch, mps=mps, cuda=cuda)
    assert resolve_device("auto").type == expected
    assert best_available_device().type == expected


def test_an_unavailable_accelerator_is_an_error_not_a_downgrade(monkeypatch):
    """Silently falling back to CPU would produce valid-looking numbers slowly."""
    _mock_availability(monkeypatch, mps=False, cuda=False)

    with pytest.raises(ValueError, match="MPS was requested"):
        resolve_device("mps")
    with pytest.raises(ValueError, match="CUDA was requested"):
        resolve_device("cuda")


def test_an_unknown_device_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")


def test_the_case_of_the_name_does_not_matter():
    assert resolve_device("CPU").type == "cpu"


def test_every_advertised_choice_is_handled(monkeypatch):
    """No choice offered on the CLI may fall through to an unhandled branch."""
    _mock_availability(monkeypatch, mps=True, cuda=True)
    for name in DEVICE_CHOICES:
        assert isinstance(resolve_device(name), torch.device)


def test_same_device_ignores_an_unset_index():
    """Tensors report 'cuda:0'; a resolved device may just say 'cuda'."""
    assert same_device(torch.device("cuda"), torch.device("cuda:0"))
    assert same_device(torch.device("cpu"), torch.device("cpu"))
    assert not same_device(torch.device("cpu"), torch.device("cuda"))
    assert not same_device(torch.device("cuda:1"), torch.device("cuda:0"))


# --------------------------------------------------------------------------
# Placement -- do the tensors land where they were told to?
# --------------------------------------------------------------------------


def test_the_counters_live_on_the_requested_device(scripted_mlp, loader_factory, cpu):
    """The accumulators a real pass produced carry the requested device."""
    from staticsbfl.suspiciousness import compute_hit_spectrum

    model = scripted_mlp()
    spectrum = compute_hit_spectrum(
        model,
        loader_factory(torch.randn(6, 4), torch.randint(0, 2, (6,)), batch_size=2),
        device=cpu,
    )

    for layer in spectrum.layers.values():
        assert same_device(layer.a_f.device, cpu)
        assert same_device(layer.a_s.device, cpu)


def test_init_counters_places_the_tensors(cpu):
    from staticsbfl.suspiciousness import init_counters

    a_f, a_s = init_counters({"fc1": (4,), "conv": (2, 3, 3)}, device=cpu)
    for tensors in (a_f, a_s):
        for tensor in tensors.values():
            assert same_device(tensor.device, cpu)


def test_the_resolved_device_is_the_one_the_pass_moves_data_to(monkeypatch, scripted_mlp, loader_factory):
    """Mock ``Tensor.to`` and assert the pass asks for the device it was given.

    This is the check that survives on a machine without an accelerator: it
    verifies the *plumbing* -- that ``device`` reaches every ``.to()`` call --
    rather than the hardware.
    """
    from staticsbfl.suspiciousness import compute_hit_spectrum

    requested = []
    original = torch.Tensor.to

    def recording_to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if isinstance(target, torch.device):
            requested.append(target)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", recording_to)

    device = torch.device("cpu")
    compute_hit_spectrum(
        scripted_mlp(),
        loader_factory(torch.randn(6, 4), torch.randint(0, 2, (6,)), batch_size=2),
        device=device,
    )

    assert requested, "the counting pass never moved a tensor to an explicit device"
    assert all(same_device(d, device) for d in requested), (
        f"some tensors were sent elsewhere: {set(map(str, requested))}"
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="no MPS backend on this machine"
)
def test_the_same_counts_come_out_on_mps_as_on_cpu(scripted_mlp, loader_factory):
    """The accelerator path must agree with the CPU path exactly.

    Counting is integer work, so 'close enough' is not the standard here: the
    two devices have to produce identical tensors.
    """
    from staticsbfl.suspiciousness import compute_hit_spectrum

    torch.manual_seed(11)
    model = scripted_mlp()
    inputs = torch.randn(12, 4)
    targets = torch.randint(0, 2, (12,))

    on_cpu = compute_hit_spectrum(
        model, loader_factory(inputs, targets, batch_size=5), device=torch.device("cpu")
    )
    on_mps = compute_hit_spectrum(
        model, loader_factory(inputs, targets, batch_size=5), device=torch.device("mps")
    )

    for name, layer in on_cpu.layers.items():
        assert same_device(on_mps.layers[name].a_f.device, torch.device("mps"))
        assert torch.equal(layer.a_f, on_mps.layers[name].a_f.cpu()), name
        assert torch.equal(layer.a_s, on_mps.layers[name].a_s.cpu()), name


def test_moving_a_spectrum_between_devices_preserves_the_counts(scripted_mlp, loader_factory, cpu):
    from staticsbfl.suspiciousness import compute_hit_spectrum

    spectrum = compute_hit_spectrum(
        scripted_mlp(),
        loader_factory(torch.randn(6, 4), torch.randint(0, 2, (6,)), batch_size=3),
        device=cpu,
    )
    moved = spectrum.to(cpu)

    assert moved.n_failing == spectrum.n_failing
    for name, layer in spectrum.layers.items():
        assert torch.equal(layer.a_f, moved.layers[name].a_f)
