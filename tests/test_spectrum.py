"""Hit-spectrum computation tests.

The counting pass is where a subtle error would be invisible in the output --
every score would still look plausible -- so it is checked against an
independent brute-force reference, against hand-scripted models whose counts are
known in advance, and against a set of invariants that must hold for any input.
"""

import pytest
import torch
import torch.nn as nn

from staticsbfl.suspiciousness import spectrum as spectrum_module
from staticsbfl.suspiciousness.spectrum import (
    COUNT_DTYPE,
    ActivationRecorder,
    active_mask,
    candidate_layers,
    compute_hit_spectrum,
    infer_layer_shapes,
    init_counters,
    select_hidden_layers,
    success_mask,
    update_counts,
)


# --------------------------------------------------------------------------
# Layer selection -- which neurons take part at all
# --------------------------------------------------------------------------


def test_declared_hidden_layers_are_honoured(scripted_mlp):
    """A model that states its scope gets exactly that scope."""
    model = scripted_mlp()
    assert [name for name, _ in select_hidden_layers(model)] == ["fc1", "fc2"]


def test_the_output_layer_is_excluded(scripted_mlp):
    """The head is a candidate layer but must never be counted.

    Output neurons *are* the prediction: 'this neuron fired on a failing sample'
    restates the misclassification instead of explaining it.
    """
    model = scripted_mlp()
    assert "head" in dict(candidate_layers(model))
    assert "head" not in dict(select_hidden_layers(model))


def test_fallback_drops_the_last_candidate_layer():
    """Without a declaration, registration order decides and the last one goes."""
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2), nn.Linear(2, 2))
    assert [name for name, _ in select_hidden_layers(model)] == ["0", "2"]


def test_explicit_names_override_the_declaration(scripted_mlp):
    model = scripted_mlp()
    assert [name for name, _ in select_hidden_layers(model, ["fc2"])] == ["fc2"]


def test_an_unknown_layer_name_is_rejected(scripted_mlp):
    with pytest.raises(ValueError, match="not Linear submodules"):
        select_hidden_layers(scripted_mlp(), ["nope"])


def test_a_model_with_one_layer_has_no_hidden_scope():
    """Nothing to localize into if the only layer is the output layer."""
    with pytest.raises(ValueError, match="at least two"):
        select_hidden_layers(nn.Sequential(nn.Linear(4, 2)))


# --------------------------------------------------------------------------
# Counter allocation -- shapes, dtype, and allocated exactly once
# --------------------------------------------------------------------------


def test_counter_shapes_match_the_layers_they_describe(cpu):
    """Every accumulator has its layer's neuron shape, with no batch axis."""
    shapes = {"fc1": (3,), "fc2": (7,)}
    a_f, a_s = init_counters(shapes, device=cpu)

    assert tuple(a_f["fc1"].shape) == (3,)
    assert tuple(a_f["fc2"].shape) == (7,)
    assert tuple(a_s["fc2"].shape) == (7,)


def test_counters_start_at_zero_and_are_int32(cpu):
    """int32: a count cannot exceed the sample count, so int64 is wasted width."""
    a_f, a_s = init_counters({"fc1": (4,)}, device=cpu)

    assert a_f["fc1"].dtype is COUNT_DTYPE is torch.int32
    assert torch.all(a_f["fc1"] == 0)
    assert torch.all(a_s["fc1"] == 0)
    # The two accumulators must not alias each other.
    a_f["fc1"] += 1
    assert torch.all(a_s["fc1"] == 0)


def test_inferred_shapes_match_the_real_layer_dimensions():
    """The probe pass reports what the layers actually produce."""
    from staticsbfl.models import SmallMLP, WideMLP

    device = torch.device("cpu")

    mlp = SmallMLP()
    shapes = infer_layer_shapes(
        mlp, torch.randn(2, 3, 32, 32), select_hidden_layers(mlp), device=device
    )
    assert shapes == {"fc1": (256,), "fc2": (128,)}

    wide = WideMLP()
    shapes = infer_layer_shapes(
        wide, torch.randn(2, 3, 32, 32), select_hidden_layers(wide), device=device
    )
    assert shapes == {"fc1": (2048,), "fc2": (1024,), "fc3": (512,)}


def test_a_layer_off_the_forward_path_is_reported():
    """A selected layer that never runs is a configuration error, not silence."""

    class Detached(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(4, 3)
            self.unused = nn.Linear(3, 3)
            self.head = nn.Linear(3, 2)

        def forward(self, x):
            return self.head(self.used(x))

    model = Detached()
    layers = select_hidden_layers(model, ["used", "unused"])
    with pytest.raises(RuntimeError, match="never executed"):
        infer_layer_shapes(model, torch.randn(2, 4), layers, device=torch.device("cpu"))


def test_counters_are_allocated_once_not_per_batch(monkeypatch, scripted_mlp, loader_factory):
    """The accumulators are created before the loop and only updated in it."""
    calls = []
    original = spectrum_module.init_counters

    def counting_init(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(spectrum_module, "init_counters", counting_init)

    model = scripted_mlp()
    inputs = torch.randn(12, 4)
    targets = torch.randint(0, 2, (12,))
    compute_hit_spectrum(
        model, loader_factory(inputs, targets, batch_size=3), device=torch.device("cpu")
    )

    assert len(calls) == 1, f"counters were allocated {len(calls)} times, expected 1"


def test_the_counters_are_updated_in_place(cpu):
    """update_counts must mutate the accumulator, not rebind it."""
    a_f, a_s = init_counters({"fc1": (3,)}, device=cpu)
    before_f, before_s = a_f["fc1"], a_s["fc1"]

    update_counts(
        a_f["fc1"],
        a_s["fc1"],
        active=torch.tensor([[True, False, True], [True, True, False]]),
        success=torch.tensor([True, False]),
    )

    assert a_f["fc1"] is before_f
    assert a_s["fc1"] is before_s
    assert a_s["fc1"].tolist() == [1, 0, 1]   # sample 0 succeeded
    assert a_f["fc1"].tolist() == [1, 1, 0]   # sample 1 failed


def test_update_counts_rejects_a_shape_mismatch(cpu):
    a_f, a_s = init_counters({"fc1": (3,)}, device=cpu)
    with pytest.raises(ValueError, match="does not match the counter shape"):
        update_counts(a_f["fc1"], a_s["fc1"], torch.zeros(2, 4, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))


def test_update_counts_rejects_a_batch_size_mismatch(cpu):
    a_f, a_s = init_counters({"fc1": (3,)}, device=cpu)
    with pytest.raises(ValueError, match="outcome flags"):
        update_counts(a_f["fc1"], a_s["fc1"], torch.zeros(2, 3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool))


# --------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------


def test_active_mask_uses_a_strict_threshold():
    """Exactly at the threshold is *not* active; ReLU's zero output must not count."""
    values = torch.tensor([-1.0, 0.0, 1e-6, 3.0])
    assert active_mask(values, threshold=0.0).tolist() == [False, False, True, True]
    assert active_mask(values, threshold=1.0).tolist() == [False, False, False, True]


def test_success_mask_is_the_argmax_agreeing_with_the_label():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    targets = torch.tensor([1, 1, 1])
    assert success_mask(logits, targets).tolist() == [True, False, True]


# --------------------------------------------------------------------------
# Scripted models -- counts known before the code runs
# --------------------------------------------------------------------------


def test_a_permanently_active_model_counts_every_sample(scripted_mlp, loader_factory):
    """Hidden bias +1 makes every neuron fire on every sample.

    Then a_f = N_f and a_s = N_s for every neuron, and n_f = n_s = 0 -- the
    'all ones' case, where the expected counts can be written down directly.
    """
    model = scripted_mlp(hidden_bias=1.0)
    model.set_prediction(0)                     # everything is predicted as class 0
    inputs = torch.randn(10, 4)
    targets = torch.tensor([0] * 6 + [1] * 4)   # 6 successes, 4 failures

    spectrum = compute_hit_spectrum(
        model, loader_factory(inputs, targets, batch_size=3), device=torch.device("cpu")
    )

    assert spectrum.n_successful == 6
    assert spectrum.n_failing == 4
    for layer in spectrum.layers.values():
        assert torch.all(layer.a_f == 4)
        assert torch.all(layer.a_s == 6)
        assert torch.all(layer.n_f == 0)
        assert torch.all(layer.n_s == 0)


def test_a_permanently_inactive_model_counts_nothing(scripted_mlp, loader_factory):
    """Hidden bias -1 (through ReLU) means nothing ever fires: the 'all zeros' case."""
    model = scripted_mlp(hidden_bias=-1.0)
    model.set_prediction(0)
    inputs = torch.randn(10, 4)
    targets = torch.tensor([0] * 6 + [1] * 4)

    spectrum = compute_hit_spectrum(
        model, loader_factory(inputs, targets, batch_size=4), device=torch.device("cpu")
    )

    for layer in spectrum.layers.values():
        assert torch.all(layer.a_f == 0)
        assert torch.all(layer.a_s == 0)
        assert torch.all(layer.n_f == 4)
        assert torch.all(layer.n_s == 6)


def test_counts_match_a_hand_built_activation_pattern(passthrough_net, loader_factory):
    """A net whose hidden activations *are* its input, so the pattern is chosen.

    Four samples, five hidden neurons. The trailing input feature fixes the
    prediction, so activation and outcome are set independently::

        sample  hidden pattern (>0 fires)   predicted  label  outcome
        0       [+, -, +, -, +]             0          0      success
        1       [+, +, -, -, +]             0          1      failure
        2       [-, -, +, +, +]             1          1      success
        3       [+, -, -, +, +]             1          0      failure

    Expected per neuron:  a_f = [2, 1, 0, 1, 2],  a_s = [1, 0, 2, 1, 2].
    """
    model = passthrough_net(n_hidden=5, num_classes=2)
    patterns = torch.tensor(
        [
            [1.0, -1.0, 1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0, 1.0, 1.0],
        ]
    )
    predictions = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    inputs = torch.cat([patterns, predictions], dim=1)
    targets = torch.tensor([0, 1, 1, 0])

    spectrum = compute_hit_spectrum(
        model, loader_factory(inputs, targets, batch_size=2), device=torch.device("cpu")
    )
    layer = spectrum.layers["fc1"]

    assert spectrum.n_failing == 2
    assert spectrum.n_successful == 2
    assert layer.a_f.tolist() == [2, 1, 0, 1, 2]
    assert layer.a_s.tolist() == [1, 0, 2, 1, 2]
    assert layer.n_f.tolist() == [0, 1, 2, 1, 0]
    assert layer.n_s.tolist() == [1, 2, 0, 1, 0]


# --------------------------------------------------------------------------
# Against an independent reference
# --------------------------------------------------------------------------


def _reference_counts(model, inputs, targets, layer_names, threshold=0.0):
    """A deliberately naive per-sample reference implementation.

    No batching, no broadcasting, no in-place accumulation -- a plain Python
    loop over samples. Slow and obviously correct, which is exactly what the
    vectorised implementation should be checked against.
    """
    model.eval()
    captured = {}
    handles = [
        module.register_forward_hook(
            lambda _m, _i, out, name=name: captured.__setitem__(name, out.detach())
        )
        for name, module in model.named_modules()
        if name in layer_names
    ]

    counts = {name: {"a_f": None, "a_s": None} for name in layer_names}
    with torch.no_grad():
        for index in range(inputs.shape[0]):
            logits = model(inputs[index : index + 1])
            correct = bool(logits.argmax(dim=1).item() == targets[index].item())
            for name in layer_names:
                active = (captured[name][0] > threshold).to(torch.int32)
                if counts[name]["a_f"] is None:
                    counts[name]["a_f"] = torch.zeros_like(active)
                    counts[name]["a_s"] = torch.zeros_like(active)
                counts[name]["a_s" if correct else "a_f"] += active

    for handle in handles:
        handle.remove()
    return counts


def test_vectorised_counting_matches_a_naive_reference(loader_factory):
    """A random model on random data: batched counting == per-sample counting."""
    from staticsbfl.models import SmallMLP

    torch.manual_seed(0)
    model = SmallMLP(num_classes=10, input_shape=(3, 32, 32))
    inputs = torch.randn(17, 3, 32, 32)
    targets = torch.randint(0, 10, (17,))
    layer_names = [name for name, _ in select_hidden_layers(model)]

    spectrum = compute_hit_spectrum(
        model, loader_factory(inputs, targets, batch_size=5), device=torch.device("cpu")
    )
    reference = _reference_counts(model, inputs, targets, layer_names)

    for name in layer_names:
        assert torch.equal(spectrum.layers[name].a_f, reference[name]["a_f"]), name
        assert torch.equal(spectrum.layers[name].a_s, reference[name]["a_s"]), name


# --------------------------------------------------------------------------
# Invariants that must hold for any input
# --------------------------------------------------------------------------


@pytest.fixture
def random_spectrum(loader_factory):
    """A spectrum over a random MLP and random data, for the invariant checks."""
    from staticsbfl.models import SmallMLP

    torch.manual_seed(7)
    model = SmallMLP(num_classes=10, input_shape=(3, 32, 32))
    inputs = torch.randn(24, 3, 32, 32)
    targets = torch.randint(0, 10, (24,))
    return compute_hit_spectrum(
        model, loader_factory(inputs, targets, batch_size=7), device=torch.device("cpu")
    )


def test_counts_never_leave_their_valid_range(random_spectrum):
    """0 <= a_f <= N_f and 0 <= a_s <= N_s, everywhere, as one reduction each."""
    for layer in random_spectrum.layers.values():
        assert torch.all(layer.a_f >= 0) and torch.all(layer.a_f <= layer.n_failing)
        assert torch.all(layer.a_s >= 0) and torch.all(layer.a_s <= layer.n_successful)
        assert torch.all(layer.n_f >= 0) and torch.all(layer.n_s >= 0)
        assert torch.all(layer.a_f + layer.a_s <= layer.n_samples)


def test_the_four_counts_sum_to_the_sample_count(random_spectrum):
    """a_f + a_s + n_f + n_s == N for every neuron -- the partition holds."""
    for layer in random_spectrum.layers.values():
        total = layer.a_f + layer.a_s + layer.n_f + layer.n_s
        assert torch.all(total == layer.n_samples)


def test_the_outcome_totals_add_up(random_spectrum):
    assert random_spectrum.n_failing + random_spectrum.n_successful == 24
    assert random_spectrum.n_samples == 24


def test_counting_is_independent_of_the_batch_size(loader_factory):
    """Batching is an implementation detail; it must not change a single count.

    The strongest single check on the accumulation logic: any error in the
    broadcasting or the in-place update would show up as a batch-size effect.
    """
    from staticsbfl.models import SmallMLP

    torch.manual_seed(3)
    model = SmallMLP()
    inputs = torch.randn(20, 3, 32, 32)
    targets = torch.randint(0, 10, (20,))

    spectra = [
        compute_hit_spectrum(
            model, loader_factory(inputs, targets, batch_size=size), device=torch.device("cpu")
        )
        for size in (1, 3, 20)
    ]

    for other in spectra[1:]:
        assert other.n_failing == spectra[0].n_failing
        for name, layer in spectra[0].layers.items():
            assert torch.equal(layer.a_f, other.layers[name].a_f), name
            assert torch.equal(layer.a_s, other.layers[name].a_s), name


def test_raising_the_threshold_can_only_reduce_the_counts(loader_factory):
    """Fewer neurons pass a stricter activation bar -- monotone by construction."""
    from staticsbfl.models import SmallMLP

    torch.manual_seed(5)
    model = SmallMLP()
    inputs = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    loader = lambda: loader_factory(inputs, targets, batch_size=4)

    low = compute_hit_spectrum(model, loader(), device=torch.device("cpu"), threshold=0.0)
    high = compute_hit_spectrum(model, loader(), device=torch.device("cpu"), threshold=0.5)

    for name, layer in low.layers.items():
        assert torch.all(high.layers[name].a_f <= layer.a_f), name
        assert torch.all(high.layers[name].a_s <= layer.a_s), name


def test_max_batches_stops_the_pass_early(scripted_mlp, loader_factory):
    model = scripted_mlp()
    inputs = torch.randn(12, 4)
    targets = torch.randint(0, 2, (12,))

    spectrum = compute_hit_spectrum(
        model,
        loader_factory(inputs, targets, batch_size=3),
        device=torch.device("cpu"),
        max_batches=2,
    )
    assert spectrum.n_samples == 6


def test_an_empty_loader_is_an_error(scripted_mlp, loader_factory):
    model = scripted_mlp()
    empty = loader_factory(torch.randn(0, 4), torch.zeros(0, dtype=torch.long))
    with pytest.raises(ValueError, match="no batches"):
        compute_hit_spectrum(model, empty, device=torch.device("cpu"))


# --------------------------------------------------------------------------
# Hooks are cleaned up
# --------------------------------------------------------------------------


def test_the_recorder_removes_its_hooks_even_on_failure(scripted_mlp):
    """A raised exception must not leave forward hooks attached to the model."""
    model = scripted_mlp()
    layers = select_hidden_layers(model)

    with pytest.raises(RuntimeError):
        with ActivationRecorder(model, layers):
            raise RuntimeError("boom")

    assert all(not module._forward_hooks for _name, module in layers)


def test_the_spectrum_leaves_no_hooks_behind(scripted_mlp, loader_factory):
    model = scripted_mlp()
    compute_hit_spectrum(
        model,
        loader_factory(torch.randn(4, 4), torch.randint(0, 2, (4,)), batch_size=2),
        device=torch.device("cpu"),
    )
    assert all(not module._forward_hooks for _name, module in select_hidden_layers(model))
