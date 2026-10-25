"""One set of counting invariants, applied to every function that counts.

One entry point produces a :class:`HitSpectrum` -- the reference counter
:func:`compute_hit_spectrum` -- and the same arithmetic has to hold for it, and
for any counter added later:

    0 <= a_f <= N_f            0 <= a_s <= N_s
    a_f + n_f == N_f           a_s + n_s == N_s
    a_f + a_s + n_f + n_s == N          for every single unit
    N_f + N_s == N == |dataset|

None of that is deep, and that is exactly why it is worth testing: a count that
drifts above the sample total, or a partition that stops adding up, produces
scores that still *look* like scores. Tarantula stays inside [0, 1] under quite
badly broken counts; the ranking is simply wrong, silently.

Two kinds of case are used together, because they fail differently.

*Scripted* models pin the degenerate corners where an off-by-one lives: every
unit always active, no unit ever active, every sample passing, every sample
failing, and a batch size that does not divide the dataset. Their expected
counts are known before the code runs, so these assert exact values, not just
consistency.

*Randomised* sweeps over seeds, sizes and batch boundaries cover the cases
nobody thought to script. They can only assert the invariants -- there is no
independent expected value -- but that is precisely their job.

:func:`test_every_counting_function_is_covered` is the guard that keeps this
honest: it discovers the counting functions by their return annotation and fails
if one is added without being registered here.
"""

import inspect
import itertools

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from staticsbfl.suspiciousness import spectrum as spectrum_module
from staticsbfl.suspiciousness.metrics import score_spectrum
from staticsbfl.suspiciousness.spectrum import (
    COUNT_DTYPE, HitSpectrum, compute_hit_spectrum, update_counts,
)

CPU = torch.device("cpu")


# --------------------------------------------------------------------------
# The shared checker
# --------------------------------------------------------------------------


def assert_count_invariants(spectrum: HitSpectrum, expected_samples: int) -> None:
    """Every arithmetic property a hit spectrum must satisfy, for every unit.

    Asserted per element rather than on a reduction: a mean or a sum could mask
    one unit that overshot by exactly as much as another undershot.
    """
    assert spectrum.n_failing >= 0 and spectrum.n_successful >= 0
    assert spectrum.n_failing + spectrum.n_successful == spectrum.n_samples
    assert spectrum.n_samples == expected_samples, (
        f"counted {spectrum.n_samples} samples, expected {expected_samples}"
    )
    assert spectrum.layers, "a spectrum with no layers proves nothing"

    for name, layer in spectrum.layers.items():
        a_f, a_s, n_f, n_s = layer.a_f, layer.a_s, layer.n_f, layer.n_s

        assert a_f.dtype == COUNT_DTYPE and a_s.dtype == COUNT_DTYPE, name
        assert a_f.shape == a_s.shape == layer.neuron_shape, name

        # Non-negative, and bounded by their own outcome total. A count above
        # its total would mean a sample was folded in twice.
        assert torch.all(a_f >= 0), f"{name}: negative a_f (overflow?)"
        assert torch.all(a_s >= 0), f"{name}: negative a_s (overflow?)"
        assert torch.all(a_f <= layer.n_failing), f"{name}: a_f exceeds N_f"
        assert torch.all(a_s <= layer.n_successful), f"{name}: a_s exceeds N_s"

        # The derived halves, which is where a wrong total would surface.
        assert torch.all(n_f >= 0), f"{name}: negative n_f"
        assert torch.all(n_s >= 0), f"{name}: negative n_s"
        assert torch.all(n_f <= layer.n_failing), f"{name}: n_f exceeds N_f"
        assert torch.all(n_s <= layer.n_successful), f"{name}: n_s exceeds N_s"

        # The two partitions, per unit.
        assert torch.all(a_f + n_f == layer.n_failing), f"{name}: a_f + n_f != N_f"
        assert torch.all(a_s + n_s == layer.n_successful), f"{name}: a_s + n_s != N_s"

        # The four counts partition the dataset, per unit.
        assert torch.all(a_f + a_s + n_f + n_s == layer.n_samples), f"{name}: four counts"
        assert torch.all(a_f + a_s <= layer.n_samples), f"{name}: active exceeds N"
        assert layer.n_samples == expected_samples, name


def assert_scores_are_well_formed(spectrum: HitSpectrum) -> None:
    """The bounded metrics stay bounded and nothing is NaN.

    Included here because a broken count is most likely to be *noticed* through a
    score, and the point of this file is that it must be caught before that.
    """
    scores = score_spectrum(spectrum, star=3)
    for metric in ("tarantula", "ochiai"):
        for name, values in scores[metric].items():
            assert not torch.isnan(values).any(), f"{metric}/{name}: NaN"
            assert torch.all(values >= 0.0) and torch.all(values <= 1.0), f"{metric}/{name}"
    for name, values in scores["dstar"].items():
        assert not torch.isnan(values).any(), f"dstar/{name}: NaN"
        assert torch.all(values >= 0.0), f"dstar/{name}: negative"


# --------------------------------------------------------------------------
# Every counting entry point, behind one signature
# --------------------------------------------------------------------------


def _count_reference(model, make_loader, layers, **kwargs):
    return compute_hit_spectrum(model, make_loader(), device=CPU, layer_names=layers, **kwargs)


#: Name -> counter. Adding a counting function without adding it here fails
#: :func:`test_every_counting_function_is_covered`.
COUNTERS = {
    "compute_hit_spectrum": _count_reference,
}

#: Counters that accept a scalar ``threshold``.
FIXED_THRESHOLD_COUNTERS = ("compute_hit_spectrum",)

every_counter = pytest.mark.parametrize("counter_name", sorted(COUNTERS))


def test_every_counting_function_is_covered():
    """Discover the counting functions and fail if one is not registered.

    A function that returns a ``HitSpectrum`` is a counting function, whatever it
    is called. Finding them by return annotation rather than by name means a
    second one cannot be added quietly and stay untested.
    """
    found = set()
    for module in (spectrum_module,):
        for name, function in vars(module).items():
            if not inspect.isfunction(function) or name.startswith("_"):
                continue
            if getattr(function, "__module__", None) != module.__name__:
                continue
            annotation = inspect.signature(function).return_annotation
            if annotation in (HitSpectrum, "HitSpectrum"):
                found.add(name)
    assert found, "the discovery itself is broken if it finds nothing"
    assert found == set(COUNTERS), (
        f"counting functions not covered by these invariants: {sorted(found - set(COUNTERS))}; "
        f"registered but no longer found: {sorted(set(COUNTERS) - found)}"
    )


# --------------------------------------------------------------------------
# Scripted corners: exact counts, not merely consistent ones
# --------------------------------------------------------------------------


def _loader_factory(inputs, targets, batch_size):
    def make():
        return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)
    return make


@every_counter
@pytest.mark.parametrize("n_samples,batch_size", [
    (8, 8),    # one full batch
    (8, 3),    # a short final batch of 2
    (7, 1),    # every sample its own batch
    (5, 16),   # batch larger than the dataset
    (1, 1),    # a single sample
])
def test_every_unit_active_on_every_sample_counts_exactly_the_totals(
    counter_name, n_samples, batch_size, scripted_mlp
):
    """hidden_bias=+1 drives every unit to a constant +1 response.

    The saturated corner, and the one where an accumulator that folded a batch in
    twice would first show up, because there is no inactive sample to absorb the
    error. The two threshold rules give *opposite* answers here, and both are
    correct: against a fixed zero threshold every unit is active on every sample,
    so a_f = N_f and a_s = N_s; against its own dataset mean a constant response
    is never strictly greater, so nothing is active at all. That second case is
    the documented "constant responses, including all-zero and identical
    all-one inputs, produce no hits" rule, and it is worth pinning here rather
    than only in the convolution tests.
    """
    model = scripted_mlp(hidden_bias=1.0)
    model.set_prediction(0)
    inputs = torch.randn(n_samples, 4)
    targets = torch.arange(n_samples) % 2  # a mix of passes and failures
    spectrum = COUNTERS[counter_name](
        model, _loader_factory(inputs, targets, batch_size), ("fc1", "fc2")
    )
    assert_count_invariants(spectrum, n_samples)

    saturated = counter_name in FIXED_THRESHOLD_COUNTERS
    for layer in spectrum.layers.values():
        if saturated:
            assert torch.all(layer.a_f == layer.n_failing)
            assert torch.all(layer.a_s == layer.n_successful)
            assert torch.all(layer.n_f == 0) and torch.all(layer.n_s == 0)
        else:
            assert torch.all(layer.a_f == 0) and torch.all(layer.a_s == 0)
            assert torch.all(layer.n_f == layer.n_failing)
            assert torch.all(layer.n_s == layer.n_successful)


@every_counter
@pytest.mark.parametrize("n_samples,batch_size", [(8, 3), (6, 6), (5, 1)])
def test_no_unit_ever_active_counts_exactly_zero(
    counter_name, n_samples, batch_size, scripted_mlp
):
    """hidden_bias=-1 never fires, so a_f=a_s=0 and n_f/n_s carry the totals.

    The dead corner, and the one that separates 'no evidence' from 'evidence of
    innocence': the four counts must still partition the dataset.
    """
    model = scripted_mlp(hidden_bias=-1.0)
    model.set_prediction(0)
    inputs = torch.randn(n_samples, 4)
    targets = torch.arange(n_samples) % 2
    spectrum = COUNTERS[counter_name](
        model, _loader_factory(inputs, targets, batch_size), ("fc1", "fc2")
    )
    assert_count_invariants(spectrum, n_samples)
    for layer in spectrum.layers.values():
        assert torch.all(layer.a_f == 0) and torch.all(layer.a_s == 0)
        assert torch.all(layer.n_f == layer.n_failing)
        assert torch.all(layer.n_s == layer.n_successful)


@every_counter
@pytest.mark.parametrize("outcome", ["all_pass", "all_fail"])
def test_a_one_sided_outcome_still_partitions_the_dataset(counter_name, outcome, scripted_mlp):
    """N_f = 0 or N_s = 0: degenerate for the metrics, but the counts must hold.

    These runs are excluded from the comparative analysis -- a spectrum with no
    failures carries no failure evidence -- but they still have to be *counted*
    correctly, and the derived n_f/n_s must not go negative when their total is
    zero.
    """
    model = scripted_mlp(hidden_bias=1.0)
    model.set_prediction(0)
    n_samples = 6
    inputs = torch.randn(n_samples, 4)
    targets = torch.zeros(n_samples, dtype=torch.long) if outcome == "all_pass" else \
        torch.ones(n_samples, dtype=torch.long)

    spectrum = COUNTERS[counter_name](
        model, _loader_factory(inputs, targets, 4), ("fc1", "fc2")
    )
    assert_count_invariants(spectrum, n_samples)
    if outcome == "all_pass":
        assert spectrum.n_failing == 0 and spectrum.n_successful == n_samples
        for layer in spectrum.layers.values():
            assert torch.all(layer.a_f == 0) and torch.all(layer.n_f == 0)
    else:
        assert spectrum.n_successful == 0 and spectrum.n_failing == n_samples
        for layer in spectrum.layers.values():
            assert torch.all(layer.a_s == 0) and torch.all(layer.n_s == 0)


@every_counter
def test_a_hand_written_activation_pattern_gives_hand_written_counts(
    counter_name, passthrough_net
):
    """fc1 is the identity, so the activation pattern is written in the input.

    Four samples, five hidden units. Unit j is active on sample i exactly when
    ``inputs[i, j] > 0``, so the expected counts can be written down by hand.
    """
    model = passthrough_net(n_hidden=5, num_classes=2)
    #                      u0    u1    u2    u3    u4   class
    inputs = torch.tensor([
        [1.0, -1.0,  1.0, -1.0,  1.0, 0.0],   # predicted 0
        [1.0,  1.0, -1.0, -1.0,  1.0, 1.0],   # predicted 1
        [-1.0, -1.0, -1.0, 1.0,  1.0, 0.0],   # predicted 0
        [1.0,  1.0,  1.0,  1.0,  1.0, 1.0],   # predicted 1
    ])
    targets = torch.tensor([0, 0, 1, 1])       # pass, fail, fail, pass
    spectrum = COUNTERS[counter_name](model, _loader_factory(inputs, targets, 2), ("fc1",))
    assert_count_invariants(spectrum, 4)
    assert spectrum.n_failing == 2 and spectrum.n_successful == 2

    if counter_name in FIXED_THRESHOLD_COUNTERS:
        layer = spectrum.layers["fc1"]
        # active where the input is positive; samples 0 and 3 pass, 1 and 2 fail.
        assert layer.a_s.tolist() == [2, 1, 2, 1, 2]
        assert layer.a_f.tolist() == [1, 1, 0, 1, 2]


@every_counter
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 8])
def test_the_counts_do_not_depend_on_where_the_batch_boundaries_fall(
    counter_name, batch_size, passthrough_net
):
    """Batching must be invisible, including when the final batch is short."""
    model = passthrough_net(n_hidden=4, num_classes=2)
    torch.manual_seed(11)
    inputs = torch.randn(7, 5)
    inputs[:, -1] = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    targets = torch.tensor([0, 0, 1, 1, 0, 1, 1])

    reference = COUNTERS[counter_name](model, _loader_factory(inputs, targets, 7), ("fc1",))
    other = COUNTERS[counter_name](model, _loader_factory(inputs, targets, batch_size), ("fc1",))
    assert_count_invariants(other, 7)
    assert other.n_failing == reference.n_failing
    assert torch.equal(other.layers["fc1"].a_f, reference.layers["fc1"].a_f)
    assert torch.equal(other.layers["fc1"].a_s, reference.layers["fc1"].a_s)


# --------------------------------------------------------------------------
# Randomised sweeps: the cases nobody scripted
# --------------------------------------------------------------------------


@every_counter
@pytest.mark.parametrize("seed,n_samples,batch_size", list(itertools.product(
    (0, 1, 2), (1, 5, 16), (1, 3, 16),
)))
def test_random_dense_models_satisfy_the_invariants(
    counter_name, seed, n_samples, batch_size, scripted_mlp
):
    torch.manual_seed(seed)
    model = scripted_mlp()
    with torch.no_grad():  # undo the scripting: random weights, random firing
        for layer in (model.fc1, model.fc2, model.head):
            layer.weight.normal_()
            layer.bias.normal_()
    inputs = torch.randn(n_samples, 4)
    targets = torch.randint(0, 2, (n_samples,))
    spectrum = COUNTERS[counter_name](
        model, _loader_factory(inputs, targets, batch_size), ("fc1", "fc2")
    )
    assert_count_invariants(spectrum, n_samples)
    assert_scores_are_well_formed(spectrum)


@pytest.mark.parametrize("counter_name", FIXED_THRESHOLD_COUNTERS)
@pytest.mark.parametrize("threshold", [-5.0, -0.5, 0.0, 0.5, 5.0])
def test_the_invariants_hold_at_every_threshold(counter_name, threshold, passthrough_net):
    """Including thresholds so extreme that every unit is dead or saturated."""
    model = passthrough_net(n_hidden=4, num_classes=2)
    torch.manual_seed(5)
    inputs = torch.randn(8, 5)
    inputs[:, -1] = torch.randint(0, 2, (8,)).float()
    targets = torch.randint(0, 2, (8,))
    spectrum = COUNTERS[counter_name](
        model, _loader_factory(inputs, targets, 3), ("fc1",), threshold=threshold
    )
    assert_count_invariants(spectrum, 8)
    assert_scores_are_well_formed(spectrum)


# --------------------------------------------------------------------------
# The accumulator itself, below the driver
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_batches,batch_size", [(1, 4), (3, 2), (5, 1)])
def test_repeated_accumulation_never_exceeds_the_samples_folded_in(n_batches, batch_size):
    """The accumulator, driven directly, over many batches.

    Checks the property the driver relies on: after folding K samples in, no
    counter exceeds K, and a_s + a_f equals the number of samples on which the
    unit was active -- recomputed here by brute force rather than assumed.
    """
    torch.manual_seed(7)
    shape = (3, 2)
    a_f = torch.zeros(shape, dtype=COUNT_DTYPE)
    a_s = torch.zeros(shape, dtype=COUNT_DTYPE)

    all_active, all_success = [], []
    for _ in range(n_batches):
        active = torch.rand(batch_size, *shape) > 0.5
        success = torch.rand(batch_size) > 0.5
        all_active.append(active)
        all_success.append(success)
        update_counts(a_f, a_s, active, success)

    total = n_batches * batch_size
    active = torch.cat(all_active)
    success = torch.cat(all_success)
    assert torch.all(a_f >= 0) and torch.all(a_s >= 0)
    assert torch.all(a_f <= total) and torch.all(a_s <= total)
    assert torch.all(a_f + a_s <= total)
    # Brute force, per unit.
    expected_s = (active & success.view(-1, 1, 1)).sum(dim=0, dtype=COUNT_DTYPE)
    expected_f = (active & ~success.view(-1, 1, 1)).sum(dim=0, dtype=COUNT_DTYPE)
    assert torch.equal(a_s, expected_s)
    assert torch.equal(a_f, expected_f)
    assert torch.equal(a_f + a_s, active.sum(dim=0, dtype=COUNT_DTYPE))


def test_a_spectrum_with_more_counts_than_samples_would_be_caught():
    """The checker has to be able to fail, or it is decoration.

    A hand-built spectrum with a_f one above its failure total must be rejected,
    and so must one whose per-unit partition does not add up.
    """
    from staticsbfl.suspiciousness.spectrum import LayerSpectrum

    broken = HitSpectrum(
        layers={"fc1": LayerSpectrum(
            name="fc1",
            a_f=torch.tensor([3], dtype=COUNT_DTYPE),   # only 2 failures exist
            a_s=torch.tensor([1], dtype=COUNT_DTYPE),
            n_failing=2, n_successful=2,
        )},
        n_failing=2, n_successful=2,
    )
    with pytest.raises(AssertionError, match="a_f exceeds N_f"):
        assert_count_invariants(broken, 4)

    miscounted = HitSpectrum(
        layers={"fc1": LayerSpectrum(
            name="fc1", a_f=torch.tensor([1], dtype=COUNT_DTYPE),
            a_s=torch.tensor([1], dtype=COUNT_DTYPE), n_failing=2, n_successful=2,
        )},
        n_failing=2, n_successful=3,  # totals disagree with the layers
    )
    with pytest.raises(AssertionError):
        assert_count_invariants(miscounted, 4)
