"""Suspiciousness metric tests.

Three layers of checking:

1. *Worked examples* -- spectra small enough to compute by hand, so a wrong
   formula fails rather than merely a changed one.
2. *Degenerate cases* -- the neuron that fires on everything, on nothing, only
   on failures, only on successes. These pin the corners of each metric.
3. *Properties* -- bounds, sign and finiteness over randomly generated spectra,
   checked as a single boolean reduction over the whole tensor.
"""

import math

import pytest
import torch

from staticsbfl.suspiciousness.metrics import (
    BOUNDED_METRICS,
    METRIC_NAMES,
    METRICS,
    dstar,
    ochiai,
    score_layer,
    score_spectrum,
    tarantula,
)
from staticsbfl.suspiciousness.spectrum import HitSpectrum, LayerSpectrum


# --------------------------------------------------------------------------
# Derived counts
# --------------------------------------------------------------------------


def test_inactive_counts_are_derived_from_the_totals(spectrum_factory):
    """n_f and n_s follow from the run totals rather than being stored."""
    spectrum = spectrum_factory(a_f=[3, 0, 5], a_s=[2, 0, 7], n_failing=5, n_successful=7)

    assert spectrum.n_f.tolist() == [2, 5, 0]
    assert spectrum.n_s.tolist() == [5, 7, 0]
    # The identity the optimisation rests on.
    assert torch.equal(spectrum.a_f + spectrum.n_f, torch.full((3,), 5, dtype=torch.int32))
    assert torch.equal(spectrum.a_s + spectrum.n_s, torch.full((3,), 7, dtype=torch.int32))
    assert spectrum.n_samples == 12


# --------------------------------------------------------------------------
# Worked examples
# --------------------------------------------------------------------------


def test_worked_example_matches_hand_computation(spectrum_factory):
    """One neuron, counts chosen so every metric can be written out by hand.

    a_f = 3, a_s = 2, N_f = 5, N_s = 10  =>  n_f = 2, n_s = 8.

        tarantula = (3/5) / (3/5 + 2/10)        = 0.6 / 0.8   = 0.75
        ochiai    = 3 / sqrt(5 * 5)             = 3 / 5       = 0.6
        D^3       = 27 / (2 + 2)                = 27 / 4      = 6.75
    """
    spectrum = spectrum_factory(a_f=[3], a_s=[2], n_failing=5, n_successful=10)

    assert tarantula(spectrum).item() == pytest.approx(0.75)
    assert ochiai(spectrum).item() == pytest.approx(0.6)
    assert dstar(spectrum, star=3).item() == pytest.approx(6.75)


def test_dstar_exponent_changes_only_the_numerator(spectrum_factory):
    """D^2 and D^3 differ by exactly one further factor of a_f."""
    spectrum = spectrum_factory(a_f=[3], a_s=[2], n_failing=5, n_successful=10)

    assert dstar(spectrum, star=2).item() == pytest.approx(9 / 4)
    assert dstar(spectrum, star=3).item() == pytest.approx(27 / 4)
    assert dstar(spectrum, star=1).item() == pytest.approx(3 / 4)


# --------------------------------------------------------------------------
# Degenerate cases -- the corners of each metric
# --------------------------------------------------------------------------


def test_perfectly_suspicious_neuron(spectrum_factory):
    """Fires on every failure and on no success: the maximum of each metric.

    Tarantula and Ochiai reach their bound of 1. D*'s denominator is
    ``a_s + n_f = 0`` with a positive numerator, which is maximal evidence and
    is reported as ``+inf`` rather than being collapsed to 0.
    """
    spectrum = spectrum_factory(a_f=[4], a_s=[0], n_failing=4, n_successful=6)

    assert tarantula(spectrum).item() == pytest.approx(1.0)
    assert ochiai(spectrum).item() == pytest.approx(1.0)
    assert math.isinf(dstar(spectrum).item())


def test_perfectly_innocent_neuron(spectrum_factory):
    """Fires only on successes: every metric bottoms out at 0."""
    spectrum = spectrum_factory(a_f=[0], a_s=[6], n_failing=4, n_successful=6)

    assert tarantula(spectrum).item() == pytest.approx(0.0)
    assert ochiai(spectrum).item() == pytest.approx(0.0)
    assert dstar(spectrum).item() == pytest.approx(0.0)


def test_neuron_that_never_fires_scores_zero_everywhere(spectrum_factory):
    """No evidence at all must not out-rank a neuron that produced some."""
    spectrum = spectrum_factory(a_f=[0], a_s=[0], n_failing=4, n_successful=6)

    for name in METRIC_NAMES:
        value = METRICS[name](spectrum)
        assert value.item() == pytest.approx(0.0), name
        assert torch.isfinite(value).all(), name


def test_always_active_neuron_has_closed_form_scores(spectrum_factory):
    """A neuron firing on every sample: predictable, and not 1.

    With a_f = N_f and a_s = N_s (so n_f = n_s = 0):

        tarantula = 1 / (1 + 1)                     = 0.5
        ochiai    = N_f / sqrt(N_f * (N_f + N_s))   = sqrt(N_f / (N_f + N_s))
        D^3       = N_f^3 / N_s
    """
    n_failing, n_successful = 4, 6
    spectrum = spectrum_factory(
        a_f=[n_failing], a_s=[n_successful], n_failing=n_failing, n_successful=n_successful
    )

    assert tarantula(spectrum).item() == pytest.approx(0.5)
    assert ochiai(spectrum).item() == pytest.approx(
        math.sqrt(n_failing / (n_failing + n_successful))
    )
    assert dstar(spectrum).item() == pytest.approx(n_failing**3 / n_successful)


def test_a_model_without_failures_scores_zero_everywhere(spectrum_factory):
    """N_f = 0: nothing can be blamed, so nothing is."""
    spectrum = spectrum_factory(a_f=[0, 0], a_s=[3, 7], n_failing=0, n_successful=7)

    for name in METRIC_NAMES:
        value = METRICS[name](spectrum)
        assert torch.all(value == 0), name


def test_tarantula_is_one_half_when_the_neuron_is_outcome_blind(spectrum_factory):
    """Equal activation *rates* on both outcomes means no signal, i.e. 0.5.

    The rates matter, not the raw counts: 2/4 failing and 5/10 successful are
    both 50%, so the neuron carries no information about the outcome.
    """
    spectrum = spectrum_factory(a_f=[2], a_s=[5], n_failing=4, n_successful=10)
    assert tarantula(spectrum).item() == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Properties over random spectra
# --------------------------------------------------------------------------


def _random_spectrum(n_neurons: int = 512, seed: int = 0) -> LayerSpectrum:
    """A random but *valid* spectrum: 0 <= a_f <= N_f and 0 <= a_s <= N_s."""
    generator = torch.Generator().manual_seed(seed)
    n_failing, n_successful = 37, 113
    return LayerSpectrum(
        name="random",
        a_f=torch.randint(0, n_failing + 1, (n_neurons,), generator=generator, dtype=torch.int32),
        a_s=torch.randint(0, n_successful + 1, (n_neurons,), generator=generator, dtype=torch.int32),
        n_failing=n_failing,
        n_successful=n_successful,
    )


@pytest.mark.parametrize("metric", BOUNDED_METRICS)
def test_bounded_metrics_stay_within_the_unit_interval(metric):
    """Tarantula and Ochiai in [0, 1], asserted as one reduction over the tensor."""
    values = METRICS[metric](_random_spectrum())

    assert torch.all((values >= 0.0) & (values <= 1.0)), (
        f"{metric} left [0, 1]: min {values.min()}, max {values.max()}"
    )
    assert torch.isfinite(values).all()


def test_dstar_is_non_negative_and_may_be_unbounded():
    """D* has no upper bound, but it is never negative and never NaN."""
    values = dstar(_random_spectrum())

    assert torch.all(values >= 0.0)
    assert not torch.isnan(values).any()


@pytest.mark.parametrize("metric", METRIC_NAMES)
def test_metrics_preserve_the_layer_shape(metric):
    """A score tensor keeps the neuron shape of the layer it describes."""
    shape = (6, 4, 4)
    spectrum = LayerSpectrum(
        name="conv",
        a_f=torch.randint(0, 5, shape, dtype=torch.int32),
        a_s=torch.randint(0, 9, shape, dtype=torch.int32),
        n_failing=4,
        n_successful=8,
    )
    assert METRICS[metric](spectrum).shape == shape


@pytest.mark.parametrize("metric", METRIC_NAMES)
def test_every_metric_rises_with_the_failing_activations(metric, spectrum_factory):
    """Holding a_s and the totals fixed, more a_f must never lower the score.

    This is the defining direction of all three metrics: firing on more of the
    failures is more incriminating, never less. Note that raising a_f also
    lowers n_f = N_f - a_f, which is why it moves D*'s numerator and denominator
    at the same time.
    """
    spectrum = spectrum_factory(
        a_f=[0, 1, 2, 3, 4], a_s=[2, 2, 2, 2, 2], n_failing=4, n_successful=6
    )
    values = METRICS[metric](spectrum)

    differences = values[1:] - values[:-1]
    assert torch.all(differences >= 0), f"{metric} is not monotone in a_f: {values}"


@pytest.mark.parametrize("metric", METRIC_NAMES)
def test_every_metric_falls_with_the_successful_activations(metric, spectrum_factory):
    """Holding a_f and the totals fixed, more a_s must never raise the score.

    The mirror image of the previous property: a neuron that also fires on the
    successes is doing less to explain the failures.
    """
    spectrum = spectrum_factory(
        a_f=[3, 3, 3, 3], a_s=[0, 2, 4, 6], n_failing=4, n_successful=6
    )
    values = METRICS[metric](spectrum)

    differences = values[1:] - values[:-1]
    assert torch.all(differences <= 0), f"{metric} is not antitone in a_s: {values}"


# --------------------------------------------------------------------------
# Dispatch and validation
# --------------------------------------------------------------------------


def test_score_layer_returns_every_requested_metric(spectrum_factory):
    spectrum = spectrum_factory(a_f=[3], a_s=[2], n_failing=5, n_successful=10)
    scores = score_layer(spectrum)
    assert set(scores) == set(METRIC_NAMES)

    subset = score_layer(spectrum, metrics=("ochiai",))
    assert set(subset) == {"ochiai"}


def test_score_layer_rejects_an_unknown_metric(spectrum_factory):
    spectrum = spectrum_factory(a_f=[1], a_s=[1], n_failing=2, n_successful=2)
    with pytest.raises(ValueError, match="Unknown metric"):
        score_layer(spectrum, metrics=("jaccard",))


@pytest.mark.parametrize("star", [0, -1, 2.5, True, "3"])
def test_dstar_rejects_a_non_positive_integer_exponent(spectrum_factory, star):
    spectrum = spectrum_factory(a_f=[1], a_s=[1], n_failing=2, n_successful=2)
    with pytest.raises(ValueError, match="positive integer"):
        dstar(spectrum, star=star)


def test_score_spectrum_covers_every_layer_and_metric(spectrum_factory):
    """The whole-model entry point maps {metric: {layer: tensor of layer shape}}."""
    spectrum = HitSpectrum(
        layers={
            "fc1": spectrum_factory([1, 2, 3], [4, 5, 6], 4, 8, name="fc1"),
            "conv1": LayerSpectrum(
                name="conv1",
                a_f=torch.ones((2, 3, 3), dtype=torch.int32),
                a_s=torch.ones((2, 3, 3), dtype=torch.int32),
                n_failing=4,
                n_successful=8,
            ),
        },
        n_failing=4,
        n_successful=8,
    )

    scores = score_spectrum(spectrum)

    assert set(scores) == set(METRIC_NAMES)
    for per_layer in scores.values():
        assert set(per_layer) == {"fc1", "conv1"}
        assert per_layer["fc1"].shape == (3,)
        assert per_layer["conv1"].shape == (2, 3, 3)
