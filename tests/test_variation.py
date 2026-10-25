"""Architecture-variation tests.

The separation between config and builder is what makes these possible: a
configuration is a plain frozen object drawn by a pure, seeded function, so
"does every sampled architecture stay inside its bounds?" is answerable without
building or training anything. That property is checked exhaustively over large
pools here -- it is cheap precisely because no torch model is involved.
"""

import random

import pytest
import torch

from staticsbfl.variation.phases import POPULATION_SIZES
from staticsbfl.variation import (
    ACTIVATIONS,
    PHASE3_BASE,
    PHASE3_EXTENSIONS,
    ArchitectureConfig,
    BoundsError,
    VariableMLP,
    VariationBounds,
    build_model,
    build_phase,
    make_activation,
    moved_axes,
    phase1_configs,
    phase2_configs,
    phase3_configs,
    PHASE_NAMES,
    DESIGNED_ARCHITECTURES,
    sample_config_pool,
    sample_widths,
    with_seed,
)


# --------------------------------------------------------------------------
# Bounds validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"min_depth": 0}, "min_depth"),
        ({"min_depth": 5, "max_depth": 2}, "max_depth"),
        ({"min_width": 0}, "min_width"),
        ({"min_width": 100, "max_width": 50}, "max_width"),
        ({"activations": ()}, "At least one activation"),
        ({"activations": ("swish",)}, "Unknown activation"),
        ({"family": "transformer"}, "Unknown family"),
    ],
)
def test_contradictory_bounds_are_rejected_at_construction(kwargs, message):
    """Bad bounds fail immediately, not once a model somewhere fails to build."""
    with pytest.raises(BoundsError, match=message):
        VariationBounds(**kwargs)


def test_a_single_point_region_is_valid():
    """Phase 1 needs min == max on every axis; that must not be an error."""
    bounds = VariationBounds(
        min_depth=3, max_depth=3, min_width=64, max_width=64, activations=("relu",)
    )
    assert bounds.describe()


# --------------------------------------------------------------------------
# The central property: sampled configs stay inside their bounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bounds",
    [
        VariationBounds(),
        VariationBounds(min_depth=1, max_depth=1, min_width=32, max_width=32),
        VariationBounds(min_depth=6, max_depth=10, min_width=50, max_width=100),
        VariationBounds(activations=("tanh", "sigmoid")),
    ],
)
def test_every_sampled_config_lies_inside_its_bounds(bounds):
    """The property the whole config/builder split exists to make testable."""
    pool = sample_config_pool(bounds, 200, rng_seed=1, unique=False)

    for config in pool:
        assert bounds.contains(config), (
            f"{config.key} escaped its region: {bounds.violations(config)}"
        )


def test_sampling_covers_the_full_admissible_range():
    """Bounds must be reachable, not merely respected.

    A sampler that always returned the same midpoint would pass the containment
    test above while making the whole variation study vacuous.
    """
    bounds = VariationBounds(min_depth=1, max_depth=10, min_width=32, max_width=512)
    pool = sample_config_pool(bounds, 500, rng_seed=2, unique=False)

    depths = {c.depth for c in pool}
    widths = {w for c in pool for w in c.hidden_widths}
    activations = {c.activation for c in pool}

    assert min(depths) == bounds.min_depth and max(depths) == bounds.max_depth
    assert min(widths) < bounds.min_width + 20
    assert max(widths) > bounds.max_width - 20
    assert activations == set(ACTIVATIONS)


def test_widths_vary_within_a_single_model():
    """Independent per-layer widths, not one constant repeated.

    If width were constant within a model, a per-neuron predictor's width
    feature could only ever explain between-model variance.
    """
    bounds = VariationBounds(min_depth=5, max_depth=5, min_width=32, max_width=512)
    pool = sample_config_pool(bounds, 50, rng_seed=3, unique=False)

    varied = [c for c in pool if len(set(c.hidden_widths)) > 1]
    assert len(varied) > 45, "widths look constant within models"


def test_depth_respects_a_pinned_value():
    rng = random.Random(0)
    bounds = VariationBounds(min_depth=1, max_depth=10)
    assert len(sample_widths(rng, bounds, depth=4)) == 4


def test_sampling_is_reproducible_from_its_seed():
    bounds = VariationBounds()
    first = sample_config_pool(bounds, 30, rng_seed=7)
    second = sample_config_pool(bounds, 30, rng_seed=7)
    third = sample_config_pool(bounds, 30, rng_seed=8)

    assert first == second
    assert first != third


def test_a_pool_holds_distinct_architectures_by_default():
    bounds = VariationBounds(min_depth=2, max_depth=6, min_width=32, max_width=512)
    pool = sample_config_pool(bounds, 100, rng_seed=4)
    signatures = {(c.hidden_widths, c.activation) for c in pool}
    assert len(signatures) == 100


def test_an_exhausted_region_reports_itself():
    """A region with only a handful of architectures cannot fill a large pool."""
    tiny = VariationBounds(
        min_depth=1, max_depth=1, min_width=10, max_width=12, activations=("relu",)
    )
    with pytest.raises(BoundsError, match="too small"):
        sample_config_pool(tiny, 50, rng_seed=5)


def test_model_seeds_are_distinct_within_a_pool():
    """Two configs sharing a seed would be needlessly correlated."""
    pool = sample_config_pool(VariationBounds(), 40, rng_seed=6)
    assert len({c.seed for c in pool}) == 40


# --------------------------------------------------------------------------
# Derived config properties
# --------------------------------------------------------------------------


def test_depth_and_width_are_read_off_the_same_tuple():
    config = ArchitectureConfig("k", "mlp", (128, 64, 32), "relu", seed=42)
    assert config.depth == 3
    assert config.total_hidden_units == 224


@pytest.mark.parametrize(
    "widths, expected",
    [
        ((128, 64, 32), "narrowing"),
        ((32, 64, 128), "widening"),
        ((64, 128, 32), "mixed"),
        ((64,), "flat"),
        ((64, 64, 64), "narrowing"),  # non-increasing counts as narrowing
    ],
)
def test_the_width_profile_classifies_the_shape(widths, expected):
    assert ArchitectureConfig("k", "mlp", widths, "relu", 42).width_profile == expected


def test_activation_ids_are_stable():
    """Cached feature tables depend on these; they may be appended to, not renumbered."""
    assert ArchitectureConfig("k", "mlp", (8,), "relu", 42).activation_id == 0
    assert [ACTIVATIONS.index(a) for a in ACTIVATIONS] == list(range(len(ACTIVATIONS)))


def test_a_config_row_carries_every_variation_axis():
    row = ArchitectureConfig("k", "mlp", (128, 64), "tanh", 42, "phase2", "train").as_row()
    for field in ("depth", "activation_id", "mean_width", "width_profile", "seed", "split"):
        assert field in row


def test_reseeding_keeps_the_architecture():
    config = ArchitectureConfig("a", "mlp", (128, 64), "relu", 42)
    reseeded = with_seed(config, 43, key="b")
    assert reseeded.hidden_widths == config.hidden_widths
    assert reseeded.seed == 43 and reseeded.key == "b"


# --------------------------------------------------------------------------
# Region overlap -- what phase 3 depends on
# --------------------------------------------------------------------------


def test_regions_are_disjoint_as_soon_as_one_axis_is():
    shallow = VariationBounds(min_depth=1, max_depth=5)
    deep = VariationBounds(min_depth=6, max_depth=10)
    assert not shallow.overlaps(deep)

    narrow = VariationBounds(min_width=50, max_width=100)
    wide = VariationBounds(min_width=101, max_width=200)
    assert not narrow.overlaps(wide)

    relu_like = VariationBounds(activations=("relu", "elu"))
    saturating = VariationBounds(activations=("tanh", "sigmoid"))
    assert not relu_like.overlaps(saturating)


def test_regions_overlapping_on_every_axis_are_reported_as_overlapping():
    assert VariationBounds().overlaps(VariationBounds(min_depth=3, max_depth=5))


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------


def test_phase1_varies_only_the_seed():
    spec = phase1_configs(hidden_widths=(256, 128, 64), n_fit=10, n_eval=10)
    configs = spec.configs()

    assert len({c.hidden_widths for c in configs}) == 1
    assert len({c.activation for c in configs}) == 1
    assert len({c.seed for c in configs}) == 20      # every seed distinct
    assert min(c.seed for c in configs) == 42


def test_phase2_splits_by_model_with_no_architecture_shared():
    """Train and test must not contain the same architecture."""
    spec = phase2_configs(n_fit=25, n_eval=25, rng_seed=1)

    train = {(c.hidden_widths, c.activation) for c in spec.train}
    test = {(c.hidden_widths, c.activation) for c in spec.test}
    assert not (train & test)
    assert len(spec.train) == len(spec.test) == 25


def test_phase2_draws_both_splits_from_the_same_region():
    spec = phase2_configs(n_fit=20, n_eval=20, rng_seed=1)
    for config in spec.configs():
        assert spec.regions[config.split].contains(config)


def test_phase3_populations_stay_in_their_own_region():
    """Every model sits in the region of the group it was drawn for."""
    spec = phase3_configs(
        base_bounds=PHASE3_BASE, extensions=PHASE3_EXTENSIONS, n_fit=20, n_eval=8
    )

    for group, configs in spec.populations.items():
        bounds = spec.regions[group]
        for config in configs:
            assert bounds.contains(config), f"{group}: {bounds.violations(config)}"


def test_phase3_validation_groups_leave_the_base_region():
    """Each validation group must be outside the region the predictor saw."""
    spec = phase3_configs(
        base_bounds=PHASE3_BASE, extensions=PHASE3_EXTENSIONS, n_fit=20, n_eval=8
    )
    for group in spec.evaluation_splits:
        for config in spec.populations[group]:
            assert not PHASE3_BASE.contains(config), f"{group}: {config.key} is inside the base region"


def test_phase3_fits_on_the_base_population_only():
    spec = phase3_configs(
        base_bounds=PHASE3_BASE, extensions=PHASE3_EXTENSIONS, n_fit=10, n_eval=10
    )
    assert spec.train_split == "base"
    assert spec.train == spec.populations["base"]
    assert set(spec.evaluation_splits) == set(PHASE3_EXTENSIONS)
    # test aggregates every evaluation group
    assert len(spec.test) == 10 * len(PHASE3_EXTENSIONS)


def test_phase3_refuses_a_group_that_overlaps_the_base():
    """Otherwise the group would not test extrapolation at all."""
    with pytest.raises(BoundsError, match="overlaps the base region"):
        phase3_configs(
            base_bounds=VariationBounds(),
            extensions={"same": VariationBounds()},
            n_fit=5, n_eval=5,
        )


def test_phase3_refuses_a_group_that_moves_two_axes():
    """A group moving depth AND width cannot attribute a drop to either."""
    base = VariationBounds(min_depth=2, max_depth=4, min_width=64, max_width=128)
    both = VariationBounds(min_depth=5, max_depth=10, min_width=129, max_width=512)
    with pytest.raises(BoundsError, match="moves 2 axes"):
        phase3_configs(base_bounds=base, extensions={"both": both}, n_fit=5, n_eval=5)


@pytest.mark.parametrize("group", sorted(PHASE3_EXTENSIONS))
def test_every_shipped_validation_group_moves_exactly_one_axis(group):
    """The whole point of the base/validation design."""
    assert moved_axes(PHASE3_BASE, PHASE3_EXTENSIONS[group]) != []
    assert len(moved_axes(PHASE3_BASE, PHASE3_EXTENSIONS[group])) == 1


def test_all_validation_groups_share_the_same_base():
    """Groups are comparable with each other only if they share a training region."""
    spec = phase3_configs(
        base_bounds=PHASE3_BASE, extensions=PHASE3_EXTENSIONS, n_fit=5, n_eval=5
    )
    assert all(spec.regions["base"] == PHASE3_BASE for _ in spec.evaluation_splits)


@pytest.mark.parametrize("name", ["phase1", "phase2", "phase3", "phase4"])
@pytest.mark.parametrize("mode", sorted(POPULATION_SIZES))
def test_every_phase_builds_in_every_mode(name, mode):
    """Every mode produces the sizes it declares -- read, not hard-coded.

    Asserting the declared numbers rather than literals means changing a
    population size is a one-line change in one place, which is what made the
    previous literals go stale.
    """
    spec = build_phase(name, mode)
    sizes = POPULATION_SIZES[mode]
    assert len(spec.train) == sizes["fit"]
    for group in spec.evaluation_splits:
        assert len(spec.populations[group]) == sizes["eval"]
    assert spec.question


@pytest.mark.parametrize("name", ["phase1", "phase2", "phase3", "phase4"])
def test_a_larger_mode_extends_a_smaller_one_rather_than_replacing_it(name):
    """Growing a population must re-use what is already trained.

    Every group of a bigger mode has to begin with exactly the group of the
    smaller one, key for key and in order, or raising the size silently
    invalidates every cached model and every spectrum computed from it. Phase 4's
    designed group used to violate this, which is why it is asserted for all of
    them now.
    """
    for smaller, bigger in (("debug", "light"), ("light", "heavy")):
        small = build_phase(name, smaller)
        large = build_phase(name, bigger)
        for group, configs in small.populations.items():
            keys = [c.key for c in configs]
            assert [c.key for c in large.populations[group]][: len(keys)] == keys, (
                f"{name}/{group}: {bigger} does not begin with {smaller}"
            )

            # A key alone is not enough. What the key *names* must be identical
            # too -- the seed and the architecture -- or a cached checkpoint gets
            # re-used for a model it was never trained as. Phase 2 failed exactly
            # here once: the pool length fed the shuffle, so p2_train_000 kept its
            # name and became a different network.
            identity = {
                c.key: (c.seed, c.hidden_widths, c.activation, c.family)
                for c in configs
            }
            for config in large.populations[group]:
                if config.key in identity:
                    assert (
                        config.seed, config.hidden_widths,
                        config.activation, config.family,
                    ) == identity[config.key], f"{name}/{group}/{config.key} changed"


def test_the_fit_population_is_larger_than_each_evaluation_group():
    """Fitting is what needs the data; an eval group only estimates a mean."""
    for name in PHASE_NAMES:
        spec = build_phase(name, "light")
        for group in spec.evaluation_splits:
            assert len(spec.train) > len(spec.populations[group]), name


def test_phase4_evaluates_on_designed_architectures():
    """The answer to 'real networks are designed, not sampled'."""
    spec = build_phase("phase4", "light")

    assert spec.train_split == "sampled"
    assert spec.evaluation_splits == ("designed",)
    designed_shapes = {c.hidden_widths for c in spec.populations["designed"]}
    for _name, widths, _activation in DESIGNED_ARCHITECTURES:
        assert widths in designed_shapes


def test_designed_models_repeat_under_several_seeds():
    """So one unlucky initialisation does not decide the result."""
    spec = build_phase("phase4", "heavy")
    by_shape = {}
    for config in spec.populations["designed"]:
        by_shape.setdefault(config.hidden_widths, set()).add(config.seed)
    assert all(len(seeds) > 1 for seeds in by_shape.values())


def test_designed_architectures_are_not_sampled_ones():
    """Every designed shape is hand-written, so none may be drawn at random."""
    spec = build_phase("phase4", "light")
    sampled = {c.hidden_widths for c in spec.populations["sampled"]}
    designed = {w for _n, w, _a in DESIGNED_ARCHITECTURES}
    assert not (sampled & designed)


def test_model_keys_are_unique_within_a_phase():
    for name in ("phase1", "phase2", "phase3"):
        keys = [c.key for c in build_phase(name, "light").configs()]
        assert len(keys) == len(set(keys)), name


def test_an_unknown_phase_or_mode_is_rejected():
    with pytest.raises(BoundsError, match="Unknown phase"):
        build_phase("phase5")
    with pytest.raises(BoundsError, match="Unknown mode"):
        build_phase("phase1", "medium")


# --------------------------------------------------------------------------
# The builder honours the config and invents nothing
# --------------------------------------------------------------------------


def test_the_built_model_has_exactly_the_configured_shape():
    config = ArchitectureConfig("k", "mlp", (128, 64, 32), "relu", 42)
    model = build_model(config, num_classes=10, input_shape=(3, 32, 32))

    assert model.hidden_layer_names == ("fc0", "fc1", "fc2")
    assert model.fc0.in_features == 3 * 32 * 32
    assert [model.fc0.out_features, model.fc1.out_features, model.fc2.out_features] == [128, 64, 32]
    assert model.head.out_features == 10


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_every_admissible_activation_builds_and_runs(activation):
    config = ArchitectureConfig("k", "mlp", (32, 16), activation, 42)
    model = build_model(config, num_classes=4, input_shape=(3, 8, 8))
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 4)


def test_an_unknown_activation_is_rejected():
    with pytest.raises(ValueError, match="Unknown activation"):
        make_activation("swish")


def test_an_unimplemented_family_says_where_to_add_it():
    config = ArchitectureConfig("k", "lstm", (32,), "relu", 42)
    with pytest.raises(NotImplementedError, match="BUILDERS"):
        build_model(config, num_classes=4, input_shape=(3, 8, 8))


def test_a_model_needs_at_least_one_hidden_layer():
    with pytest.raises(ValueError, match="at least one hidden layer"):
        VariableMLP((), num_classes=4, input_shape=(3, 8, 8))


def test_the_output_layer_is_not_a_hidden_layer():
    """The spectrum must never count the head, whatever the depth."""
    from staticsbfl.suspiciousness import select_hidden_layers

    for depth in (1, 3, 10):
        config = ArchitectureConfig("k", "mlp", tuple([32] * depth), "relu", 42)
        model = build_model(config, num_classes=10, input_shape=(3, 8, 8))
        selected = [name for name, _ in select_hidden_layers(model)]
        assert selected == list(model.hidden_layer_names)
        assert "head" not in selected
        assert len(selected) == depth


def test_a_configured_model_is_reproducible_from_its_seed():
    from staticsbfl.utils.seeding import set_seed

    config = ArchitectureConfig("k", "mlp", (64, 32), "relu", 42)
    set_seed(config.seed)
    first = build_model(config, num_classes=10, input_shape=(3, 8, 8))
    set_seed(config.seed)
    second = build_model(config, num_classes=10, input_shape=(3, 8, 8))

    for (_, a), (_, b) in zip(first.state_dict().items(), second.state_dict().items()):
        assert torch.equal(a, b)
