"""Tests for the model-to-table transformation.

The transformation is where a silent error would be least visible: every
downstream number would still be computable, just describing the wrong neuron.
So the checks here are exact rather than statistical -- a hand-built model whose
weights are known lets the expected feature values be written down in advance.
"""

import numpy as np
import pytest
import torch

from staticsbfl.dataset import (
    ARCHITECTURE,
    EXCLUDED_FEATURES,
    FEATURE_NAMES,
    FEATURES,
    NEURON_STATIC,
    TARGET_METRICS,
    TARGET_NAMES,
    NeuronDataset,
    build_model_rows,
    combine,
    model_feature_rows,
    model_target_rows,
    normalized_ranks,
    target_names,
)
from staticsbfl.persistence import SuspiciousnessBundle
from staticsbfl.suspiciousness import compute_hit_spectrum, score_spectrum
from staticsbfl.suspiciousness.spectrum import HitSpectrum, LayerSpectrum
from staticsbfl.variation import ArchitectureConfig, build_model


# --------------------------------------------------------------------------
# Fixtures: a model small enough to write the expected table out by hand
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_config():
    """Two hidden layers of 3 and 2 units, so the table must have 5 rows."""
    return ArchitectureConfig("tiny", "mlp", (3, 2), "relu", seed=42)


@pytest.fixture
def tiny_model(tiny_config):
    """The model with hand-set weights, so every feature has a known value.

    fc0: 3 units over a flattened input of 4; unit j has every incoming weight
    equal to ``j + 1`` and bias ``10 * (j + 1)``.
    fc1: 2 units over 3; head: 2 classes over 2.
    """
    model = build_model(tiny_config, num_classes=2, input_shape=(4,))
    with torch.no_grad():
        for unit in range(3):
            model.fc0.weight[unit].fill_(float(unit + 1))
            model.fc0.bias[unit] = 10.0 * (unit + 1)
        model.fc1.weight.fill_(0.5)
        model.fc1.bias.zero_()
        model.head.weight.fill_(2.0)
        model.head.bias.zero_()
    return model


def make_bundle(key, layer_scores):
    """A SuspiciousnessBundle carrying the given per-layer score tensors."""
    layers = {
        name: LayerSpectrum(
            name=name,
            a_f=torch.zeros(values.shape, dtype=torch.int32),
            a_s=torch.zeros(values.shape, dtype=torch.int32),
            n_failing=4,
            n_successful=6,
        )
        for name, values in layer_scores.items()
    }
    spectrum = HitSpectrum(layers=layers, n_failing=4, n_successful=6)
    scores = {metric: dict(layer_scores) for metric in TARGET_METRICS}
    return SuspiciousnessBundle(key=key, hit_spectrum=spectrum, scores=scores)


# --------------------------------------------------------------------------
# Row count: one row per hidden neuron, no more and no fewer
# --------------------------------------------------------------------------


def test_the_table_has_exactly_one_row_per_hidden_neuron(tiny_model, tiny_config):
    """3 + 2 hidden units must give exactly 5 rows."""
    features, layers = model_feature_rows(
        tiny_model, tiny_config, input_size=4, num_classes=2
    )

    assert features.shape == (5, len(FEATURE_NAMES))
    assert len(layers) == 5
    assert tiny_config.total_hidden_units == 5


@pytest.mark.parametrize("widths", [(1,), (8,), (4, 4), (16, 8, 4), (7, 3, 9, 2)])
def test_the_row_count_equals_the_sum_of_the_hidden_widths(widths):
    """Holds for any depth and any width combination."""
    config = ArchitectureConfig("k", "mlp", widths, "relu", 42)
    model = build_model(config, num_classes=10, input_shape=(3, 8, 8))

    features, layers = model_feature_rows(
        model, config, input_size=192, num_classes=10
    )

    assert features.shape[0] == sum(widths)
    assert len(layers) == sum(widths)


def test_the_output_layer_contributes_no_rows(tiny_model, tiny_config):
    """The head is not a hidden layer, so none of its units may appear."""
    _features, layers = model_feature_rows(
        tiny_model, tiny_config, input_size=4, num_classes=2
    )
    assert "head" not in set(layers)
    assert set(layers) == {"fc0", "fc1"}


def test_rows_are_ordered_by_layer_then_by_unit(tiny_model, tiny_config):
    _features, layers = model_feature_rows(
        tiny_model, tiny_config, input_size=4, num_classes=2
    )
    assert layers == ["fc0", "fc0", "fc0", "fc1", "fc1"]


# --------------------------------------------------------------------------
# Exact feature values on the hand-built model
# --------------------------------------------------------------------------


def test_the_features_of_a_hand_built_model_are_exactly_as_expected(tiny_model, tiny_config):
    """Every column of fc0's second unit, written out in advance.

    Unit 1 of fc0 has four incoming weights all equal to 2.0 and bias 20.0. Its
    outgoing weights are the second column of fc1's matrix: two entries of 0.5.
    """
    features, layers = model_feature_rows(
        tiny_model, tiny_config, input_size=4, num_classes=2
    )
    row = dict(zip(FEATURE_NAMES, features[1]))
    assert layers[1] == "fc0"

    # incoming: four weights of 2.0
    assert row["in_weight_mean"] == pytest.approx(2.0)
    assert row["in_weight_std"] == pytest.approx(0.0)
    assert row["in_weight_l1"] == pytest.approx(8.0)
    assert row["in_weight_l2"] == pytest.approx(4.0)          # sqrt(4 * 2^2)
    assert row["in_weight_absmax"] == pytest.approx(2.0)
    assert row["in_weight_frac_small"] == pytest.approx(0.0)
    assert row["bias"] == pytest.approx(20.0)

    # outgoing: two weights of 0.5
    assert row["out_weight_mean"] == pytest.approx(0.5)
    assert row["out_weight_l1"] == pytest.approx(1.0)
    assert row["out_weight_l2"] == pytest.approx(np.sqrt(2 * 0.25))
    assert row["out_weight_absmax"] == pytest.approx(0.5)

    assert row["fan_in"] == pytest.approx(4.0)                # flattened input
    assert row["fan_out"] == pytest.approx(2.0)               # fc1 has 2 units


def test_the_architecture_columns_describe_the_model(tiny_model, tiny_config):
    features, _ = model_feature_rows(
        tiny_model, tiny_config, input_size=4, num_classes=2
    )
    first = dict(zip(FEATURE_NAMES, features[0]))    # fc0, unit 0
    last = dict(zip(FEATURE_NAMES, features[-1]))    # fc1, unit 1

    assert first["depth"] == 2 and last["depth"] == 2
    assert first["layer_index"] == 0 and last["layer_index"] == 1
    assert first["layer_position"] == pytest.approx(0.0)
    assert last["layer_position"] == pytest.approx(1.0)
    assert first["layer_width"] == 3 and last["layer_width"] == 2
    assert first["prev_layer_width"] == 4     # the flattened input
    assert first["next_layer_width"] == 2     # fc1
    assert last["next_layer_width"] == 2      # the class count
    assert first["total_hidden_units"] == 5


def test_the_bias_column_is_per_neuron_not_per_layer(tiny_model, tiny_config):
    """The defect of the earlier feature set, asserted against directly.

    Layer-aggregated statistics gave every neuron of a layer the same value, so
    a predictor could not tell two neurons of one layer apart.
    """
    features, _ = model_feature_rows(
        tiny_model, tiny_config, input_size=4, num_classes=2
    )
    bias_column = features[:3, FEATURE_NAMES.index("bias")]
    assert list(bias_column) == pytest.approx([10.0, 20.0, 30.0])
    assert len(set(bias_column)) == 3


def test_neuron_static_features_vary_within_a_layer(tiny_model, tiny_config):
    """Every neuron-static column must distinguish the units of one layer.

    A column that is constant across a layer carries no within-layer
    information, which is exactly what made the earlier feature set unable to
    rank neurons inside a layer.
    """
    features, layers = model_feature_rows(
        tiny_model, tiny_config, input_size=4, num_classes=2
    )
    fc0 = features[np.array(layers) == "fc0"]

    varying = []
    for spec in FEATURES:
        if spec.group != NEURON_STATIC:
            continue
        column = fc0[:, FEATURE_NAMES.index(spec.name)]
        if len(np.unique(column)) > 1:
            varying.append(spec.name)

    # in this model the weights differ per unit, so the weight-derived
    # statistics and the bias must all vary; fan_in/fan_out are structural.
    assert "bias" in varying
    assert "in_weight_l2" in varying
    assert "in_weight_mean" in varying


def test_architecture_features_are_constant_within_a_layer(tiny_model, tiny_config):
    """Legitimately so -- they describe the layer, not the neuron."""
    features, layers = model_feature_rows(
        tiny_model, tiny_config, input_size=4, num_classes=2
    )
    fc0 = features[np.array(layers) == "fc0"]

    for spec in FEATURES:
        if spec.group != ARCHITECTURE:
            continue
        column = fc0[:, FEATURE_NAMES.index(spec.name)]
        assert len(np.unique(column)) == 1, f"{spec.name} varies within a layer"


def test_no_excluded_feature_leaked_into_the_schema():
    """The index features in particular must stay out; see EXCLUDED_FEATURES."""
    for name in EXCLUDED_FEATURES:
        assert name not in FEATURE_NAMES


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------


def test_normalized_ranks_span_zero_to_one():
    # ascending order is 1.0, 3.0, 5.0, 9.0 -> ranks 0, 1, 2, 3 -> /3
    ranks = normalized_ranks(np.array([5.0, 1.0, 3.0, 9.0]))
    assert ranks.min() == 0.0 and ranks.max() == 1.0
    assert list(ranks) == pytest.approx([2 / 3, 0.0, 1 / 3, 1.0])


def test_normalized_ranks_are_order_only():
    """A monotone transform of the values must not change the ranks."""
    values = np.array([0.1, 0.4, 0.2, 0.9])
    assert normalized_ranks(values) == pytest.approx(normalized_ranks(values * 100 + 7))


def test_tied_values_share_an_averaged_rank():
    ranks = normalized_ranks(np.array([1.0, 1.0, 2.0]))
    assert ranks[0] == ranks[1]
    assert ranks[2] == 1.0


def test_infinite_values_rank_at_the_top():
    """D* is +inf for a neuron firing exactly on the failures."""
    ranks = normalized_ranks(np.array([0.0, 5.0, np.inf]))
    assert ranks[2] == 1.0
    assert np.isfinite(ranks).all()


def test_a_single_neuron_layer_ranks_one():
    assert normalized_ranks(np.array([0.7])) == pytest.approx([1.0])


def test_every_metric_yields_a_raw_and_two_rank_targets():
    scores = {"fc0": torch.tensor([0.1, 0.9, 0.5]), "fc1": torch.tensor([0.2, 0.8])}
    targets = model_target_rows(make_bundle("k", scores), ["fc0", "fc1"])

    assert set(targets) == set(TARGET_NAMES)
    for metric in TARGET_METRICS:
        raw, model_rank, layer_rank = target_names(metric)
        assert targets[raw].shape == (5,)
        assert targets[model_rank].max() == pytest.approx(1.0)
        # fc1's two neurons rank within their own layer, not against fc0
        assert sorted(targets[layer_rank][3:]) == pytest.approx([0.0, 1.0])


def test_the_raw_target_preserves_the_spectrum_values():
    scores = {"fc0": torch.tensor([0.1, 0.9, 0.5]), "fc1": torch.tensor([0.2, 0.8])}
    targets = model_target_rows(make_bundle("k", scores), ["fc0", "fc1"])
    assert targets["tarantula"] == pytest.approx([0.1, 0.9, 0.5, 0.2, 0.8])


def test_layer_ranks_and_model_ranks_differ():
    """Within-layer ranking removes the between-layer offset; that is the point."""
    scores = {"fc0": torch.tensor([0.1, 0.2]), "fc1": torch.tensor([0.8, 0.9])}
    targets = model_target_rows(make_bundle("k", scores), ["fc0", "fc1"])

    assert targets["ochiai_rank_model"] == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert targets["ochiai_rank_layer"] == pytest.approx([0.0, 1.0, 0.0, 1.0])


def test_a_missing_layer_in_the_spectrum_is_reported():
    scores = {"fc0": torch.tensor([0.1, 0.2])}
    with pytest.raises(KeyError, match="no tarantula scores"):
        model_target_rows(make_bundle("k", scores), ["fc0", "fc1"])


# --------------------------------------------------------------------------
# Features and targets aligned
# --------------------------------------------------------------------------


def test_features_and_targets_line_up_row_for_row(tiny_model, tiny_config):
    scores = {"fc0": torch.tensor([0.1, 0.9, 0.5]), "fc1": torch.tensor([0.2, 0.8])}
    features, targets, layers = build_model_rows(
        tiny_model, tiny_config, make_bundle("tiny", scores),
        input_size=4, num_classes=2,
    )

    assert features.shape[0] == 5
    assert all(v.shape[0] == 5 for v in targets.values())
    assert layers == ["fc0", "fc0", "fc0", "fc1", "fc1"]
    # row 1 is fc0 unit 1: bias 20, tarantula 0.9
    assert features[1, FEATURE_NAMES.index("bias")] == pytest.approx(20.0)
    assert targets["tarantula"][1] == pytest.approx(0.9)


def test_a_spectrum_from_a_different_model_is_rejected(tiny_model, tiny_config):
    """Guards against pairing a cached model with the wrong cached spectrum."""
    wrong = {"fc0": torch.tensor([0.1, 0.2]), "fc1": torch.tensor([0.3, 0.4])}
    with pytest.raises(ValueError, match="not the same model"):
        build_model_rows(
            tiny_model, tiny_config, make_bundle("tiny", wrong),
            input_size=4, num_classes=2,
        )


def test_the_row_count_matches_a_real_spectrum_end_to_end(tiny_config):
    """The count agreement holds against a genuinely computed spectrum too."""
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    model = build_model(tiny_config, num_classes=2, input_shape=(4,))
    loader = DataLoader(
        TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,))), batch_size=4
    )
    spectrum = compute_hit_spectrum(model, loader, device=torch.device("cpu"))
    bundle = SuspiciousnessBundle("tiny", spectrum, score_spectrum(spectrum))

    features, targets, _ = build_model_rows(
        model, tiny_config, bundle, input_size=4, num_classes=2
    )

    assert features.shape[0] == spectrum.n_neurons == 5
    assert targets["ochiai"].shape[0] == 5


# --------------------------------------------------------------------------
# The assembled dataset
# --------------------------------------------------------------------------


@pytest.fixture
def two_model_dataset(tiny_model, tiny_config):
    scores = {"fc0": torch.tensor([0.1, 0.9, 0.5]), "fc1": torch.tensor([0.2, 0.8])}
    parts = []
    for key, split in (("m1", "base"), ("m2", "deeper")):
        features, targets, layers = build_model_rows(
            tiny_model, tiny_config, make_bundle(key, scores),
            input_size=4, num_classes=2,
        )
        parts.append((features, targets, layers, key, split))
    return combine(parts)


def test_combining_models_keeps_every_row(two_model_dataset):
    assert len(two_model_dataset) == 10
    assert two_model_dataset.n_models == 2


def test_every_row_records_the_model_it_came_from(two_model_dataset):
    """The grouping column: the unit of independent evidence is the model."""
    assert list(two_model_dataset.model_keys[:5]) == ["m1"] * 5
    assert list(two_model_dataset.model_keys[5:]) == ["m2"] * 5


def test_a_split_can_be_selected_without_touching_the_others(two_model_dataset):
    base = two_model_dataset.for_split("base")
    assert len(base) == 5
    assert set(base.model_keys) == {"m1"}
    assert set(base.splits) == {"base"}


def test_a_feature_subset_selects_the_named_columns(two_model_dataset):
    """Feature-group ablations are a column filter, not a rebuild."""
    names = [f.name for f in FEATURES if f.group == ARCHITECTURE]
    subset = two_model_dataset.feature_subset(names)
    assert subset.shape == (10, len(names))

    with pytest.raises(KeyError, match="Unknown feature"):
        two_model_dataset.feature_subset(["not_a_feature"])


def test_the_dataset_survives_a_save_and_load(two_model_dataset, tmp_path):
    path = tmp_path / "neurons.npz"
    two_model_dataset.save(path)
    loaded = NeuronDataset.load(path)

    assert loaded.feature_names == two_model_dataset.feature_names
    assert np.array_equal(loaded.features, two_model_dataset.features)
    assert list(loaded.model_keys) == list(two_model_dataset.model_keys)
    for name in TARGET_NAMES:
        assert np.array_equal(loaded.targets[name], two_model_dataset.targets[name])


def test_a_malformed_dataset_is_rejected():
    with pytest.raises(ValueError, match="feature columns"):
        NeuronDataset(
            features=np.zeros((3, 2)),
            targets={},
            model_keys=np.array(["a"] * 3),
            layer_names=np.array(["fc0"] * 3),
            splits=np.array(["base"] * 3),
        )


def test_combining_nothing_is_an_error():
    with pytest.raises(ValueError, match="No models to combine"):
        combine([])


@pytest.mark.parametrize("widths,activation", [((300, 100), "relu"), ((64, 512, 32, 128), "tanh"), ((257,), "elu")])
def test_vectorised_features_are_bit_identical_to_the_per_unit_helpers(widths, activation):
    """The layer-wise feature build must reproduce the per-unit one exactly.

    Tables built before the vectorisation (and the predictors fitted on them)
    stay valid only if not a single bit changed, including the strided-column
    mean of the outgoing weights.
    """
    from staticsbfl.dataset.build import _outgoing_statistics, _weight_statistics, model_feature_rows

    torch.manual_seed(0)
    config = ArchitectureConfig(key="v", family="mlp", hidden_widths=widths, activation=activation, seed=0)
    model = build_model(config, num_classes=10, input_shape=(3, 8, 8))
    features, _ = model_feature_rows(model, config, input_size=192, num_classes=10)

    state = model.state_dict()
    names = list(model.hidden_layer_names) + ["head"]
    expected = []
    for index, name in enumerate(names[:-1]):
        weight, bias, following = state[f"{name}.weight"], state[f"{name}.bias"], state[f"{names[index + 1]}.weight"]
        for unit in range(weight.shape[0]):
            expected.append(_weight_statistics(weight[unit]) + [float(bias[unit])]
                            + _outgoing_statistics(following[:, unit]))
    expected = np.asarray(expected, dtype=np.float64)
    assert np.array_equal(features[:, : expected.shape[1]], expected)
