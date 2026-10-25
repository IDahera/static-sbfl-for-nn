"""Training-metric tests.

The loops themselves are ordinary torch; what is worth pinning is the metric
arithmetic, because a confusion matrix built the wrong way round produces
plausible-looking numbers that quietly transpose precision and recall.
"""

import pytest
import torch

from staticsbfl.training import confusion_matrix, evaluate, macro_scores, train_epochs


def test_the_confusion_matrix_has_true_classes_on_its_rows():
    """Row = label, column = prediction. Transposing it swaps P and R."""
    predictions = torch.tensor([0, 1, 1, 2])
    targets = torch.tensor([0, 1, 2, 2])

    matrix = confusion_matrix(predictions, targets, num_classes=3)

    assert matrix.tolist() == [
        [1, 0, 0],   # true 0: one correct
        [0, 1, 0],   # true 1: one correct
        [0, 1, 1],   # true 2: one predicted as 1, one correct
    ]
    assert int(matrix.sum()) == 4
    assert int(matrix.diagonal().sum()) == 3


def test_perfect_predictions_score_one_across_the_board():
    matrix = torch.eye(4, dtype=torch.long) * 5
    assert macro_scores(matrix) == pytest.approx((1.0, 1.0, 1.0))


def test_macro_scores_match_a_hand_computation():
    """Two classes, deliberately asymmetric so P and R differ.

        true 0: 8 correct, 2 predicted as 1
        true 1: 3 predicted as 0, 7 correct

        precision = [8/11, 7/9],  recall = [8/10, 7/10]
    """
    matrix = torch.tensor([[8, 2], [3, 7]])
    precision, recall, f1 = macro_scores(matrix)

    expected_precision = (8 / 11 + 7 / 9) / 2
    expected_recall = (0.8 + 0.7) / 2
    assert precision == pytest.approx(expected_precision)
    assert recall == pytest.approx(expected_recall)
    assert 0.0 <= f1 <= 1.0


def test_a_class_that_never_appears_contributes_zero_not_nan():
    """Matching scikit-learn's zero_division=0 rather than poisoning the mean."""
    matrix = torch.tensor([[4, 0, 0], [0, 4, 0], [0, 0, 0]])
    precision, recall, f1 = macro_scores(matrix)

    for value in (precision, recall, f1):
        assert value == pytest.approx(2 / 3)   # two perfect classes, one absent


def test_evaluation_reports_the_accuracy_the_matrix_implies(scripted_mlp, loader_factory, cpu):
    """A model forced to predict class 0 on a 6/4 split scores exactly 0.6."""
    model = scripted_mlp()
    model.set_prediction(0)
    targets = torch.tensor([0] * 6 + [1] * 4)

    result = evaluate(
        model,
        loader_factory(torch.randn(10, 4), targets, batch_size=3),
        device=cpu,
        num_classes=2,
    )

    assert result.n_samples == 10
    assert result.accuracy == pytest.approx(0.6)
    assert result.loss > 0
    assert 0.0 <= result.macro_f1 <= 1.0


def test_training_yields_one_result_per_epoch(scripted_mlp, loader_factory, cpu):
    """train_epochs is a generator so a caller can log or snapshot in between."""
    from staticsbfl.models import SmallMLP

    torch.manual_seed(0)
    model = SmallMLP()
    loader = loader_factory(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)), batch_size=4)

    results = list(train_epochs(model, loader, epochs=3, device=cpu))

    assert [r.epoch for r in results] == [1, 2, 3]
    assert all(r.duration_s >= 0 for r in results)
    assert all(r.train_loss > 0 for r in results)


def test_training_actually_moves_the_weights(loader_factory, cpu):
    from staticsbfl.models import SmallMLP

    torch.manual_seed(0)
    model = SmallMLP()
    before = model.fc1.weight.detach().clone()

    list(
        train_epochs(
            model,
            loader_factory(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)), batch_size=4),
            epochs=1,
            device=cpu,
        )
    )

    assert not torch.equal(before, model.fc1.weight.detach().cpu())
