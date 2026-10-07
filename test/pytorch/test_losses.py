"""Spec for the loss translations, where the oracle is the numpy loss itself.

Two implementations of one definition, written by different hands in
different libraries, held to each other on the value and on the gradient.
The value half says the criterion measures what the numpy loss measures. The
gradient half is the more interesting one, because the numpy gradient is
written by hand and the torch one comes from autograd, so agreement there is
automatic differentiation vouching for five derivations in ``loss.py``.

The reduction test is the reason most of the criteria exist. Four of the five
stock torch criteria answer one number per entry and average over entries,
rows times outputs, where this library divides by rows alone. The test builds
a two-output problem that misses exactly as badly as a one-output one, asserts
the library's reading that the cost doubles, and asserts the stock criterion
disagrees, so that nobody replaces these with ``nn.MSELoss`` on the grounds
that it looks the same. ``nn.CrossEntropyLoss`` is the exception, answering
one number per row, and a test pins that it already agrees with this library.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.network.loss import (
    AbsoluteError,
    BinaryCrossEntropy,
    HuberError,
    Loss,
    SoftmaxCrossEntropy,
    SquaredError,
)
from oop_ml.pytorch.losses import torch_loss_for
from oop_ml.pytorch.plumbing import as_array, as_tensor

N_ROWS = 7


def regression_blocks(n_outputs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    outputs = generator.normal(size=(N_ROWS, n_outputs))
    targets = generator.normal(size=(N_ROWS, n_outputs))
    return outputs, targets


def binary_blocks(seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    outputs = generator.normal(size=(N_ROWS, 1)) * 2.0
    targets = generator.integers(0, 2, size=(N_ROWS, 1)).astype(np.float64)
    return outputs, targets


def softmax_blocks(seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    outputs = generator.normal(size=(N_ROWS, 4)) * 2.0
    targets = SoftmaxCrossEntropy.one_hot(generator.integers(0, 4, size=N_ROWS), 4)
    return outputs, targets


CASES: list[tuple[Loss, np.ndarray, np.ndarray]] = [
    (SquaredError(), *regression_blocks(3, 1)),
    (AbsoluteError(), *regression_blocks(3, 2)),
    (HuberError(threshold=0.8), *regression_blocks(3, 3)),
    (BinaryCrossEntropy(), *binary_blocks(4)),
    (SoftmaxCrossEntropy(), *softmax_blocks(5)),
]
CASE_IDS = [type(case[0]).__name__ for case in CASES]


class TestEveryCriterionMeasuresWhatItsLossMeasures:
    @pytest.mark.parametrize(("loss", "outputs", "targets"), CASES, ids=CASE_IDS)
    def test_the_value_agrees(
        self, loss: Loss, outputs: np.ndarray, targets: np.ndarray
    ) -> None:
        expected = loss.measure(outputs, targets).value
        measured = torch_loss_for(loss)(as_tensor(outputs), as_tensor(targets))

        assert float(measured) == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize(("loss", "outputs", "targets"), CASES, ids=CASE_IDS)
    def test_autograd_agrees_with_the_hand_written_gradient(
        self, loss: Loss, outputs: np.ndarray, targets: np.ndarray
    ) -> None:
        """The derivation in loss.py, checked by a differentiator that never read it."""
        expected = loss.measure(outputs, targets).gradient

        scores = as_tensor(outputs).requires_grad_(True)
        torch_loss_for(loss)(scores, as_tensor(targets)).backward()
        assert scores.grad is not None

        assert np.max(np.abs(as_array(scores.grad) - expected)) < 1e-12


class TestTheReductionIsByRowsAndNotByEntries:
    """A two-output problem that misses as badly as a one-output one costs twice."""

    def test_the_cost_doubles_with_the_width(self) -> None:
        outputs, targets = regression_blocks(1, 6)
        wide_outputs = np.hstack([outputs, outputs])
        wide_targets = np.hstack([targets, targets])

        narrow = float(
            torch_loss_for(SquaredError())(as_tensor(outputs), as_tensor(targets))
        )
        wide = float(
            torch_loss_for(SquaredError())(
                as_tensor(wide_outputs), as_tensor(wide_targets)
            )
        )

        assert wide == pytest.approx(2.0 * narrow)

    def test_the_stock_criterion_does_not_and_that_is_why_these_exist(self) -> None:
        """nn.MSELoss averages over entries, so it reports the same number at
        both widths, and it also lacks the half."""
        outputs, targets = regression_blocks(1, 6)
        wide_outputs = np.hstack([outputs, outputs])
        wide_targets = np.hstack([targets, targets])
        stock = nn.MSELoss()

        narrow = float(stock(as_tensor(outputs), as_tensor(targets)))
        wide = float(stock(as_tensor(wide_outputs), as_tensor(wide_targets)))

        assert wide == pytest.approx(narrow)
        assert narrow != pytest.approx(SquaredError().measure(outputs, targets).value)

    def test_cross_entropy_is_the_one_stock_criterion_that_already_agrees(
        self,
    ) -> None:
        """Cross entropy answers one number per row, so torch's mean over its
        output is already a mean over rows and the stock criterion coincides
        with this library's reading. Pinned so that the exception the module
        docstring claims is a measured one."""
        outputs, targets = softmax_blocks(8)

        stock = float(nn.CrossEntropyLoss()(as_tensor(outputs), as_tensor(targets)))
        expected = SoftmaxCrossEntropy().measure(outputs, targets).value

        assert stock == pytest.approx(expected, abs=1e-12)


class TestHuber:
    def test_the_threshold_is_carried(self) -> None:
        outputs, targets = regression_blocks(2, 7)
        tight = torch_loss_for(HuberError(threshold=0.1))
        loose = torch_loss_for(HuberError(threshold=10.0))

        assert float(tight(as_tensor(outputs), as_tensor(targets))) != pytest.approx(
            float(loose(as_tensor(outputs), as_tensor(targets)))
        )

    def test_the_two_branches_meet_at_the_threshold(self) -> None:
        """torch's branch is strict where numpy's is inclusive, and at the one
        point that could matter both branches give 0.5 * threshold ** 2."""
        threshold = 0.75
        outputs = np.array([[threshold]])
        targets = np.array([[0.0]])

        expected = HuberError(threshold=threshold).measure(outputs, targets).value
        measured = torch_loss_for(HuberError(threshold=threshold))(
            as_tensor(outputs), as_tensor(targets)
        )

        assert float(measured) == pytest.approx(expected, abs=1e-15)
        assert expected == pytest.approx(0.5 * threshold**2)


class TestWhereTheTwoDeliberatelyPart:
    def test_at_an_absurd_score_torch_keeps_growing_where_numpy_clips(self) -> None:
        """Recorded rather than hidden. numpy clips probabilities away from zero
        so a confidently wrong row costs a bounded amount; torch's log-sum-exp
        keeps the true value, which grows linearly in the score. torch is the
        more faithful of the two, and the agreement tests above sit at scores
        where it does not matter."""
        outputs = np.array([[800.0]])
        targets = np.array([[0.0]])

        clipped = BinaryCrossEntropy().measure(outputs, targets).value
        faithful = float(
            torch_loss_for(BinaryCrossEntropy())(as_tensor(outputs), as_tensor(targets))
        )

        assert faithful > clipped
        assert faithful == pytest.approx(800.0, abs=1e-6)


class TestWhatIsRefused:
    def test_a_loss_the_backend_was_never_told_about(self) -> None:
        class SomethingElse(Loss):
            def _measure_aligned(self, outputs, targets):  # type: ignore[override]
                raise NotImplementedError

            @property
            def description(self) -> str:
                return "something else"

        with pytest.raises(InvalidValuesError):
            torch_loss_for(SomethingElse())

    def test_every_criterion_is_a_torch_module(self) -> None:
        for loss, _, _ in CASES:
            assert isinstance(torch_loss_for(loss), torch.nn.Module)
