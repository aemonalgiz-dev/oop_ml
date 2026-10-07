"""Each of this library's losses, as a torch criterion that measures the same number.

Why not the stock criteria with their defaults
-----------------------------------------------
torch ships ``MSELoss``, ``L1Loss``, ``HuberLoss``, ``BCEWithLogitsLoss`` and
``CrossEntropyLoss``, and every one of them averages by default. The first
four answer one number per entry and so average over *every entry*, which is
rows times outputs, where this library's losses divide by the row count
alone. On a single-output problem the two agree; on a two-output problem the
stock criterion reports half of what the numpy backend reports, and the
gradient it hands to autograd is half the size, which is a learning rate
quietly halved by the width of the answer. ``CrossEntropyLoss`` is the
exception, and a measured one. It answers one number per row, so its mean is
already a mean over rows and it agrees with this library's reading exactly.
Its criterion is here so that all five reduce by one rule written in one
place.

:class:`~oop_ml.core.network.loss.SquaredError` then halves as well, so that
its derivative comes out as a clean subtraction, and no stock criterion does.

So each criterion here is written to the numpy definition exactly: a sum over
everything, divided by the rows, halved where the library halves. The
arithmetic underneath is torch's own, which is what makes these criteria
differentiable by autograd rather than by hand, and the agreement test in
``test/pytorch/test_losses.py`` holds each one to its numpy namesake to the
last bits, value and gradient both.

What is translated rather than reimplemented
---------------------------------------------
The two cross entropies. The numpy backend clips probabilities away from zero
before taking logarithms, because a confidently wrong row would otherwise
contribute ``-inf``. torch reaches the same safety by a different route, the
log-sum-exp inside ``binary_cross_entropy_with_logits`` and ``cross_entropy``,
and at moderate scores the two agree to rounding. At scores past about 700
they part: the numpy value saturates at a clipped ceiling and torch's keeps
growing linearly, which is torch's being the more faithful of the two and is
recorded here rather than papered over.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.network.loss import (
    AbsoluteError,
    BinaryCrossEntropy,
    HuberError,
    Loss,
    SoftmaxCrossEntropy,
    SquaredError,
)


class SquaredErrorCriterion(nn.Module):
    """``sum((p - y)**2) / (2n)``, the numpy backend's halved squared error."""

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (outputs - targets).pow(2).sum() / (2 * outputs.shape[0])


class AbsoluteErrorCriterion(nn.Module):
    """``sum(|p - y|) / n``. Autograd's sign at an exact tie is zero, as numpy's."""

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (outputs - targets).abs().sum() / outputs.shape[0]


class HuberErrorCriterion(nn.Module):
    """Squared inside the threshold, linear outside, summed and divided by rows.

    torch's branch is ``|e| < delta`` where numpy's is ``|e| <= delta``. At
    the one point where that matters, both branches give the same value and
    the same slope, which is the continuity the threshold exists for, so the
    two are equal everywhere.
    """

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = float(threshold)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        summed = functional.huber_loss(
            outputs, targets, reduction="sum", delta=self.threshold
        )
        return summed / outputs.shape[0]


class BinaryCrossEntropyCriterion(nn.Module):
    """Log-loss over raw scores, sigmoid applied inside, divided by rows."""

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        summed = functional.binary_cross_entropy_with_logits(
            outputs, targets, reduction="sum"
        )
        return summed / outputs.shape[0]


class SoftmaxCrossEntropyCriterion(nn.Module):
    """Softmax across the classes of raw scores, scored against one-hot targets.

    ``cross_entropy`` reads a floating target block of the same shape as the
    scores as per-class probabilities, which a one-hot block is. That reading
    arrived in torch 1.10 and is the reason the backend's version floor sits
    where it does.
    """

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        summed = functional.cross_entropy(outputs, targets, reduction="sum")
        return summed / outputs.shape[0]


def torch_loss_for(loss: Loss) -> nn.Module:
    """The criterion that measures exactly what this loss measures.

    Parameters
    ----------
    loss:
        One of this library's five losses.

    Returns
    -------
    torch.nn.Module
        A criterion taking ``(outputs, targets)`` tensors and answering the
        scalar the numpy loss's ``measure`` would answer, differentiable by
        autograd.

    Raises
    ------
    InvalidValuesError
        If the loss is one this backend has no criterion for. Every loss the
        library ships has one, so reaching this means a loss subclass this
        backend has not been told about.
    """
    if isinstance(loss, SquaredError):
        return SquaredErrorCriterion()
    if isinstance(loss, AbsoluteError):
        return AbsoluteErrorCriterion()
    if isinstance(loss, HuberError):
        return HuberErrorCriterion(loss.threshold)
    if isinstance(loss, BinaryCrossEntropy):
        return BinaryCrossEntropyCriterion()
    if isinstance(loss, SoftmaxCrossEntropy):
        return SoftmaxCrossEntropyCriterion()
    raise InvalidValuesError(
        f"the torch backend has no criterion for {type(loss).__name__}; the "
        "five losses this library ships all translate, so this is a loss the "
        "backend has not been told about"
    )
