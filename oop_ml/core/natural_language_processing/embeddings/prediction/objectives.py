"""The two ways word2vec avoids a full softmax, as one binary logistic step.

Both objectives are the same arithmetic
---------------------------------------
Negative sampling asks, for a hidden vector ``h`` and a handful of output rows,
"is this row the true neighbour or a random word", and trains each row as a
binary logistic classifier with label 1 for the true neighbour and 0 for each
negative. Hierarchical softmax asks, for the same ``h`` and the internal nodes
on a word's Huffman path, "does this word go left or right here", and trains
each node as a binary logistic classifier whose label is the bit. Different
rows, different labels, identical arithmetic: a score ``s_k = u_k . h``, a
probability ``sigma(s_k)``, a loss ``-log sigma(s_k)`` for a label of 1 and
``-log(1 - sigma(s_k))`` for a label of 0, and gradients

    d loss / d h    =  sum_k (sigma(s_k) - y_k) u_k
    d loss / d u_k  =  (sigma(s_k) - y_k) h

So there is one function, :func:`binary_logistic_gradients`, and the two
objectives differ only in which rows and labels they hand it. A finite
difference of the loss it reports against the gradients it returns is the test
that pins the arithmetic, in the same way the network package pins every
backward pass.

The gradients are of the loss, so the caller subtracts ``learning_rate`` times
each. The reference implementation writes the same step as an addition of
``(label - sigma) * alpha``, which is this with the sign folded in.

The loss is computed stably
---------------------------
``-log sigma(s)`` is ``log(1 + exp(-s))``, which overflows for a large negative
score if written literally; ``numpy.logaddexp(0, -s)`` is the same number
without the overflow. The probability itself comes from
:func:`~oop_ml.core.logistic.stable_logistic`, for the reason that module gives.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.logistic import stable_logistic
from oop_ml.core.natural_language_processing.embeddings.prediction.huffman import (
    HuffmanTree,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.sampling import (
    UnigramSampler,
)
from oop_ml.core.types import FloatArray, IndexArray


class PairGradients:
    """What one training pair asks the model to change, and what it cost.

    Parameters
    ----------
    output_ids:
        Which rows of the output table were scored, in the order the gradients
        are given.
    hidden_gradient:
        ``d loss / d hidden``, one vector.
    output_gradients:
        ``d loss / d output_row``, one row per output id.
    loss:
        The binary logistic loss summed over the rows, non-negative.

    Raises
    ------
    ShapeMismatchError
        If the output gradients do not match the ids in count or the hidden
        gradient in width.
    InvalidValuesError
        If the loss is negative or not finite.
    """

    __slots__ = ("_hidden_gradient", "_loss", "_output_gradients", "_output_ids")

    def __init__(
        self,
        output_ids: IndexArray,
        hidden_gradient: FloatArray,
        output_gradients: FloatArray,
        loss: float,
    ) -> None:
        if output_gradients.shape != (len(output_ids), hidden_gradient.shape[0]):
            raise ShapeMismatchError(
                f"expected output gradients of shape "
                f"{(len(output_ids), hidden_gradient.shape[0])}, got "
                f"{output_gradients.shape}"
            )

        if not np.isfinite(loss) or loss < 0.0:
            raise InvalidValuesError(
                f"a logistic loss is finite and non-negative, got {loss}"
            )

        self._output_ids = np.asarray(output_ids, dtype=np.intp)
        self._hidden_gradient = hidden_gradient
        self._output_gradients = output_gradients
        self._loss = float(loss)

    @property
    def output_ids(self) -> IndexArray:
        """Which output rows were scored."""
        return self._output_ids

    @property
    def hidden_gradient(self) -> FloatArray:
        """``d loss / d hidden``."""
        return self._hidden_gradient

    @property
    def output_gradients(self) -> FloatArray:
        """``d loss / d output_row``, aligned with :attr:`output_ids`."""
        return self._output_gradients

    @property
    def loss(self) -> float:
        """The pair's binary logistic loss."""
        return self._loss

    def __repr__(self) -> str:
        return (
            f"PairGradients(n_outputs={len(self._output_ids)}, loss={self._loss:.4f})"
        )


def binary_logistic_gradients(
    hidden: FloatArray,
    output_ids: IndexArray,
    outputs: FloatArray,
    labels: Sequence[float] | FloatArray,
) -> PairGradients:
    """Score ``hidden`` against each output row and differentiate the loss.

    Parameters
    ----------
    hidden:
        ``(dimension,)``, the vector being trained.
    output_ids:
        Which rows ``outputs`` are, for the caller's bookkeeping.
    outputs:
        ``(n_rows, dimension)``, the rows to score.
    labels:
        One label per row, each 0 or 1.

    Raises
    ------
    ShapeMismatchError
        If the rows are not of the hidden vector's width, or the labels or ids
        do not match the rows in count.
    InvalidValuesError
        If a label is not 0 or 1.
    """
    label_array = np.asarray(labels, dtype=np.float64)
    if outputs.ndim != 2 or outputs.shape[1] != hidden.shape[0]:
        raise ShapeMismatchError(
            f"output rows of width {outputs.shape[-1] if outputs.ndim else '?'} "
            f"against a hidden vector of width {hidden.shape[0]}"
        )
    if label_array.shape != (outputs.shape[0],) or len(output_ids) != outputs.shape[0]:
        raise ShapeMismatchError(
            f"{outputs.shape[0]} rows need {outputs.shape[0]} labels and ids, got "
            f"{label_array.shape[0]} and {len(output_ids)}"
        )
    if not np.all((label_array == 0.0) | (label_array == 1.0)):
        raise InvalidValuesError("every label is 0 or 1")

    scores = outputs @ hidden
    probabilities = stable_logistic(scores)
    errors = probabilities - label_array
    signs = 2.0 * label_array - 1.0
    loss = float(np.sum(np.logaddexp(0.0, -signs * scores)))

    return PairGradients(
        output_ids=np.asarray(output_ids, dtype=np.intp),
        hidden_gradient=errors @ outputs,
        output_gradients=errors[:, None] * hidden[None, :],
        loss=loss,
    )


def negative_sampling_gradients(
    hidden: FloatArray,
    target_id: int,
    output_vectors: FloatArray,
    sampler: UnigramSampler,
    n_negative_samples: int,
    generator: np.random.Generator,
) -> PairGradients:
    """One true neighbour against ``n_negative_samples`` random words.

    The target's row is scored with label 1 and each drawn word's row with
    label 0; the draw never returns the target itself.
    """
    negatives = sampler.draw(generator, n_negative_samples, excluding=target_id)
    output_ids = np.concatenate(([target_id], negatives)).astype(np.intp)
    labels = np.zeros(len(output_ids))
    labels[0] = 1.0
    return binary_logistic_gradients(
        hidden, output_ids, output_vectors[output_ids], labels
    )


def hierarchical_softmax_gradients(
    hidden: FloatArray,
    target_id: int,
    node_vectors: FloatArray,
    tree: HuffmanTree,
) -> PairGradients:
    """The internal nodes on the target's Huffman path, each labelled by its bit.

    A node is trained to say the bit the path takes there, so the label of
    node ``k`` is the code's ``k``-th bit and the probability of the word is
    the product along the path. The reference implementation labels a node
    with ``1 - bit`` instead; that is the same tree with its branches renamed,
    and every probability comes out identical.
    """
    code = tree.code_of(target_id)
    output_ids = np.asarray(code.nodes, dtype=np.intp)
    labels = np.asarray(code.bits, dtype=np.float64)
    return binary_logistic_gradients(
        hidden, output_ids, node_vectors[output_ids], labels
    )
