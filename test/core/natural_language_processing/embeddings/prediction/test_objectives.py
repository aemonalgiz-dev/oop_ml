"""Spec for the two word2vec objectives and the one logistic step beneath them.

The finite-difference checks carry this module, as they do for every backward
pass in the network package: the oracle is the definition of a derivative
applied to the loss the function itself reports, never a formula that could
have been copied from the implementation. The remaining tests pin which rows
each objective scores and with which labels, which is the only way the two
differ.
"""

from __future__ import annotations

import numpy as np
import pytest

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.natural_language_processing.embeddings.prediction.huffman import (
    HuffmanTree,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.objectives import (
    PairGradients,
    binary_logistic_gradients,
    hierarchical_softmax_gradients,
    negative_sampling_gradients,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.sampling import (
    UnigramSampler,
)
from oop_ml.core.types import FloatArray

STEP = 1e-6
LABELS = np.array([1.0, 0.0, 0.0, 1.0])
OUTPUT_IDS = np.array([2, 5, 7, 9], dtype=np.intp)


def sample_hidden() -> FloatArray:
    return np.random.default_rng(3).normal(size=6)


def sample_outputs() -> FloatArray:
    return np.random.default_rng(4).normal(size=(4, 6))


def loss_at(hidden: FloatArray, outputs: FloatArray) -> float:
    return binary_logistic_gradients(hidden, OUTPUT_IDS, outputs, LABELS).loss


class TestBinaryLogisticGradients:
    def test_the_hidden_gradient_matches_a_finite_difference(self) -> None:
        hidden, outputs = sample_hidden(), sample_outputs()
        claimed = binary_logistic_gradients(
            hidden, OUTPUT_IDS, outputs, LABELS
        ).hidden_gradient

        measured = np.empty_like(hidden)
        for position in range(hidden.shape[0]):
            up, down = hidden.copy(), hidden.copy()
            up[position] += STEP
            down[position] -= STEP
            measured[position] = (loss_at(up, outputs) - loss_at(down, outputs)) / (
                2.0 * STEP
            )

        assert np.allclose(claimed, measured, atol=1e-7)

    def test_the_output_gradients_match_a_finite_difference(self) -> None:
        hidden, outputs = sample_hidden(), sample_outputs()
        claimed = binary_logistic_gradients(
            hidden, OUTPUT_IDS, outputs, LABELS
        ).output_gradients

        measured = np.empty_like(outputs)
        for row in range(outputs.shape[0]):
            for column in range(outputs.shape[1]):
                up, down = outputs.copy(), outputs.copy()
                up[row, column] += STEP
                down[row, column] -= STEP
                measured[row, column] = (
                    loss_at(hidden, up) - loss_at(hidden, down)
                ) / (2.0 * STEP)

        assert np.allclose(claimed, measured, atol=1e-7)

    def test_a_single_label_one_row_costs_log_one_plus_exp_minus_score(self) -> None:
        """``hidden . row = 0.5 + 0.5 = 1``, so the loss is ``log(1 + e^-1)``."""
        gradients = binary_logistic_gradients(
            np.array([1.0, 2.0]), np.array([0]), np.array([[0.5, 0.25]]), [1.0]
        )

        assert gradients.loss == pytest.approx(np.log(1.0 + np.exp(-1.0)))

    def test_a_single_label_zero_row_costs_log_one_plus_exp_score(self) -> None:
        gradients = binary_logistic_gradients(
            np.array([1.0, 2.0]), np.array([0]), np.array([[0.5, 0.25]]), [0.0]
        )

        assert gradients.loss == pytest.approx(np.log(1.0 + np.exp(1.0)))

    def test_the_loss_is_summed_over_the_rows(self) -> None:
        hidden, outputs = sample_hidden(), sample_outputs()
        together = binary_logistic_gradients(hidden, OUTPUT_IDS, outputs, LABELS).loss
        apart = sum(
            binary_logistic_gradients(
                hidden,
                OUTPUT_IDS[row : row + 1],
                outputs[row : row + 1],
                LABELS[row : row + 1],
            ).loss
            for row in range(4)
        )

        assert together == pytest.approx(apart)

    def test_against_zero_rows_the_output_gradient_is_half_minus_the_label_times_hidden(
        self,
    ) -> None:
        """Every score is zero, every probability a half, so ``sigma - y`` is
        ``+0.5`` for a negative and ``-0.5`` for the true neighbour."""
        hidden = np.array([1.0, -2.0, 3.0])
        gradients = binary_logistic_gradients(
            hidden, np.array([0, 1]), np.zeros((2, 3)), [1.0, 0.0]
        )

        assert np.allclose(gradients.output_gradients[0], -0.5 * hidden)
        assert np.allclose(gradients.output_gradients[1], 0.5 * hidden)
        assert np.allclose(gradients.hidden_gradient, 0.0)
        assert gradients.loss == pytest.approx(2.0 * np.log(2.0))

    def test_the_output_ids_ride_through_unchanged(self) -> None:
        gradients = binary_logistic_gradients(
            sample_hidden(), OUTPUT_IDS, sample_outputs(), LABELS
        )

        assert np.array_equal(gradients.output_ids, OUTPUT_IDS)
        assert gradients.output_ids.dtype == np.intp

    def test_a_large_negative_score_does_not_overflow(self) -> None:
        """``-log sigma(-1000)`` is about 1000; the literal formula overflows."""
        gradients = binary_logistic_gradients(
            np.array([1000.0]), np.array([0]), np.array([[-1.0]]), [1.0]
        )

        assert np.isfinite(gradients.loss)
        assert gradients.loss == pytest.approx(1000.0, abs=1e-9)
        assert np.allclose(gradients.hidden_gradient, [1.0])


class TestNegativeSampling:
    def test_the_target_comes_first_and_is_never_a_negative(self) -> None:
        """Word 0 takes 97% of the unigram draws, so without the exclusion it
        would be drawn as its own negative almost every time."""
        sampler = UnigramSampler([1000, 1, 1], 0.75)
        gradients = negative_sampling_gradients(
            np.ones(3), 0, np.zeros((3, 3)), sampler, 5, np.random.default_rng(0)
        )

        assert gradients.output_ids[0] == 0
        assert 0 not in gradients.output_ids[1:]
        assert len(gradients.output_ids) == 6

    def test_the_target_is_labelled_one_and_every_negative_zero(self) -> None:
        hidden = np.array([1.0, -2.0, 3.0])
        gradients = negative_sampling_gradients(
            hidden,
            0,
            np.zeros((3, 3)),
            UnigramSampler([1, 1, 1], 0.75),
            4,
            np.random.default_rng(0),
        )

        assert np.allclose(gradients.output_gradients[0], -0.5 * hidden)
        assert np.allclose(gradients.output_gradients[1:], 0.5 * hidden)

    def test_repeated_negatives_are_kept_for_the_caller_to_add_at(self) -> None:
        """With one other word every negative is that word, five times over."""
        gradients = negative_sampling_gradients(
            np.ones(2),
            0,
            np.zeros((2, 2)),
            UnigramSampler([1, 1000], 0.75),
            5,
            np.random.default_rng(0),
        )

        assert list(gradients.output_ids) == [0, 1, 1, 1, 1, 1]

    def test_it_is_the_binary_step_on_the_rows_it_drew(self) -> None:
        generator = np.random.default_rng(5)
        sampler = UnigramSampler([5, 3, 2, 1, 1], 0.75)
        output_vectors = np.random.default_rng(6).normal(size=(5, 4))
        hidden = np.random.default_rng(7).normal(size=4)

        drawn = negative_sampling_gradients(
            hidden, 1, output_vectors, sampler, 3, generator
        )
        labels = np.zeros(4)
        labels[0] = 1.0
        by_hand = binary_logistic_gradients(
            hidden, drawn.output_ids, output_vectors[drawn.output_ids], labels
        )

        assert drawn.loss == by_hand.loss
        assert np.array_equal(drawn.hidden_gradient, by_hand.hidden_gradient)
        assert np.array_equal(drawn.output_gradients, by_hand.output_gradients)


class TestHierarchicalSoftmax:
    def test_it_scores_exactly_the_codes_nodes_with_the_codes_bits(self) -> None:
        """Word 2 of ``5, 2, 1, 1`` walks nodes ``2, 1, 0`` with bits ``0, 1, 0``;
        against zero node vectors the gradient at a node is ``(0.5 - bit) h``."""
        tree = HuffmanTree.from_counts([5, 2, 1, 1])
        hidden = np.array([1.0, -2.0, 3.0])
        gradients = hierarchical_softmax_gradients(hidden, 2, np.zeros((3, 3)), tree)

        assert list(gradients.output_ids) == [2, 1, 0]
        assert np.allclose(gradients.output_gradients[0], 0.5 * hidden)
        assert np.allclose(gradients.output_gradients[1], -0.5 * hidden)
        assert np.allclose(gradients.output_gradients[2], 0.5 * hidden)

    def test_the_commonest_word_touches_only_the_root(self) -> None:
        tree = HuffmanTree.from_counts([5, 2, 1, 1])
        gradients = hierarchical_softmax_gradients(
            np.ones(3), 0, np.zeros((3, 3)), tree
        )

        assert list(gradients.output_ids) == [2]
        assert gradients.output_gradients.shape == (1, 3)

    def test_the_loss_is_minus_log_the_probability_of_the_path(self) -> None:
        """At each node the path takes bit 1 with probability ``sigma(s)`` and
        bit 0 with ``1 - sigma(s)``; the word's probability is the product."""
        tree = HuffmanTree.from_counts([5, 2, 1, 1])
        node_vectors = np.random.default_rng(8).normal(size=(3, 4))
        hidden = np.random.default_rng(9).normal(size=4)
        code = tree.code_of(3)

        path_probability = 1.0
        for node, bit in zip(code.nodes, code.bits, strict=True):
            probability_of_one = 1.0 / (1.0 + np.exp(-node_vectors[node] @ hidden))
            path_probability *= probability_of_one if bit else 1.0 - probability_of_one

        gradients = hierarchical_softmax_gradients(hidden, 3, node_vectors, tree)

        assert gradients.loss == pytest.approx(-np.log(path_probability))

    def test_the_paths_probabilities_sum_to_one_over_the_vocabulary(self) -> None:
        """What makes it a softmax: the leaves' probabilities are a distribution."""
        tree = HuffmanTree.from_counts([5, 2, 1, 1])
        node_vectors = np.random.default_rng(8).normal(size=(3, 4))
        hidden = np.random.default_rng(9).normal(size=4)

        total = sum(
            np.exp(
                -hierarchical_softmax_gradients(hidden, word, node_vectors, tree).loss
            )
            for word in range(4)
        )

        assert total == pytest.approx(1.0)

    def test_it_is_the_binary_step_on_the_codes_nodes(self) -> None:
        tree = HuffmanTree.from_counts([5, 2, 1, 1])
        node_vectors = np.random.default_rng(8).normal(size=(3, 4))
        hidden = np.random.default_rng(9).normal(size=4)
        code = tree.code_of(2)

        walked = hierarchical_softmax_gradients(hidden, 2, node_vectors, tree)
        by_hand = binary_logistic_gradients(
            hidden,
            np.asarray(code.nodes),
            node_vectors[list(code.nodes)],
            list(code.bits),
        )

        assert walked.loss == by_hand.loss
        assert np.array_equal(walked.hidden_gradient, by_hand.hidden_gradient)
        assert np.array_equal(walked.output_gradients, by_hand.output_gradients)


class TestPairGradients:
    def test_output_gradients_must_match_the_ids_and_the_hidden_width(self) -> None:
        with pytest.raises(ShapeMismatchError):
            PairGradients(np.array([0, 1]), np.zeros(3), np.zeros((3, 3)), 0.0)
        with pytest.raises(ShapeMismatchError):
            PairGradients(np.array([0, 1]), np.zeros(3), np.zeros((2, 4)), 0.0)

    @pytest.mark.parametrize("loss", [-0.1, float("nan"), float("inf")])
    def test_a_loss_that_is_negative_or_not_finite_is_refused(
        self, loss: float
    ) -> None:
        with pytest.raises(InvalidValuesError):
            PairGradients(np.array([0, 1]), np.zeros(3), np.zeros((2, 3)), loss)

    def test_repr_names_the_rows_and_the_loss(self) -> None:
        gradients = PairGradients(np.array([0, 1]), np.zeros(3), np.zeros((2, 3)), 1.5)

        assert repr(gradients) == "PairGradients(n_outputs=2, loss=1.5000)"


class TestRefusals:
    def test_rows_of_another_width_are_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            binary_logistic_gradients(
                np.ones(3), np.array([0, 1]), np.zeros((2, 4)), [1.0, 0.0]
            )

    def test_a_one_dimensional_output_block_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            binary_logistic_gradients(np.ones(3), np.array([0]), np.zeros(3), [1.0])

    def test_labels_must_match_the_rows(self) -> None:
        with pytest.raises(ShapeMismatchError):
            binary_logistic_gradients(
                np.ones(3), np.array([0, 1]), np.zeros((2, 3)), [1.0]
            )

    def test_ids_must_match_the_rows(self) -> None:
        with pytest.raises(ShapeMismatchError):
            binary_logistic_gradients(
                np.ones(3), np.array([0]), np.zeros((2, 3)), [1.0, 0.0]
            )

    @pytest.mark.parametrize("label", [2.0, 0.5, -1.0])
    def test_a_label_other_than_zero_or_one_is_refused(self, label: float) -> None:
        with pytest.raises(InvalidValuesError):
            binary_logistic_gradients(
                np.ones(3), np.array([0, 1]), np.zeros((2, 3)), [1.0, label]
            )
