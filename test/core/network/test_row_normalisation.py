"""Spec for the within-row normalisers, where most of the file is a family claim.

Two classes are specified here and they are one class with a boolean flipped.
:class:`~oop_ml.core.network.row_normalisation.LayerNormalization` subtracts the
row's mean before dividing and
:class:`~oop_ml.core.network.row_normalisation.RMSNormalization` does not, and
beneath that one difference sit the same width, the same affine, the same
refusals, the same step and the same backward derivation. So nearly every claim
below is parametrized over both, because a claim about a family that is only
ever asserted of one member is a claim nobody is keeping.

The claim the module exists to have
------------------------------------
``purpose`` is ignored, and training and predicting return bit-identical
answers. That is not a convenience, it is the whole argument. A row is
normalised by its own contents, so nothing about why the pass is happening could
change what comes out, and therefore there are no running statistics to keep, no
:class:`~oop_ml.core.network.purpose.PassPurpose` branch to get wrong, no
response subclass to carry a batch's figures and no state that is not learned by
gradient. The batch sibling needs all four and its spec spends most of its length
on them. Asserted here with ``array_equal`` rather than ``allclose``, since the
two passes are not two calculations that agree, they are one calculation.

The same fact read from the other side is that a batch of one and a batch of a
thousand answer identically for the row they share. Batch normalisation cannot
say that at all, and the running statistics exist precisely because it cannot.

What separates the two classes
-------------------------------
Both leave every row with a root mean square of one, so that reading cannot tell
them apart and is asserted of both. What can is the row *mean*. Layer
normalisation leaves every row centred, to 1e-12; RMS normalisation leaves the
row's mean where it was, divided by the row's root mean square, which on the
uncentred fixture is 0.98, 0.997, -0.99 and 0.82. The pair is stated in both
directions, so that the first test is asserting a difference rather than a
constant. And on a row that was already centred the two agree bit for bit, which
is the cleanest statement of what the difference is.

The finite-difference check, and what it is blind to
-----------------------------------------------------
The naive backward pass treats the centre and the deviation as constants and
answers ``scaled / deviation``. It has the right shape, plausible magnitudes, and
it trains -- installed over the real class, the test asserting that one step
lowers the summary still passes. Nothing about the shape of the answer says which
version produced it, so the oracle has to be the definition of a derivative.

Installed over the real class and run against this whole file, the naive reading
fails 11 of 210 tests. Worst absolute disagreement with the central differences,
honest against naive, over the three fixtures and both classes::

    five by three     LayerNormalization   6.9e-10     0.849
    three by nine     LayerNormalization   2.8e-09     0.686
    flat row          LayerNormalization   3.3e-09     4.063
    five by three     RMSNormalization     5.7e-10     0.411
    three by nine     RMSNormalization     3.4e-09     0.565
    flat row          RMSNormalization     6.8e-10     0.496

The tolerance is 1e-7, which leaves thirty times of room beneath the honest
figures and six orders of magnitude above the broken ones.

The check runs over ``passed_down`` specifically. ``d scale`` and ``d shift`` are
sums down the rows and owe nothing to the routes the break drops, so they come
back bit-identical under it, and a spec checking only those would pass a body
with the wrong backward pass inside it.

Two structural readings catch the same break more cheaply. Every one of these
layers rescales a row by that row's own root mean square, so blame that merely
stretches a row along the direction it already points buys nothing, and
``sum(passed_down * normalised)`` along a row is zero but for a remainder
proportional to epsilon. Measured at the default 1e-5, 1.6e-05 for the centring
class and 1.8e-06 for the other, against 1.236 and 0.754 under the break. And
shifting a whole row by a constant cannot change what layer normalisation
answers, so its blame sums to zero along every row *exactly* -- 4.4e-16 honest,
1.234 naive. That second claim is the exact analogue of the column sums in the
batch spec, and it belongs to the centring class alone, since RMS normalisation
is not shift-invariant and asserting it there would assert something false. Its
row sums are 0.977, and a test says so.

Swapping ``centres`` fails 28 tests, and the gradient check is not one of them
-------------------------------------------------------------------------------
The second break makes the centring class stop centring and the other start,
forward and backward alike, since both passes read the same property. It fails 28
of 210, and the useful part is which 28. Every hand-worked forward number, every
comparison against the plain Python oracle, the identity claim, the constant-row
pair, the row-mean pair, the row-sum pair, and the claim that a step returns a
layer of the same concrete class.

**Every test in the gradient check passes.** That is not a weakness in it. The
forward and backward halves both read ``centres``, so a layer with it flipped is
a perfectly differentiated implementation of the other member of the family, and
a finite difference of its own forward pass has no opinion about which member it
was meant to be. What catches it is arithmetic worked on paper and the row means,
which is the same argument the dropout spec makes for a hand-built fixture that a
gradient check would never generate.

How the numbers are arrived at
-------------------------------
The forward figures are worked by hand where they can be and checked against a
plain Python loop over the definition where they cannot. That loop takes the
centring boolean from :data:`CENTRES_BY_DEFINITION` in this file rather than from
the layer, so a class answering the wrong boolean fails the oracle rather than
agreeing with itself.

Both hand-worked blocks are two rows that are permutations of one another, so the
rows share a statistic while no column shares anything with them. A body reducing
down the rows instead of across them meets column means of 4, 2, 5 and 5 on the
first block and cannot reproduce the answer. The epsilons are 11.0 and 1.0 so
that ``sqrt(5 + 11)`` is exactly 4 and ``sqrt(3 + 1)`` is exactly 2, which is
larger than any real layer would use and is the point, since every normalised
value is then exact in binary.

The identity claim needs a block whose rows share a statistic, and that is a real
difference from the batch sibling rather than an awkward fixture. There the
statistic is one value per feature and so is the scale, so any block undoes
itself. Here the statistic is one value per *row* while the scale is still one
value per feature, so an undoing scale exists for a whole block only where every
row agrees about what it is. Permuted rows agree, whether the centre is the mean
or is zero, and the epsilons are chosen so the undoing scale is exactly 4 for the
centring class and exactly 5 for the other.

Two shapes are pinned on their own. The gradient's weight block is
``(n_features, 1)``, one row per feature holding that feature's single
multiplier, and the transposed ``(1, n_features)`` is the tempting misreading. At
a width of one the two agree, so every test that pins it uses a width of three.
"""

import math

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
)
from oop_ml.core.network.activation import HyperbolicTangent, Identity
from oop_ml.core.network.gradient import LayerGradient
from oop_ml.core.network.layer import DenseLayer, LayerResponse
from oop_ml.core.network.loss import SquaredError
from oop_ml.core.network.neuron import Neuron
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.row_normalisation import (
    LayerNormalization,
    RMSNormalization,
    WithinRowNormalization,
)
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.network.stack import LayerStack

#: The step a central difference takes on either side.
NUDGE = 1e-6

#: Every concrete within-row normaliser. A claim about the family is
#: parametrized over this rather than asserted of whichever subclass was written
#: first, and the ids are the class names so a failure says which one broke it.
NORMALIZERS = [
    pytest.param(LayerNormalization, id="LayerNormalization"),
    pytest.param(RMSNormalization, id="RMSNormalization"),
]

#: Which of them subtracts the row's mean, stated here from the definition
#: rather than read off ``centres``. The oracles below take this rather than
#: asking the layer, so a class that answered the wrong boolean fails the whole
#: file rather than agreeing with itself.
CENTRES_BY_DEFINITION: dict[type[WithinRowNormalization], bool] = {
    LayerNormalization: True,
    RMSNormalization: False,
}

#: Two rows worked on paper for layer normalisation. Each is a permutation of
#: ``1, 3, 5, 7``, so both share a mean of 4 and a spread of 5, while no column
#: shares anything with them. A body reducing down the rows instead of across
#: them meets column means of 4, 2, 5 and 5 and cannot reproduce the answer.
LAYER_HAND_BLOCK = np.array([[1.0, 3.0, 5.0, 7.0], [7.0, 1.0, 5.0, 3.0]])

#: Chosen so that ``sqrt(5 + 11)`` is exactly 4 and every normalised value is a
#: quarter or three quarters, one way or the other. Far larger than a real layer
#: would use, which is what makes the hand arithmetic exact rather than nearly
#: right.
LAYER_HAND_EPSILON = 11.0

#: The same exercise for RMS normalisation, whose spread is the mean square
#: rather than the variance. Each row is a permutation of ``1, 1, 1, 3``, so the
#: mean square is 3 and ``sqrt(3 + 1)`` is exactly 2. The row mean is 1.5 rather
#: than zero, which is what makes this block tell the two classes apart.
RMS_HAND_BLOCK = np.array([[1.0, 1.0, 1.0, 3.0], [3.0, 1.0, 1.0, 1.0]])

#: See :data:`RMS_HAND_BLOCK`.
RMS_HAND_EPSILON = 1.0

#: The scale and shift the hand-worked passes carry. None of them is a default,
#: and one scale is negative, because a scale of one lets the affine's own
#: contribution to the backward pass hide behind the arriving block.
HAND_SCALE = np.array([2.0, 10.0, -1.0, 0.5])
HAND_SHIFT = np.array([1.0, -1.0, 0.25, 0.0])

#: A block with nothing convenient about it, for the plain Python oracle. Every
#: row has a mean well away from zero, so the two classes answer differently on
#: all four of them.
AWKWARD_BLOCK = np.array(
    [
        [1.5, -2.0, 0.125],
        [-0.5, 4.0, 1.25],
        [3.25, 1.0, 0.5],
        [0.75, -0.5, -2.0],
    ]
)

#: Four rows whose means are 4, 10, -6 and 4, so none of them is anywhere near
#: centred. That is what the claim separating the two classes needs, since on a
#: row that was already centred they are one function.
UNCENTRED_BLOCK = np.array(
    [
        [3.0, 4.0, 5.0],
        [10.0, 11.0, 9.0],
        [-6.0, -5.0, -7.0],
        [2.0, 2.0, 8.0],
    ]
)

#: Rows that are permutations of one another, so every row shares a centre and a
#: spread. That is what lets the identity claim be made on a block at all, since
#: the scale is one value per feature while the statistic is one value per row.
PERMUTED_BLOCK = np.array(
    [
        [1.0, 3.0, 5.0, 7.0],
        [7.0, 5.0, 3.0, 1.0],
        [5.0, 1.0, 7.0, 3.0],
    ]
)

#: One epsilon per class, chosen so the undoing scale is a whole number. Layer
#: normalisation sees a spread of 5 and undoes itself with a scale of 4; RMS
#: normalisation sees a mean square of 21 and undoes itself with a scale of 5.
IDENTITY_EPSILON: dict[type[WithinRowNormalization], float] = {
    LayerNormalization: 11.0,
    RMSNormalization: 4.0,
}


def row_centres_by_hand(block: np.ndarray, centres: bool) -> list[float]:
    """What each row is centred by, totalled in Python from the definition.

    Deliberately not :func:`numpy.mean`. The implementation reaches for that,
    and an oracle reaching for the same thing asserts only that numpy is
    self-consistent.
    """
    return [
        sum(float(value) for value in row) / len(row) if centres else 0.0
        for row in block.tolist()
    ]


def row_spreads_by_hand(block: np.ndarray, centres: bool) -> list[float]:
    """Each row's mean squared departure from its centre.

    For a centring layer that is the row's variance; for one that does not
    centre it is the row's plain mean square, which is the same expression with
    the centre held at zero. Writing it once is the oracle's version of the
    argument the module makes for writing the layer once.
    """
    return [
        sum((float(value) - centre) ** 2 for value in row) / len(row)
        for row, centre in zip(
            block.tolist(), row_centres_by_hand(block, centres), strict=False
        )
    ]


def standardised_by_hand(
    block: np.ndarray,
    centres: bool,
    epsilon: float,
    scale: np.ndarray,
    shift: np.ndarray,
) -> np.ndarray:
    """``scale * (value - centre) / sqrt(spread + epsilon) + shift``, in Python.

    One value at a time, with :func:`math.sqrt` rather than the array call, so
    that nothing about how the answer is arranged is borrowed from the thing
    being checked.
    """
    centres_of = row_centres_by_hand(block, centres)
    spreads_of = row_spreads_by_hand(block, centres)

    answer = np.empty_like(block)
    for row in range(block.shape[0]):
        deviation = math.sqrt(spreads_of[row] + epsilon)
        for feature in range(block.shape[1]):
            answer[row, feature] = float(scale[feature]) * (
                float(block[row, feature]) - centres_of[row]
            ) / deviation + float(shift[feature])
    return answer


def row_means_by_hand(block: np.ndarray) -> list[float]:
    """Each row's mean, for the claim that separates the two classes."""
    return [sum(float(value) for value in row) / len(row) for row in block.tolist()]


def row_root_mean_squares_by_hand(block: np.ndarray) -> list[float]:
    """Each row's root mean square, which is what every one of them normalises to."""
    return [
        math.sqrt(sum(float(value) ** 2 for value in row) / len(row))
        for row in block.tolist()
    ]


def bent_layer(normalizer: type[WithinRowNormalization]) -> WithinRowNormalization:
    """Three features whose scale and shift are none of them the defaults.

    A scale of one and a shift of zero would let the affine's own contribution
    to the backward pass hide, since ``scaled = arriving * scale`` is then just
    the arriving block. One scale is negative for the same reason.
    """
    return normalizer(
        n_features=3,
        scale=np.array([1.3, -0.7, 0.4]),
        shift=np.array([0.2, 0.9, -0.5]),
    )


def gradient_check_rows() -> np.ndarray:
    """Five rows of three features, none of them centred on zero.

    Rows that already had a mean of zero would make the two classes agree
    exactly, so a gradient check run on them would pass a centring layer that
    had quietly stopped centring.
    """
    generator = np.random.default_rng(21)
    return generator.normal(size=(5, 3)) * np.array([1.0, 3.0, 0.5]) + np.array(
        [2.0, -1.0, 1.0]
    )


def gradient_check_weights() -> np.ndarray:
    """The summary's weights, which are also the arriving block."""
    return np.random.default_rng(22).normal(size=(5, 3))


def summary_of(
    layer: WithinRowNormalization,
    rows: np.ndarray,
    weights: np.ndarray,
    purpose: PassPurpose = PassPurpose.PREDICTING,
) -> float:
    """A scalar reading of a whole pass, whose slope is ``weights``.

    ``sum(weights * outputs)`` differentiates to ``weights`` at every output, so
    the same block is both the scalar's definition and the arriving block a
    backward pass is handed, and the two claims are directly comparable without
    a loss object in between.

    The purpose is defaulted rather than stated, which is the difference from
    the batch normalisation spec. There the check had to say ``TRAINING`` or the
    batch route would not have been exercised at all. Here there is one route.
    """
    return float(np.sum(weights * layer.respond_to(rows, purpose).outputs))


def layer_carrying(
    layer: WithinRowNormalization, scale: np.ndarray, shift: np.ndarray
) -> WithinRowNormalization:
    """The same layer with different learned parameters, for a nudged pass."""
    return type(layer)(
        n_features=layer.n_features,
        epsilon=layer.epsilon,
        scale=scale,
        shift=shift,
    )


def gradient_for(
    layer: WithinRowNormalization, rows: np.ndarray, weights: np.ndarray
) -> LayerGradient:
    """The gradient a forward pass and its backward step produce."""
    correction = layer.correction_for(layer.respond_to(rows), weights)
    assert correction.gradient is not None
    return correction.gradient


#: The two per-feature vectors the constructor validates, so every refusal can
#: be asked of both without a builder each.
VECTOR_NAMES = [pytest.param("scale", id="scale"), pytest.param("shift", id="shift")]


def built_with(
    normalizer: type[WithinRowNormalization], vector: str, values: object
) -> WithinRowNormalization:
    """A layer of three features whose named vector is ``values``."""
    if vector == "scale":
        return normalizer(n_features=3, scale=values)  # type: ignore[arg-type]
    return normalizer(n_features=3, shift=values)  # type: ignore[arg-type]


class TestThePurposeIsIgnored:
    """The single most important claim in the file.

    Everything the module argues for rests on this. A row is normalised by its
    own contents, so there is nothing about why the pass is happening that could
    change the answer, and therefore no running statistics, no
    :class:`~oop_ml.core.network.purpose.PassPurpose` branch, no response
    subclass and no state that is not learned. Batch normalisation needs all
    four and this is the sentence that says why it does not.

    Asserted bit for bit rather than approximately, because the two passes are
    not two calculations that happen to agree, they are one calculation.
    """

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_training_and_predicting_answer_identically(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = bent_layer(normalizer)
        rows = gradient_check_rows()

        training = layer.respond_to(rows, PassPurpose.TRAINING)
        predicting = layer.respond_to(rows, PassPurpose.PREDICTING)

        assert np.array_equal(training.outputs, predicting.outputs)
        assert np.array_equal(training.scores, predicting.scores)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_caller_who_says_nothing_gets_the_same_answer_again(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """Which is the whole of what forgetting the keyword costs here.

        For batch normalisation forgetting it is the difference between the
        batch's statistics and the running ones, which is why the default there
        protects against the worse mistake. There is no worse mistake here.
        """
        layer = bent_layer(normalizer)
        rows = gradient_check_rows()

        assert np.array_equal(
            layer.respond_to(rows).outputs,
            layer.respond_to(rows, PassPurpose.TRAINING).outputs,
        )

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_backward_step_reads_either_response(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """Batch normalisation refuses a prediction-pass response outright, since
        it standardised by different numbers. There are no different numbers."""
        layer = bent_layer(normalizer)
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        training = layer.correction_for(
            layer.respond_to(rows, PassPurpose.TRAINING), weights
        )
        predicting = layer.correction_for(
            layer.respond_to(rows, PassPurpose.PREDICTING), weights
        )

        assert np.array_equal(training.passed_down, predicting.passed_down)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_response_needs_no_subclass_of_its_own(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """A plain response, because there is nothing extra to carry. The
        deviation is a function of the block the response already holds."""
        layer = bent_layer(normalizer)

        response = layer.respond_to(gradient_check_rows())

        assert type(response) is LayerResponse


class TestARowIsNormalisedByItsOwnContents:
    """A batch of one and a batch of a thousand answer the same, which is the point.

    Batch normalisation cannot say this at all. Its training pass standardises
    by the batch, so a row's answer moves with whichever other rows travelled
    with it, and the running statistics exist precisely to give prediction
    something that does not. Nothing here needs them.
    """

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_one_row_alone_answers_what_it_answered_in_a_crowd(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = bent_layer(normalizer)
        alone = np.array([[1.0, 2.0, 3.0]])
        crowd = np.vstack(
            [np.random.default_rng(7).normal(size=(1000, 3)) * 40.0, alone]
        )

        in_a_crowd = layer.respond_to(crowd).outputs[-1]

        assert np.array_equal(layer.respond_to(alone).outputs[0], in_a_crowd)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_changing_one_row_leaves_every_other_row_alone(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = bent_layer(normalizer)
        rows = gradient_check_rows()
        moved = np.array(rows, copy=True)
        moved[0] = moved[0] * 500.0 - 90.0

        assert np.array_equal(
            layer.respond_to(rows).outputs[1:], layer.respond_to(moved).outputs[1:]
        )

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_single_row_is_an_ordinary_case_rather_than_a_degenerate_one(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """Batch normalisation training on one row has zero variance everywhere
        and answers with the shift. Here one row carries its own spread."""
        layer = bent_layer(normalizer)

        answer = layer.respond_to(np.array([[1.0, 5.0, -2.0]]), PassPurpose.TRAINING)

        assert np.all(np.isfinite(answer.outputs))
        assert not np.allclose(answer.outputs, layer.shift)


class TestEveryOutputRowIsScaledToOne:
    """The family's forward invariant, and the half both classes agree on.

    Whether or not the mean came off first, what is divided out is the root mean
    square of what is left, so every normalised row comes back with a root mean
    square of one. Epsilon shrinks it by ``sqrt(spread / (spread + epsilon))``,
    which on these rows is a few parts in a million.
    """

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_normalised_block_has_unit_root_mean_square_in_every_row(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = normalizer(n_features=3)

        response = layer.respond_to(AWKWARD_BLOCK)

        assert np.allclose(
            row_root_mean_squares_by_hand(np.asarray(response.scores)), 1.0, atol=1e-4
        )

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_outputs_carry_it_too_at_unit_scale_and_zero_shift(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """The defaults are exactly the affine that changes nothing."""
        layer = normalizer(n_features=3)

        answer = layer.respond_to(AWKWARD_BLOCK)

        assert np.allclose(
            row_root_mean_squares_by_hand(np.asarray(answer.outputs)), 1.0, atol=1e-4
        )

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_row_on_a_wildly_different_scale_is_still_brought_to_one(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """A body taking one statistic for the whole block would leave the first
        row near zero and the second enormous."""
        layer = normalizer(n_features=3)
        block = np.array([[1.0, 2.0, 3.0], [4000.0, -1000.0, 9000.0]])

        answer = layer.respond_to(block)

        assert np.allclose(
            row_root_mean_squares_by_hand(np.asarray(answer.outputs)), 1.0, atol=1e-4
        )


class TestWhereTheTwoClassesPart:
    """The one term that separates them, asserted rather than described.

    Layer normalisation subtracts the row's mean, so every row it answers with
    is centred. RMS normalisation does not, so its rows keep whatever centre
    they arrived with, rescaled. Both then have a root mean square of one, which
    is why the row mean is the reading that tells them apart and the root mean
    square is not.
    """

    def test_layer_normalisation_leaves_every_row_centred(self) -> None:
        layer = LayerNormalization(n_features=3)

        answer = layer.respond_to(UNCENTRED_BLOCK)

        assert np.allclose(
            row_means_by_hand(np.asarray(answer.outputs)), 0.0, atol=1e-12
        )

    def test_rms_normalisation_demonstrably_does_not(self) -> None:
        """Not merely "is not asserted to be zero". Every row of this block has a
        mean far from zero and the layer only rescales it, so what comes out is
        the row's mean divided by its root mean square, which on these four rows
        is 0.98, 0.997, -0.99 and 0.82."""
        layer = RMSNormalization(n_features=3)

        answer = layer.respond_to(UNCENTRED_BLOCK)

        means = np.asarray(row_means_by_hand(np.asarray(answer.outputs)))
        assert np.all(np.abs(means) > 0.8)
        assert np.allclose(
            means,
            np.asarray(row_means_by_hand(UNCENTRED_BLOCK))
            / np.asarray(row_root_mean_squares_by_hand(UNCENTRED_BLOCK)),
            atol=1e-5,
        )

    def test_the_two_answer_differently_on_an_uncentred_block(self) -> None:
        centring = LayerNormalization(n_features=3)
        plain = RMSNormalization(n_features=3)

        assert not np.allclose(
            centring.respond_to(UNCENTRED_BLOCK).outputs,
            plain.respond_to(UNCENTRED_BLOCK).outputs,
        )

    def test_on_a_row_that_was_already_centred_they_agree_exactly(self) -> None:
        """Which is the cleanest statement of what the difference is. Subtracting
        a mean of zero is subtracting nothing, so the two become one function."""
        centred = np.array([[-3.0, -1.0, 1.0, 3.0], [2.0, -6.0, 6.0, -2.0]])

        assert np.array_equal(
            LayerNormalization(n_features=4).respond_to(centred).outputs,
            RMSNormalization(n_features=4).respond_to(centred).outputs,
        )

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_each_class_reports_which_one_it_is(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        assert normalizer(n_features=2).centres is CENTRES_BY_DEFINITION[normalizer]

    def test_shifting_a_whole_row_changes_nothing_for_the_centring_one(self) -> None:
        """Centring removes the offset, so it cannot survive the layer."""
        layer = LayerNormalization(n_features=3)
        moved = AWKWARD_BLOCK + np.array([[100.0], [-0.5], [6.0], [0.0]])

        assert np.allclose(
            layer.respond_to(AWKWARD_BLOCK).outputs, layer.respond_to(moved).outputs
        )

    def test_shifting_a_whole_row_changes_everything_for_the_other(self) -> None:
        """The same comparison the other way round, so the test above is asserting
        a difference between the two classes rather than a constant."""
        layer = RMSNormalization(n_features=3)
        moved = AWKWARD_BLOCK + np.array([[100.0], [-0.5], [6.0], [0.0]])

        assert not np.allclose(
            layer.respond_to(AWKWARD_BLOCK).outputs, layer.respond_to(moved).outputs
        )


class TestTheLayerNormalisationForwardPassWorkedByHand:
    def layer(self) -> LayerNormalization:
        return LayerNormalization(
            n_features=4,
            epsilon=LAYER_HAND_EPSILON,
            scale=HAND_SCALE,
            shift=HAND_SHIFT,
        )

    def test_the_normalised_block_is_the_numbers_computed_on_paper(self) -> None:
        """Mean 4, spread 5, epsilon 11, deviation 4, so every normalised value
        is a quarter or three quarters one way or the other."""
        response = self.layer().respond_to(LAYER_HAND_BLOCK)

        assert np.allclose(
            response.scores,
            [[-0.75, -0.25, 0.25, 0.75], [0.75, -0.75, 0.25, -0.25]],
        )

    def test_the_outputs_are_those_numbers_through_the_affine(self) -> None:
        answer = self.layer().respond_to(LAYER_HAND_BLOCK)

        assert np.allclose(
            answer.outputs,
            [[-0.5, -3.5, 0.0, 0.375], [2.5, -8.5, 0.0, -0.125]],
        )

    def test_the_spread_is_the_biased_one(self) -> None:
        """Dividing by ``n - 1`` would give 20 over 3 rather than 5, so the
        deviation would be ``sqrt(11 + 20 / 3)`` rather than exactly 4."""
        response = self.layer().respond_to(LAYER_HAND_BLOCK)

        assert row_spreads_by_hand(LAYER_HAND_BLOCK, centres=True) == [5.0, 5.0]
        assert np.allclose(np.abs(response.scores).max(), 0.75)


class TestTheRMSNormalisationForwardPassWorkedByHand:
    def layer(self) -> RMSNormalization:
        return RMSNormalization(
            n_features=4,
            epsilon=RMS_HAND_EPSILON,
            scale=HAND_SCALE,
            shift=HAND_SHIFT,
        )

    def test_the_normalised_block_is_the_numbers_computed_on_paper(self) -> None:
        """Mean square 3, epsilon 1, deviation 2, so the answer is the row
        halved. The row mean of 1.5 is left exactly where it was, scaled."""
        response = self.layer().respond_to(RMS_HAND_BLOCK)

        assert np.allclose(
            response.scores, [[0.5, 0.5, 0.5, 1.5], [1.5, 0.5, 0.5, 0.5]]
        )

    def test_the_outputs_are_those_numbers_through_the_affine(self) -> None:
        answer = self.layer().respond_to(RMS_HAND_BLOCK)

        assert np.allclose(
            answer.outputs, [[2.0, 4.0, -0.25, 0.75], [4.0, 4.0, -0.25, 0.25]]
        )

    def test_the_mean_was_never_taken(self) -> None:
        """A centring body would find a mean of 1.5, a spread of 0.75, and a
        deviation of ``sqrt(1.75)``. Nothing about the shape of the answer says
        which of those two happened, which is why the numbers are pinned."""
        response = self.layer().respond_to(RMS_HAND_BLOCK)

        assert row_spreads_by_hand(RMS_HAND_BLOCK, centres=False) == [3.0, 3.0]
        assert response.scores[0, 0] == pytest.approx(0.5)


class TestAgainstAPlainPythonOracle:
    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    @pytest.mark.parametrize(
        ("scale", "shift"),
        [
            (np.ones(3), np.zeros(3)),
            (np.array([2.0, -1.5, 0.25]), np.array([1.0, 0.0, -3.0])),
            (np.array([-1.0, -1.0, -1.0]), np.array([5.0, 5.0, 5.0])),
        ],
        ids=["defaults", "mixed", "reflected"],
    )
    def test_the_outputs_agree_with_the_definition(
        self,
        normalizer: type[WithinRowNormalization],
        scale: np.ndarray,
        shift: np.ndarray,
    ) -> None:
        """The oracle is written from the definition, one value at a time, and
        takes the centring boolean from this file rather than from the layer."""
        layer = normalizer(n_features=3, scale=scale, shift=shift)

        answer = layer.respond_to(AWKWARD_BLOCK)

        assert np.allclose(
            answer.outputs,
            standardised_by_hand(
                AWKWARD_BLOCK,
                CENTRES_BY_DEFINITION[normalizer],
                layer.epsilon,
                scale,
                shift,
            ),
        )

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_scores_are_the_standardisation_before_the_affine(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """Here the affine plays the part an activation plays elsewhere, so the
        pre-affine block is what a score has always been."""
        layer = normalizer(n_features=3)

        response = layer.respond_to(AWKWARD_BLOCK)

        assert np.allclose(
            response.scores,
            standardised_by_hand(
                AWKWARD_BLOCK,
                CENTRES_BY_DEFINITION[normalizer],
                layer.epsilon,
                np.ones(3),
                np.zeros(3),
            ),
        )

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_response_carries_the_block_that_was_read(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        response = normalizer(n_features=3).respond_to(AWKWARD_BLOCK)

        assert np.allclose(response.inputs, AWKWARD_BLOCK)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_it_answers_with_the_arrangement_it_read(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        answer = normalizer(n_features=3).respond_to(AWKWARD_BLOCK)

        assert answer.outputs.shape == AWKWARD_BLOCK.shape


class TestTheAffineCanUndoIt:
    """The claim that normalisation costs the layer no expressiveness.

    Standardising alone would be a constraint, forbidding an off-centre answer
    the network may genuinely want. With the affine restored the layer can
    represent the identity exactly, and this is that sentence run as arithmetic.

    It needs a block whose rows share a statistic, and that is a real difference
    from the batch sibling rather than an inconvenience of the fixture. There
    the statistic is one value per feature and so is the scale, so any block will
    do. Here the statistic is one value per *row* while the scale is still one
    value per feature, so the undoing scale exists for a whole block only when
    every row agrees about what it is. Rows that are permutations of one another
    agree, whether the centre is the mean or is zero.
    """

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_undoing_scale_and_shift_return_the_inputs(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        centres = CENTRES_BY_DEFINITION[normalizer]
        epsilon = IDENTITY_EPSILON[normalizer]
        spreads = row_spreads_by_hand(PERMUTED_BLOCK, centres)
        row_centres = row_centres_by_hand(PERMUTED_BLOCK, centres)
        assert spreads == [spreads[0]] * len(spreads)
        assert row_centres == [row_centres[0]] * len(row_centres)

        width = PERMUTED_BLOCK.shape[1]
        undoing = normalizer(
            n_features=width,
            epsilon=epsilon,
            scale=np.full(width, math.sqrt(spreads[0] + epsilon)),
            shift=np.full(width, row_centres[0]),
        )

        assert np.allclose(undoing.respond_to(PERMUTED_BLOCK).outputs, PERMUTED_BLOCK)

    @pytest.mark.parametrize(
        ("normalizer", "undoing_scale"),
        [(LayerNormalization, 4.0), (RMSNormalization, 5.0)],
        ids=["LayerNormalization", "RMSNormalization"],
    )
    def test_the_undoing_scale_is_the_whole_number_the_epsilon_was_chosen_for(
        self, normalizer: type[WithinRowNormalization], undoing_scale: float
    ) -> None:
        """Spread 5 plus 11 is 16 for the centring one, mean square 21 plus 4 is
        25 for the other. Stated so that the fixture above is reproducible on
        paper rather than only by running it."""
        centres = CENTRES_BY_DEFINITION[normalizer]
        spread = row_spreads_by_hand(PERMUTED_BLOCK, centres)[0]

        assert math.sqrt(spread + IDENTITY_EPSILON[normalizer]) == undoing_scale


class TestARowWithNothingInIt:
    """What epsilon is for, and why it sits inside the square root.

    The degenerate row is not the same row for the two classes, which is worth
    a test rather than a sentence. A centring layer divides by the row's spread
    about its own mean, so any constant row is degenerate. One that does not
    centre divides by the row's mean square, so only the row of zeros is.
    """

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_row_of_zeros_normalises_to_zeros_rather_than_nan(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = normalizer(n_features=4)

        response = layer.respond_to(np.array([[0.0, 0.0, 0.0, 0.0]]))

        assert np.all(np.isfinite(response.outputs))
        assert np.allclose(response.scores, 0.0)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_row_of_zeros_answers_with_the_shift(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """Zero normalised times any scale is zero, so all that is left is the
        offset, which is the honest answer for a row carrying nothing."""
        layer = normalizer(
            n_features=3,
            scale=np.array([2.0, -1.0, 7.0]),
            shift=np.array([0.5, 1.5, -2.5]),
        )

        answer = layer.respond_to(np.zeros((1, 3)))

        assert np.allclose(answer.outputs, [[0.5, 1.5, -2.5]])

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_layer_of_one_feature_still_answers(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """A single value has no spread about its own mean at all, which makes
        this the degenerate case for the centring class and an ordinary one for
        the other. Both must answer a number."""
        layer = normalizer(n_features=1, scale=np.array([3.0]), shift=np.array([-1.0]))

        answer = layer.respond_to(np.array([[2.0], [-5.0], [0.0]]))

        assert np.all(np.isfinite(answer.outputs))

    def test_a_constant_row_answers_with_the_shift_when_the_layer_centres(self) -> None:
        block = np.array([[5.0, 5.0, 5.0], [1.5, 1.5, 1.5]])
        layer = LayerNormalization(
            n_features=3,
            scale=np.array([1.0, 3.0, -2.0]),
            shift=np.array([0.0, -2.0, 4.0]),
        )

        answer = layer.respond_to(block)

        assert np.allclose(answer.outputs, [[0.0, -2.0, 4.0], [0.0, -2.0, 4.0]])

    def test_a_constant_row_is_nothing_special_when_it_does_not(self) -> None:
        """The counterpart, and the reason the claim above cannot be a family one.
        A constant non-zero row has a mean square equal to its own square, so it
        normalises to ones rather than to zeros."""
        layer = RMSNormalization(n_features=3)

        answer = layer.respond_to(np.array([[5.0, 5.0, 5.0]]))

        assert np.allclose(answer.outputs, 1.0, atol=1e-5)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_single_feature_layer_centring_or_not_is_finite(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = normalizer(n_features=1)

        answer = layer.respond_to(np.array([[1e-12], [1e12], [0.0]]))

        assert np.all(np.isfinite(answer.outputs))


class TestConstruction:
    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_its_shape_is_the_same_width_on_both_sides(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """This layer moves no join, which is what lets it go anywhere."""
        layer = normalizer(n_features=7)

        assert layer.shape == LayerShape(n_inputs=7, n_outputs=7)
        assert layer.shape.reads == layer.shape.answers

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_it_reports_the_width_it_was_given(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        assert normalizer(n_features=12).n_features == 12

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_scale_starts_at_one_and_the_shift_at_zero(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """Which starts the layer standardising and doing nothing else. RMS
        normalisation conventionally has no shift at all, and zeros are exactly
        that model, which is why one class shape serves both."""
        layer = normalizer(n_features=4)

        assert np.allclose(layer.scale, 1.0)
        assert np.allclose(layer.shift, 0.0)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_both_default_vectors_are_one_value_per_feature(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = normalizer(n_features=5)

        assert layer.scale.shape == (5,)
        assert layer.shift.shape == (5,)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_epsilon_default_is_reported(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        assert normalizer(n_features=2).epsilon == pytest.approx(1e-5)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_supplied_vectors_are_kept(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = bent_layer(normalizer)

        assert np.allclose(layer.scale, [1.3, -0.7, 0.4])
        assert np.allclose(layer.shift, [0.2, 0.9, -0.5])

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    @pytest.mark.parametrize("vector", VECTOR_NAMES)
    def test_a_supplied_vector_is_copied_rather_than_aliased(
        self, normalizer: type[WithinRowNormalization], vector: str
    ) -> None:
        """A caller who kept the array must not be able to move the layer."""
        supplied = np.array([2.0, 3.0, 4.0])
        layer = built_with(normalizer, vector, supplied)
        before = np.array(getattr(layer, vector), copy=True)

        supplied[0] = 99.0

        assert np.array_equal(getattr(layer, vector), before)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    @pytest.mark.parametrize("vector", VECTOR_NAMES)
    def test_the_learned_vectors_are_frozen(
        self, normalizer: type[WithinRowNormalization], vector: str
    ) -> None:
        layer = normalizer(n_features=3)

        with pytest.raises(ValueError):
            getattr(layer, vector)[0] = 1.0

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_it_names_its_own_class_when_printed(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """The base builds the text, so a subclass that inherited a hard-coded
        name would misreport which normaliser a reader was looking at."""
        assert repr(normalizer(n_features=3)).startswith(normalizer.__name__)


class TestRefusedConfigurations:
    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    @pytest.mark.parametrize("epsilon", [0.0, -1e-5, -1.0, np.nan, np.inf])
    def test_an_epsilon_that_is_not_strictly_positive_is_refused(
        self, normalizer: type[WithinRowNormalization], epsilon: float
    ) -> None:
        """Zero is the division by zero this parameter exists to prevent."""
        with pytest.raises(InvalidValuesError):
            normalizer(n_features=3, epsilon=epsilon)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    @pytest.mark.parametrize("value", ["small", None])
    def test_an_epsilon_that_is_not_a_number_is_refused(
        self, normalizer: type[WithinRowNormalization], value: object
    ) -> None:
        with pytest.raises(InvalidValuesError):
            normalizer(n_features=3, epsilon=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    @pytest.mark.parametrize("n_features", [0, -1, -8])
    def test_a_width_below_one_is_refused(
        self, normalizer: type[WithinRowNormalization], n_features: int
    ) -> None:
        with pytest.raises(InvalidValuesError):
            normalizer(n_features=n_features)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    @pytest.mark.parametrize("vector", VECTOR_NAMES)
    @pytest.mark.parametrize("length", [2, 4, 0])
    def test_a_vector_of_the_wrong_length_is_refused(
        self, normalizer: type[WithinRowNormalization], vector: str, length: int
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            built_with(normalizer, vector, np.ones(length))

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    @pytest.mark.parametrize("vector", VECTOR_NAMES)
    def test_a_vector_that_is_not_one_dimensional_is_refused(
        self, normalizer: type[WithinRowNormalization], vector: str
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            built_with(normalizer, vector, np.ones((3, 1)))

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    @pytest.mark.parametrize("vector", VECTOR_NAMES)
    @pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
    def test_a_vector_carrying_a_non_finite_entry_is_refused(
        self, normalizer: type[WithinRowNormalization], vector: str, poison: float
    ) -> None:
        values = np.ones(3)
        values[1] = poison

        with pytest.raises(InvalidValuesError):
            built_with(normalizer, vector, values)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    @pytest.mark.parametrize("vector", VECTOR_NAMES)
    def test_a_vector_that_is_not_numbers_at_all_is_refused(
        self, normalizer: type[WithinRowNormalization], vector: str
    ) -> None:
        with pytest.raises(InvalidValuesError):
            built_with(normalizer, vector, "per feature")


class TestRefusedBlocks:
    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_block_of_the_wrong_width_is_refused(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            normalizer(n_features=3).respond_to(np.zeros((4, 5)))

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_block_of_no_rows_is_refused(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        with pytest.raises(EmptyValuesError):
            normalizer(n_features=3).respond_to(np.zeros((0, 3)))

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_non_finite_entry_is_refused_where_it_enters(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """One ``nan`` would otherwise poison its whole row's statistic and travel
        to every feature that shares it."""
        poisoned = np.array(AWKWARD_BLOCK, copy=True)
        poisoned[2, 1] = np.nan

        with pytest.raises(InvalidValuesError):
            normalizer(n_features=3).respond_to(poisoned)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_block_that_is_not_numbers_at_all_is_refused(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        with pytest.raises(InvalidValuesError):
            normalizer(n_features=3).respond_to("a batch")  # type: ignore[arg-type]

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_one_dimensional_block_is_refused(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """A single row is ``(1, n_features)`` and not ``(n_features,)``. The
        leading axis is always the rows, which is the one rule every layer here
        shares."""
        with pytest.raises(ShapeMismatchError):
            normalizer(n_features=3).respond_to(np.array([1.0, 2.0, 3.0]))


class TestTheGradientCheck:
    """Nudge one number, watch the summary, compare against the claim.

    Run over the inputs, the scale and the shift in turn, for both classes. The
    inputs are the one that matters, because the naive reading of the backward
    pass leaves the other two bit-identical and would sail through a spec that
    checked only those. See the module docstring for the measured disagreement.
    """

    def measured_input_gradient(
        self, layer: WithinRowNormalization, rows: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        measured = np.zeros_like(rows)
        for row in range(rows.shape[0]):
            for feature in range(rows.shape[1]):
                moved = []
                for direction in (+NUDGE, -NUDGE):
                    nudged = np.array(rows, copy=True)
                    nudged[row, feature] += direction
                    moved.append(summary_of(layer, nudged, weights))
                measured[row, feature] = (moved[0] - moved[1]) / (2.0 * NUDGE)
        return measured

    def measured_vector_gradient(
        self,
        layer: WithinRowNormalization,
        rows: np.ndarray,
        weights: np.ndarray,
        vector: str,
    ) -> np.ndarray:
        measured = np.zeros(layer.n_features)
        for feature in range(layer.n_features):
            moved = []
            for direction in (+NUDGE, -NUDGE):
                scale = np.array(layer.scale, copy=True)
                shift = np.array(layer.shift, copy=True)
                if vector == "scale":
                    scale[feature] += direction
                else:
                    shift[feature] += direction
                moved.append(
                    summary_of(layer_carrying(layer, scale, shift), rows, weights)
                )
            measured[feature] = (moved[0] - moved[1]) / (2.0 * NUDGE)
        return measured

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_every_input_slope_matches_a_finite_difference(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """The routes, collected. This is the test the module is about."""
        layer = bent_layer(normalizer)
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        correction = layer.correction_for(layer.respond_to(rows), weights)
        measured = self.measured_input_gradient(layer, rows, weights)

        assert np.allclose(correction.passed_down, measured, atol=1e-7)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_every_scale_slope_matches_a_finite_difference(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = bent_layer(normalizer)
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        gradient = gradient_for(layer, rows, weights)
        measured = self.measured_vector_gradient(layer, rows, weights, "scale")

        assert np.allclose(gradient.weights[:, 0], measured, atol=1e-7)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_every_shift_slope_matches_a_finite_difference(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = bent_layer(normalizer)
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        gradient = gradient_for(layer, rows, weights)
        measured = self.measured_vector_gradient(layer, rows, weights, "shift")

        assert np.allclose(gradient.biases, measured, atol=1e-7)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_input_slopes_hold_on_a_wider_row(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """Three rows of nine features, so the ``1 / n`` inside the subtracted
        terms is a different number than in the fixture above, and the reduction
        runs over an axis long enough for a misplaced one to show."""
        generator = np.random.default_rng(31)
        layer = normalizer(
            n_features=9,
            scale=generator.normal(size=9),
            shift=generator.normal(size=9),
        )
        rows = np.random.default_rng(32).normal(size=(3, 9)) * 2.0 + 1.5
        weights = np.random.default_rng(33).normal(size=(3, 9))

        correction = layer.correction_for(layer.respond_to(rows), weights)
        measured = self.measured_input_gradient(layer, rows, weights)

        assert np.allclose(correction.passed_down, measured, atol=1e-7)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_input_slopes_hold_through_a_row_with_no_spread(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """Where the deviation is epsilon alone. A larger epsilon keeps the
        slopes near one rather than near ``1 / sqrt(1e-5)``, so a genuine
        disagreement is not hidden inside a relative tolerance. The flat row is
        constant and non-zero, so it is degenerate for one class and ordinary
        for the other, and both are exercised."""
        layer = normalizer(
            n_features=3,
            epsilon=0.25,
            scale=np.array([1.5, -0.8, 2.0]),
            shift=np.array([0.1, 0.2, 0.3]),
        )
        rows = np.array([[4.0, 4.0, 4.0], [1.0, 2.0, -0.5], [0.0, 0.0, 0.0]])
        weights = np.random.default_rng(34).normal(size=(3, 3))

        correction = layer.correction_for(layer.respond_to(rows), weights)
        measured = self.measured_input_gradient(layer, rows, weights)

        assert np.allclose(correction.passed_down, measured, atol=1e-7)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_slopes_do_not_depend_on_the_purpose_either(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """The check is run against a defaulted pass, so this is what says the
        stated one would have given the same answer."""
        layer = bent_layer(normalizer)
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        correction = layer.correction_for(layer.respond_to(rows), weights)
        measured = self.measured_input_gradient(layer, rows, weights)

        assert np.allclose(correction.passed_down, measured, atol=1e-7)
        assert np.allclose(
            [summary_of(layer, rows, weights, PassPurpose.TRAINING)],
            [summary_of(layer, rows, weights, PassPurpose.PREDICTING)],
        )


class TestTheStructuralSignaturesOfTheBackwardPass:
    """Two cheap readings that the naive pass cannot fake.

    The first is the family's. Every one of these layers rescales a row by that
    row's own root mean square, so blame that merely stretches a row along the
    direction it already points buys nothing, and ``sum(passed_down *
    normalised)`` down a row is zero but for an epsilon-sized remainder. The
    naive pass leaves the whole of that term in.

    The second belongs to the centring class alone, and is the exact analogue of
    the column sums in the batch spec. Shifting a whole row by a constant cannot
    change what layer normalisation answers, so its blame sums to zero along
    every row, exactly. RMS normalisation has no such claim, because shifting a
    row does change what it answers, and asserting it there would assert
    something false.
    """

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_blame_is_orthogonal_to_the_normalised_row(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """Zero but for an epsilon-sized remainder, which is why the tolerance is
        loose and is stated as a measurement rather than as a rounding. The exact
        leftover is ``n * mean(scaled * normalised) * epsilon / deviation ** 3``,
        so it shrinks with epsilon rather than with the arithmetic. Measured at
        the default 1e-5 it is 1.6e-05 for the centring class and 1.8e-06 for the
        other, where the naive backward pass gives 1.236 and 0.754.
        """
        layer = bent_layer(normalizer)
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        response = layer.respond_to(rows)
        correction = layer.correction_for(response, weights)

        along = (np.asarray(correction.passed_down) * np.asarray(response.scores)).sum(
            axis=1
        )
        assert np.allclose(along, 0.0, atol=1e-3)

    def test_the_blame_sums_to_zero_along_every_row_when_the_layer_centres(
        self,
    ) -> None:
        """The structural signature of the three-route form, and the cheapest way
        to see the naive one fail."""
        layer = bent_layer(LayerNormalization)
        rows = gradient_check_rows()

        correction = layer.correction_for(
            layer.respond_to(rows), gradient_check_weights()
        )

        assert np.allclose(
            np.asarray(correction.passed_down).sum(axis=1), 0.0, atol=1e-12
        )

    def test_it_does_not_sum_to_zero_when_the_layer_does_not_centre(self) -> None:
        """The counterpart, so the claim above is asserting a difference between
        the two classes rather than a constant. RMS normalisation is not
        invariant to a shift, so its blame along a row has no reason to cancel
        and does not."""
        layer = bent_layer(RMSNormalization)
        rows = gradient_check_rows()

        correction = layer.correction_for(
            layer.respond_to(rows), gradient_check_weights()
        )

        assert not np.allclose(
            np.asarray(correction.passed_down).sum(axis=1), 0.0, atol=1e-6
        )


class TestTheGradientObject:
    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_gradient_is_a_plain_layer_gradient(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """Nothing rides alongside it, because nothing here is remembered between
        batches. Batch normalisation needs a subclass carrying the batch's own
        statistics, and this is the same contrast in one assertion."""
        layer = bent_layer(normalizer)

        correction = layer.correction_for(
            layer.respond_to(gradient_check_rows()), gradient_check_weights()
        )

        assert type(correction.gradient) is LayerGradient

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_weight_block_is_one_row_per_feature(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """``(n_features, 1)``, not the transposed ``(1, n_features)``.

        Each feature is a unit reading a single number, since the answer for
        feature ``j`` is ``scale_j * normalised_j + shift_j``, so this is
        ``LayerGradient``'s ordinary ``(n_neurons, n_inputs)`` rather than a
        shape bent to fit. The width is three deliberately, because at a width
        of one the two readings agree and the mistake survives.
        """
        gradient = gradient_for(
            bent_layer(normalizer), gradient_check_rows(), gradient_check_weights()
        )

        assert gradient.weights.shape == (3, 1)
        assert gradient.biases.shape == (3,)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_blame_passed_down_has_the_shape_of_what_was_read(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = bent_layer(normalizer)
        rows = gradient_check_rows()

        correction = layer.correction_for(
            layer.respond_to(rows), gradient_check_weights()
        )

        assert correction.passed_down.shape == rows.shape

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_shift_slope_is_the_blame_summed_down_the_rows(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """One shift serves every row, so every row it answered contributes."""
        gradient = gradient_for(
            normalizer(n_features=3), AWKWARD_BLOCK, np.ones_like(AWKWARD_BLOCK)
        )

        assert np.allclose(gradient.biases, AWKWARD_BLOCK.shape[0])


class TestRefusedBackwardSteps:
    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_an_arriving_block_of_the_wrong_width_is_refused(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = bent_layer(normalizer)
        response = layer.respond_to(gradient_check_rows())

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((5, 4)))

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_an_arriving_block_of_the_wrong_row_count_is_refused(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = bent_layer(normalizer)
        response = layer.respond_to(gradient_check_rows())

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((4, 3)))

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_response_from_a_differently_shaped_layer_is_refused(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """It carries real numbers of a plausible shape, so nothing else would
        catch it and the blame would be routed to arbitrary places."""
        layer = bent_layer(normalizer)
        narrower = normalizer(n_features=2)
        response = narrower.respond_to(
            np.array([[1.0, 2.0], [3.0, 5.0], [0.5, -1.0], [2.0, 2.5], [1.0, 0.0]])
        )

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((5, 3)))

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_an_arriving_block_that_is_not_numbers_is_refused(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = bent_layer(normalizer)
        response = layer.respond_to(gradient_check_rows())

        with pytest.raises(InvalidValuesError):
            layer.correction_for(response, "blame")  # type: ignore[arg-type]


class TestStepping:
    def stepped(
        self, layer: WithinRowNormalization, learning_rate: float = 0.1
    ) -> WithinRowNormalization:
        gradient = gradient_for(layer, gradient_check_rows(), gradient_check_weights())
        stepped = layer.stepped_by(gradient, learning_rate)
        assert isinstance(stepped, WithinRowNormalization)
        return stepped

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_scale_and_shift_move_against_their_slopes(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """An ordinary subtract-the-slope, which is the whole of it. Batch
        normalisation's step also folds in an exponential moving average of the
        batch, because it remembers something between batches. Nothing here
        does."""
        layer = bent_layer(normalizer)
        gradient = gradient_for(layer, gradient_check_rows(), gradient_check_weights())

        stepped = layer.stepped_by(gradient, learning_rate=0.1)

        assert np.allclose(stepped.scale, layer.scale - 0.1 * gradient.weights[:, 0])
        assert np.allclose(stepped.shift, layer.shift - 0.1 * gradient.biases)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_stepped_layer_is_the_same_concrete_class(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """The base rebuilds through ``type(self)``, and a hard-coded class there
        would silently turn every RMS layer into a centring one on its first
        step, or the other way round."""
        stepped = self.stepped(bent_layer(normalizer))

        assert type(stepped) is normalizer
        assert stepped.centres is CENTRES_BY_DEFINITION[normalizer]

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_stepped_layer_is_a_new_object(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = bent_layer(normalizer)

        assert self.stepped(layer) is not layer

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_original_layer_is_untouched(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """Training does not mutate, which is what keeps every block frozen."""
        layer = bent_layer(normalizer)
        scale_before = np.array(layer.scale, copy=True)
        shift_before = np.array(layer.shift, copy=True)

        self.stepped(layer, learning_rate=0.5)

        assert np.array_equal(layer.scale, scale_before)
        assert np.array_equal(layer.shift, shift_before)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_the_step_keeps_the_configuration_it_was_not_asked_to_change(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        layer = normalizer(
            n_features=3,
            epsilon=0.5,
            scale=np.array([1.3, -0.7, 0.4]),
            shift=np.array([0.2, 0.9, -0.5]),
        )

        stepped = self.stepped(layer)

        assert stepped.epsilon == pytest.approx(0.5)
        assert stepped.n_features == 3
        assert stepped.shape == layer.shape

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_stepping_by_nothing_is_refused(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """This layer has parameters, so a missing gradient is a mistake rather
        than a layer with nothing to learn."""
        with pytest.raises(ShapeMismatchError):
            normalizer(n_features=3).stepped_by(None, learning_rate=0.1)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    @pytest.mark.parametrize("width", [2, 4])
    def test_a_gradient_of_the_wrong_width_is_refused(
        self, normalizer: type[WithinRowNormalization], width: int
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            normalizer(n_features=3).stepped_by(
                LayerGradient(weights=np.zeros((width, 1)), biases=np.zeros(width)),
                learning_rate=0.1,
            )

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_transposed_gradient_is_refused(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """``(1, n_features)`` is the tempting misreading, and at a width of three
        it names one unit reading three numbers rather than three units reading
        one each."""
        with pytest.raises(ShapeMismatchError):
            normalizer(n_features=3).stepped_by(
                LayerGradient(weights=np.zeros((1, 3)), biases=np.zeros(1)),
                learning_rate=0.1,
            )

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_plain_layer_gradient_is_what_it_wants(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """The contrast with the batch sibling, which refuses one. There are no
        running figures to update, so the parameter slopes are the whole of what
        a step needs."""
        layer = normalizer(n_features=3)

        stepped = layer.stepped_by(
            LayerGradient(weights=np.ones((3, 1)), biases=np.ones(3)),
            learning_rate=0.25,
        )

        assert np.allclose(stepped.scale, 0.75)
        assert np.allclose(stepped.shift, -0.25)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_step_lowers_the_summary_it_was_asked_to_lower(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """The point of the whole exercise, on one small step."""
        layer = bent_layer(normalizer)
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        stepped = self.stepped(layer, learning_rate=0.05)

        assert summary_of(stepped, rows, weights) < summary_of(layer, rows, weights)


class TestInAStack:
    def stack(
        self, normalizer: type[WithinRowNormalization], seed: int = 4
    ) -> LayerStack:
        """Four inputs, a bent middle of three, normalisation, two outputs."""
        generator = np.random.default_rng(seed)
        hidden = DenseLayer(
            [
                Neuron(
                    generator.normal(size=4) * 0.5,
                    bias=float(generator.normal()) * 0.5,
                    activation=HyperbolicTangent(),
                )
                for _ in range(3)
            ]
        )
        output = DenseLayer(
            [
                Neuron(
                    generator.normal(size=3) * 0.5,
                    bias=float(generator.normal()) * 0.5,
                    activation=Identity(),
                )
                for _ in range(2)
            ]
        )
        return LayerStack([hidden, normalizer(n_features=3), output])

    def rows(self, seed: int = 5) -> np.ndarray:
        return np.random.default_rng(seed).normal(size=(8, 4))

    def targets(self, seed: int = 6) -> np.ndarray:
        return np.random.default_rng(seed).normal(size=(8, 2))

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_it_moves_no_join(self, normalizer: type[WithinRowNormalization]) -> None:
        """Same width on both sides, so it can be inserted into a sound stack
        without making it unsound."""
        assert self.stack(normalizer).shape == LayerShape(n_inputs=4, n_outputs=2)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_it_can_sit_at_the_bottom_of_a_stack_too(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        hidden = DenseLayer(
            [
                Neuron(np.ones(4) * 0.1, bias=0.0, activation=Identity())
                for _ in range(2)
            ]
        )

        stack = LayerStack([normalizer(n_features=4), hidden])

        assert stack.shape == LayerShape(n_inputs=4, n_outputs=2)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_backward_pass_produces_a_plain_gradient_in_layer_order(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        backward = self.stack(normalizer).backward_pass(
            self.rows(), self.targets(), SquaredError()
        )

        assert len(backward) == 3
        assert type(backward[1]) is LayerGradient

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_repeated_steps_lower_the_loss(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        stack = self.stack(normalizer)
        rows = self.rows()
        targets = self.targets()
        loss = SquaredError()

        first = stack.backward_pass(rows, targets, loss).loss
        for _ in range(20):
            stack = stack.stepped_by(
                stack.backward_pass(rows, targets, loss), learning_rate=0.2
            )
        last = stack.backward_pass(rows, targets, loss).loss

        assert last < first * 0.9

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_trained_stack_predicts_one_row_exactly_as_it_predicts_eight(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """The whole difference from the batch sibling, at the end of a training
        run rather than in the abstract. There the answer for one row is only
        available because running statistics were kept for it. Here the row
        carries its own.

        ``allclose`` rather than the bit-for-bit equality the layer on its own
        keeps, and the difference is not the normaliser. The dense layer beneath
        it is a matrix multiply, and BLAS sums the products in a different order
        for a block of one row than for a block of eight. Measured, the two
        answers agree to 1.1e-16.
        """
        stack = self.stack(normalizer)
        rows = self.rows()
        targets = self.targets()
        loss = SquaredError()

        for _ in range(5):
            stack = stack.stepped_by(
                stack.backward_pass(rows, targets, loss), learning_rate=0.2
            )

        alone = stack.respond_to(rows[:1], PassPurpose.PREDICTING).outputs
        together = stack.respond_to(rows, PassPurpose.PREDICTING).outputs

        assert np.allclose(alone[0], together[0], rtol=0.0, atol=1e-14)

    @pytest.mark.parametrize("normalizer", NORMALIZERS)
    def test_a_trained_stack_answers_the_same_thing_while_training(
        self, normalizer: type[WithinRowNormalization]
    ) -> None:
        """No forward pass anywhere in the stack was ever a different function,
        so nothing had to be remembered to make the two agree."""
        stack = self.stack(normalizer)
        rows = self.rows()
        targets = self.targets()
        loss = SquaredError()

        for _ in range(5):
            stack = stack.stepped_by(
                stack.backward_pass(rows, targets, loss), learning_rate=0.2
            )

        assert np.array_equal(
            stack.respond_to(rows, PassPurpose.TRAINING).outputs,
            stack.respond_to(rows, PassPurpose.PREDICTING).outputs,
        )
