"""Spec for dropout, where the sharpest test is about a number nobody looks at.

Dropout is small enough that most of it can be checked by reading. The forward
pass silences a unit or it does not, the backward pass multiplies by the same
thing, and the shape never moves. What makes the file worth its length is that
its three real failure modes all run, all produce finite and plausible numbers,
and none of them raise.

The first is reading the mask back off the answers. A zero in the output looks
like a unit that was dropped, and recovering the mask that way is cheap and
removes a stored block. It is also wrong, because an input that was genuinely
zero and genuinely kept leaves a zero too, and those are not rare -- they are
what a rectified unit produces for every negative score, and most of the border
of a scanned digit. ``TestTheZeroInputTrap`` feeds blocks whose kept entries are
exactly zero and asserts they are still paid. Run against an implementation that
recovers the mask from ``outputs != 0``, 135 of the 138 tests here pass and only
that class fails, with the whole block of blame missing.

The second is the scale on the way back. A survivor is amplified by
``1 / (1 - p)`` going up, so its blame is amplified by the same factor coming
down, and a backward pass that multiplies by a bare mask instead answers a
gradient uniformly too small by ``1 - p``. Nothing about that announces itself.
The network trains, the loss falls, and the effective learning rate is quietly
not the one that was configured. So the oracle is a finite difference, computed
from forward passes alone and knowing nothing about how the backward pass was
derived. Worst absolute disagreement with the finite differences, over the four
fixtures here that drop anything at all -- at ``p = 0`` there is no scale to
forget -- is 3.8e-10 for the real backward pass, between 0.46 and 2.3 for one
that forgets the scale, and between 1.8 and 2.5 for one that ignores the mask
entirely. The tolerance is 1e-8, so there is a factor of 26 of headroom beneath
it and more than seven orders of magnitude of clearance above.

That measurement also says something the finite differences cannot, and it is
why the zero-input class exists as well as this one. On these fixtures every
input is non-zero, so a mask recovered from ``outputs != 0`` is the right mask,
and the shortcut implementation agrees with the finite differences to the same
3.8e-10 the correct one does. A gradient check is blind to it by construction,
because the fixture that would expose it is the fixture a gradient check avoids.

Holding the mask still is the one piece of machinery this file needs that the
other layer specs do not. A forward pass advances the layer's generator, so the
same layer asked twice answers differently by design, and a finite difference
over a moving target measures nothing. Every forward pass in the gradient check
therefore runs on a layer rebuilt from the same seed, which makes the first draw
identical every time and the summary a genuine function of the inputs alone.

The third is the generator. A layer that rebuilt it per pass, or a
``stepped_by`` that answered with a new layer rather than ``self``, would draw
the identical mask forever, and that is not dropout at all -- it is a smaller
network with a strange initialisation, and it trains perfectly happily.
``TestConsecutivePassesDiffer`` and ``TestSteppingKeepsTheStream`` are what
stand between those two mistakes and a green suite.

The expectation test is the one place a tolerance is chosen rather than forced,
and the arithmetic behind the choice is written beside it. It is seeded, so it
is deterministic rather than merely unlikely to fail.
"""

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
)
from oop_ml.core.network.activation import HyperbolicTangent, Identity
from oop_ml.core.network.dropout import Dropout, DropoutResponse
from oop_ml.core.network.layer import DenseLayer, LayerResponse
from oop_ml.core.network.loss import SquaredError
from oop_ml.core.network.neuron import Neuron
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.network.stack import LayerStack

#: The step a central difference takes on either side.
NUDGE = 1e-6

#: Drop probabilities whose ``1 / (1 - p)`` is a power of two, so that
#: ``value * scale`` and ``value / keep_probability`` agree bit for bit and an
#: exactness assertion is testing the layer rather than the rounding.
EXACTLY_SCALED = (0.0, 0.5, 0.75)


def training_response(layer: Dropout, block: np.ndarray) -> DropoutResponse:
    """One training pass, narrowed so the mask it carries can be read.

    ``respond_to`` promises a ``LayerResponse``, which is the honest signature
    since a prediction pass really does answer with a plain one. Under
    ``TRAINING`` it is always the richer type, and this narrowing is what lets a
    test ask for the mask at all.
    """
    response = layer.respond_to(block, PassPurpose.TRAINING)
    assert isinstance(response, DropoutResponse)
    return response


def mask_of(layer: Dropout, block: np.ndarray) -> np.ndarray:
    """The scaled mask one training pass drew, as a plain array."""
    return np.asarray(training_response(layer, block).kept)


def summary_of(
    reads: int | tuple[int, ...],
    drop_probability: float,
    seed: int,
    block: np.ndarray,
    weights: np.ndarray,
) -> float:
    """A scalar reading of one training pass, on a layer whose draw is pinned.

    ``sum(weights * outputs)`` differentiates to ``weights`` at every output, so
    the same block is both the scalar's definition and the arriving block the
    backward pass is handed, and the two claims are comparable with no loss
    object in between.

    The layer is built here rather than passed in, and that is the whole trick.
    A forward pass advances the generator, so a layer reused across the nudges
    would draw a different mask for each of them and the difference measured
    would be the difference between two networks. Rebuilding from the same seed
    against a block of the same shape reproduces the first draw exactly, so the
    summary is a function of ``block`` alone.
    """
    layer = Dropout(reads=reads, drop_probability=drop_probability, random_seed=seed)
    outputs = layer.respond_to(block, PassPurpose.TRAINING).outputs
    return float(np.sum(weights * outputs))


def small_regression_batch() -> tuple[np.ndarray, np.ndarray]:
    """Forty rows of four features, and a target that is a plain sum of them."""
    generator = np.random.default_rng(90)
    rows = generator.normal(size=(40, 4))
    targets = (rows @ np.array([1.0, -2.0, 0.5, 0.25])).reshape(40, 1)
    return rows, targets


def dense_layer(n_inputs: int, n_neurons: int, seed: int, bent: bool) -> DenseLayer:
    """A small dense layer, weights drawn small enough to start near linear."""
    generator = np.random.default_rng(seed)
    return DenseLayer(
        [
            Neuron(
                generator.normal(size=n_inputs) * 0.4,
                bias=float(generator.normal()) * 0.4,
                activation=HyperbolicTangent() if bent else Identity(),
            )
            for _ in range(n_neurons)
        ]
    )


class TestConstruction:
    def test_the_shape_is_the_same_on_both_sides(self) -> None:
        """The reason it can be inserted anywhere in a sound stack."""
        layer = Dropout(reads=512)

        assert layer.shape == LayerShape(n_inputs=512, n_outputs=512)
        assert layer.shape.reads == layer.shape.answers

    def test_an_arranged_shape_is_carried_through_untouched(self) -> None:
        layer = Dropout(reads=(8, 26, 26))

        assert layer.shape.reads == (8, 26, 26)
        assert layer.shape.answers == (8, 26, 26)

    def test_it_follows_itself(self) -> None:
        """Two of these back to back join, which no other layer here promises."""
        layer = Dropout(reads=(4, 7, 7))

        assert layer.shape.follows(layer.shape)

    def test_the_element_counts_agree(self) -> None:
        layer = Dropout(reads=(2, 8, 8))

        assert layer.shape.n_inputs == 128
        assert layer.shape.n_outputs == 128

    def test_the_default_probability_is_a_half(self) -> None:
        assert Dropout(reads=8).drop_probability == 0.5

    @pytest.mark.parametrize("probability", [0.0, 0.1, 0.25, 0.5, 0.9, 0.999])
    def test_the_keep_probability_is_the_complement(self, probability: float) -> None:
        layer = Dropout(reads=8, drop_probability=probability)

        assert layer.drop_probability == probability
        assert layer.keep_probability == pytest.approx(1.0 - probability)

    def test_zero_is_permitted_because_a_control_needs_it(self) -> None:
        """An expensive identity, which is what a search over ``p`` compares to."""
        layer = Dropout(reads=8, drop_probability=0.0)

        assert layer.drop_probability == 0.0
        assert layer.keep_probability == 1.0

    def test_an_integer_probability_of_zero_is_read_as_a_float(self) -> None:
        assert Dropout(reads=8, drop_probability=0).keep_probability == 1.0

    def test_its_repr_names_its_configuration(self) -> None:
        text = repr(Dropout(reads=8, drop_probability=0.25, random_seed=3))

        assert text == "Dropout(reads=(8,), drop_probability=0.25, random_seed=3)"

    def test_its_repr_says_when_no_seed_was_fixed(self) -> None:
        text = repr(Dropout(reads=(3, 4), drop_probability=0.5))

        assert text == "Dropout(reads=(3, 4), drop_probability=0.5, random_seed=None)"


class TestConstructionRefusals:
    def test_a_probability_of_one_is_refused(self) -> None:
        """It silences everything, and makes the ``1 / (1 - p)`` scale infinite."""
        with pytest.raises(InvalidValuesError):
            Dropout(reads=8, drop_probability=1.0)

    @pytest.mark.parametrize("probability", [1.0000001, 1.5, 2.0, 100.0])
    def test_a_probability_above_one_is_refused(self, probability: float) -> None:
        with pytest.raises(InvalidValuesError):
            Dropout(reads=8, drop_probability=probability)

    @pytest.mark.parametrize("probability", [-1e-9, -0.1, -1.0])
    def test_a_negative_probability_is_refused(self, probability: float) -> None:
        with pytest.raises(InvalidValuesError):
            Dropout(reads=8, drop_probability=probability)

    @pytest.mark.parametrize("probability", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_probability_is_refused(self, probability: float) -> None:
        """``nan`` compares false against both bounds, so the range test alone
        would let it through and every draw would then survive."""
        with pytest.raises(InvalidValuesError):
            Dropout(reads=8, drop_probability=probability)

    @pytest.mark.parametrize("probability", [None, "half", object(), [0.5]])
    def test_a_probability_that_is_not_a_number_is_refused(
        self, probability: object
    ) -> None:
        with pytest.raises(InvalidValuesError):
            Dropout(reads=8, drop_probability=probability)  # type: ignore[arg-type]

    def test_a_probability_just_under_one_is_allowed(self) -> None:
        """Extreme and legal. The scale is 1000, which is large and finite."""
        layer = Dropout(reads=8, drop_probability=0.999)

        assert layer.keep_probability == pytest.approx(0.001)

    def test_a_zero_extent_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Dropout(reads=0)

    def test_a_negative_extent_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Dropout(reads=-4)

    def test_a_zero_extent_inside_an_arrangement_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Dropout(reads=(8, 0, 26))

    def test_a_fractional_extent_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Dropout(reads=8.5)  # type: ignore[arg-type]

    def test_a_boolean_extent_is_refused(self) -> None:
        """``True`` indexes as 1 and would otherwise pass as a width of one."""
        with pytest.raises(InvalidValuesError):
            Dropout(reads=True)  # type: ignore[arg-type]

    def test_an_empty_arrangement_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Dropout(reads=())

    def test_an_arrangement_that_is_not_numbers_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Dropout(reads="wide")  # type: ignore[arg-type]


class TestForwardRefusals:
    def test_a_block_of_the_wrong_width_is_refused(self) -> None:
        layer = Dropout(reads=4, drop_probability=0.5, random_seed=0)

        with pytest.raises(ShapeMismatchError):
            layer.respond_to(np.ones((3, 5)), PassPurpose.TRAINING)

    def test_a_one_dimensional_block_is_refused(self) -> None:
        """One row is still a block, and its first axis is still the rows."""
        layer = Dropout(reads=4, drop_probability=0.5, random_seed=0)

        with pytest.raises(ShapeMismatchError):
            layer.respond_to(np.ones(4), PassPurpose.TRAINING)

    def test_a_block_with_no_rows_is_refused(self) -> None:
        layer = Dropout(reads=4, drop_probability=0.5, random_seed=0)

        with pytest.raises(EmptyValuesError):
            layer.respond_to(np.zeros((0, 4)), PassPurpose.TRAINING)

    @pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_entry_is_refused(self, poison: float) -> None:
        layer = Dropout(reads=4, drop_probability=0.5, random_seed=0)
        block = np.ones((2, 4))
        block[1, 2] = poison

        with pytest.raises(InvalidValuesError):
            layer.respond_to(block, PassPurpose.TRAINING)

    def test_a_block_that_is_not_numeric_is_refused(self) -> None:
        layer = Dropout(reads=4, drop_probability=0.5, random_seed=0)

        with pytest.raises(InvalidValuesError):
            layer.respond_to("not a block", PassPurpose.TRAINING)  # type: ignore[arg-type]

    def test_nested_lists_are_accepted(self) -> None:
        """Coercing at the boundary is what makes a small example writable."""
        layer = Dropout(reads=3, drop_probability=0.0, random_seed=0)
        block = np.array([[1.0, 2.0, 3.0]])

        response = layer.respond_to(block.tolist(), PassPurpose.TRAINING)

        assert np.allclose(response.outputs, block)


class TestPredictingIsAPassThrough:
    """The whole payoff of inverted dropout, which is that it is not there."""

    def test_the_outputs_are_the_inputs_bit_for_bit(self) -> None:
        layer = Dropout(reads=6, drop_probability=0.5, random_seed=1)
        block = np.random.default_rng(2).normal(size=(5, 6))

        response = layer.respond_to(block, PassPurpose.PREDICTING)

        assert np.array_equal(np.asarray(response.outputs), block)

    def test_predicting_is_the_default_purpose(self) -> None:
        """Forgetting to say why costs a slower descent, never a random answer."""
        layer = Dropout(reads=6, drop_probability=0.9, random_seed=1)
        block = np.random.default_rng(3).normal(size=(5, 6))

        assert np.array_equal(np.asarray(layer.respond_to(block).outputs), block)

    @pytest.mark.parametrize("probability", [0.0, 0.25, 0.5, 0.9, 0.999])
    def test_nothing_is_dropped_at_any_probability(self, probability: float) -> None:
        layer = Dropout(reads=8, drop_probability=probability, random_seed=4)
        block = np.random.default_rng(5).normal(size=(20, 8))

        response = layer.respond_to(block, PassPurpose.PREDICTING)

        assert np.count_nonzero(np.asarray(response.outputs)) == np.count_nonzero(block)

    def test_the_response_carries_no_mask(self) -> None:
        """A plain response, which is what makes a backward step from it refuse."""
        layer = Dropout(reads=6, drop_probability=0.5, random_seed=6)

        response = layer.respond_to(np.ones((3, 6)), PassPurpose.PREDICTING)

        assert isinstance(response, LayerResponse)
        assert not isinstance(response, DropoutResponse)

    def test_two_prediction_passes_answer_identically(self) -> None:
        """Nothing about a prediction may depend on a draw, so nothing is drawn."""
        layer = Dropout(reads=64, drop_probability=0.5, random_seed=7)
        block = np.random.default_rng(8).normal(size=(10, 64))

        first = np.asarray(layer.respond_to(block, PassPurpose.PREDICTING).outputs)
        second = np.asarray(layer.respond_to(block, PassPurpose.PREDICTING).outputs)

        assert np.array_equal(first, second)

    def test_an_arranged_block_passes_through_unchanged(self) -> None:
        layer = Dropout(reads=(2, 4, 4), drop_probability=0.5, random_seed=9)
        block = np.random.default_rng(10).normal(size=(3, 2, 4, 4))

        response = layer.respond_to(block, PassPurpose.PREDICTING)

        assert np.array_equal(np.asarray(response.outputs), block)


class TestTrainingDropsAndRescales:
    def test_the_response_carries_the_mask(self) -> None:
        layer = Dropout(reads=6, drop_probability=0.5, random_seed=11)

        response = layer.respond_to(np.ones((4, 6)), PassPurpose.TRAINING)

        assert isinstance(response, DropoutResponse)
        assert np.asarray(response.kept).shape == (4, 6)

    @pytest.mark.parametrize("probability", EXACTLY_SCALED)
    def test_every_answer_is_its_input_seen_through_the_mask(
        self, probability: float
    ) -> None:
        """The relationship, read off the mask the response itself reports.

        Written this way rather than by drawing a mask and re-doing the
        multiply, because re-doing the multiply is running the implementation
        again and would agree with it however wrong it was.
        """
        layer = Dropout(reads=12, drop_probability=probability, random_seed=12)
        block = np.random.default_rng(13).normal(size=(8, 12))

        response = training_response(layer, block)

        expected = np.asarray(response.inputs) * np.asarray(response.kept)
        assert np.array_equal(np.asarray(response.outputs), expected)

    @pytest.mark.parametrize("probability", [0.1, 0.3, 0.4, 0.6, 0.8])
    def test_the_relationship_holds_at_awkward_probabilities_too(
        self, probability: float
    ) -> None:
        """``approx`` here, because ``1 / (1 - p)`` is not exact for these."""
        layer = Dropout(reads=12, drop_probability=probability, random_seed=14)
        block = np.random.default_rng(15).normal(size=(8, 12))

        response = training_response(layer, block)

        expected = np.asarray(response.inputs) * np.asarray(response.kept)
        assert np.allclose(np.asarray(response.outputs), expected)

    def test_a_dropped_entry_is_exactly_zero(self) -> None:
        layer = Dropout(reads=20, drop_probability=0.5, random_seed=16)
        block = np.random.default_rng(17).normal(size=(10, 20)) + 5.0

        response = training_response(layer, block)

        dropped = np.asarray(response.kept) == 0.0
        assert int(np.count_nonzero(dropped)) > 0
        answers = np.asarray(response.outputs)[dropped]
        assert np.array_equal(answers, np.zeros(answers.shape))

    @pytest.mark.parametrize("probability", [0.5, 0.75])
    def test_a_surviving_entry_is_exactly_its_input_divided_by_the_keep(
        self, probability: float
    ) -> None:
        layer = Dropout(reads=20, drop_probability=probability, random_seed=18)
        block = np.random.default_rng(19).normal(size=(10, 20)) + 5.0

        response = training_response(layer, block)

        surviving = np.asarray(response.kept) != 0.0
        assert int(np.count_nonzero(surviving)) > 0
        assert np.array_equal(
            np.asarray(response.outputs)[surviving],
            np.asarray(response.inputs)[surviving] / layer.keep_probability,
        )

    @pytest.mark.parametrize("probability", [0.1, 0.5, 0.75, 0.9])
    def test_the_mask_holds_only_two_values(self, probability: float) -> None:
        """Silenced, or amplified by the reciprocal of the keep. Nothing else."""
        layer = Dropout(reads=40, drop_probability=probability, random_seed=20)

        kept = mask_of(layer, np.ones((25, 40)))

        scale = 1.0 / (1.0 - probability)
        assert np.all((kept == 0.0) | np.isclose(kept, scale))

    def test_the_scores_and_the_outputs_are_the_same_block(self) -> None:
        """No bend here, so there is no pre-activation value to keep apart."""
        layer = Dropout(reads=6, drop_probability=0.5, random_seed=21)

        response = training_response(layer, np.ones((3, 6)))

        assert response.scores is response.outputs

    def test_the_response_carries_the_block_that_was_read(self) -> None:
        layer = Dropout(reads=6, drop_probability=0.5, random_seed=22)
        block = np.random.default_rng(23).normal(size=(3, 6))

        response = training_response(layer, block)

        assert np.array_equal(np.asarray(response.inputs), block)

    def test_an_arranged_block_keeps_its_arrangement(self) -> None:
        layer = Dropout(reads=(2, 4, 4), drop_probability=0.5, random_seed=24)

        response = training_response(layer, np.ones((3, 2, 4, 4)))

        assert np.asarray(response.outputs).shape == (3, 2, 4, 4)
        assert np.asarray(response.kept).shape == (3, 2, 4, 4)

    def test_the_fraction_silenced_is_the_probability_asked_for(self) -> None:
        """One hundred thousand draws at ``p = 0.3``, whose standard error is
        ``sqrt(0.3 * 0.7 / 1e5)``, about 0.0014. The tolerance is 0.01, seven
        of those, and the layer is seeded so the number is fixed rather than
        merely probable."""
        layer = Dropout(reads=500, drop_probability=0.3, random_seed=25)

        kept = mask_of(layer, np.ones((200, 500)))

        silenced = float(np.count_nonzero(kept == 0.0)) / kept.size
        assert silenced == pytest.approx(0.3, abs=0.01)


class TestDroppingNothing:
    """``p = 0.0`` has to be an identity, or the control in a search is a lie."""

    def test_nothing_is_silenced(self) -> None:
        layer = Dropout(reads=50, drop_probability=0.0, random_seed=26)

        kept = mask_of(layer, np.ones((20, 50)))

        assert int(np.count_nonzero(kept == 0.0)) == 0

    def test_the_mask_is_exactly_one_everywhere(self) -> None:
        layer = Dropout(reads=50, drop_probability=0.0, random_seed=27)

        kept = mask_of(layer, np.ones((20, 50)))

        assert np.array_equal(kept, np.ones((20, 50)))

    def test_the_answers_are_the_inputs_bit_for_bit(self) -> None:
        layer = Dropout(reads=50, drop_probability=0.0, random_seed=28)
        block = np.random.default_rng(29).normal(size=(20, 50))

        response = training_response(layer, block)

        assert np.array_equal(np.asarray(response.outputs), block)

    def test_the_blame_comes_back_untouched(self) -> None:
        layer = Dropout(reads=50, drop_probability=0.0, random_seed=30)
        response = training_response(layer, np.ones((20, 50)))
        arriving = np.random.default_rng(31).normal(size=(20, 50))

        correction = layer.correction_for(response, arriving)

        assert np.array_equal(np.asarray(correction.passed_down), arriving)


class TestTheExpectationIsPreserved:
    """The entire justification for the ``1 / (1 - p)`` scale, measured.

    A unit contributes ``0`` with probability ``p`` and ``value / (1 - p)`` with
    probability ``1 - p``, so its expected contribution is ``value``. If that
    holds then a network trained under dropout sees inputs at the same scale
    when it is later asked to predict with every unit present, which is what
    makes the prediction pass an identity rather than an approximation.
    """

    @pytest.mark.parametrize("probability", [0.2, 0.5, 0.8])
    def test_the_mean_of_the_answers_is_the_mean_of_the_inputs(
        self, probability: float
    ) -> None:
        """One hundred thousand entries, spread evenly over ``[0.5, 1.5]``.

        The variance of one answer is ``value^2 * p / (1 - p)``, so the standard
        error of the mean over ``N`` of them is
        ``sqrt(p / (1 - p) * mean(value^2) / N)``. Here ``mean(value^2)`` is
        1.083 and ``N`` is 1e5, which puts the worst case, ``p = 0.8``, at
        0.0066. The tolerance is 0.04, six of those, and the layer is seeded so
        the number is settled rather than probable.

        The headroom above matters more than the headroom below. An
        implementation that never applied the scale would report a mean of
        ``1 - p``, which is 0.2 at the worst case here, off by 0.8 and twenty
        times the tolerance. One that scaled by ``1 - p`` instead of its
        reciprocal would report 0.04 and be off by 0.96.
        """
        block = np.linspace(0.5, 1.5, 100_000).reshape(400, 250)
        layer = Dropout(reads=250, drop_probability=probability, random_seed=32)

        response = training_response(layer, block)

        assert float(np.mean(np.asarray(response.outputs))) == pytest.approx(
            float(np.mean(block)), abs=0.04
        )

    def test_the_total_is_preserved_as_well_as_the_mean(self) -> None:
        """The sum is what the next layer's weighted read actually depends on."""
        block = np.full((400, 250), 2.0)
        layer = Dropout(reads=250, drop_probability=0.5, random_seed=33)

        response = training_response(layer, block)

        assert float(np.sum(np.asarray(response.outputs))) == pytest.approx(
            float(np.sum(block)), rel=0.02
        )


class TestTheDrawIsPerUnitAndPerRow:
    """A shared mask would let a unit rely on a partner for the whole batch."""

    def test_two_rows_in_one_block_are_thinned_differently(self) -> None:
        """Four hundred independent draws per row. Two rows agreeing everywhere
        by chance has probability ``2 ** -400``, so a single ``array_equal``
        catches a shared mask essentially always rather than merely usually."""
        layer = Dropout(reads=400, drop_probability=0.5, random_seed=34)

        kept = mask_of(layer, np.ones((3, 400)))

        assert not np.array_equal(kept[0], kept[1])
        assert not np.array_equal(kept[1], kept[2])
        assert not np.array_equal(kept[0], kept[2])

    def test_no_two_rows_of_a_large_block_share_a_mask(self) -> None:
        layer = Dropout(reads=400, drop_probability=0.5, random_seed=35)

        kept = mask_of(layer, np.ones((30, 400)))

        distinct = {row.tobytes() for row in kept}
        assert len(distinct) == 30

    def test_every_row_is_partly_silenced_and_partly_kept(self) -> None:
        """A draw made once per row would leave whole rows on or wholly off."""
        layer = Dropout(reads=400, drop_probability=0.5, random_seed=36)

        kept = mask_of(layer, np.ones((10, 400)))

        for row in kept:
            assert 0 < int(np.count_nonzero(row)) < 400

    def test_the_arrangement_is_drawn_over_too(self) -> None:
        """A convolution's channels and pixels are units like any other."""
        layer = Dropout(reads=(4, 10, 10), drop_probability=0.5, random_seed=37)

        kept = mask_of(layer, np.ones((2, 4, 10, 10)))

        assert not np.array_equal(kept[0], kept[1])
        assert not np.array_equal(kept[0, 0], kept[0, 1])


class TestSeeding:
    """Reproducible over a sequence of passes, which is what a stream allows."""

    def test_one_seed_gives_one_sequence_of_masks(self) -> None:
        block = np.ones((4, 64))
        first = Dropout(reads=64, drop_probability=0.5, random_seed=38)
        second = Dropout(reads=64, drop_probability=0.5, random_seed=38)

        for _ in range(5):
            assert np.array_equal(mask_of(first, block), mask_of(second, block))

    def test_the_answers_agree_too_and_not_only_the_masks(self) -> None:
        block = np.random.default_rng(39).normal(size=(4, 64))
        first = Dropout(reads=64, drop_probability=0.5, random_seed=40)
        second = Dropout(reads=64, drop_probability=0.5, random_seed=40)

        for _ in range(3):
            assert np.array_equal(
                np.asarray(training_response(first, block).outputs),
                np.asarray(training_response(second, block).outputs),
            )

    def test_a_prediction_pass_in_the_same_place_does_not_break_the_agreement(
        self,
    ) -> None:
        """The sequence is what is reproducible, so both layers get the same one.

        Whether a prediction pass consumes from the stream is deliberately not
        asserted, since nothing in the contract says it must and an
        implementation that draws and discards would be no less correct.
        """
        block = np.ones((4, 64))
        first = Dropout(reads=64, drop_probability=0.5, random_seed=41)
        second = Dropout(reads=64, drop_probability=0.5, random_seed=41)

        for layer in (first, second):
            layer.respond_to(block, PassPurpose.TRAINING)
            layer.respond_to(block, PassPurpose.PREDICTING)

        assert np.array_equal(mask_of(first, block), mask_of(second, block))

    def test_two_seeds_give_two_sequences(self) -> None:
        block = np.ones((4, 64))
        first = Dropout(reads=64, drop_probability=0.5, random_seed=42)
        second = Dropout(reads=64, drop_probability=0.5, random_seed=43)

        assert not np.array_equal(mask_of(first, block), mask_of(second, block))

    def test_no_seed_draws_from_fresh_entropy(self) -> None:
        """Two unseeded layers agreeing on 256 draws is a ``2 ** -256`` event."""
        block = np.ones((4, 64))
        first = Dropout(reads=64, drop_probability=0.5)
        second = Dropout(reads=64, drop_probability=0.5)

        assert not np.array_equal(mask_of(first, block), mask_of(second, block))


class TestConsecutivePassesDiffer:
    """What catches a layer that rebuilds its generator instead of advancing it."""

    def test_a_seeded_layer_does_not_draw_the_same_mask_twice(self) -> None:
        """Silencing the same units forever is a smaller network, not dropout.

        And it trains, converges and reports a plausible loss, which is why this
        needs asserting rather than assuming.
        """
        layer = Dropout(reads=256, drop_probability=0.5, random_seed=44)
        block = np.ones((2, 256))

        assert not np.array_equal(mask_of(layer, block), mask_of(layer, block))

    def test_ten_passes_give_ten_different_masks(self) -> None:
        layer = Dropout(reads=256, drop_probability=0.5, random_seed=45)
        block = np.ones((2, 256))

        drawn = {mask_of(layer, block).tobytes() for _ in range(10)}

        assert len(drawn) == 10

    def test_the_answers_differ_and_not_only_the_masks(self) -> None:
        layer = Dropout(reads=256, drop_probability=0.5, random_seed=46)
        block = np.random.default_rng(47).normal(size=(2, 256))

        first = np.asarray(training_response(layer, block).outputs)
        second = np.asarray(training_response(layer, block).outputs)

        assert not np.array_equal(first, second)


class TestSteppingKeepsTheStream:
    """``stepped_by`` answers with ``self``, and that is load-bearing."""

    def test_a_step_answers_with_the_very_same_layer(self) -> None:
        layer = Dropout(reads=8, drop_probability=0.5, random_seed=48)

        assert layer.stepped_by(None, learning_rate=0.1) is layer

    @pytest.mark.parametrize("learning_rate", [0.0, 0.1, 1000.0])
    def test_no_learning_rate_changes_that(self, learning_rate: float) -> None:
        layer = Dropout(reads=8, drop_probability=0.5, random_seed=49)

        assert layer.stepped_by(None, learning_rate=learning_rate) is layer

    def test_stepping_does_not_reset_the_stream(self) -> None:
        """A rebuilt layer would answer the same mask on either side of a step."""
        layer = Dropout(reads=256, drop_probability=0.5, random_seed=50)
        block = np.ones((2, 256))
        before = mask_of(layer, block)

        stepped = layer.stepped_by(None, learning_rate=0.1)

        assert isinstance(stepped, Dropout)
        assert not np.array_equal(before, mask_of(stepped, block))

    def test_a_network_still_draws_fresh_masks_after_several_steps(self) -> None:
        """The one that matters. Every step rebuilds the stack from its layers,
        so a dropout that answered with a new instance would hand each epoch an
        untouched generator and silence the identical units all the way through
        training."""
        rows, targets = small_regression_batch()
        dropout = Dropout(reads=8, drop_probability=0.4, random_seed=51)
        stack = LayerStack(
            [
                dense_layer(4, 8, seed=52, bent=True),
                dropout,
                dense_layer(8, 1, seed=53, bent=False),
            ]
        )

        for _ in range(5):
            stack = stack.stepped_by(
                stack.backward_pass(rows, targets, SquaredError()), learning_rate=0.05
            )

        assert stack[1] is dropout
        first = stack.respond_to(rows, PassPurpose.TRAINING)[1]
        second = stack.respond_to(rows, PassPurpose.TRAINING)[1]
        assert isinstance(first, DropoutResponse)
        assert isinstance(second, DropoutResponse)
        assert not np.array_equal(np.asarray(first.kept), np.asarray(second.kept))


class TestTheBackwardPass:
    def test_a_dropped_unit_is_owed_exactly_nothing(self) -> None:
        """It contributed nothing to the loss, so it collects nothing back."""
        layer = Dropout(reads=12, drop_probability=0.5, random_seed=54)
        response = training_response(layer, np.full((5, 12), 3.0))
        arriving = np.arange(60, dtype=float).reshape(5, 12) + 1.0

        correction = layer.correction_for(response, arriving)

        dropped = np.asarray(response.kept) == 0.0
        assert int(np.count_nonzero(dropped)) > 0
        owed = np.asarray(correction.passed_down)[dropped]
        assert np.array_equal(owed, np.zeros(owed.shape))

    def test_a_surviving_unit_is_owed_its_blame_amplified(self) -> None:
        """The mistake the module docstring names. A backward pass that dropped
        the scale would report exactly ``arriving`` here, which is smaller by
        the keep probability and raises nothing ever."""
        layer = Dropout(reads=12, drop_probability=0.5, random_seed=55)
        response = training_response(layer, np.full((5, 12), 3.0))
        arriving = np.arange(60, dtype=float).reshape(5, 12) + 1.0

        correction = layer.correction_for(response, arriving)

        surviving = np.asarray(response.kept) != 0.0
        assert int(np.count_nonzero(surviving)) > 0
        assert np.array_equal(
            np.asarray(correction.passed_down)[surviving],
            arriving[surviving] / layer.keep_probability,
        )
        assert not np.array_equal(
            np.asarray(correction.passed_down)[surviving], arriving[surviving]
        )

    def test_the_whole_block_is_the_arriving_one_seen_through_the_mask(self) -> None:
        layer = Dropout(reads=12, drop_probability=0.5, random_seed=56)
        block = np.random.default_rng(57).normal(size=(5, 12))
        response = training_response(layer, block)
        arriving = np.random.default_rng(58).normal(size=(5, 12))

        correction = layer.correction_for(response, arriving)

        expected = arriving * np.asarray(response.kept)
        assert np.allclose(np.asarray(correction.passed_down), expected)

    def test_the_block_passed_down_is_the_shape_the_layer_read(self) -> None:
        layer = Dropout(reads=(2, 5, 5), drop_probability=0.5, random_seed=59)
        response = training_response(layer, np.ones((3, 2, 5, 5)))

        correction = layer.correction_for(response, np.ones((3, 2, 5, 5)))

        assert np.asarray(correction.passed_down).shape == (3, 2, 5, 5)

    def test_it_reports_no_gradient(self) -> None:
        """``None`` rather than a block of zeros, which would claim it learns."""
        layer = Dropout(reads=6, drop_probability=0.5, random_seed=60)
        response = training_response(layer, np.ones((3, 6)))

        correction = layer.correction_for(response, np.ones((3, 6)))

        assert correction.gradient is None
        assert correction.learns is False

    def test_the_same_response_can_be_corrected_twice_alike(self) -> None:
        """The mask is on the response, so a second backward step cannot draw a
        different one out of a generator that has since moved on."""
        layer = Dropout(reads=64, drop_probability=0.5, random_seed=61)
        response = training_response(layer, np.ones((4, 64)))
        arriving = np.random.default_rng(62).normal(size=(4, 64))

        first = layer.correction_for(response, arriving).passed_down
        layer.respond_to(np.ones((4, 64)), PassPurpose.TRAINING)
        second = layer.correction_for(response, arriving).passed_down

        assert np.array_equal(np.asarray(first), np.asarray(second))


class TestBackwardRefusals:
    def test_a_response_from_a_prediction_pass_is_refused(self) -> None:
        """It carries no mask, and guessing one from the answers is the quiet
        mistake this whole file is arranged around."""
        layer = Dropout(reads=4, drop_probability=0.5, random_seed=63)
        response = layer.respond_to(np.ones((3, 4)), PassPurpose.PREDICTING)

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((3, 4)))

    def test_a_plain_response_of_exactly_the_right_shape_is_refused(self) -> None:
        """Everything ``_checked_arriving`` looks at agrees here. The mask is the
        one thing it cannot see, which is why there is a second guard."""
        layer = Dropout(reads=4, drop_probability=0.5, random_seed=64)
        response = LayerResponse(
            inputs=np.ones((3, 4)), scores=np.ones((3, 4)), outputs=np.ones((3, 4))
        )

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((3, 4)))

    def test_an_arriving_block_of_the_wrong_width_is_refused(self) -> None:
        layer = Dropout(reads=4, drop_probability=0.5, random_seed=65)
        response = training_response(layer, np.ones((3, 4)))

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((3, 5)))

    def test_an_arriving_block_with_the_wrong_row_count_is_refused(self) -> None:
        layer = Dropout(reads=4, drop_probability=0.5, random_seed=66)
        response = training_response(layer, np.ones((3, 4)))

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((2, 4)))

    def test_a_one_dimensional_arriving_block_is_refused(self) -> None:
        layer = Dropout(reads=4, drop_probability=0.5, random_seed=67)
        response = training_response(layer, np.ones((3, 4)))

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones(4))

    def test_an_arriving_block_that_is_not_numeric_is_refused(self) -> None:
        layer = Dropout(reads=4, drop_probability=0.5, random_seed=68)
        response = training_response(layer, np.ones((3, 4)))

        with pytest.raises(InvalidValuesError):
            layer.correction_for(response, "not a slope")  # type: ignore[arg-type]

    def test_a_response_from_a_differently_shaped_layer_is_refused(self) -> None:
        """A pairing mistake, and it would route real numbers to arbitrary places."""
        wide = Dropout(reads=6, drop_probability=0.5, random_seed=69)
        narrow = Dropout(reads=4, drop_probability=0.5, random_seed=70)
        response = wide.respond_to(np.ones((3, 6)), PassPurpose.TRAINING)

        with pytest.raises(ShapeMismatchError):
            narrow.correction_for(response, np.ones((3, 4)))

    def test_a_response_from_a_differently_arranged_layer_is_refused(self) -> None:
        """Same element count, different arrangement, and they still do not pair."""
        arranged = Dropout(reads=(2, 3, 4), drop_probability=0.5, random_seed=71)
        flat = Dropout(reads=24, drop_probability=0.5, random_seed=72)
        response = arranged.respond_to(np.ones((3, 2, 3, 4)), PassPurpose.TRAINING)

        with pytest.raises(ShapeMismatchError):
            flat.correction_for(response, np.ones((3, 24)))


class TestTheZeroInputTrap:
    """The sharpest test here, and the only one a shortcut implementation fails.

    Recovering the mask from ``outputs != 0`` passes every other test in this
    file. It fails these, because an input that was genuinely zero and genuinely
    kept leaves a zero in the answer and would be recorded as silenced, taking
    its gradient with it.
    """

    def test_a_block_of_nothing_but_zeros_is_still_blamed(self) -> None:
        """Every answer is zero, so a mask read off the answers says everything
        was dropped and the whole block of blame vanishes. The real mask says
        otherwise, and with an arriving block of ones the blame passed down *is*
        the mask."""
        layer = Dropout(reads=16, drop_probability=0.5, random_seed=73)
        response = training_response(layer, np.zeros((4, 16)))

        correction = layer.correction_for(response, np.ones((4, 16)))

        kept = np.asarray(response.kept)
        assert int(np.count_nonzero(kept)) > 0
        assert np.array_equal(np.asarray(response.outputs), np.zeros((4, 16)))
        assert np.array_equal(np.asarray(correction.passed_down), kept)

    def test_the_kept_zeros_of_a_rectified_block_are_paid_in_full(self) -> None:
        """The realistic version. Half of what a rectified layer hands upward is
        exactly zero, so this is the ordinary case rather than a contrived one.
        """
        generator = np.random.default_rng(74)
        block = np.maximum(generator.normal(size=(8, 24)), 0.0)
        layer = Dropout(reads=24, drop_probability=0.4, random_seed=75)
        response = training_response(layer, block)
        arriving = generator.normal(size=(8, 24)) + 3.0

        correction = layer.correction_for(response, arriving)

        zero_and_kept = (block == 0.0) & (np.asarray(response.kept) != 0.0)
        assert int(np.count_nonzero(zero_and_kept)) > 0
        assert np.allclose(
            np.asarray(correction.passed_down)[zero_and_kept],
            arriving[zero_and_kept] / layer.keep_probability,
        )

    def test_a_kept_zero_and_a_dropped_zero_are_told_apart(self) -> None:
        """Two positions with the identical output, owed entirely different
        amounts. Nothing in the answers distinguishes them, which is the whole
        argument for storing the mask."""
        layer = Dropout(reads=32, drop_probability=0.5, random_seed=76)
        response = training_response(layer, np.zeros((4, 32)))

        correction = layer.correction_for(response, np.full((4, 32), 7.0))

        kept = np.asarray(response.kept)
        passed_down = np.asarray(correction.passed_down)
        outputs = np.asarray(response.outputs)
        assert int(np.count_nonzero(kept)) > 0
        assert int(np.count_nonzero(kept == 0.0)) > 0
        # Every answer is zero, kept and dropped alike, so the outputs hold
        # nothing that could tell the two apart.
        assert np.array_equal(outputs, np.zeros((4, 32)))
        assert np.allclose(passed_down[kept != 0.0], 14.0)
        assert np.array_equal(
            passed_down[kept == 0.0], np.zeros(int(np.count_nonzero(kept == 0.0)))
        )


class TestTheGradientCheck:
    """Nudge each input by a hair, watch a scalar summary, compare the claim.

    The oracle. It is computed from forward passes alone, so a backward pass
    that agrees with it is right for reasons owing nothing to how it was
    derived, and in particular it settles the ``1 / (1 - p)`` factor that no
    shape test can see.

    The blocks are small and centred near zero on purpose. A central difference
    divides by ``2e-6``, which multiplies the summary's own rounding by 5e5, and
    a block of values in the hundreds would put that amplified noise at the
    tolerance itself.
    """

    def measured_slopes(
        self,
        reads: int | tuple[int, ...],
        drop_probability: float,
        seed: int,
        block: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """How the summary really moves when each input value is nudged."""
        measured = np.empty_like(block)

        for position in np.ndindex(*block.shape):
            moved = []
            for direction in (+NUDGE, -NUDGE):
                nudged = block.copy()
                nudged[position] += direction
                moved.append(summary_of(reads, drop_probability, seed, nudged, weights))
            measured[position] = (moved[0] - moved[1]) / (2.0 * NUDGE)

        return measured

    def claimed_slopes(
        self,
        reads: int | tuple[int, ...],
        drop_probability: float,
        seed: int,
        block: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """What the backward pass says, on a layer drawing that same first mask."""
        layer = Dropout(
            reads=reads, drop_probability=drop_probability, random_seed=seed
        )
        response = layer.respond_to(block, PassPurpose.TRAINING)
        return np.asarray(layer.correction_for(response, weights).passed_down)

    def test_it_matches_a_finite_difference_on_a_flat_block(self) -> None:
        block = np.random.default_rng(77).normal(size=(4, 6)) * 0.5
        weights = np.random.default_rng(78).normal(size=(4, 6))

        claimed = self.claimed_slopes(6, 0.5, 79, block, weights)
        measured = self.measured_slopes(6, 0.5, 79, block, weights)

        assert np.allclose(claimed, measured, atol=1e-8)
        # A survivor's slope is its weight doubled at this probability, so a
        # backward pass that dropped the scale would agree with neither.
        assert float(np.max(np.abs(claimed))) > float(np.max(np.abs(weights)))

    @pytest.mark.parametrize("probability", [0.0, 0.25, 0.5, 0.75])
    def test_it_matches_a_finite_difference_at_several_probabilities(
        self, probability: float
    ) -> None:
        block = np.random.default_rng(80).normal(size=(3, 5)) * 0.5
        weights = np.random.default_rng(81).normal(size=(3, 5))

        claimed = self.claimed_slopes(5, probability, 82, block, weights)
        measured = self.measured_slopes(5, probability, 82, block, weights)

        assert np.allclose(claimed, measured, atol=1e-8)

    def test_it_matches_a_finite_difference_on_an_arranged_block(self) -> None:
        """Rows and channels must not bleed into each other, and only a block
        with more than one of each says so."""
        block = np.random.default_rng(83).normal(size=(2, 2, 3, 3)) * 0.5
        weights = np.random.default_rng(84).normal(size=(2, 2, 3, 3))

        claimed = self.claimed_slopes((2, 3, 3), 0.5, 85, block, weights)
        measured = self.measured_slopes((2, 3, 3), 0.5, 85, block, weights)

        assert np.allclose(claimed, measured, atol=1e-8)

    def test_the_silenced_positions_measure_exactly_zero(self) -> None:
        """The finite difference agrees that a silenced unit changes nothing,
        which is the same fact the routing test asserts from the other side."""
        block = np.random.default_rng(86).normal(size=(4, 6)) * 0.5
        weights = np.random.default_rng(87).normal(size=(4, 6))

        measured = self.measured_slopes(6, 0.5, 88, block, weights)
        layer = Dropout(reads=6, drop_probability=0.5, random_seed=88)
        kept = mask_of(layer, block)

        assert int(np.count_nonzero(kept == 0.0)) > 0
        assert np.allclose(measured[kept == 0.0], 0.0, atol=1e-8)


class TestTwoResponsesAreTheSameOnlyIfTheyThinnedAlike:
    """The mask is the whole reason this class exists, so equality must read it.

    Two responses can hold identical answers and have arrived at them by
    silencing different units, since the inputs that produced them differ too.
    Calling those the same response is a claim that the thing distinguishing
    them does not count, and the thing distinguishing them is the only thing
    this class adds.

    The last test here is the one that would have failed before the base
    learned to compare types. A subclass cannot fix this alone: comparing
    against a plain response it answers ``NotImplemented``, Python retries the
    other way round, and the base finds matching outputs and says True.
    """

    def test_the_same_mask_and_the_same_answers_are_equal(self) -> None:
        block = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[2.0, 0.0], [0.0, 2.0]])

        first = DropoutResponse.recording(
            inputs=block.copy(), outputs=block.copy(), kept=mask.copy()
        )
        second = DropoutResponse.recording(
            inputs=block.copy(), outputs=block.copy(), kept=mask.copy()
        )

        assert first == second

    def test_the_same_answers_under_a_different_mask_are_not(self) -> None:
        block = np.array([[1.0, 2.0], [3.0, 4.0]])

        kept_left = DropoutResponse.recording(
            inputs=block.copy(),
            outputs=block.copy(),
            kept=np.array([[2.0, 0.0], [0.0, 2.0]]),
        )
        kept_right = DropoutResponse.recording(
            inputs=block.copy(),
            outputs=block.copy(),
            kept=np.array([[0.0, 2.0], [2.0, 0.0]]),
        )

        assert kept_left != kept_right

    def test_it_is_not_equal_to_a_plain_response_with_the_same_answers(
        self,
    ) -> None:
        block = np.array([[1.0, 2.0], [3.0, 4.0]])
        carrying = DropoutResponse.recording(
            inputs=block.copy(), outputs=block.copy(), kept=np.full((2, 2), 2.0)
        )
        plain = LayerResponse(inputs=block, scores=block, outputs=block)

        assert carrying != plain
        assert plain != carrying

    def test_it_defers_to_anything_that_is_not_a_response(self) -> None:
        block = np.array([[1.0, 2.0]])
        carrying = DropoutResponse.recording(
            inputs=block.copy(), outputs=block.copy(), kept=np.ones((1, 2))
        )

        assert carrying.__eq__("response") is NotImplemented


class TestTheResponse:
    def test_the_mask_is_frozen(self) -> None:
        """A caller editing it would change what the backward pass believes
        happened on a forward pass that has already run."""
        layer = Dropout(reads=6, drop_probability=0.5, random_seed=89)
        response = training_response(layer, np.ones((3, 6)))

        with pytest.raises(ValueError):
            response.kept[0, 0] = 99.0

    def test_the_outputs_are_frozen(self) -> None:
        layer = Dropout(reads=6, drop_probability=0.5, random_seed=91)
        response = training_response(layer, np.ones((3, 6)))

        with pytest.raises(ValueError):
            response.outputs[0, 0] = 99.0

    def test_the_inputs_are_frozen(self) -> None:
        layer = Dropout(reads=6, drop_probability=0.5, random_seed=92)
        response = training_response(layer, np.ones((3, 6)))

        with pytest.raises(ValueError):
            response.inputs[0, 0] = 99.0

    def test_the_callers_own_block_is_not_frozen_underneath_them(self) -> None:
        layer = Dropout(reads=6, drop_probability=0.5, random_seed=93)
        block = np.ones((3, 6))

        layer.respond_to(block, PassPurpose.TRAINING)
        block[0, 0] = 99.0

        assert block[0, 0] == 99.0

    def test_it_reports_its_rows_and_its_width(self) -> None:
        layer = Dropout(reads=6, drop_probability=0.5, random_seed=94)

        response = training_response(layer, np.ones((3, 6)))

        assert response.n_rows == 3
        assert response.n_neurons == 6

    def test_its_repr_counts_what_survived(self) -> None:
        layer = Dropout(reads=4, drop_probability=0.5, random_seed=95)
        response = training_response(layer, np.ones((2, 4)))

        surviving = int(np.count_nonzero(np.asarray(response.kept)))
        assert repr(response) == f"DropoutResponse(n_rows=2, n_kept={surviving})"


class TestInAStack:
    """It moves no join, so a sound stack stays sound with one inserted."""

    def test_it_can_be_inserted_without_changing_the_network_shape(self) -> None:
        beneath = dense_layer(4, 8, seed=96, bent=True)
        above = dense_layer(8, 1, seed=97, bent=False)

        without = LayerStack([beneath, above])
        with_dropout = LayerStack([beneath, Dropout(reads=8), above])

        assert with_dropout.shape == without.shape
        assert len(with_dropout) == 3

    def test_two_of_them_back_to_back_still_join(self) -> None:
        stack = LayerStack(
            [
                dense_layer(4, 8, seed=98, bent=True),
                Dropout(reads=8, drop_probability=0.2),
                Dropout(reads=8, drop_probability=0.2),
                dense_layer(8, 1, seed=99, bent=False),
            ]
        )

        assert len(stack) == 4

    def test_a_stack_containing_it_predicts_deterministically(self) -> None:
        rows, _ = small_regression_batch()
        stack = LayerStack(
            [
                dense_layer(4, 8, seed=100, bent=True),
                Dropout(reads=8, drop_probability=0.5, random_seed=101),
                dense_layer(8, 1, seed=102, bent=False),
            ]
        )

        first = np.asarray(stack.respond_to(rows).outputs)
        second = np.asarray(stack.respond_to(rows).outputs)

        assert np.array_equal(first, second)

    def test_a_stack_containing_it_perturbs_while_training(self) -> None:
        """``backward_pass`` states ``TRAINING`` for itself, so the answers it
        scores are the thinned ones rather than the deterministic ones."""
        rows, _ = small_regression_batch()
        stack = LayerStack(
            [
                dense_layer(4, 8, seed=103, bent=True),
                Dropout(reads=8, drop_probability=0.5, random_seed=104),
                dense_layer(8, 1, seed=105, bent=False),
            ]
        )

        predicted = np.asarray(stack.respond_to(rows).outputs)
        trained = np.asarray(stack.respond_to(rows, PassPurpose.TRAINING).outputs)

        assert not np.allclose(predicted, trained)

    def test_a_stack_containing_it_trains(self) -> None:
        """The loss is read under ``PREDICTING`` on both sides, so what falls is
        the network rather than the luck of one draw."""
        rows, targets = small_regression_batch()
        loss = SquaredError()
        stack = LayerStack(
            [
                dense_layer(4, 8, seed=106, bent=True),
                Dropout(reads=8, drop_probability=0.2, random_seed=107),
                dense_layer(8, 1, seed=108, bent=False),
            ]
        )

        before = loss.measure(stack.respond_to(rows).outputs, targets).value
        for _ in range(200):
            stack = stack.stepped_by(
                stack.backward_pass(rows, targets, loss), learning_rate=0.05
            )
        after = loss.measure(stack.respond_to(rows).outputs, targets).value

        assert after < before * 0.5

    def test_the_gradients_come_back_one_per_layer_with_a_hole_for_dropout(
        self,
    ) -> None:
        rows, targets = small_regression_batch()
        stack = LayerStack(
            [
                dense_layer(4, 8, seed=109, bent=True),
                Dropout(reads=8, drop_probability=0.3, random_seed=110),
                dense_layer(8, 1, seed=111, bent=False),
            ]
        )

        backward = stack.backward_pass(rows, targets, SquaredError())

        assert len(backward) == 3
        assert backward[0] is not None
        assert backward[1] is None
        assert backward[2] is not None
