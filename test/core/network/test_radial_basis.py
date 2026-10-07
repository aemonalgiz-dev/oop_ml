"""Spec for the radial basis layer, where one test stands behind the whole file.

The forward half is the easy half and is worked by hand. A unit answers
``exp(-||row - centre|| ** 2 / (2 * width ** 2))``, which is a number a person
can compute on paper, and three claims about it are worth pinning separately: a
unit sitting exactly on its own centre answers exactly one, the answer falls as
the row moves away, and the width is what sets how fast it falls. The third is
the one an implementation gets wrong quietly, because a width squared where a
width was meant, or a factor of two dropped from the denominator, still produces
a bell that peaks at one and decays -- it decays at the wrong rate, and nothing
about the shape of the curve says so. So the width claim is measured against
``exp(-0.5)`` at a distance of exactly one width, which is the one point where
the arithmetic can be checked without recomputing it.

The backward half is where the file earns its keep, for the reason
``test_convolution.py`` and ``test_pooling.py`` both give: almost every wrong
backward pass still runs. This layer adds three of its own ways to be wrong, and
all three are finite, correctly shaped and plausible.

The first is the sign of the centre slope. The score here is a *distance*, so it
falls as the centre approaches the row, where a dense layer's score rises as its
weights do. A sign error therefore pushes every centre away from the data it was
supposed to describe, and the loss curve still moves, just upward on the whole
and downward on some steps.

The second is the width slope, which is the only place in this library where a
parameter appears cubed. ``d output / d width`` is ``output * score / width ** 3``
and the tempting reading is ``output * score / width``, which has the right sign,
the right shape and the right zero at the centre, and is wrong by a factor of
``width ** 2`` -- exactly one at the default width, which is why the fixtures
below deliberately use widths that are not one.

The third is the block passed down, which is the negative of the per-row centre
contribution. The two are the same numbers with opposite signs, so a body that
returns the wrong one of them is off by a sign on every input slope and by
nothing at all on the shapes.

So the finite-difference check is run three ways -- over the centres, over the
widths and over the inputs -- and twice over, on two layers whose widths differ
from one another and from one. It knows nothing about how any of the three
slopes was derived; it runs the layer forward many times and watches.
"""

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    MLLibError,
    ShapeMismatchError,
)
from oop_ml.core.network.gradient import LayerGradient
from oop_ml.core.network.radial_basis import RadialBasisLayer
from oop_ml.core.network.shape import LayerShape

#: The step a central difference takes on either side.
NUDGE = 1e-6


def hand_worked_layer() -> RadialBasisLayer:
    """Three units over two features, every centre and width chosen on paper.

    The first unit sits at the origin with a width of one, the second at
    ``(3, 4)`` -- five away, by the oldest triangle there is -- with a width of
    two, and the third at ``(1, 0)`` with a width of a half. No two units share
    a centre or a width, so a block filled in the wrong order, or a width
    broadcast across the wrong axis, cannot pass by symmetry.
    """
    layer = RadialBasisLayer(reads=2, n_units=3)
    return layer.with_parameters(
        centres=np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]]),
        widths=np.array([1.0, 2.0, 0.5]),
    )


def summary_of(layer: RadialBasisLayer, rows: np.ndarray, weights: np.ndarray) -> float:
    """A scalar reading of the whole answer, whose slope is ``weights`` exactly.

    ``sum(weights * outputs)`` differentiates to ``weights`` at every output, so
    the same block is both the scalar's definition and the arriving block the
    backward pass is handed. That is what makes the finite difference and the
    claim comparable without a loss object standing between them.
    """
    return float(np.sum(weights * layer.respond_to(rows).outputs))


class TestConstruction:
    def test_its_shape_is_the_row_it_reads_and_the_units_it_holds(self) -> None:
        layer = RadialBasisLayer(reads=8, n_units=5)

        assert layer.shape == LayerShape(n_inputs=8, n_outputs=5)

    @pytest.mark.parametrize(
        ("n_features", "n_units"), [(1, 1), (2, 7), (16, 3), (64, 64)]
    )
    def test_the_shape_is_settled_before_a_row_arrives(
        self, n_features: int, n_units: int
    ) -> None:
        layer = RadialBasisLayer(reads=n_features, n_units=n_units)

        assert layer.shape.reads == (n_features,)
        assert layer.shape.answers == (n_units,)

    def test_a_bare_width_and_a_one_extent_tuple_are_the_same_layer(self) -> None:
        """A row is one extent, so both spellings name the same arrangement."""
        bare = RadialBasisLayer(reads=4, n_units=2)
        arranged = RadialBasisLayer(reads=(4,), n_units=2)

        assert bare.shape == arranged.shape

    def test_a_centre_holds_one_coordinate_per_input(self) -> None:
        layer = RadialBasisLayer(reads=6, n_units=4)

        assert layer.centres.shape == (4, 6)

    def test_a_width_is_one_number_per_unit(self) -> None:
        layer = RadialBasisLayer(reads=6, n_units=4)

        assert layer.widths.shape == (4,)

    @pytest.mark.parametrize("width", [0.5, 1.0, 3.25])
    def test_every_width_starts_at_the_stated_value(self, width: float) -> None:
        layer = RadialBasisLayer(reads=3, n_units=5, width=width)

        assert np.allclose(layer.widths, width)

    def test_the_centres_are_not_all_the_same_place(self) -> None:
        """Symmetry breaking. Identical centres are one unit copied five times."""
        layer = RadialBasisLayer(reads=3, n_units=5)

        assert float(np.std(layer.centres)) > 0.0

    def test_the_same_seed_gives_the_same_layer(self) -> None:
        first = RadialBasisLayer(reads=3, n_units=5, random_seed=17)
        second = RadialBasisLayer(reads=3, n_units=5, random_seed=17)

        assert np.array_equal(first.centres, second.centres)

    def test_a_different_seed_gives_a_different_layer(self) -> None:
        first = RadialBasisLayer(reads=3, n_units=5, random_seed=17)
        second = RadialBasisLayer(reads=3, n_units=5, random_seed=18)

        assert not np.array_equal(first.centres, second.centres)

    def test_the_learned_arrays_are_frozen(self) -> None:
        layer = RadialBasisLayer(reads=3, n_units=2)

        with pytest.raises(ValueError):
            layer.centres[0, 0] = 1.0
        with pytest.raises(ValueError):
            layer.widths[0] = 1.0

    def test_the_arrays_a_replacement_was_built_from_are_not_shared(self) -> None:
        """A caller keeping the block it handed in cannot write into the layer."""
        centres = np.zeros((2, 3))
        widths = np.ones(2)
        layer = RadialBasisLayer(reads=3, n_units=2).with_parameters(centres, widths)

        centres[0, 0] = 99.0
        widths[0] = 99.0

        assert float(layer.centres[0, 0]) == 0.0
        assert float(layer.widths[0]) == 1.0


class TestRefusedConfigurations:
    @pytest.mark.parametrize("n_units", [0, -3])
    def test_a_unit_count_below_one_is_refused(self, n_units: int) -> None:
        with pytest.raises(InvalidValuesError):
            RadialBasisLayer(reads=3, n_units=n_units)

    @pytest.mark.parametrize("n_features", [0, -4])
    def test_an_input_width_below_one_is_refused(self, n_features: int) -> None:
        with pytest.raises(InvalidValuesError):
            RadialBasisLayer(reads=n_features, n_units=2)

    def test_a_boolean_unit_count_is_refused(self) -> None:
        """``True`` is an ``int`` in Python, and a layer of one unit by accident."""
        with pytest.raises(InvalidValuesError):
            RadialBasisLayer(reads=3, n_units=True)  # type: ignore[arg-type]

    def test_a_fractional_unit_count_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            RadialBasisLayer(reads=3, n_units=2.5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("reads", [(3, 3), (1, 28, 28)])
    def test_a_picture_is_refused_because_this_layer_reads_a_row(
        self, reads: tuple[int, ...]
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            RadialBasisLayer(reads=reads, n_units=2)

    @pytest.mark.parametrize(
        "reads", [None, 3.5, "four"], ids=["nothing", "a fraction", "a word"]
    )
    def test_reads_that_is_not_an_extent_is_refused_by_name(
        self, reads: object
    ) -> None:
        """The refusal has to be one of this library's own, not builtins'.

        ``tuple(5)`` raises ``TypeError: 'int' object is not iterable`` and
        ``float('four')`` raises a bare ``ValueError``, and neither is an
        ``MLLibError``, so neither may escape from a constructor here.
        """
        with pytest.raises(MLLibError):
            RadialBasisLayer(reads=reads, n_units=2)  # type: ignore[arg-type]

    @pytest.mark.parametrize("width", [0.0, -1.0])
    def test_a_width_that_is_not_above_zero_is_refused(self, width: float) -> None:
        """A width of zero divides by nothing; a negative one climbs the loss."""
        with pytest.raises(InvalidValuesError):
            RadialBasisLayer(reads=3, n_units=2, width=width)

    @pytest.mark.parametrize("width", [float("nan"), float("inf")])
    def test_a_width_that_is_not_finite_is_refused(self, width: float) -> None:
        with pytest.raises(InvalidValuesError):
            RadialBasisLayer(reads=3, n_units=2, width=width)

    def test_a_width_that_is_not_a_number_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            RadialBasisLayer(reads=3, n_units=2, width="wide")  # type: ignore[arg-type]

    def test_a_negative_seed_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            RadialBasisLayer(reads=3, n_units=2, random_seed=-1)

    def test_a_fractional_seed_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            RadialBasisLayer(reads=3, n_units=2, random_seed=1.5)  # type: ignore[arg-type]


class TestTheForwardPassWorkedByHand:
    def test_a_unit_at_its_own_centre_answers_exactly_one(self) -> None:
        """Zero distance, zero exponent, and ``exp(0)`` is exactly 1.0.

        Exactly, not approximately: the squared distance is a sum of exact
        zeros, and ``numpy.exp(-0.0)`` is 1.0 to the last bit.
        """
        layer = hand_worked_layer()

        answer = layer.respond_to(np.array([[0.0, 0.0]])).outputs

        assert float(answer[0, 0]) == 1.0

    @pytest.mark.parametrize("unit", [0, 1, 2])
    def test_every_unit_answers_one_on_its_own_centre(self, unit: int) -> None:
        layer = hand_worked_layer()

        answer = layer.respond_to(np.array([layer.centres[unit]])).outputs

        assert float(answer[0, unit]) == 1.0

    def test_the_scores_are_the_squared_distances(self) -> None:
        """From the origin: 0 to itself, 25 to ``(3, 4)``, 1 to ``(1, 0)``."""
        response = hand_worked_layer().respond_to(np.array([[0.0, 0.0]]))

        assert np.allclose(response.scores[0], [0.0, 25.0, 1.0])

    def test_the_outputs_are_the_bell_taken_at_those_scores(self) -> None:
        """``exp(-score / (2 * width ** 2))``, with the widths 1, 2 and 0.5."""
        response = hand_worked_layer().respond_to(np.array([[0.0, 0.0]]))

        assert np.allclose(
            response.outputs[0],
            [np.exp(-0.0), np.exp(-25.0 / 8.0), np.exp(-1.0 / 0.5)],
        )

    def test_the_answer_falls_as_the_row_moves_away(self) -> None:
        """Walked out along one axis from a single unit sitting at the origin."""
        layer = RadialBasisLayer(reads=1, n_units=1).with_parameters(
            centres=np.zeros((1, 1)), widths=np.ones(1)
        )
        rows = np.array([[0.0], [0.5], [1.0], [2.0], [4.0], [8.0]])

        answers = layer.respond_to(rows).outputs[:, 0]

        assert np.all(np.diff(answers) < 0.0)

    def test_the_answer_depends_only_on_the_distance(self) -> None:
        """A radial function has no opinion about direction, which is its name."""
        layer = RadialBasisLayer(reads=2, n_units=1).with_parameters(
            centres=np.zeros((1, 2)), widths=np.full(1, 1.5)
        )
        ring = np.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [1.8, 2.4]])

        answers = layer.respond_to(ring).outputs[:, 0]

        assert np.allclose(answers, answers[0])

    @pytest.mark.parametrize("width", [0.25, 0.5, 1.0, 2.0, 5.0])
    def test_one_width_out_is_always_the_same_fraction_of_the_peak(
        self, width: float
    ) -> None:
        """At a distance of exactly one width the answer is ``exp(-0.5)``.

        The claim that pins the *rate*. A body squaring the width where it
        should not, or dropping the two from the denominator, still peaks at one
        and still decays, and is caught only here.
        """
        layer = RadialBasisLayer(reads=1, n_units=1).with_parameters(
            centres=np.zeros((1, 1)), widths=np.full(1, width)
        )

        answer = layer.respond_to(np.array([[width]])).outputs

        assert float(answer[0, 0]) == pytest.approx(float(np.exp(-0.5)))

    def test_a_wider_unit_falls_off_more_slowly(self) -> None:
        """The whole point of the width, read at one fixed distance."""
        layer = RadialBasisLayer(reads=1, n_units=3).with_parameters(
            centres=np.zeros((3, 1)), widths=np.array([0.5, 1.0, 2.0])
        )

        answers = layer.respond_to(np.array([[1.0]])).outputs[0]

        assert np.allclose(answers, [np.exp(-2.0), np.exp(-0.5), np.exp(-0.125)])
        assert np.all(np.diff(answers) > 0.0)

    def test_the_answer_has_the_shape_the_layer_promised(self) -> None:
        layer = hand_worked_layer()

        answer = layer.respond_to(np.zeros((4, 2))).outputs

        assert answer.shape == (4, *layer.shape.answers)

    def test_a_second_row_is_answered_independently(self) -> None:
        layer = hand_worked_layer()
        rows = np.array([[0.0, 0.0], [3.0, 4.0]])

        answers = layer.respond_to(rows).outputs

        assert float(answers[0, 0]) == 1.0
        assert float(answers[1, 1]) == 1.0

    def test_the_response_carries_the_block_that_was_read(self) -> None:
        rows = np.array([[1.0, 2.0], [3.0, 4.0]])

        response = hand_worked_layer().respond_to(rows)

        assert np.allclose(response.inputs, rows)

    def test_a_far_row_answers_something_small_rather_than_failing(self) -> None:
        """The exponent is never positive, so this underflows and never blows up."""
        layer = hand_worked_layer()

        answers = layer.respond_to(np.array([[1e6, -1e6]])).outputs

        assert np.all(np.isfinite(answers))
        assert np.allclose(answers, 0.0)


class TestRefusedBlocks:
    def test_a_block_of_the_wrong_arrangement_is_refused(self) -> None:
        """A picture handed to a layer that reads a row."""
        layer = hand_worked_layer()

        with pytest.raises(ShapeMismatchError):
            layer.respond_to(np.zeros((2, 1, 2)))

    def test_a_row_of_the_wrong_width_is_refused(self) -> None:
        layer = hand_worked_layer()

        with pytest.raises(ShapeMismatchError):
            layer.respond_to(np.zeros((2, 5)))

    def test_a_block_of_no_rows_is_refused(self) -> None:
        layer = hand_worked_layer()

        with pytest.raises(EmptyValuesError):
            layer.respond_to(np.zeros((0, 2)))

    def test_a_non_finite_value_is_refused_where_it_enters(self) -> None:
        layer = hand_worked_layer()

        with pytest.raises(InvalidValuesError):
            layer.respond_to(np.array([[np.nan, 0.0]]))

    def test_a_block_that_is_not_numbers_at_all_is_refused(self) -> None:
        layer = hand_worked_layer()

        with pytest.raises(InvalidValuesError):
            layer.respond_to("a row")  # type: ignore[arg-type]

    def test_centres_of_the_wrong_shape_are_refused(self) -> None:
        layer = hand_worked_layer()

        with pytest.raises(ShapeMismatchError):
            layer.with_parameters(np.zeros((3, 5)), np.ones(3))

    def test_widths_of_the_wrong_length_are_refused(self) -> None:
        layer = hand_worked_layer()

        with pytest.raises(ShapeMismatchError):
            layer.with_parameters(np.zeros((3, 2)), np.ones(4))

    @pytest.mark.parametrize("width", [0.0, -0.5])
    def test_a_replacement_width_that_is_not_above_zero_is_refused(
        self, width: float
    ) -> None:
        """A step large enough to cross zero is refused rather than absorbed."""
        layer = hand_worked_layer()

        with pytest.raises(InvalidValuesError):
            layer.with_parameters(np.zeros((3, 2)), np.array([1.0, width, 1.0]))

    def test_a_non_finite_centre_is_refused(self) -> None:
        layer = hand_worked_layer()
        centres = np.zeros((3, 2))
        centres[1, 1] = np.inf

        with pytest.raises(InvalidValuesError):
            layer.with_parameters(centres, np.ones(3))


class TestTheCorrectionsShape:
    def test_the_centre_gradient_is_one_row_per_unit(self) -> None:
        """Exactly the arrangement ``LayerGradient`` already carries."""
        layer = RadialBasisLayer(reads=4, n_units=3)
        rows = np.random.default_rng(0).normal(size=(5, 4))
        response = layer.respond_to(rows)

        correction = layer.correction_for(response, np.ones((5, 3)))

        assert correction.gradient is not None
        assert correction.gradient.weights.shape == (3, 4)
        assert correction.gradient.biases.shape == (3,)

    def test_the_blame_passed_down_has_the_shape_of_what_was_read(self) -> None:
        layer = RadialBasisLayer(reads=4, n_units=3)
        rows = np.random.default_rng(0).normal(size=(5, 4))
        response = layer.respond_to(rows)

        correction = layer.correction_for(response, np.ones((5, 3)))

        assert correction.passed_down.shape == rows.shape

    def test_a_unit_sitting_exactly_on_a_row_wants_to_move_nowhere(self) -> None:
        """Every slope carries the gap as a factor, which is zero at the centre.

        The width slope carries the squared distance instead, and that is zero
        there too, so a unit perfectly placed on the only row it sees is at a
        stationary point of all three.
        """
        layer = RadialBasisLayer(reads=2, n_units=1).with_parameters(
            centres=np.zeros((1, 2)), widths=np.ones(1)
        )
        response = layer.respond_to(np.zeros((1, 2)))

        correction = layer.correction_for(response, np.ones((1, 1)))

        assert correction.gradient is not None
        assert np.allclose(correction.gradient.weights, 0.0)
        assert np.allclose(correction.gradient.biases, 0.0)
        assert np.allclose(correction.passed_down, 0.0)

    def test_the_blame_passed_down_is_the_centres_own_share_reversed(self) -> None:
        """The row and the centre enter the distance with opposite signs.

        With one row and one unit the two blocks hold the same numbers negated,
        which is the cheapest available statement that the layer did not return
        one of them where it meant the other.
        """
        layer = RadialBasisLayer(reads=3, n_units=1, random_seed=2)
        rows = np.random.default_rng(3).normal(size=(1, 3))
        response = layer.respond_to(rows)

        correction = layer.correction_for(response, np.full((1, 1), 0.7))

        assert correction.gradient is not None
        assert np.allclose(correction.passed_down[0], -correction.gradient.weights[0])

    def test_an_arriving_block_of_the_wrong_shape_is_refused(self) -> None:
        layer = hand_worked_layer()
        response = layer.respond_to(np.zeros((2, 2)))

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((2, 5)))

    def test_an_arriving_block_of_the_wrong_row_count_is_refused(self) -> None:
        layer = hand_worked_layer()
        response = layer.respond_to(np.zeros((2, 2)))

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((3, 3)))

    def test_an_arriving_block_that_is_not_numbers_is_refused(self) -> None:
        layer = hand_worked_layer()
        response = layer.respond_to(np.zeros((2, 2)))

        with pytest.raises(InvalidValuesError):
            layer.correction_for(response, "blame")  # type: ignore[arg-type]


class TestTheGradientCheck:
    """Nudge one number, watch the summary, compare against the claim.

    Run over the centres, the widths and the inputs in turn, because they are
    three separate derivations sharing only the bell's own slope, and a mistake
    in any one of them leaves the other two right. Run on two layers, because
    the width slope is the one that carries ``width ** 3`` and a body that wrote
    ``width`` instead agrees exactly at a width of one.
    """

    def narrow_layer(self) -> RadialBasisLayer:
        """Four units over three features, every width well under one."""
        return RadialBasisLayer(reads=3, n_units=4, width=0.75, random_seed=5)

    def wide_layer(self) -> RadialBasisLayer:
        """Three units over two features, every width well over one."""
        return RadialBasisLayer(reads=2, n_units=3, width=2.5, random_seed=6)

    def batch_for(
        self, layer: RadialBasisLayer, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Some rows, and the summary weights that are also the blame."""
        generator = np.random.default_rng(seed)
        rows = generator.normal(size=(4, *layer.shape.reads))
        weights = generator.normal(size=(4, *layer.shape.answers))
        return rows, weights

    def measured_centre_gradient(
        self, layer: RadialBasisLayer, rows: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        measured = np.zeros_like(layer.centres)
        for unit in range(layer.centres.shape[0]):
            for coordinate in range(layer.centres.shape[1]):
                moved = []
                for direction in (+NUDGE, -NUDGE):
                    centres = np.array(layer.centres, copy=True)
                    centres[unit, coordinate] += direction
                    moved.append(
                        summary_of(
                            layer.with_parameters(centres, layer.widths), rows, weights
                        )
                    )
                measured[unit, coordinate] = (moved[0] - moved[1]) / (2.0 * NUDGE)
        return measured

    def measured_width_gradient(
        self, layer: RadialBasisLayer, rows: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        measured = np.zeros_like(layer.widths)
        for unit in range(layer.widths.shape[0]):
            moved = []
            for direction in (+NUDGE, -NUDGE):
                widths = np.array(layer.widths, copy=True)
                widths[unit] += direction
                moved.append(
                    summary_of(
                        layer.with_parameters(layer.centres, widths), rows, weights
                    )
                )
            measured[unit] = (moved[0] - moved[1]) / (2.0 * NUDGE)
        return measured

    def measured_input_gradient(
        self, layer: RadialBasisLayer, rows: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        measured = np.zeros_like(rows)
        for row in range(rows.shape[0]):
            for coordinate in range(rows.shape[1]):
                moved = []
                for direction in (+NUDGE, -NUDGE):
                    nudged = np.array(rows, copy=True)
                    nudged[row, coordinate] += direction
                    moved.append(summary_of(layer, nudged, weights))
                measured[row, coordinate] = (moved[0] - moved[1]) / (2.0 * NUDGE)
        return measured

    @pytest.mark.parametrize("which", ["narrow", "wide"])
    def test_every_centre_slope_matches_a_finite_difference(self, which: str) -> None:
        layer = self.narrow_layer() if which == "narrow" else self.wide_layer()
        rows, weights = self.batch_for(layer, seed=11)

        correction = layer.correction_for(layer.respond_to(rows), weights)
        measured = self.measured_centre_gradient(layer, rows, weights)

        assert correction.gradient is not None
        assert np.allclose(correction.gradient.weights, measured, atol=1e-7)

    @pytest.mark.parametrize("which", ["narrow", "wide"])
    def test_every_width_slope_matches_a_finite_difference(self, which: str) -> None:
        """The slope carrying ``width ** 3``, checked at two widths that are not one."""
        layer = self.narrow_layer() if which == "narrow" else self.wide_layer()
        rows, weights = self.batch_for(layer, seed=11)

        correction = layer.correction_for(layer.respond_to(rows), weights)
        measured = self.measured_width_gradient(layer, rows, weights)

        assert correction.gradient is not None
        assert np.allclose(correction.gradient.biases, measured, atol=1e-7)

    @pytest.mark.parametrize("which", ["narrow", "wide"])
    def test_every_input_slope_matches_a_finite_difference(self, which: str) -> None:
        """The half that travels downward, and the half a sign error reverses."""
        layer = self.narrow_layer() if which == "narrow" else self.wide_layer()
        rows, weights = self.batch_for(layer, seed=11)

        correction = layer.correction_for(layer.respond_to(rows), weights)
        measured = self.measured_input_gradient(layer, rows, weights)

        assert np.allclose(correction.passed_down, measured, atol=1e-7)

    def test_the_widths_really_do_differ_from_one(self) -> None:
        """Without this the width check is run twice at the one width that hides.

        ``output * score / width ** 3`` and ``output * score / width`` agree
        exactly at a width of one, so a fixture that used the default would pass
        a body that is wrong everywhere else.
        """
        assert not np.allclose(self.narrow_layer().widths, 1.0)
        assert not np.allclose(self.wide_layer().widths, 1.0)


class TestStepping:
    def test_a_step_moves_against_the_slope(self) -> None:
        layer = RadialBasisLayer(reads=3, n_units=2, width=1.5, random_seed=3)
        rows = np.random.default_rng(4).normal(size=(4, 3))
        correction = layer.correction_for(layer.respond_to(rows), np.ones((4, 2)))
        assert correction.gradient is not None

        stepped = layer.stepped_by(correction.gradient, learning_rate=0.01)

        assert np.allclose(
            stepped.centres, layer.centres - 0.01 * correction.gradient.weights
        )
        assert np.allclose(
            stepped.widths, layer.widths - 0.01 * correction.gradient.biases
        )

    def test_a_step_lowers_the_summary_it_was_asked_to_lower(self) -> None:
        """The point of the whole exercise, on one small step."""
        layer = RadialBasisLayer(reads=3, n_units=4, width=1.25, random_seed=8)
        generator = np.random.default_rng(21)
        rows = generator.normal(size=(6, *layer.shape.reads))
        weights = generator.normal(size=(6, *layer.shape.answers))

        correction = layer.correction_for(layer.respond_to(rows), weights)
        stepped = layer.stepped_by(correction.gradient, learning_rate=0.01)

        assert summary_of(stepped, rows, weights) < summary_of(layer, rows, weights)

    def test_a_step_returns_a_new_layer(self) -> None:
        layer = RadialBasisLayer(reads=3, n_units=2, random_seed=3)
        rows = np.random.default_rng(4).normal(size=(4, 3))
        correction = layer.correction_for(layer.respond_to(rows), np.ones((4, 2)))

        stepped = layer.stepped_by(correction.gradient, learning_rate=0.01)

        assert stepped is not layer
        assert isinstance(stepped, RadialBasisLayer)

    def test_the_original_layer_is_untouched(self) -> None:
        """Training does not mutate, which is what keeps the arrays frozen."""
        layer = RadialBasisLayer(reads=3, n_units=2, width=1.5, random_seed=3)
        centres_before = np.array(layer.centres, copy=True)
        widths_before = np.array(layer.widths, copy=True)
        rows = np.random.default_rng(5).normal(size=(4, 3))
        correction = layer.correction_for(layer.respond_to(rows), np.ones((4, 2)))

        layer.stepped_by(correction.gradient, learning_rate=0.05)

        assert np.array_equal(layer.centres, centres_before)
        assert np.array_equal(layer.widths, widths_before)

    def test_stepping_by_nothing_is_refused(self) -> None:
        """This layer has parameters, so a missing gradient is a mistake."""
        layer = RadialBasisLayer(reads=3, n_units=2)

        with pytest.raises(ShapeMismatchError):
            layer.stepped_by(None, learning_rate=0.1)

    def test_a_gradient_of_the_wrong_width_is_refused(self) -> None:
        layer = RadialBasisLayer(reads=3, n_units=2)

        with pytest.raises(ShapeMismatchError):
            layer.stepped_by(
                LayerGradient(weights=np.zeros((2, 9)), biases=np.zeros(2)),
                learning_rate=0.1,
            )

    def test_a_gradient_naming_the_wrong_number_of_units_is_refused(self) -> None:
        layer = RadialBasisLayer(reads=3, n_units=2)

        with pytest.raises(ShapeMismatchError):
            layer.stepped_by(
                LayerGradient(weights=np.zeros((5, 3)), biases=np.zeros(5)),
                learning_rate=0.1,
            )

    def test_a_step_that_would_drive_a_width_through_zero_is_refused(self) -> None:
        """The guard is reachable, so the spec reaches it rather than assuming.

        A learning rate large enough to overshoot leaves a width at or below
        zero, which is a unit that answers a delta rather than a bell and
        divides by nothing on its next pass. Refusing names the step that did
        it, where clamping would leave a layer nobody chose.
        """
        layer = RadialBasisLayer(reads=2, n_units=1, width=0.5)

        with pytest.raises(InvalidValuesError):
            layer.stepped_by(
                LayerGradient(weights=np.zeros((1, 2)), biases=np.full(1, 1.0)),
                learning_rate=1.0,
            )

    def test_the_stepped_layer_keeps_its_shape(self) -> None:
        layer = RadialBasisLayer(reads=4, n_units=3, width=2.0, random_seed=6)
        rows = np.random.default_rng(6).normal(size=(3, 4))
        correction = layer.correction_for(layer.respond_to(rows), np.ones((3, 3)))

        stepped = layer.stepped_by(correction.gradient, learning_rate=0.01)

        assert stepped.shape == layer.shape
        assert stepped.centres.shape == layer.centres.shape
        assert stepped.widths.shape == layer.widths.shape
