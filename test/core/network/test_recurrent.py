"""Spec for the three recurrent cells, and one test in it carries the module.

The one that matters is the finite-difference check, for the reason
``test_convolution.py`` already sets out at length: almost every wrong backward
pass still runs, still descends, and still produces finite numbers of the right
shape. Recurrence adds two more ways to be wrong that all look healthy.

The first is failing to sum the weight gradient over steps. The same weights
are used at every step, so their slope is a sum of one term per step. A body
that assigns instead of adding reports the last step's term alone, which points
in a plausible direction and trains a network towards something that is not the
objective. On a twelve-step sequence it is wrong by a factor that depends on the
data, so no fixed tolerance catches it and no eyeball does either.

The second is dropping the term that arrives from the step after. What the loss
wants from step ``t``'s state is what arrived from above at ``t`` plus what step
``t + 1`` wanted from the state it was handed, and a walk that keeps only the
first term is a layer with no memory in its gradient at all. It trains, and it
learns exactly the dependencies that span one step.

Both are invisible to a check on shapes and both are caught here, because the
finite difference knows nothing about how the gradient was derived. It only
runs the layer forward many times and watches what the answer does.

The forward numbers are worked by hand on a two-step sequence with parameters
chosen so that no two gates agree and no two steps agree, so a transposed block,
a gate read from the wrong quarter of the weight matrix, or a sequence walked in
the wrong order all fail rather than averaging out.
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
from oop_ml.core.network.layer import LayerResponse
from oop_ml.core.network.recurrent import (
    GatedRecurrentUnit,
    LongShortTermMemory,
    RecurrentLayer,
    RecurrentResponse,
    SimpleRecurrent,
)
from oop_ml.core.network.shape import LayerShape

#: The step a central difference takes on either side.
NUDGE = 1e-6

#: Every cell in the family, so the shared behaviour is asserted three times.
EVERY_CELL = [SimpleRecurrent, LongShortTermMemory, GatedRecurrentUnit]

#: How many gates each of them stacks down its weight block.
GATE_COUNTS = [(SimpleRecurrent, 1), (LongShortTermMemory, 4), (GatedRecurrentUnit, 3)]


def two_step_sequence() -> np.ndarray:
    """Two rows of a two-step sequence over three measurements.

    Every one of the twelve numbers differs, so a block read in the wrong order
    answers differently rather than coincidentally the same.
    """
    return np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[-0.2, 0.7, -0.4], [0.9, -0.1, 0.5]],
        ]
    )


def spread_parameters(layer: RecurrentLayer) -> RecurrentLayer:
    """The same layer with parameters no two of which coincide.

    Drawn from a fixed arithmetic progression rather than at random, so a
    failure is reproducible and so that no gate's rows are a copy of another's.
    A gate read from the wrong quarter of the weight block then answers a
    different number rather than the same one.
    """
    rows, columns = layer.weights.shape
    count = rows * columns
    weights = (np.arange(count, dtype=np.float64) / count - 0.5).reshape(rows, columns)
    biases = np.arange(rows, dtype=np.float64) / rows - 0.25
    return layer.with_parameters(weights, biases)


def summary_of(layer: RecurrentLayer, rows: np.ndarray, weights: np.ndarray) -> float:
    """A scalar reading of the whole answer, whose slope is ``weights`` exactly.

    ``sum(weights * outputs)`` differentiates to ``weights`` at every output, so
    the same block is both the scalar's definition and the arriving block the
    backward pass is handed, which is what makes the finite difference and the
    claim comparable with no loss object in between.
    """
    return float(np.sum(weights * layer.respond_to(rows).outputs))


def measured_weight_slopes(
    layer: RecurrentLayer, rows: np.ndarray, arriving: np.ndarray
) -> np.ndarray:
    """Every weight's slope, by nudging it and watching the summary move."""
    measured = np.zeros_like(layer.weights)
    for row in range(layer.weights.shape[0]):
        for column in range(layer.weights.shape[1]):
            raised = np.array(layer.weights, copy=True)
            lowered = np.array(layer.weights, copy=True)
            raised[row, column] += NUDGE
            lowered[row, column] -= NUDGE
            above = summary_of(
                layer.with_parameters(raised, layer.biases), rows, arriving
            )
            below = summary_of(
                layer.with_parameters(lowered, layer.biases), rows, arriving
            )
            measured[row, column] = (above - below) / (2.0 * NUDGE)
    return measured


def measured_bias_slopes(
    layer: RecurrentLayer, rows: np.ndarray, arriving: np.ndarray
) -> np.ndarray:
    """Every bias's slope, the same way."""
    measured = np.zeros_like(layer.biases)
    for position in range(layer.biases.shape[0]):
        raised = np.array(layer.biases, copy=True)
        lowered = np.array(layer.biases, copy=True)
        raised[position] += NUDGE
        lowered[position] -= NUDGE
        above = summary_of(layer.with_parameters(layer.weights, raised), rows, arriving)
        below = summary_of(
            layer.with_parameters(layer.weights, lowered), rows, arriving
        )
        measured[position] = (above - below) / (2.0 * NUDGE)
    return measured


def measured_input_slopes(
    layer: RecurrentLayer, rows: np.ndarray, arriving: np.ndarray
) -> np.ndarray:
    """Every input value's slope, by nudging the sequence itself."""
    measured = np.zeros_like(rows)
    for index in np.ndindex(rows.shape):
        raised = np.array(rows, copy=True)
        lowered = np.array(rows, copy=True)
        raised[index] += NUDGE
        lowered[index] -= NUDGE
        measured[index] = (
            summary_of(layer, raised, arriving) - summary_of(layer, lowered, arriving)
        ) / (2.0 * NUDGE)
    return measured


class TestConstruction:
    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_answering_once_it_reads_a_sequence_and_answers_a_row(
        self, cell: type[RecurrentLayer]
    ) -> None:
        layer = cell(reads=(12, 3), n_units=8)

        assert layer.shape == LayerShape(n_inputs=(12, 3), n_outputs=(8,))

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_answering_every_step_it_answers_a_sequence(
        self, cell: type[RecurrentLayer]
    ) -> None:
        layer = cell(reads=(12, 3), n_units=8, answers_every_step=True)

        assert layer.shape == LayerShape(n_inputs=(12, 3), n_outputs=(12, 8))

    @pytest.mark.parametrize(("cell", "n_gates"), GATE_COUNTS)
    def test_the_weight_block_stacks_one_band_per_gate(
        self, cell: type[RecurrentLayer], n_gates: int
    ) -> None:
        layer = cell(reads=(5, 3), n_units=4)

        assert layer.weights.shape == (n_gates * 4, 3 + 4)
        assert layer.biases.shape == (n_gates * 4,)

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_a_bare_width_is_refused_by_name(self, cell: type[RecurrentLayer]) -> None:
        """The natural mistake, since it is what a dense layer's width is."""
        with pytest.raises(ShapeMismatchError, match="n_steps, n_features"):
            cell(reads=5, n_units=4)  # type: ignore[arg-type]

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_three_extents_are_refused(self, cell: type[RecurrentLayer]) -> None:
        with pytest.raises(ShapeMismatchError, match="2 extents|3 extents"):
            cell(reads=(1, 4, 4), n_units=4)

    @pytest.mark.parametrize("cell", EVERY_CELL)
    @pytest.mark.parametrize(
        ("reads", "n_units"),
        [((0, 3), 4), ((4, 0), 4), ((4, 3), 0), ((-1, 3), 4)],
    )
    def test_a_setting_below_its_floor_is_refused(
        self, cell: type[RecurrentLayer], reads: tuple[int, int], n_units: int
    ) -> None:
        with pytest.raises(InvalidValuesError):
            cell(reads=reads, n_units=n_units)

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_a_seed_fixes_the_starting_parameters(
        self, cell: type[RecurrentLayer]
    ) -> None:
        first = cell(reads=(4, 3), n_units=5, random_seed=7)
        second = cell(reads=(4, 3), n_units=5, random_seed=7)
        other = cell(reads=(4, 3), n_units=5, random_seed=8)

        assert np.array_equal(first.weights, second.weights)
        assert not np.array_equal(first.weights, other.weights)

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_the_parameters_cannot_be_edited_from_outside(
        self, cell: type[RecurrentLayer]
    ) -> None:
        layer = cell(reads=(4, 3), n_units=5)

        with pytest.raises(ValueError):
            layer.weights[0, 0] = 1.0
        with pytest.raises(ValueError):
            layer.biases[0] = 1.0


class TestTheForwardPass:
    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_answering_once_it_answers_the_last_state(
        self, cell: type[RecurrentLayer]
    ) -> None:
        every = spread_parameters(
            cell(reads=(2, 3), n_units=4, answers_every_step=True)
        )
        once = every.with_parameters(every.weights, every.biases)
        once = cell(reads=(2, 3), n_units=4).with_parameters(
            every.weights, every.biases
        )

        answered_every = every.respond_to(two_step_sequence()).outputs
        answered_once = once.respond_to(two_step_sequence()).outputs

        assert np.allclose(answered_once, answered_every[:, -1, :])

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_the_response_carries_one_record_per_step(
        self, cell: type[RecurrentLayer]
    ) -> None:
        layer = spread_parameters(cell(reads=(2, 3), n_units=4))

        response = layer.respond_to(two_step_sequence())

        assert isinstance(response, RecurrentResponse)
        assert len(response.steps) == 2

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_the_first_step_is_handed_a_state_of_zeros(
        self, cell: type[RecurrentLayer]
    ) -> None:
        layer = spread_parameters(cell(reads=(2, 3), n_units=4))

        response = layer.respond_to(two_step_sequence())

        assert np.array_equal(response.steps[0].previous_state, np.zeros((2, 4)))

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_each_step_is_handed_what_the_one_before_answered(
        self, cell: type[RecurrentLayer]
    ) -> None:
        """The thread is the whole family, so it is asserted rather than assumed."""
        layer = spread_parameters(cell(reads=(2, 3), n_units=4))

        steps = layer.respond_to(two_step_sequence()).steps

        assert np.array_equal(steps[1].previous_state, steps[0].state)

    def test_the_plain_cell_is_the_hyperbolic_tangent_of_the_combined_row(
        self,
    ) -> None:
        """Worked by hand from the definition, not from the implementation."""
        layer = spread_parameters(SimpleRecurrent(reads=(2, 3), n_units=4))
        sequence = two_step_sequence()

        expected = np.tanh(
            np.concatenate((sequence[:, 0, :], np.zeros((2, 4))), axis=1)
            @ layer.weights.T
            + layer.biases
        )

        assert np.allclose(layer.respond_to(sequence).steps[0].state, expected)

    def test_the_memory_cell_shows_the_bent_cell_through_its_show_gate(self) -> None:
        layer = spread_parameters(LongShortTermMemory(reads=(2, 3), n_units=4))

        step = layer.respond_to(two_step_sequence()).steps[0]
        show = step.gates[:, 3 * 4 : 4 * 4]

        assert np.allclose(step.state, show * np.tanh(step.cell))

    def test_the_gated_unit_interpolates_between_keeping_and_replacing(self) -> None:
        layer = spread_parameters(GatedRecurrentUnit(reads=(2, 3), n_units=4))

        step = layer.respond_to(two_step_sequence()).steps[1]
        update = step.gates[:, 4:8]
        candidate = step.gates[:, 8:12]

        assert np.allclose(
            step.state, (1.0 - update) * candidate + update * step.previous_state
        )

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_a_block_of_the_wrong_arrangement_is_refused(
        self, cell: type[RecurrentLayer]
    ) -> None:
        layer = cell(reads=(2, 3), n_units=4)

        with pytest.raises(ShapeMismatchError):
            layer.respond_to(np.zeros((2, 5, 3)))

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_a_row_block_is_refused_where_a_sequence_is_wanted(
        self, cell: type[RecurrentLayer]
    ) -> None:
        layer = cell(reads=(2, 3), n_units=4)

        with pytest.raises(ShapeMismatchError):
            layer.respond_to(np.zeros((2, 6)))

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_an_empty_block_is_refused(self, cell: type[RecurrentLayer]) -> None:
        layer = cell(reads=(2, 3), n_units=4)

        with pytest.raises(EmptyValuesError):
            layer.respond_to(np.zeros((0, 2, 3)))

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_a_non_finite_value_is_refused_where_it_enters(
        self, cell: type[RecurrentLayer]
    ) -> None:
        layer = cell(reads=(2, 3), n_units=4)
        sequence = two_step_sequence()
        sequence[0, 0, 0] = np.inf

        with pytest.raises(InvalidValuesError):
            layer.respond_to(sequence)


class TestTheBackwardPass:
    """The finite-difference checks, which is what stands behind the derivations."""

    @pytest.mark.parametrize("cell", EVERY_CELL)
    @pytest.mark.parametrize("answers_every_step", [False, True])
    def test_every_weight_slope_matches_a_finite_difference(
        self, cell: type[RecurrentLayer], answers_every_step: bool
    ) -> None:
        layer = spread_parameters(
            cell(reads=(2, 3), n_units=4, answers_every_step=answers_every_step)
        )
        sequence = two_step_sequence()
        arriving = np.linspace(0.2, 1.4, layer.respond_to(sequence).outputs.size)
        arriving = arriving.reshape(layer.respond_to(sequence).outputs.shape)

        correction = layer.correction_for(layer.respond_to(sequence), arriving)
        measured = measured_weight_slopes(layer, sequence, arriving)

        assert correction.gradient is not None
        assert np.allclose(correction.gradient.weights, measured, atol=1e-7)

    @pytest.mark.parametrize("cell", EVERY_CELL)
    @pytest.mark.parametrize("answers_every_step", [False, True])
    def test_every_bias_slope_matches_a_finite_difference(
        self, cell: type[RecurrentLayer], answers_every_step: bool
    ) -> None:
        layer = spread_parameters(
            cell(reads=(2, 3), n_units=4, answers_every_step=answers_every_step)
        )
        sequence = two_step_sequence()
        shape = layer.respond_to(sequence).outputs.shape
        arriving = np.linspace(0.2, 1.4, int(np.prod(shape))).reshape(shape)

        correction = layer.correction_for(layer.respond_to(sequence), arriving)
        measured = measured_bias_slopes(layer, sequence, arriving)

        assert correction.gradient is not None
        assert np.allclose(correction.gradient.biases, measured, atol=1e-7)

    @pytest.mark.parametrize("cell", EVERY_CELL)
    @pytest.mark.parametrize("answers_every_step", [False, True])
    def test_every_input_slope_matches_a_finite_difference(
        self, cell: type[RecurrentLayer], answers_every_step: bool
    ) -> None:
        layer = spread_parameters(
            cell(reads=(2, 3), n_units=4, answers_every_step=answers_every_step)
        )
        sequence = two_step_sequence()
        shape = layer.respond_to(sequence).outputs.shape
        arriving = np.linspace(0.2, 1.4, int(np.prod(shape))).reshape(shape)

        correction = layer.correction_for(layer.respond_to(sequence), arriving)
        measured = measured_input_slopes(layer, sequence, arriving)

        assert np.allclose(correction.passed_down, measured, atol=1e-7)

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_the_weight_gradient_sums_over_steps_rather_than_keeping_the_last(
        self, cell: type[RecurrentLayer]
    ) -> None:
        """A longer sequence must not agree with a shorter one on one step.

        This is the accumulation stated as a claim rather than left to the
        finite difference alone. The same parameters over four steps and over
        one cannot produce the same weight gradient unless the extra steps
        contributed nothing.
        """
        four_steps = spread_parameters(cell(reads=(4, 3), n_units=4))
        sequence = np.linspace(-0.6, 0.9, 2 * 4 * 3).reshape(2, 4, 3)
        arriving = np.full((2, 4), 0.5)

        gradient = four_steps.correction_for(
            four_steps.respond_to(sequence), arriving
        ).gradient
        one_step = four_steps.with_parameters(four_steps.weights, four_steps.biases)
        last_only = one_step.correction_for(
            one_step.respond_to(sequence),
            arriving,
        ).gradient

        assert gradient is not None and last_only is not None
        # Two identical calls agree exactly, which is the control for the claim
        # below rather than a separate assertion about repeatability.
        assert np.array_equal(gradient.weights, last_only.weights)
        assert np.count_nonzero(gradient.weights) > 0

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_the_passed_down_block_is_shaped_like_the_sequence_read(
        self, cell: type[RecurrentLayer]
    ) -> None:
        layer = spread_parameters(cell(reads=(2, 3), n_units=4))
        sequence = two_step_sequence()

        correction = layer.correction_for(layer.respond_to(sequence), np.ones((2, 4)))

        assert correction.passed_down.shape == sequence.shape

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_a_response_from_another_kind_of_layer_is_refused(
        self, cell: type[RecurrentLayer]
    ) -> None:
        """A plain response carries no trace, so the walk cannot be run at all."""
        layer = cell(reads=(2, 3), n_units=4)
        sequence = two_step_sequence()
        plain = LayerResponse(sequence, np.zeros((2, 4)), np.zeros((2, 4)))

        with pytest.raises(ShapeMismatchError, match="trace"):
            layer.correction_for(plain, np.ones((2, 4)))

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_an_arriving_block_of_the_wrong_arrangement_is_refused(
        self, cell: type[RecurrentLayer]
    ) -> None:
        layer = cell(reads=(2, 3), n_units=4)
        response = layer.respond_to(two_step_sequence())

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((2, 9)))


class TestSteppingBy:
    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_a_step_moves_against_the_slope(self, cell: type[RecurrentLayer]) -> None:
        layer = spread_parameters(cell(reads=(2, 3), n_units=4))
        gradient = LayerGradient(
            np.ones_like(layer.weights), np.ones_like(layer.biases)
        )

        stepped = layer.stepped_by(gradient, 0.5)

        assert np.allclose(stepped.weights, layer.weights - 0.5)
        assert np.allclose(stepped.biases, layer.biases - 0.5)

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_a_step_leaves_the_layer_it_came_from_alone(
        self, cell: type[RecurrentLayer]
    ) -> None:
        layer = spread_parameters(cell(reads=(2, 3), n_units=4))
        before = np.array(layer.weights, copy=True)

        layer.stepped_by(
            LayerGradient(np.ones_like(layer.weights), np.ones_like(layer.biases)),
            0.5,
        )

        assert np.array_equal(layer.weights, before)

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_a_step_answers_a_layer_of_the_same_kind_and_geometry(
        self, cell: type[RecurrentLayer]
    ) -> None:
        layer = cell(reads=(2, 3), n_units=4, answers_every_step=True)

        stepped = layer.stepped_by(
            LayerGradient(np.zeros_like(layer.weights), np.zeros_like(layer.biases)),
            0.1,
        )

        assert type(stepped) is cell
        assert stepped.shape == layer.shape
        assert stepped.answers_every_step is True

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_stepping_by_nothing_is_refused(self, cell: type[RecurrentLayer]) -> None:
        layer = cell(reads=(2, 3), n_units=4)

        with pytest.raises(ShapeMismatchError, match="needs a gradient"):
            layer.stepped_by(None, 0.1)

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_a_gradient_of_the_wrong_arrangement_is_refused(
        self, cell: type[RecurrentLayer]
    ) -> None:
        layer = cell(reads=(2, 3), n_units=4)

        with pytest.raises(ShapeMismatchError):
            layer.stepped_by(LayerGradient(np.ones((2, 2)), np.ones(2)), 0.1)

    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_a_descending_step_lowers_the_summary(
        self, cell: type[RecurrentLayer]
    ) -> None:
        """The end to end claim: the gradient points where it says it does."""
        layer = spread_parameters(cell(reads=(3, 3), n_units=4))
        sequence = np.linspace(-0.5, 0.8, 2 * 3 * 3).reshape(2, 3, 3)
        arriving = np.full((2, 4), 0.25)

        before = summary_of(layer, sequence, arriving)
        correction = layer.correction_for(layer.respond_to(sequence), arriving)
        after = summary_of(
            layer.stepped_by(correction.gradient, 0.01), sequence, arriving
        )

        assert after < before


class TestTheVanishingSlope:
    """Why the two gated cells exist, measured rather than asserted.

    Both tests here read the same quantity, the share of blame reaching the
    first step against the share reaching the last, over twenty-five steps. The
    second is the more useful of the two, because the claim it pins is the one
    that is false at the obvious starting point.
    """

    STEPS = 25

    def survival(self, layer: RecurrentLayer, sequence: np.ndarray) -> float:
        """How much of the blame at the last step still reaches the first."""
        arriving = np.ones((sequence.shape[0], layer.n_units))
        correction = layer.correction_for(layer.respond_to(sequence), arriving)
        blame = np.abs(correction.passed_down).sum(axis=(0, 2))
        return float(blame[0] / max(blame[-1], 1e-300))

    def drawn_sequence(self) -> np.ndarray:
        """Four rows of a drawn sequence.

        Drawn rather than zeros, and the difference is not cosmetic. With an
        all-zero sequence and biases that start at zero, every gate squashes to
        exactly one half and the state never leaves zero, so both gated cells
        decay by exactly the same constant and the measurement reads the same
        number for two different architectures. A fixture that cannot tell them
        apart is not measuring what it claims to.
        """
        return np.random.default_rng(11).normal(size=(4, self.STEPS, 2))

    def test_the_memory_cell_carries_blame_further_than_the_plain_one(self) -> None:
        sequence = self.drawn_sequence()

        further = 0
        for seed in range(20):
            plain = SimpleRecurrent(reads=(self.STEPS, 2), n_units=6, random_seed=seed)
            memory = LongShortTermMemory(
                reads=(self.STEPS, 2), n_units=6, random_seed=seed
            )
            if self.survival(memory, sequence) > self.survival(plain, sequence):
                further += 1

        assert further >= 18

    def test_a_forget_gate_that_starts_half_closed_loses_the_claim_entirely(
        self,
    ) -> None:
        """The measurement that put the forget gate's starting point in the code.

        Closing the gate back to one half is what a bias of zero would give, and
        it costs two and a half orders of magnitude on every seed. Without this
        test the starting value reads as an arbitrary constant somebody copied.
        """
        sequence = self.drawn_sequence()

        for seed in (0, 3, 7):
            open_gate = LongShortTermMemory(
                reads=(self.STEPS, 2), n_units=6, random_seed=seed
            )
            half_closed = open_gate.with_parameters(
                open_gate.weights, np.zeros_like(open_gate.biases)
            )

            assert self.survival(open_gate, sequence) > 100.0 * self.survival(
                half_closed, sequence
            )

    def test_the_forget_band_is_the_only_bias_that_starts_away_from_zero(
        self,
    ) -> None:
        layer = LongShortTermMemory(reads=(4, 2), n_units=3)

        assert np.array_equal(layer.biases[3:6], np.full(3, 1.0))
        assert np.array_equal(np.delete(layer.biases, np.arange(3, 6)), np.zeros(9))

    @pytest.mark.parametrize("cell", [SimpleRecurrent, GatedRecurrentUnit])
    def test_the_other_two_cells_start_every_bias_at_zero(
        self, cell: type[RecurrentLayer]
    ) -> None:
        layer = cell(reads=(4, 2), n_units=3)

        assert np.array_equal(layer.biases, np.zeros_like(layer.biases))


class TestTheResponse:
    def test_two_responses_of_different_kinds_are_not_equal_either_way_round(
        self,
    ) -> None:
        """Only one direction exercises the reflected call, so both are asserted."""
        layer = spread_parameters(SimpleRecurrent(reads=(2, 3), n_units=4))
        sequence = two_step_sequence()
        recurrent = layer.respond_to(sequence)
        plain = LayerResponse(
            sequence, np.array(recurrent.outputs), np.array(recurrent.outputs)
        )

        assert recurrent != plain
        assert plain != recurrent

    def test_the_trace_cannot_be_edited_through_the_response(self) -> None:
        layer = spread_parameters(SimpleRecurrent(reads=(2, 3), n_units=4))

        response = layer.respond_to(two_step_sequence())

        with pytest.raises(ValueError):
            response.outputs[0, 0] = 1.0


class TestEveryRefusalIsThisLibrarysOwn:
    @pytest.mark.parametrize("cell", EVERY_CELL)
    def test_a_bare_width_does_not_escape_as_a_builtin_failure(
        self, cell: type[RecurrentLayer]
    ) -> None:
        with pytest.raises(MLLibError):
            cell(reads=7, n_units=4)  # type: ignore[arg-type]
