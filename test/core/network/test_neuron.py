"""Spec for the neuron -- red until ``_response_for`` lands.

Three of these are worth reading rather than skimming.

The first is the agreement with ``LogisticRegression``. A neuron carrying a
sigmoid is that model, so the two are fitted and asked the same question and
have to answer identically. It is the same shape of test as the two tree routes
that had to agree on an exact tie, and it is what stops the neuron drifting into
being a second, subtly different logistic model.

The second is the exclusive-or ceiling, asserted rather than described. Four
corners are pushed through a neuron under every activation offered, and the
test asserts no weighting can separate the two diagonals. The argument in the
module docstring holds for any monotone bend, so a body that somehow passed
this would have found a genuine counterexample to it.

The third is the frozen weight buffer. A caller who reaches through the column
and writes to it would be editing a fitted parameter in place, which is the
encapsulation rule the library pinned once already for probabilities and leaf
shares.
"""

import numpy as np
import pytest

from oop_ml.core.data.column import Column
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
)
from oop_ml.core.network.activation import (
    Activation,
    HyperbolicTangent,
    Identity,
    RectifiedLinear,
    Sigmoid,
)
from oop_ml.core.network.neuron import Neuron, NeuronResponse
from oop_ml.core.validation import ValueRole

EVERY_ACTIVATION = [Identity(), RectifiedLinear(), Sigmoid(), HyperbolicTangent()]

EXCLUSIVE_OR_CORNERS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]


class TestConstruction:
    def test_the_weight_count_is_the_input_width(self) -> None:
        neuron = Neuron([0.5, -1.5, 2.0], bias=0.0, activation=Identity())

        assert neuron.n_inputs == 3

    def test_a_neuron_with_no_inputs_is_refused(self) -> None:
        """It would be a constant wearing a neuron's name."""
        with pytest.raises(EmptyValuesError):
            Neuron([], bias=0.0, activation=Identity())

    @pytest.mark.parametrize("bad_weight", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_weight_is_refused(self, bad_weight: float) -> None:
        with pytest.raises(InvalidValuesError):
            Neuron([1.0, bad_weight], bias=0.0, activation=Identity())

    @pytest.mark.parametrize("bad_bias", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_bias_is_refused(self, bad_bias: float) -> None:
        with pytest.raises(InvalidValuesError):
            Neuron([1.0, 2.0], bias=bad_bias, activation=Identity())

    def test_a_validated_column_passes_through_unchanged(self) -> None:
        """``Column.of`` is idempotent, so handing one in costs nothing."""
        weights = Column([0.25, 0.75], ValueRole.WEIGHT_VALUES)
        neuron = Neuron(weights, bias=0.0, activation=Identity())

        assert np.allclose(neuron.weights.values, [0.25, 0.75])

    def test_it_keeps_what_it_was_given(self) -> None:
        neuron = Neuron([1.5, -0.5], bias=0.25, activation=Sigmoid())

        assert np.allclose(neuron.weights.values, [1.5, -0.5])
        assert neuron.bias == 0.25
        assert neuron.activation == Sigmoid()


class TestEncapsulation:
    def test_the_weights_cannot_be_written_through(self) -> None:
        neuron = Neuron([1.0, 2.0], bias=0.0, activation=Identity())

        with pytest.raises(ValueError):
            neuron.weights.values[0] = 99.0

    def test_mutating_the_caller_s_list_does_not_reach_the_neuron(self) -> None:
        supplied = [1.0, 2.0]
        neuron = Neuron(supplied, bias=0.0, activation=Identity())

        supplied[0] = 99.0

        assert np.allclose(neuron.weights.values, [1.0, 2.0])


class TestResponding:
    def test_it_scores_then_bends(self) -> None:
        """Worked by hand: 2*3 + (-1)*4 + 0.5 = 2.5, and identity keeps it."""
        neuron = Neuron([2.0, -1.0], bias=0.5, activation=Identity())

        response = neuron.respond_to([3.0, 4.0])

        assert response.score == pytest.approx(2.5)
        assert response.output == pytest.approx(2.5)

    def test_the_bend_is_applied_to_the_score(self) -> None:
        """Same sum, rectified. The negative score has to come back as zero."""
        neuron = Neuron([1.0, 1.0], bias=-5.0, activation=RectifiedLinear())

        response = neuron.respond_to([1.0, 2.0])

        assert response.score == pytest.approx(-2.0)
        assert response.output == pytest.approx(0.0)

    @pytest.mark.parametrize("activation", EVERY_ACTIVATION)
    def test_the_score_ignores_the_activation(self, activation: Activation) -> None:
        neuron = Neuron([1.0, 1.0], bias=0.0, activation=activation)

        assert neuron.respond_to([0.5, 0.25]).score == pytest.approx(0.75)

    @pytest.mark.parametrize("activation", EVERY_ACTIVATION)
    def test_it_answers_python_floats(self, activation: Activation) -> None:
        response = Neuron([1.0], bias=0.0, activation=activation).respond_to([2.0])

        assert type(response.score) is float
        assert type(response.output) is float

    def test_a_row_of_the_wrong_width_is_refused(self) -> None:
        neuron = Neuron([1.0, 2.0, 3.0], bias=0.0, activation=Identity())

        with pytest.raises(NonEqualArrayLengthError):
            neuron.respond_to([1.0, 2.0])

    def test_a_column_row_costs_nothing_extra(self) -> None:
        neuron = Neuron([2.0, 2.0], bias=0.0, activation=Identity())
        row = Column([1.0, 3.0], ValueRole.INPUT_VALUES)

        assert neuron.respond_to(row).output == pytest.approx(8.0)


class TestItIsLogisticRegression:
    """A neuron with a sigmoid is the model this library already had."""

    def test_it_agrees_with_a_fitted_logistic_regression(self) -> None:
        from oop_ml import Feature, LogisticRegression

        hours = Feature("hours", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        passed = Feature("passed", [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        model = LogisticRegression(learning_rate=0.1).fit([hours], passed)

        neuron = Neuron(
            [float(next(iter(model.coefficients)).value)],
            bias=float(model.intercept),
            activation=Sigmoid(),
        )

        expected = float(
            np.asarray(
                model.predict_probability([Feature("hours", [5.0])]), dtype=np.float64
            )[0]
        )

        assert neuron.respond_to([5.0]).output == pytest.approx(expected, abs=1e-12)


class TestOneNeuronCannotSeparateExclusiveOr:
    """The ceiling from the module docstring, asserted rather than described."""

    @pytest.mark.parametrize("activation", EVERY_ACTIVATION)
    @pytest.mark.parametrize(
        ("first_weight", "second_weight", "bias"),
        [
            (1.0, 1.0, -0.5),
            (1.0, -1.0, 0.0),
            (-2.0, 3.0, 0.7),
            (5.0, 5.0, -7.5),
            (0.0, 0.0, 0.0),
        ],
    )
    def test_no_weighting_lifts_one_diagonal_above_the_other(
        self,
        activation: Activation,
        first_weight: float,
        second_weight: float,
        bias: float,
    ) -> None:
        neuron = Neuron([first_weight, second_weight], bias=bias, activation=activation)
        outputs = {
            corner: neuron.respond_to(list(corner)).output
            for corner in EXCLUSIVE_OR_CORNERS
        }

        exclusive_or_true = min(outputs[(0.0, 1.0)], outputs[(1.0, 0.0)])
        exclusive_or_false = max(outputs[(0.0, 0.0)], outputs[(1.0, 1.0)])

        assert not exclusive_or_true > exclusive_or_false

    @pytest.mark.parametrize(
        ("first_weight", "second_weight", "bias"),
        [(1.0, 1.0, -0.5), (2.0, -3.0, 0.7), (-1.5, 0.4, 2.0)],
    )
    def test_the_two_diagonals_carry_the_same_score_total(
        self, first_weight: float, second_weight: float, bias: float
    ) -> None:
        """The identity the impossibility rests on, checked directly."""
        neuron = Neuron([first_weight, second_weight], bias=bias, activation=Identity())
        scores = {
            corner: neuron.respond_to(list(corner)).score
            for corner in EXCLUSIVE_OR_CORNERS
        }

        assert scores[(0.0, 1.0)] + scores[(1.0, 0.0)] == pytest.approx(
            scores[(0.0, 0.0)] + scores[(1.0, 1.0)]
        )


class TestNeuronResponse:
    def test_it_pairs_the_two_numbers(self) -> None:
        response = NeuronResponse(score=2.5, output=0.924)

        assert response.score == 2.5
        assert response.output == 0.924

    def test_two_alike_are_equal(self) -> None:
        assert NeuronResponse(1.0, 0.5) == NeuronResponse(1.0, 0.5)
        assert hash(NeuronResponse(1.0, 0.5)) == hash(NeuronResponse(1.0, 0.5))

    def test_it_defers_to_anything_that_is_not_a_response(self) -> None:
        assert NeuronResponse(1.0, 0.5).__eq__((1.0, 0.5)) is NotImplemented
