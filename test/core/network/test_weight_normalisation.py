"""Spec for weight normalisation, which is two claims and a projection.

The two claims are opposite and the file is built around holding both at once.
The first is that this layer *is* a dense layer, value for value, because the
reparameterisation rebuilds a weight matrix and then does exactly what a dense
layer does with one. That is
:meth:`TestTheForwardPassIsADenseLayers.test_it_answers_what_a_dense_layer_holding_those_weights_answers`
and it is the most valuable test here, since if it fails the layer is not a
reparameterisation of anything. The second is that it is nevertheless a
different model to fit, because the same slope and the same learning rate land
its effective weights somewhere a dense layer's step would not. A change of
coordinates that moved the parameters to the same place would be a renaming, so
that one is asserted too.

The defining property sits between them. Only each direction row's *direction*
reaches the answer, its length having been divided out, so multiplying a row by
any positive constant must change nothing at all. A negative constant is
asserted to change everything, because a test that passed under both would be
asserting that the layer ignores its directions.

The projection term, and what the check catches
------------------------------------------------
The backward pass is a dense layer's followed by a change of coordinates, and
the second half is where the mistake lives. The direction's slope is the weight
slope with its radial component projected out and then rescaled::

    direction_slope = (magnitude / norm)
                      * (weight_slope - (weight_slope . unit) * unit)

Dropping the subtraction leaves ``(magnitude / norm) * weight_slope``. It
conforms, its magnitudes are plausible, and a network built on it descends. So
the oracle is the definition of a derivative rather than any rearrangement of
the formula -- nudge one stored number, watch the summary, compare.

That check was run against a deliberately unprojected body to confirm it
discriminates, and the numbers are worth writing down. On the wide fixture
below, five rows of four inputs into three neurons, whose largest true direction
slope is 2.3730, the projected form disagrees with a central finite difference
by 3.5e-10 and the unprojected one by 0.6123. On the narrow fixture, three rows
of two inputs into two neurons, the unprojected error is 0.2202 against a
largest true slope of 0.9015. A quarter of the answer, both times, which is what
makes it worth a spec -- an error that size is not a rounding and not a blow-up,
it is a wrong direction of travel that a falling loss curve will not show.

The second reading is structural and costs one dot product. Scaling a direction
by a positive constant cannot change what this layer answers, so the loss is
flat along the direction itself and the gradient can carry no component along
it -- ``direction_slope . direction`` is zero for every neuron, exactly.
Measured, the projected form gives 5.8e-16 there and the unprojected one 2.0155.
It is the cheapest way to watch the break fail and it is the same shape of
argument as batch normalisation's column sums.

The check has to run over the *directions* specifically. The magnitude and bias
slopes come back bit-identical under both bodies, because neither passes through
the projection, so a spec that checked only those would sail past an unprojected
backward pass. Run against the break, 5 of this file's tests fail -- the three
finite-difference checks over the directions and both orthogonality claims --
and the other 113 pass, the forward-pass and stepping tests among them.

The forward numbers are worked by hand where they can be. The hand fixture's
rows are ``(3, 4, 0)`` and ``(0, -6, 8)``, whose lengths are exactly 5 and 10, so
every unit direction is a fifth or a tenth and the effective weights come out
whole. The two rows are chosen to differ in which components they use, so a
transposed or a broadcast-across-neurons reading fails rather than agreeing by
symmetry.
"""

import math
from collections.abc import Callable

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
)
from oop_ml.core.network.activation import (
    Activation,
    HyperbolicTangent,
    Identity,
    RectifiedLinear,
    Sigmoid,
)
from oop_ml.core.network.gradient import LayerGradient
from oop_ml.core.network.layer import DenseLayer, LayerResponse
from oop_ml.core.network.loss import SquaredError
from oop_ml.core.network.neuron import Neuron
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.network.stack import LayerStack
from oop_ml.core.network.weight_normalisation import (
    ReparameterisedGradient,
    WeightNormalization,
)

#: The step a central difference takes on either side.
NUDGE = 1e-6

#: Two neurons over three inputs, whose lengths are exactly 5 and 10.
#:
#: The rows lean on different components, so a body that broadcast one neuron's
#: length across the whole block, or read the matrix transposed, produces
#: something else rather than the same answer by symmetry.
HAND_WORKED_DIRECTIONS = np.array([[3.0, 4.0, 0.0], [0.0, -6.0, 8.0]])

#: Lengths chosen so the effective weights come out whole: ``10 / 5`` doubles
#: the first row and ``5 / 10`` halves the second.
HAND_WORKED_MAGNITUDES = np.array([10.0, 5.0])

#: ``magnitude * direction / norm(direction)``, worked on paper.
HAND_WORKED_EFFECTIVE = np.array([[6.0, 8.0, 0.0], [0.0, -3.0, 4.0]])

#: A block with nothing convenient about it, for the dense-layer comparison.
AWKWARD_ROWS = np.array(
    [
        [1.5, -2.0, 0.125],
        [-0.5, 4.0, 0.75],
        [3.25, 0.5, -1.0],
        [0.0, -0.25, 2.5],
    ]
)


def hand_worked_layer(activation: Activation | None = None) -> WeightNormalization:
    """The paper fixture, with biases that are none of them zero."""
    return WeightNormalization(
        directions=HAND_WORKED_DIRECTIONS,
        magnitudes=HAND_WORKED_MAGNITUDES,
        biases=np.array([1.0, -2.0]),
        activation=activation,
    )


def dense_twin(layer: WeightNormalization) -> DenseLayer:
    """A dense layer holding exactly the weights the reparameterisation produced.

    Built from :attr:`WeightNormalization.effective_weights` rather than from the
    directions and magnitudes again, because rebuilding them here would make the
    comparison an assertion that one formula agrees with a copy of itself.
    """
    return DenseLayer(
        [
            Neuron(row, bias=float(bias), activation=layer.activation)
            for row, bias in zip(layer.effective_weights, layer.biases, strict=True)
        ]
    )


def effective_weights_by_hand(
    directions: np.ndarray, magnitudes: np.ndarray
) -> np.ndarray:
    """``magnitude * direction / norm(direction)``, one number at a time.

    Deliberately not :func:`numpy.linalg.norm` and not a broadcast division. The
    implementation reaches for both, and an oracle reaching for the same things
    asserts only that numpy is self-consistent.
    """
    n_neurons, n_inputs = directions.shape
    answer = np.empty_like(directions)
    for neuron in range(n_neurons):
        length = math.sqrt(
            sum(float(directions[neuron, column]) ** 2 for column in range(n_inputs))
        )
        for column in range(n_inputs):
            answer[neuron, column] = (
                float(magnitudes[neuron]) * float(directions[neuron, column]) / length
            )
    return answer


def gradient_check_layer() -> WeightNormalization:
    """Three neurons over four inputs, with a bend and nothing at a default.

    The magnitudes are none of them the directions' own lengths, so a body that
    quietly used the norm where the magnitude belongs answers differently. One
    is negative, which is legal -- a negative length simply reverses the
    neuron -- and it keeps the sign of ``magnitude / norm`` from being a
    constant everywhere.
    """
    generator = np.random.default_rng(41)
    return WeightNormalization(
        directions=generator.normal(size=(3, 4)),
        magnitudes=np.array([1.7, -0.6, 2.3]),
        biases=np.array([0.3, -0.8, 0.15]),
        activation=HyperbolicTangent(),
    )


def gradient_check_rows() -> np.ndarray:
    """Five rows of four inputs, each input on its own scale and centre."""
    generator = np.random.default_rng(42)
    return generator.normal(size=(5, 4)) * np.array([1.0, 2.5, 0.4, 1.5]) + np.array(
        [0.5, -1.0, 0.0, 2.0]
    )


def gradient_check_weights() -> np.ndarray:
    """The summary's weights, which are also the arriving block."""
    return np.random.default_rng(43).normal(size=(5, 3))


def narrow_layer() -> WeightNormalization:
    """Two neurons over two inputs, so ``1 / norm`` is a different number."""
    return WeightNormalization(
        directions=np.array([[0.6, -0.8], [2.0, 1.0]]),
        magnitudes=np.array([0.9, -1.4]),
        biases=np.array([0.25, 0.0]),
        activation=Sigmoid(),
    )


def narrow_rows() -> np.ndarray:
    """Three rows of two inputs, none of them near the others."""
    return np.array([[1.0, -0.5], [-2.0, 3.0], [0.25, 0.75]])


def narrow_weights() -> np.ndarray:
    return np.array([[1.0, -2.0], [0.5, 1.5], [-1.0, 0.25]])


def summary_of(
    layer: WeightNormalization, rows: np.ndarray, weights: np.ndarray
) -> float:
    """A scalar reading of a forward pass, whose slope is ``weights``.

    ``sum(weights * outputs)`` differentiates to ``weights`` at every output, so
    the same block is both the scalar's definition and the arriving block the
    backward pass is handed, and the two claims compare directly without a loss
    object in between.
    """
    return float(np.sum(weights * layer.respond_to(rows).outputs))


def layer_carrying(
    layer: WeightNormalization,
    directions: np.ndarray,
    magnitudes: np.ndarray,
    biases: np.ndarray,
) -> WeightNormalization:
    """The same layer with different parameters, for a nudged pass."""
    return WeightNormalization(
        directions=directions,
        magnitudes=magnitudes,
        biases=biases,
        activation=layer.activation,
    )


def gradient_of(
    layer: WeightNormalization, rows: np.ndarray, weights: np.ndarray
) -> ReparameterisedGradient:
    """The gradient one forward pass and its backward step produce."""
    correction = layer.correction_for(layer.respond_to(rows), weights)
    assert isinstance(correction.gradient, ReparameterisedGradient)
    return correction.gradient


#: One builder per per-neuron vector, so each refusal can be asked of both.
PER_NEURON_BUILDERS = [
    pytest.param(
        lambda values: WeightNormalization(
            directions=HAND_WORKED_DIRECTIONS, magnitudes=values
        ),
        id="magnitudes",
    ),
    pytest.param(
        lambda values: WeightNormalization(
            directions=HAND_WORKED_DIRECTIONS, biases=values
        ),
        id="biases",
    ),
]


class TestConstruction:
    def test_its_shape_is_the_inputs_it_reads_and_the_neurons_it_holds(self) -> None:
        layer = WeightNormalization(directions=np.ones((5, 8)))

        assert layer.shape == LayerShape(n_inputs=8, n_outputs=5)

    def test_it_reports_both_widths(self) -> None:
        layer = WeightNormalization(directions=np.ones((5, 8)))

        assert layer.n_neurons == 5
        assert layer.n_inputs == 8

    def test_the_effective_weights_are_the_numbers_computed_on_paper(self) -> None:
        """Lengths of exactly 5 and 10, so ``10 / 5`` doubles the first row and
        ``5 / 10`` halves the second."""
        layer = hand_worked_layer()

        assert np.allclose(layer.effective_weights, HAND_WORKED_EFFECTIVE)

    def test_the_effective_weights_agree_with_a_plain_python_oracle(self) -> None:
        generator = np.random.default_rng(44)
        directions = generator.normal(size=(4, 6))
        magnitudes = generator.normal(size=4)

        layer = WeightNormalization(directions=directions, magnitudes=magnitudes)

        assert np.allclose(
            layer.effective_weights,
            effective_weights_by_hand(directions, magnitudes),
        )

    def test_every_effective_row_has_the_length_it_was_given(self) -> None:
        """Which is the whole of what the magnitude parameter means."""
        layer = hand_worked_layer()

        lengths = np.linalg.norm(layer.effective_weights, axis=1)

        assert np.allclose(lengths, np.abs(HAND_WORKED_MAGNITUDES))

    def test_the_magnitudes_default_to_the_lengths_the_directions_already_had(
        self,
    ) -> None:
        """Which is what lets a caller hand in a weight matrix and get a layer
        whose effective weights are that matrix."""
        weights = np.array([[3.0, 4.0], [-5.0, 12.0]])

        layer = WeightNormalization(directions=weights)

        assert np.allclose(layer.magnitudes, [5.0, 13.0])
        assert np.allclose(layer.effective_weights, weights)

    def test_the_biases_default_to_zero(self) -> None:
        layer = WeightNormalization(directions=HAND_WORKED_DIRECTIONS)

        assert np.allclose(layer.biases, 0.0)

    def test_the_activation_defaults_to_the_identity(self) -> None:
        """So an output layer predicting a quantity is not a special case."""
        layer = WeightNormalization(directions=HAND_WORKED_DIRECTIONS)

        assert layer.activation == Identity()

    def test_supplied_parameters_are_kept(self) -> None:
        layer = hand_worked_layer(activation=RectifiedLinear())

        assert np.allclose(layer.directions, HAND_WORKED_DIRECTIONS)
        assert np.allclose(layer.magnitudes, HAND_WORKED_MAGNITUDES)
        assert np.allclose(layer.biases, [1.0, -2.0])
        assert layer.activation == RectifiedLinear()

    def test_the_directions_are_kept_at_the_length_they_arrived_with(self) -> None:
        """The layer divides the length out on the way to an answer and does not
        rewrite what it was handed, so a caller can still see what they passed."""
        layer = hand_worked_layer()

        assert np.allclose(np.linalg.norm(layer.directions, axis=1), [5.0, 10.0])

    @pytest.mark.parametrize(
        "supply",
        [
            lambda block: WeightNormalization(directions=block),
            lambda block: WeightNormalization(
                directions=HAND_WORKED_DIRECTIONS[:, :2], magnitudes=block[0]
            ),
            lambda block: WeightNormalization(
                directions=HAND_WORKED_DIRECTIONS[:, :2], biases=block[0]
            ),
        ],
        ids=["directions", "magnitudes", "biases"],
    )
    def test_a_supplied_block_is_copied_rather_than_aliased(
        self, supply: Callable[[np.ndarray], WeightNormalization]
    ) -> None:
        """A caller who kept the array must not be able to move the layer."""
        supplied = np.array([[3.0, 4.0], [1.0, 1.0]])
        layer = supply(supplied)
        before = np.array(layer.effective_weights, copy=True)

        supplied[0, 0] = 99.0

        assert np.allclose(layer.effective_weights, before)

    @pytest.mark.parametrize(
        "read",
        [
            lambda layer: layer.directions,
            lambda layer: layer.magnitudes,
            lambda layer: layer.biases,
            lambda layer: layer.effective_weights,
        ],
        ids=["directions", "magnitudes", "biases", "effective_weights"],
    )
    def test_every_block_it_hands_out_is_frozen(
        self, read: Callable[[WeightNormalization], np.ndarray]
    ) -> None:
        layer = hand_worked_layer()

        with pytest.raises(ValueError):
            read(layer)[0] = 1.0

    def test_it_repeats_its_two_widths(self) -> None:
        assert repr(WeightNormalization(directions=np.ones((2, 3)))) == (
            "WeightNormalization(n_inputs=3, n_neurons=2)"
        )


class TestRefusedConfigurations:
    @pytest.mark.parametrize("shape", [(3,), (2, 2, 2)])
    def test_a_direction_block_that_is_not_a_matrix_is_refused(
        self, shape: tuple[int, ...]
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            WeightNormalization(directions=np.ones(shape))

    @pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
    def test_a_direction_block_carrying_a_non_finite_entry_is_refused(
        self, poison: float
    ) -> None:
        directions = np.ones((2, 3))
        directions[1, 2] = poison

        with pytest.raises(InvalidValuesError):
            WeightNormalization(directions=directions)

    def test_a_direction_block_that_is_not_numbers_at_all_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            WeightNormalization(directions="due north")  # type: ignore[arg-type]

    def test_a_direction_of_zero_is_refused(self) -> None:
        """A zero vector does not name a direction badly, it names none at all,
        and both slopes divide by the length it does not have."""
        with pytest.raises(InvalidValuesError):
            WeightNormalization(directions=np.array([[3.0, 4.0], [0.0, 0.0]]))

    def test_a_block_of_nothing_but_zeros_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            WeightNormalization(directions=np.zeros((3, 4)))

    def test_a_direction_that_is_merely_tiny_is_allowed(self) -> None:
        """The refusal is of the undefined case and not of a small number. This
        one has a length of 1e-150 and normalises perfectly well."""
        layer = WeightNormalization(
            directions=np.array([[1e-150, 0.0], [0.0, 1.0]]),
            magnitudes=np.array([2.0, 3.0]),
        )

        assert np.allclose(layer.effective_weights, [[2.0, 0.0], [0.0, 3.0]])

    @pytest.mark.parametrize("n_neurons", [0])
    def test_a_layer_with_no_neurons_is_refused(self, n_neurons: int) -> None:
        with pytest.raises(InvalidValuesError):
            WeightNormalization(directions=np.ones((n_neurons, 3)))

    def test_a_layer_with_no_inputs_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            WeightNormalization(directions=np.ones((3, 0)))

    @pytest.mark.parametrize("build", PER_NEURON_BUILDERS)
    @pytest.mark.parametrize("length", [1, 3, 0])
    def test_a_vector_of_the_wrong_length_is_refused(
        self, build: Callable[[np.ndarray], WeightNormalization], length: int
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            build(np.ones(length))

    @pytest.mark.parametrize("build", PER_NEURON_BUILDERS)
    def test_a_vector_that_is_not_one_dimensional_is_refused(
        self, build: Callable[[np.ndarray], WeightNormalization]
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            build(np.ones((2, 1)))

    @pytest.mark.parametrize("build", PER_NEURON_BUILDERS)
    @pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
    def test_a_vector_carrying_a_non_finite_entry_is_refused(
        self, build: Callable[[np.ndarray], WeightNormalization], poison: float
    ) -> None:
        values = np.ones(2)
        values[1] = poison

        with pytest.raises(InvalidValuesError):
            build(values)

    @pytest.mark.parametrize("build", PER_NEURON_BUILDERS)
    def test_a_vector_that_is_not_numbers_at_all_is_refused(
        self, build: Callable[[np.ndarray], WeightNormalization]
    ) -> None:
        with pytest.raises(InvalidValuesError):
            build("per neuron")  # type: ignore[arg-type]

    def test_a_magnitude_of_zero_is_allowed(self) -> None:
        """A neuron that answers nothing but its bias is an ordinary state for
        one to be in, and the gradient can still move it out again, because the
        magnitude's slope does not divide by the magnitude."""
        layer = WeightNormalization(
            directions=HAND_WORKED_DIRECTIONS, magnitudes=np.zeros(2)
        )

        assert np.allclose(layer.effective_weights, 0.0)


class TestRefusedBlocks:
    def test_a_block_of_the_wrong_width_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            hand_worked_layer().respond_to(np.zeros((4, 5)))

    def test_a_block_of_no_rows_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            hand_worked_layer().respond_to(np.zeros((0, 3)))

    def test_a_non_finite_entry_is_refused_where_it_enters(self) -> None:
        poisoned = np.array(AWKWARD_ROWS, copy=True)
        poisoned[2, 1] = np.nan

        with pytest.raises(InvalidValuesError):
            hand_worked_layer().respond_to(poisoned)

    def test_a_block_that_is_not_numbers_at_all_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            hand_worked_layer().respond_to("a batch")  # type: ignore[arg-type]


class TestTheForwardPassIsADenseLayers:
    """The claim the whole layer rests on, and the reason it is first.

    A reparameterisation changes where the parameters live and nothing about the
    function. If this layer answered anything a dense layer holding its effective
    weights would not, it would not be a reparameterisation of a dense layer at
    all, and every other test here would be describing something else.
    """

    @pytest.mark.parametrize(
        "activation",
        [Identity(), HyperbolicTangent(), Sigmoid(), RectifiedLinear()],
        ids=["identity", "tanh", "sigmoid", "relu"],
    )
    def test_it_answers_what_a_dense_layer_holding_those_weights_answers(
        self, activation: Activation
    ) -> None:
        layer = hand_worked_layer(activation=activation)

        answer = layer.respond_to(AWKWARD_ROWS)
        twin = dense_twin(layer).respond_to(AWKWARD_ROWS)

        assert np.allclose(answer.outputs, twin.outputs)
        assert np.allclose(answer.scores, twin.scores)

    def test_it_agrees_with_a_dense_twin_on_an_awkward_layer_too(self) -> None:
        layer = gradient_check_layer()

        answer = layer.respond_to(gradient_check_rows())
        twin = dense_twin(layer).respond_to(gradient_check_rows())

        assert np.allclose(answer.outputs, twin.outputs)

    def test_the_outputs_are_the_numbers_computed_on_paper(self) -> None:
        """Effective weights ``(6, 8, 0)`` and ``(0, -3, 4)``, biases 1 and -2,
        against the row ``(1, 1, 1)`` and the row ``(0, 2, 0)``."""
        answer = hand_worked_layer().respond_to(
            np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 0.0]])
        )

        assert np.allclose(answer.outputs, [[15.0, -1.0], [17.0, -8.0]])

    def test_the_scores_are_before_the_bend_and_the_outputs_after(self) -> None:
        layer = hand_worked_layer(activation=RectifiedLinear())

        answer = layer.respond_to(np.array([[0.0, 0.0, 0.0]]))

        assert np.allclose(answer.scores, [[1.0, -2.0]])
        assert np.allclose(answer.outputs, [[1.0, 0.0]])

    def test_the_response_carries_the_block_that_was_read(self) -> None:
        response = hand_worked_layer().respond_to(AWKWARD_ROWS)

        assert np.allclose(response.inputs, AWKWARD_ROWS)

    def test_it_answers_with_a_plain_layer_response(self) -> None:
        """There is nothing extra to carry. Nothing was measured from the batch
        and nothing was drawn, so the response needs no subclass."""
        response = hand_worked_layer().respond_to(AWKWARD_ROWS)

        assert type(response) is LayerResponse

    @pytest.mark.parametrize(
        "purpose",
        [PassPurpose.TRAINING, PassPurpose.PREDICTING],
        ids=["train", "predict"],
    )
    def test_the_purpose_makes_no_difference(self, purpose: PassPurpose) -> None:
        """Nothing here reads the batch and nothing is random, so there is no
        second variant of this pass for the purpose to choose between."""
        layer = hand_worked_layer(activation=HyperbolicTangent())

        assert np.allclose(
            layer.respond_to(AWKWARD_ROWS, purpose).outputs,
            layer.respond_to(AWKWARD_ROWS).outputs,
        )


class TestOnlyTheDirectionMatters:
    """The defining property, since the length is divided straight back out.

    Multiplying a stored direction by a positive constant is the one change to
    this layer's parameters that must make no difference whatsoever. That is what
    separates the reparameterisation from an ordinary weight matrix, and it is
    also the reason the direction's gradient has no radial component.
    """

    @pytest.mark.parametrize("factor", [1e-8, 0.25, 3.0, 1e8])
    def test_scaling_a_direction_leaves_the_effective_weights_alone(
        self, factor: float
    ) -> None:
        stretched = np.array(HAND_WORKED_DIRECTIONS, copy=True)
        stretched[0] *= factor

        layer = WeightNormalization(
            directions=stretched, magnitudes=HAND_WORKED_MAGNITUDES
        )

        assert np.allclose(layer.effective_weights, HAND_WORKED_EFFECTIVE)

    @pytest.mark.parametrize("factor", [1e-8, 0.25, 3.0, 1e8])
    def test_scaling_every_direction_leaves_the_answers_alone(
        self, factor: float
    ) -> None:
        layer = hand_worked_layer(activation=HyperbolicTangent())
        stretched = layer_carrying(
            layer, HAND_WORKED_DIRECTIONS * factor, layer.magnitudes, layer.biases
        )

        assert np.allclose(
            stretched.respond_to(AWKWARD_ROWS).outputs,
            layer.respond_to(AWKWARD_ROWS).outputs,
        )

    def test_a_negative_factor_does_change_the_answers(self) -> None:
        """Otherwise the test above would be asserting that the layer ignores
        its directions altogether. A negative constant reverses the direction,
        which is a different neuron and must read as one."""
        layer = hand_worked_layer()
        reversed_first = np.array(HAND_WORKED_DIRECTIONS, copy=True)
        reversed_first[0] *= -1.0

        flipped = layer_carrying(layer, reversed_first, layer.magnitudes, layer.biases)

        assert not np.allclose(
            flipped.respond_to(AWKWARD_ROWS).outputs,
            layer.respond_to(AWKWARD_ROWS).outputs,
        )

    def test_changing_a_magnitude_does_change_the_answers(self) -> None:
        """The other half of the same argument. The length is a real parameter,
        it is simply not the length of the block that was handed in."""
        layer = hand_worked_layer()
        louder = layer_carrying(
            layer, layer.directions, layer.magnitudes * 2.0, layer.biases
        )

        assert not np.allclose(
            louder.respond_to(AWKWARD_ROWS).outputs,
            layer.respond_to(AWKWARD_ROWS).outputs,
        )

    def test_one_neuron_s_direction_does_not_reach_another(self) -> None:
        """A body that took one length over the whole block rather than one per
        row would fail this, and every shape would still conform."""
        layer = hand_worked_layer()
        moved = np.array(HAND_WORKED_DIRECTIONS, copy=True)
        moved[0] = [50.0, -1.0, 7.0]

        changed = layer_carrying(layer, moved, layer.magnitudes, layer.biases)

        assert np.allclose(
            changed.respond_to(AWKWARD_ROWS).outputs[:, 1],
            layer.respond_to(AWKWARD_ROWS).outputs[:, 1],
        )


class TestTheGradientCheck:
    """Nudge one stored number, watch the summary, compare against the claim.

    Run over the directions, the magnitudes, the biases and the inputs in turn.
    The directions are the one that matters, because an unprojected backward
    pass leaves the other three bit-identical and would sail through a spec that
    checked only those. See the module docstring for the measured disagreement.
    """

    def measured_direction_gradient(
        self, layer: WeightNormalization, rows: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        measured = np.zeros_like(layer.directions)
        for neuron in range(layer.n_neurons):
            for column in range(layer.n_inputs):
                moved = []
                for direction in (+NUDGE, -NUDGE):
                    nudged = np.array(layer.directions, copy=True)
                    nudged[neuron, column] += direction
                    moved.append(
                        summary_of(
                            layer_carrying(
                                layer, nudged, layer.magnitudes, layer.biases
                            ),
                            rows,
                            weights,
                        )
                    )
                measured[neuron, column] = (moved[0] - moved[1]) / (2.0 * NUDGE)
        return measured

    def measured_magnitude_gradient(
        self, layer: WeightNormalization, rows: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        measured = np.zeros_like(layer.magnitudes)
        for neuron in range(layer.n_neurons):
            moved = []
            for direction in (+NUDGE, -NUDGE):
                nudged = np.array(layer.magnitudes, copy=True)
                nudged[neuron] += direction
                moved.append(
                    summary_of(
                        layer_carrying(layer, layer.directions, nudged, layer.biases),
                        rows,
                        weights,
                    )
                )
            measured[neuron] = (moved[0] - moved[1]) / (2.0 * NUDGE)
        return measured

    def measured_bias_gradient(
        self, layer: WeightNormalization, rows: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        measured = np.zeros_like(layer.biases)
        for neuron in range(layer.n_neurons):
            moved = []
            for direction in (+NUDGE, -NUDGE):
                nudged = np.array(layer.biases, copy=True)
                nudged[neuron] += direction
                moved.append(
                    summary_of(
                        layer_carrying(
                            layer, layer.directions, layer.magnitudes, nudged
                        ),
                        rows,
                        weights,
                    )
                )
            measured[neuron] = (moved[0] - moved[1]) / (2.0 * NUDGE)
        return measured

    def measured_input_gradient(
        self, layer: WeightNormalization, rows: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        measured = np.zeros_like(rows)
        for row in range(rows.shape[0]):
            for column in range(rows.shape[1]):
                moved = []
                for direction in (+NUDGE, -NUDGE):
                    nudged = np.array(rows, copy=True)
                    nudged[row, column] += direction
                    moved.append(summary_of(layer, nudged, weights))
                measured[row, column] = (moved[0] - moved[1]) / (2.0 * NUDGE)
        return measured

    def test_every_direction_slope_matches_a_finite_difference(self) -> None:
        """The projection, checked. This is the test the module is about."""
        layer = gradient_check_layer()
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        gradient = gradient_of(layer, rows, weights)
        measured = self.measured_direction_gradient(layer, rows, weights)

        assert np.allclose(gradient.directions, measured, atol=1e-7)

    def test_every_magnitude_slope_matches_a_finite_difference(self) -> None:
        layer = gradient_check_layer()
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        gradient = gradient_of(layer, rows, weights)
        measured = self.measured_magnitude_gradient(layer, rows, weights)

        assert np.allclose(gradient.magnitudes, measured, atol=1e-7)

    def test_every_bias_slope_matches_a_finite_difference(self) -> None:
        layer = gradient_check_layer()
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        gradient = gradient_of(layer, rows, weights)
        measured = self.measured_bias_gradient(layer, rows, weights)

        assert np.allclose(gradient.biases, measured, atol=1e-7)

    def test_every_input_slope_matches_a_finite_difference(self) -> None:
        """What the layer beneath is told, which the reparameterisation leaves
        exactly as a dense layer's."""
        layer = gradient_check_layer()
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        correction = layer.correction_for(layer.respond_to(rows), weights)
        measured = self.measured_input_gradient(layer, rows, weights)

        assert np.allclose(correction.passed_down, measured, atol=1e-7)

    def test_the_direction_slopes_hold_on_the_narrow_fixture(self) -> None:
        """Two inputs rather than four and a different bend, so ``1 / norm`` and
        the activation slope are both different numbers."""
        layer = narrow_layer()
        rows = narrow_rows()
        weights = narrow_weights()

        gradient = gradient_of(layer, rows, weights)
        measured = self.measured_direction_gradient(layer, rows, weights)

        assert np.allclose(gradient.directions, measured, atol=1e-7)

    def test_the_direction_slopes_hold_when_the_directions_are_long(self) -> None:
        """The length is divided out of the answer and *not* out of the
        gradient, which carries ``magnitude / norm``. Stretching the stored
        block by a thousand must therefore shrink the direction slopes by a
        thousand while changing nothing the layer answers."""
        layer = gradient_check_layer()
        stretched = layer_carrying(
            layer, layer.directions * 1000.0, layer.magnitudes, layer.biases
        )
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        gradient = gradient_of(stretched, rows, weights)
        measured = self.measured_direction_gradient(stretched, rows, weights)

        assert np.allclose(gradient.directions, measured, atol=1e-9)

    @pytest.mark.parametrize(
        ("build", "rows", "weights"),
        [
            pytest.param(
                gradient_check_layer,
                gradient_check_rows,
                gradient_check_weights,
                id="wide",
            ),
            pytest.param(narrow_layer, narrow_rows, narrow_weights, id="narrow"),
        ],
    )
    def test_the_direction_slope_is_orthogonal_to_the_direction(
        self,
        build: Callable[[], WeightNormalization],
        rows: Callable[[], np.ndarray],
        weights: Callable[[], np.ndarray],
    ) -> None:
        """The structural signature of the projection, and the cheapest way to
        watch an unprojected body fail. Scaling a direction cannot change what
        this layer answers, so the loss is flat along the direction and the
        gradient can carry no component along it. Measured, the unprojected
        reading gives 2.0155 on the wide fixture and 0.5506 on the narrow one.
        """
        layer = build()

        gradient = gradient_of(layer, rows(), weights())

        along = (gradient.directions * layer.directions).sum(axis=1)
        assert np.allclose(along, 0.0, atol=1e-12)


class TestTheGradientObject:
    def test_the_gradient_is_a_reparameterised_one(self) -> None:
        """A plain ``LayerGradient`` would leave the lengths unable to move,
        which is the parameter this whole reparameterisation exists to expose."""
        layer = gradient_check_layer()

        correction = layer.correction_for(
            layer.respond_to(gradient_check_rows()), gradient_check_weights()
        )

        assert isinstance(correction.gradient, ReparameterisedGradient)

    def test_it_is_a_layer_gradient(self) -> None:
        """So a stack that knows nothing about this layer can still thread it."""
        gradient = gradient_of(
            gradient_check_layer(), gradient_check_rows(), gradient_check_weights()
        )

        assert isinstance(gradient, LayerGradient)

    def test_the_directions_are_the_weight_block_under_its_own_name(self) -> None:
        """``LayerGradient`` calls that block ``weights`` because that is what it
        means everywhere else. Here it is the slope for the directions, which is
        not the slope for the effective weight matrix, and the second name is
        what keeps a reader from confusing the two."""
        gradient = gradient_of(
            gradient_check_layer(), gradient_check_rows(), gradient_check_weights()
        )

        assert gradient.directions is gradient.weights

    def test_the_three_blocks_have_the_shapes_of_what_they_describe(self) -> None:
        gradient = gradient_of(
            gradient_check_layer(), gradient_check_rows(), gradient_check_weights()
        )

        assert gradient.directions.shape == (3, 4)
        assert gradient.magnitudes.shape == (3,)
        assert gradient.biases.shape == (3,)

    def test_the_blame_passed_down_has_the_shape_of_what_was_read(self) -> None:
        layer = gradient_check_layer()
        rows = gradient_check_rows()

        correction = layer.correction_for(
            layer.respond_to(rows), gradient_check_weights()
        )

        assert correction.passed_down.shape == rows.shape

    def test_the_largest_movement_reads_the_magnitudes_too(self) -> None:
        """The base reads two blocks and this object carries three. A
        convergence check blind to the lengths would call a network settled
        while every neuron's sharpness was still moving."""
        gradient = ReparameterisedGradient(
            weights=np.zeros((2, 3)),
            biases=np.zeros(2),
            magnitudes=np.array([0.0, -7.0]),
        )

        assert gradient.largest_movement == pytest.approx(7.0)

    def test_the_largest_movement_still_reads_the_other_two(self) -> None:
        gradient = ReparameterisedGradient(
            weights=np.array([[0.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
            biases=np.array([0.0, -1.0]),
            magnitudes=np.zeros(2),
        )

        assert gradient.largest_movement == pytest.approx(4.0)

    def test_the_magnitudes_are_frozen(self) -> None:
        gradient = ReparameterisedGradient(
            weights=np.zeros((2, 3)), biases=np.zeros(2), magnitudes=np.zeros(2)
        )

        with pytest.raises(ValueError):
            gradient.magnitudes[0] = 1.0

    @pytest.mark.parametrize("length", [1, 3])
    def test_a_magnitude_block_of_the_wrong_length_is_refused(
        self, length: int
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            ReparameterisedGradient(
                weights=np.zeros((2, 3)),
                biases=np.zeros(2),
                magnitudes=np.zeros(length),
            )

    def test_a_magnitude_block_that_is_not_one_dimensional_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            ReparameterisedGradient(
                weights=np.zeros((2, 3)),
                biases=np.zeros(2),
                magnitudes=np.zeros((2, 1)),
            )

    @pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_magnitude_block_is_refused(self, poison: float) -> None:
        with pytest.raises(InvalidValuesError):
            ReparameterisedGradient(
                weights=np.zeros((2, 3)),
                biases=np.zeros(2),
                magnitudes=np.array([0.0, poison]),
            )

    def test_it_repeats_the_arrangement_it_describes(self) -> None:
        gradient = ReparameterisedGradient(
            weights=np.zeros((2, 3)), biases=np.zeros(2), magnitudes=np.zeros(2)
        )

        assert repr(gradient) == "ReparameterisedGradient(shape=(2, 3))"


class TestRefusedBackwardSteps:
    def test_an_arriving_block_of_the_wrong_width_is_refused(self) -> None:
        layer = gradient_check_layer()
        response = layer.respond_to(gradient_check_rows())

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((5, 4)))

    def test_an_arriving_block_of_the_wrong_row_count_is_refused(self) -> None:
        layer = gradient_check_layer()
        response = layer.respond_to(gradient_check_rows())

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((4, 3)))

    def test_a_response_from_a_differently_shaped_layer_is_refused(self) -> None:
        """It carries real numbers of a plausible shape, so nothing else would
        catch it and the blame would be routed to arbitrary places."""
        layer = gradient_check_layer()
        narrower = WeightNormalization(directions=np.ones((3, 2)))
        response = narrower.respond_to(np.ones((5, 2)))

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((5, 3)))

    def test_an_arriving_block_that_is_not_numbers_is_refused(self) -> None:
        layer = gradient_check_layer()
        response = layer.respond_to(gradient_check_rows())

        with pytest.raises(InvalidValuesError):
            layer.correction_for(response, "blame")  # type: ignore[arg-type]


class TestStepping:
    def stepped(
        self, layer: WeightNormalization, learning_rate: float = 0.05
    ) -> WeightNormalization:
        gradient = gradient_of(layer, gradient_check_rows(), gradient_check_weights())
        stepped = layer.stepped_by(gradient, learning_rate)
        assert isinstance(stepped, WeightNormalization)
        return stepped

    def test_all_three_parameter_sets_move_against_their_slopes(self) -> None:
        layer = gradient_check_layer()
        gradient = gradient_of(layer, gradient_check_rows(), gradient_check_weights())

        stepped = layer.stepped_by(gradient, learning_rate=0.05)
        assert isinstance(stepped, WeightNormalization)

        assert np.allclose(
            stepped.directions, layer.directions - 0.05 * gradient.directions
        )
        assert np.allclose(
            stepped.magnitudes, layer.magnitudes - 0.05 * gradient.magnitudes
        )
        assert np.allclose(stepped.biases, layer.biases - 0.05 * gradient.biases)

    def test_the_step_keeps_the_bend_it_was_not_asked_to_change(self) -> None:
        layer = gradient_check_layer()

        assert self.stepped(layer).activation == HyperbolicTangent()

    def test_the_stepped_layer_has_the_same_shape(self) -> None:
        layer = gradient_check_layer()

        assert self.stepped(layer).shape == layer.shape

    def test_the_stepped_layer_is_a_new_object(self) -> None:
        layer = gradient_check_layer()

        assert self.stepped(layer) is not layer

    def test_the_original_layer_is_untouched(self) -> None:
        """Training does not mutate, which is what keeps every block frozen."""
        layer = gradient_check_layer()
        directions_before = np.array(layer.directions, copy=True)
        magnitudes_before = np.array(layer.magnitudes, copy=True)
        biases_before = np.array(layer.biases, copy=True)

        self.stepped(layer, learning_rate=0.5)

        assert np.array_equal(layer.directions, directions_before)
        assert np.array_equal(layer.magnitudes, magnitudes_before)
        assert np.array_equal(layer.biases, biases_before)

    def test_a_step_lowers_the_summary_it_was_asked_to_lower(self) -> None:
        """The point of the whole exercise, on one small step."""
        layer = gradient_check_layer()
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        stepped = self.stepped(layer, learning_rate=0.01)

        assert summary_of(stepped, rows, weights) < summary_of(layer, rows, weights)

    def test_a_step_that_lands_a_direction_on_zero_is_refused(self) -> None:
        """Rather than building a layer that answers ``nan`` to everything from
        then on. The refusal is the constructor's, inherited by rebuilding
        through it rather than restated here."""
        layer = hand_worked_layer()
        gradient = ReparameterisedGradient(
            weights=np.array(layer.directions, copy=True),
            biases=np.zeros(2),
            magnitudes=np.zeros(2),
        )

        with pytest.raises(InvalidValuesError):
            layer.stepped_by(gradient, learning_rate=1.0)

    def test_stepping_by_nothing_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            hand_worked_layer().stepped_by(None, learning_rate=0.1)

    def test_stepping_by_a_plain_layer_gradient_is_refused(self) -> None:
        """It carries no lengths, so a layer stepped by one would be learning
        half of what it was told and nothing would say so."""
        with pytest.raises(ShapeMismatchError):
            hand_worked_layer().stepped_by(
                LayerGradient(weights=np.zeros((2, 3)), biases=np.zeros(2)),
                learning_rate=0.1,
            )

    @pytest.mark.parametrize("shape", [(1, 3), (3, 3), (2, 4)])
    def test_a_gradient_of_the_wrong_arrangement_is_refused(
        self, shape: tuple[int, int]
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            hand_worked_layer().stepped_by(
                ReparameterisedGradient(
                    weights=np.zeros(shape),
                    biases=np.zeros(shape[0]),
                    magnitudes=np.zeros(shape[0]),
                ),
                learning_rate=0.1,
            )


class TestTheSurfaceIsNotTheDenseOne:
    """The second half of the claim, and the reason the layer is worth having.

    The function is a dense layer's and the *fit* is not. A dense layer's step
    subtracts the learning rate times the slope of the loss at each effective
    weight; this one subtracts it in direction-and-length coordinates and then
    rebuilds the weights, and the two land in different places. If they landed
    in the same place the reparameterisation would be a renaming.
    """

    def test_a_step_lands_the_effective_weights_somewhere_a_dense_step_would_not(
        self,
    ) -> None:
        layer = gradient_check_layer()
        rows = gradient_check_rows()
        weights = gradient_check_weights()
        twin = dense_twin(layer)

        stepped = self.stepped_layer(layer, rows, weights)
        dense_gradient = twin.correction_for(twin.respond_to(rows), weights).gradient
        assert dense_gradient is not None
        stepped_twin = twin.stepped_by(dense_gradient, learning_rate=0.05)

        assert not np.allclose(stepped.effective_weights, stepped_twin.weight_matrix)

    def test_a_direction_step_alone_cannot_change_a_neuron_s_length(self) -> None:
        """The separation the reparameterisation buys, stated as arithmetic. The
        stored direction moves and its length changes with it, and none of that
        reaches the answer, because the length is divided out again."""
        layer = gradient_check_layer()
        gradient = gradient_of(layer, gradient_check_rows(), gradient_check_weights())
        directions_only = ReparameterisedGradient(
            weights=gradient.directions,
            biases=np.zeros(layer.n_neurons),
            magnitudes=np.zeros(layer.n_neurons),
        )

        stepped = layer.stepped_by(directions_only, learning_rate=0.05)
        assert isinstance(stepped, WeightNormalization)

        assert np.allclose(
            np.linalg.norm(stepped.effective_weights, axis=1),
            np.abs(layer.magnitudes),
        )

    def test_a_dense_step_of_the_same_slope_does_change_every_length(self) -> None:
        """The comparison the test above is making, run the other way, so that
        it is asserting a difference rather than an arithmetic identity."""
        layer = gradient_check_layer()
        rows = gradient_check_rows()
        weights = gradient_check_weights()
        twin = dense_twin(layer)

        dense_gradient = twin.correction_for(twin.respond_to(rows), weights).gradient
        assert dense_gradient is not None
        stepped_twin = twin.stepped_by(dense_gradient, learning_rate=0.05)

        assert not np.allclose(
            np.linalg.norm(stepped_twin.weight_matrix, axis=1),
            np.abs(layer.magnitudes),
        )

    def stepped_layer(
        self, layer: WeightNormalization, rows: np.ndarray, weights: np.ndarray
    ) -> WeightNormalization:
        gradient = gradient_of(layer, rows, weights)
        stepped = layer.stepped_by(gradient, learning_rate=0.05)
        assert isinstance(stepped, WeightNormalization)
        return stepped


class TestInAStack:
    def stack(self, seed: int = 7) -> LayerStack:
        """Four inputs, a bent weight-normalised middle of three, two outputs."""
        generator = np.random.default_rng(seed)
        hidden = WeightNormalization(
            directions=generator.normal(size=(3, 4)),
            magnitudes=np.array([0.8, 1.2, -0.5]),
            biases=generator.normal(size=3) * 0.5,
            activation=HyperbolicTangent(),
        )
        output = WeightNormalization(
            directions=generator.normal(size=(2, 3)),
            biases=np.zeros(2),
            activation=Identity(),
        )
        return LayerStack([hidden, output])

    def rows(self, seed: int = 8) -> np.ndarray:
        return np.random.default_rng(seed).normal(size=(8, 4))

    def targets(self, seed: int = 9) -> np.ndarray:
        return np.random.default_rng(seed).normal(size=(8, 2))

    def test_the_stack_reports_the_joins_it_was_built_from(self) -> None:
        assert self.stack().shape == LayerShape(n_inputs=4, n_outputs=2)

    def test_it_sits_between_dense_layers_without_moving_a_join(self) -> None:
        dense = DenseLayer(
            [
                Neuron(np.ones(3) * 0.1, bias=0.0, activation=Identity())
                for _ in range(2)
            ]
        )

        stack = LayerStack(
            [WeightNormalization(directions=np.ones((3, 4)) + np.eye(3, 4)), dense]
        )

        assert stack.shape == LayerShape(n_inputs=4, n_outputs=2)

    def test_a_backward_pass_produces_its_gradient_in_layer_order(self) -> None:
        stack = self.stack()

        backward = stack.backward_pass(self.rows(), self.targets(), SquaredError())

        assert len(backward) == 2
        assert isinstance(backward[0], ReparameterisedGradient)
        assert isinstance(backward[1], ReparameterisedGradient)

    def test_repeated_steps_lower_the_loss(self) -> None:
        stack = self.stack()
        rows = self.rows()
        targets = self.targets()
        loss = SquaredError()

        first = stack.backward_pass(rows, targets, loss).loss
        for _ in range(30):
            stack = stack.stepped_by(
                stack.backward_pass(rows, targets, loss), learning_rate=0.2
            )
        last = stack.backward_pass(rows, targets, loss).loss

        assert last < first * 0.9
