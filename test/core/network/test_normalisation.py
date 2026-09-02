"""Spec for batch normalisation, and one test in it carries the whole module.

The one that matters is the finite-difference check over ``passed_down``, for
the reason the module's own docstring names. The tempting backward pass reads
the normalised block as ``(value - mean) / deviation`` with the mean and the
deviation held constant, which gives ``arriving * scale / deviation``. That has
the right shape, plausible magnitudes, and it trains. It is not the gradient of
the loss.

The mean and the deviation are functions of every row in the batch, so each
input reaches the loss by three routes rather than one, and the two missing
terms are the mean route and the variance route. Nothing about the shape of the
answer says which version produced it, so the oracle has to be the definition of
a derivative. Nudge one input, watch the summary, compare against the claim.

That check was run against a deliberately naive body to confirm it
discriminates, and the numbers are worth writing down. On the five-by-three
fixture below, whose largest true slope is 1.32, the three-term form disagrees
with the finite difference by 1.3e-09 and the naive one by 1.85, which is an
error larger than the largest slope it was trying to report. On the
nine-by-four batch the naive error is 0.855, and on the block with a constant
feature it is 2.94. The column sums are the second reading. Shifting a whole
feature by a constant cannot change what this layer answers, so every column of
``passed_down`` sums to zero exactly; measured, the correct form gives 2.2e-16
and the naive one 0.601.

The check runs over ``passed_down`` specifically because the naive reading
leaves the other two gradients bit-identical. ``d scale`` and ``d shift`` are
sums down the rows and owe nothing to the three routes, so a spec that checked
only those would pass a body with the wrong backward pass inside it.

The forward numbers are worked by hand rather than generated, and where they are
not, the oracle is a plain Python loop over the definition rather than a second
call to numpy by the route the implementation would take. The hand fixture's
epsilon is 7.0 so that ``sqrt(9 + 7)`` is exactly 4, which is larger than any
real layer would use and is the point. Every normalised value is then 0.75 one
way or the other and the whole block is exact in binary. The two features share
a variance and differ in how their rows pair up, so a transposed answer fails
rather than agreeing by symmetry.

The rest divides into the two halves the module warns about. Training reads the
batch and prediction reads the running figures, so a row's answer must depend on
its company in the first case and must not in the second, and both directions
are asserted here. The running figures then move in
:meth:`~oop_ml.core.network.normalisation.BatchNormalization.stepped_by` rather
than in the forward pass, which is the one place in this package where a step
does something besides subtract a slope.

One shape is pinned on its own because it survived a first reading. The
gradient's weight block is ``(n_features, 1)``, one row per feature holding that
feature's single weight, and the transposed ``(1, n_features)`` is the tempting
misreading. At a width of one the two agree, so the test that pins it uses a
width of three.
"""

import math
from collections.abc import Callable, Sequence

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
from oop_ml.core.network.normalisation import (
    BatchNormalization,
    BatchStatistics,
    NormalisationResponse,
)
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.network.stack import LayerStack

#: The step a central difference takes on either side.
NUDGE = 1e-6

#: Four rows and two features, chosen so every step of the arithmetic is exact.
#:
#: The first feature is ``-3, -3, 3, 3``, so its mean is 0 and its biased
#: variance is 9. The second is ``7, 13, 7, 13``, so its mean is 10 and its
#: variance is also 9, but its rows alternate where the first pairs up. A block
#: filled in the wrong order therefore fails rather than agreeing by symmetry.
HAND_WORKED_BLOCK = np.array(
    [
        [-3.0, 7.0],
        [-3.0, 13.0],
        [3.0, 7.0],
        [3.0, 13.0],
    ]
)

#: Chosen so that ``sqrt(9 + 7)`` is exactly 4 and every normalised value is
#: 0.75 one way or the other. Far larger than a real layer would use, which is
#: what makes the hand arithmetic exact rather than approximately right.
HAND_WORKED_EPSILON = 7.0

#: A block with nothing convenient about it, for the plain Python oracle.
AWKWARD_BLOCK = np.array(
    [
        [1.5, -20.0, 0.125],
        [-0.5, 4.0, 0.125],
        [3.25, 11.0, 0.5],
        [0.75, -7.5, -2.0],
        [-2.0, 0.5, 1.375],
    ]
)


def column_means_by_hand(block: np.ndarray) -> np.ndarray:
    """Each column's mean, summed in Python from the definition.

    Deliberately not :func:`numpy.mean`. The implementation will reach for that
    and an oracle that reaches for the same thing asserts only that numpy is
    self-consistent.
    """
    n_rows = block.shape[0]
    return np.array(
        [
            sum(float(block[row, column]) for row in range(n_rows)) / n_rows
            for column in range(block.shape[1])
        ]
    )


def column_variances_by_hand(block: np.ndarray) -> np.ndarray:
    """Each column's *biased* variance, dividing by ``n`` rather than ``n - 1``.

    The biased form is what the forward pass uses and therefore what the
    backward pass differentiates, so an oracle using the unbiased one would
    disagree by ``n / (n - 1)`` and blame the layer for it.
    """
    n_rows = block.shape[0]
    means = column_means_by_hand(block)
    return np.array(
        [
            sum(
                (float(block[row, column]) - float(means[column])) ** 2
                for row in range(n_rows)
            )
            / n_rows
            for column in range(block.shape[1])
        ]
    )


def standardised_by_hand(
    block: np.ndarray, epsilon: float, scale: np.ndarray, shift: np.ndarray
) -> np.ndarray:
    """``scale * (value - mean) / sqrt(variance + epsilon) + shift``, in Python.

    One value at a time, with :func:`math.sqrt` rather than the array call, so
    that nothing about how the answer is arranged is borrowed from the thing
    being checked.
    """
    n_rows = block.shape[0]
    means = column_means_by_hand(block)
    variances = column_variances_by_hand(block)

    answer = np.empty_like(block)
    for column in range(block.shape[1]):
        deviation = math.sqrt(float(variances[column]) + epsilon)
        for row in range(n_rows):
            answer[row, column] = float(scale[column]) * (
                float(block[row, column]) - float(means[column])
            ) / deviation + float(shift[column])
    return answer


def predicted_by_hand(block: np.ndarray, layer: BatchNormalization) -> np.ndarray:
    """The prediction-time answer, worked from the running figures in Python.

    A different formula from :func:`standardised_by_hand` rather than the same
    one fed different numbers, because the whole claim being checked is that
    prediction reads statistics the batch had no part in.
    """
    answer = np.empty_like(block)
    for column in range(block.shape[1]):
        deviation = math.sqrt(float(layer.running_variance[column]) + layer.epsilon)
        for row in range(block.shape[0]):
            answer[row, column] = float(layer.scale[column]) * (
                float(block[row, column]) - float(layer.running_mean[column])
            ) / deviation + float(layer.shift[column])
    return answer


def hand_worked_layer() -> BatchNormalization:
    """Two features, an epsilon of 7.0, and a scale and shift chosen on paper."""
    return BatchNormalization(
        n_features=2,
        epsilon=HAND_WORKED_EPSILON,
        scale=np.array([2.0, 10.0]),
        shift=np.array([1.0, -1.0]),
    )


def predicting_layer() -> BatchNormalization:
    """Running figures that are nothing like any batch it will be shown.

    The defaults are zeros and ones, which a body that quietly standardised by
    the batch would still roughly reproduce on centred data. These cannot be
    reached by accident.
    """
    return BatchNormalization(
        n_features=3,
        scale=np.array([2.0, 0.5, -1.0]),
        shift=np.array([1.0, -1.0, 0.25]),
        running_mean=np.array([10.0, -4.0, 0.5]),
        running_variance=np.array([4.0, 9.0, 0.25]),
    )


def bent_layer() -> BatchNormalization:
    """Three features whose scale and shift are none of them the defaults.

    A scale of one and a shift of zero would let the affine's own contribution
    to the backward pass hide, since ``d = arriving * scale`` is then just
    ``arriving``. One scale is negative for the same reason.
    """
    return BatchNormalization(
        n_features=3,
        scale=np.array([1.3, -0.7, 0.4]),
        shift=np.array([0.2, 0.9, -0.5]),
    )


def gradient_check_rows() -> np.ndarray:
    """Five rows of three features, each feature on its own scale and centre.

    Features that shared a mean and a variance would let a body that computed
    one statistic for the whole block pass, which is the sort of thing only a
    fixture can rule out.
    """
    generator = np.random.default_rng(21)
    return generator.normal(size=(5, 3)) * np.array([1.0, 3.0, 0.5]) + np.array(
        [2.0, -1.0, 0.0]
    )


def gradient_check_weights() -> np.ndarray:
    """The summary's weights, which are also the arriving block."""
    return np.random.default_rng(22).normal(size=(5, 3))


def summary_of(
    layer: BatchNormalization, rows: np.ndarray, weights: np.ndarray
) -> float:
    """A scalar reading of a whole training pass, whose slope is ``weights``.

    ``sum(weights * outputs)`` differentiates to ``weights`` at every output, so
    the same block is both the scalar's definition and the arriving block a
    backward pass is handed, and the two claims are directly comparable without
    a loss object in between.

    The pass is stated as training rather than defaulted. Under prediction the
    batch is not read at all and every derivative with respect to another row
    would be zero, so the check would pass on a body with no batch route in it.
    """
    return float(np.sum(weights * layer.respond_to(rows, PassPurpose.TRAINING).outputs))


def layer_carrying(
    layer: BatchNormalization, scale: np.ndarray, shift: np.ndarray
) -> BatchNormalization:
    """The same layer with different learned parameters, for a nudged pass."""
    return BatchNormalization(
        n_features=layer.n_features,
        momentum=layer.momentum,
        epsilon=layer.epsilon,
        scale=scale,
        shift=shift,
        running_mean=layer.running_mean,
        running_variance=layer.running_variance,
    )


def statistics_for(
    layer: BatchNormalization, rows: np.ndarray, weights: np.ndarray
) -> BatchStatistics:
    """The gradient a training pass and its backward step produce."""
    correction = layer.correction_for(
        layer.respond_to(rows, PassPurpose.TRAINING), weights
    )
    assert isinstance(correction.gradient, BatchStatistics)
    return correction.gradient


#: One builder per per-feature vector, so each refusal can be asked of all four.
PER_FEATURE_BUILDERS = [
    pytest.param(
        lambda values: BatchNormalization(n_features=3, scale=values), id="scale"
    ),
    pytest.param(
        lambda values: BatchNormalization(n_features=3, shift=values), id="shift"
    ),
    pytest.param(
        lambda values: BatchNormalization(n_features=3, running_mean=values),
        id="running_mean",
    ),
    pytest.param(
        lambda values: BatchNormalization(n_features=3, running_variance=values),
        id="running_variance",
    ),
]


class TestConstruction:
    def test_its_shape_is_the_same_width_on_both_sides(self) -> None:
        """This layer moves no join, which is what lets it go anywhere."""
        layer = BatchNormalization(n_features=7)

        assert layer.shape == LayerShape(n_inputs=7, n_outputs=7)
        assert layer.shape.reads == layer.shape.answers

    def test_it_reports_the_width_it_was_given(self) -> None:
        assert BatchNormalization(n_features=12).n_features == 12

    def test_the_scale_starts_at_one_and_the_shift_at_zero(self) -> None:
        """Which starts the layer standardising and doing nothing else."""
        layer = BatchNormalization(n_features=4)

        assert np.allclose(layer.scale, 1.0)
        assert np.allclose(layer.shift, 0.0)

    def test_the_running_figures_start_at_zero_and_one(self) -> None:
        """What an untrained layer believes, which makes prediction near enough
        an identity rather than an error."""
        layer = BatchNormalization(n_features=4)

        assert np.allclose(layer.running_mean, 0.0)
        assert np.allclose(layer.running_variance, 1.0)

    def test_every_default_vector_is_one_value_per_feature(self) -> None:
        layer = BatchNormalization(n_features=5)

        for vector in (
            layer.scale,
            layer.shift,
            layer.running_mean,
            layer.running_variance,
        ):
            assert vector.shape == (5,)

    def test_the_momentum_and_epsilon_defaults_are_reported(self) -> None:
        layer = BatchNormalization(n_features=2)

        assert layer.momentum == pytest.approx(0.9)
        assert layer.epsilon == pytest.approx(1e-5)

    def test_supplied_vectors_are_kept(self) -> None:
        layer = predicting_layer()

        assert np.allclose(layer.scale, [2.0, 0.5, -1.0])
        assert np.allclose(layer.shift, [1.0, -1.0, 0.25])
        assert np.allclose(layer.running_mean, [10.0, -4.0, 0.5])
        assert np.allclose(layer.running_variance, [4.0, 9.0, 0.25])

    def test_a_supplied_vector_is_copied_rather_than_aliased(self) -> None:
        """A caller who kept the array must not be able to move the layer."""
        supplied = np.array([2.0, 3.0])
        layer = BatchNormalization(n_features=2, scale=supplied)

        supplied[0] = 99.0

        assert np.allclose(layer.scale, [2.0, 3.0])

    @pytest.mark.parametrize(
        "read",
        [
            lambda layer: layer.scale,
            lambda layer: layer.shift,
            lambda layer: layer.running_mean,
            lambda layer: layer.running_variance,
        ],
        ids=["scale", "shift", "running_mean", "running_variance"],
    )
    def test_the_learned_and_running_vectors_are_frozen(
        self, read: Callable[[BatchNormalization], np.ndarray]
    ) -> None:
        layer = BatchNormalization(n_features=3)

        with pytest.raises(ValueError):
            read(layer)[0] = 1.0

    def test_a_running_variance_of_zero_is_allowed(self) -> None:
        """Zero is what a feature that never varied really had, and epsilon is
        what keeps it from dividing by nothing."""
        layer = BatchNormalization(n_features=2, running_variance=np.zeros(2))

        assert np.allclose(layer.running_variance, 0.0)


class TestRefusedConfigurations:
    @pytest.mark.parametrize("momentum", [-0.1, -1.0, 1.0, 1.5, np.nan, np.inf])
    def test_a_momentum_outside_its_range_is_refused(self, momentum: float) -> None:
        """One never learns anything from the batch, which is the open end."""
        with pytest.raises(InvalidValuesError):
            BatchNormalization(n_features=3, momentum=momentum)

    @pytest.mark.parametrize("momentum", [0.0, 0.5, 0.999])
    def test_a_momentum_inside_its_range_is_allowed(self, momentum: float) -> None:
        assert BatchNormalization(n_features=3, momentum=momentum).momentum == momentum

    @pytest.mark.parametrize("epsilon", [0.0, -1e-5, -1.0, np.nan, np.inf])
    def test_an_epsilon_that_is_not_strictly_positive_is_refused(
        self, epsilon: float
    ) -> None:
        """Zero is the division by zero this parameter exists to prevent."""
        with pytest.raises(InvalidValuesError):
            BatchNormalization(n_features=3, epsilon=epsilon)

    @pytest.mark.parametrize("value", ["high", None])
    def test_a_momentum_that_is_not_a_number_is_refused(self, value: object) -> None:
        with pytest.raises(InvalidValuesError):
            BatchNormalization(n_features=3, momentum=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", ["small", None])
    def test_an_epsilon_that_is_not_a_number_is_refused(self, value: object) -> None:
        with pytest.raises(InvalidValuesError):
            BatchNormalization(n_features=3, epsilon=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("n_features", [0, -1, -8])
    def test_a_width_below_one_is_refused(self, n_features: int) -> None:
        with pytest.raises(InvalidValuesError):
            BatchNormalization(n_features=n_features)

    @pytest.mark.parametrize("build", PER_FEATURE_BUILDERS)
    @pytest.mark.parametrize("length", [2, 4, 0])
    def test_a_vector_of_the_wrong_length_is_refused(
        self,
        build: Callable[[np.ndarray], BatchNormalization],
        length: int,
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            build(np.ones(length))

    @pytest.mark.parametrize("build", PER_FEATURE_BUILDERS)
    def test_a_vector_that_is_not_one_dimensional_is_refused(
        self, build: Callable[[np.ndarray], BatchNormalization]
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            build(np.ones((3, 1)))

    @pytest.mark.parametrize("build", PER_FEATURE_BUILDERS)
    @pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
    def test_a_vector_carrying_a_non_finite_entry_is_refused(
        self,
        build: Callable[[np.ndarray], BatchNormalization],
        poison: float,
    ) -> None:
        values = np.ones(3)
        values[1] = poison

        with pytest.raises(InvalidValuesError):
            build(values)

    @pytest.mark.parametrize("build", PER_FEATURE_BUILDERS)
    def test_a_vector_that_is_not_numbers_at_all_is_refused(
        self, build: Callable[[np.ndarray], BatchNormalization]
    ) -> None:
        with pytest.raises(InvalidValuesError):
            build("per feature")  # type: ignore[arg-type]

    def test_a_negative_running_variance_is_refused(self) -> None:
        """A variance is a mean of squares, so a negative one is not a small
        number, it is a different quantity."""
        with pytest.raises(InvalidValuesError):
            BatchNormalization(
                n_features=3, running_variance=np.array([1.0, -1.0, 1.0])
            )


class TestTheTrainingForwardPassWorkedByHand:
    def test_the_outputs_are_the_numbers_computed_on_paper(self) -> None:
        """Variance 9, epsilon 7, deviation 4, so every normalised value is
        0.75 one way or the other before the affine."""
        answer = hand_worked_layer().respond_to(HAND_WORKED_BLOCK, PassPurpose.TRAINING)

        assert np.allclose(
            answer.outputs,
            [[-0.5, -8.5], [-0.5, 6.5], [2.5, -8.5], [2.5, 6.5]],
        )

    def test_the_normalised_block_is_the_standardisation_before_the_affine(
        self,
    ) -> None:
        response = hand_worked_layer().respond_to(
            HAND_WORKED_BLOCK, PassPurpose.TRAINING
        )
        assert isinstance(response, NormalisationResponse)

        assert np.allclose(
            response.normalised,
            [[-0.75, -0.75], [-0.75, 0.75], [0.75, -0.75], [0.75, 0.75]],
        )

    def test_it_answers_with_the_arrangement_it_read(self) -> None:
        answer = hand_worked_layer().respond_to(HAND_WORKED_BLOCK, PassPurpose.TRAINING)

        assert answer.outputs.shape == HAND_WORKED_BLOCK.shape

    def test_the_variance_is_the_biased_one(self) -> None:
        """The unbiased form would divide by 3 rather than 4 here, giving 12
        rather than 9 and a deviation of ``sqrt(19)`` rather than 4."""
        response = hand_worked_layer().respond_to(
            HAND_WORKED_BLOCK, PassPurpose.TRAINING
        )
        assert isinstance(response, NormalisationResponse)

        assert np.allclose(response.batch_variance, [9.0, 9.0])
        assert np.allclose(response.deviation, [4.0, 4.0])

    @pytest.mark.parametrize(
        ("scale", "shift"),
        [
            (np.ones(3), np.zeros(3)),
            (np.array([2.0, -1.5, 0.25]), np.array([1.0, 0.0, -3.0])),
            (np.array([-1.0, -1.0, -1.0]), np.array([5.0, 5.0, 5.0])),
        ],
        ids=["defaults", "mixed", "reflected"],
    )
    def test_it_agrees_with_a_plain_python_oracle(
        self, scale: np.ndarray, shift: np.ndarray
    ) -> None:
        """The oracle is written from the definition, one value at a time."""
        layer = BatchNormalization(n_features=3, scale=scale, shift=shift)

        answer = layer.respond_to(AWKWARD_BLOCK, PassPurpose.TRAINING)

        assert np.allclose(
            answer.outputs,
            standardised_by_hand(AWKWARD_BLOCK, layer.epsilon, scale, shift),
        )

    def test_the_response_carries_the_block_that_was_read(self) -> None:
        response = hand_worked_layer().respond_to(
            HAND_WORKED_BLOCK, PassPurpose.TRAINING
        )

        assert np.allclose(response.inputs, HAND_WORKED_BLOCK)


class TestEachFeatureIsStandardisedOnItsOwn:
    """The property the layer exists for, asserted rather than described."""

    def test_every_column_comes_out_centred(self) -> None:
        layer = BatchNormalization(n_features=3)

        answer = layer.respond_to(AWKWARD_BLOCK, PassPurpose.TRAINING)

        assert np.allclose(column_means_by_hand(answer.outputs), 0.0, atol=1e-12)

    def test_every_column_comes_out_with_unit_variance(self) -> None:
        """Epsilon shrinks it by ``variance / (variance + epsilon)``, which on
        this block is at most four parts in a million."""
        layer = BatchNormalization(n_features=3)

        answer = layer.respond_to(AWKWARD_BLOCK, PassPurpose.TRAINING)

        assert np.allclose(column_variances_by_hand(answer.outputs), 1.0, atol=1e-4)

    def test_a_column_is_standardised_by_its_own_statistics(self) -> None:
        """One column on a wildly different scale must not drag the others.

        A body that took one mean over the whole block rather than one per
        column would leave the first two columns badly off centre here.
        """
        block = np.array([[0.0, 1000.0], [1.0, 1002.0], [2.0, 1004.0], [3.0, 1006.0]])
        layer = BatchNormalization(n_features=2)

        answer = layer.respond_to(block, PassPurpose.TRAINING)

        assert np.allclose(column_means_by_hand(answer.outputs), 0.0, atol=1e-10)
        assert np.allclose(column_variances_by_hand(answer.outputs), 1.0, atol=1e-4)

    def test_changing_one_feature_leaves_the_others_alone(self) -> None:
        layer = BatchNormalization(n_features=3)
        moved = np.array(AWKWARD_BLOCK, copy=True)
        moved[:, 0] = moved[:, 0] * 40.0 + 7.0

        first = layer.respond_to(AWKWARD_BLOCK, PassPurpose.TRAINING).outputs
        second = layer.respond_to(moved, PassPurpose.TRAINING).outputs

        assert np.allclose(first[:, 1:], second[:, 1:])

    def test_shifting_a_whole_feature_changes_nothing(self) -> None:
        """Standardising removes the centre, so the offset cannot survive it."""
        layer = BatchNormalization(n_features=3)
        moved = AWKWARD_BLOCK + np.array([100.0, -0.5, 6.0])

        first = layer.respond_to(AWKWARD_BLOCK, PassPurpose.TRAINING).outputs
        second = layer.respond_to(moved, PassPurpose.TRAINING).outputs

        assert np.allclose(first, second)


class TestTheAffineCanUndoIt:
    """The claim that normalisation costs the layer no expressiveness.

    Standardising alone would be a constraint, forbidding an off-centre answer
    the network may genuinely want. With the affine restored the layer can
    represent the identity exactly, and this is that sentence run as arithmetic.
    """

    def test_the_identity_scale_and_shift_return_the_inputs(self) -> None:
        variances = column_variances_by_hand(AWKWARD_BLOCK)
        means = column_means_by_hand(AWKWARD_BLOCK)
        layer = BatchNormalization(n_features=3)
        undoing = BatchNormalization(
            n_features=3,
            epsilon=layer.epsilon,
            scale=np.sqrt(variances + layer.epsilon),
            shift=means,
        )

        answer = undoing.respond_to(AWKWARD_BLOCK, PassPurpose.TRAINING)

        assert np.allclose(answer.outputs, AWKWARD_BLOCK)

    def test_it_holds_on_the_hand_worked_block_too(self) -> None:
        """Here the undoing scale is exactly 4 and the shift exactly the mean."""
        undoing = BatchNormalization(
            n_features=2,
            epsilon=HAND_WORKED_EPSILON,
            scale=np.array([4.0, 4.0]),
            shift=np.array([0.0, 10.0]),
        )

        answer = undoing.respond_to(HAND_WORKED_BLOCK, PassPurpose.TRAINING)

        assert np.allclose(answer.outputs, HAND_WORKED_BLOCK)


class TestAConstantFeature:
    """What epsilon is for, and the reason it sits inside the square root."""

    def test_a_constant_column_normalises_to_zero_rather_than_nan(self) -> None:
        """A rectified unit that is off for every row in the batch is exactly
        this, so it is an ordinary case rather than an edge one."""
        block = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
        layer = BatchNormalization(n_features=2)

        response = layer.respond_to(block, PassPurpose.TRAINING)
        assert isinstance(response, NormalisationResponse)

        assert np.all(np.isfinite(response.outputs))
        assert np.allclose(response.normalised[:, 1], 0.0)

    def test_a_constant_column_answers_with_its_shift(self) -> None:
        """Zero normalised times any scale is zero, so all that is left is the
        offset, which is the honest answer for a feature carrying nothing."""
        block = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        layer = BatchNormalization(
            n_features=2, scale=np.array([1.0, 3.0]), shift=np.array([0.0, -2.0])
        )

        answer = layer.respond_to(block, PassPurpose.TRAINING)

        assert np.allclose(answer.outputs[:, 1], -2.0)

    def test_the_neighbouring_column_is_unaffected_by_it(self) -> None:
        block = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
        layer = BatchNormalization(n_features=2)

        answer = layer.respond_to(block, PassPurpose.TRAINING)

        assert column_variances_by_hand(answer.outputs)[0] == pytest.approx(
            1.0, abs=1e-4
        )

    def test_a_single_row_makes_every_feature_constant(self) -> None:
        """One row has no spread anywhere, so the whole answer is the shift."""
        layer = BatchNormalization(
            n_features=3,
            scale=np.array([2.0, -1.0, 7.0]),
            shift=np.array([0.5, 1.5, -2.5]),
        )

        answer = layer.respond_to(np.array([[3.0, -8.0, 0.25]]), PassPurpose.TRAINING)

        assert np.all(np.isfinite(answer.outputs))
        assert np.allclose(answer.outputs, [[0.5, 1.5, -2.5]])

    def test_the_recorded_variance_of_a_constant_feature_is_zero(self) -> None:
        block = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])

        response = BatchNormalization(n_features=2).respond_to(
            block, PassPurpose.TRAINING
        )
        assert isinstance(response, NormalisationResponse)

        assert response.batch_variance[1] == pytest.approx(0.0, abs=1e-15)
        assert response.deviation[1] == pytest.approx(math.sqrt(1e-5))


class TestPredicting:
    def test_it_standardises_by_the_running_figures(self) -> None:
        """Worked by hand from the running mean and variance, not the batch."""
        layer = predicting_layer()
        block = np.array([[12.0, -1.0, 1.0], [10.0, -4.0, 0.5]])

        answer = layer.respond_to(block, PassPurpose.PREDICTING)

        assert np.allclose(answer.outputs, predicted_by_hand(block, layer))

    def test_a_row_that_is_the_running_mean_answers_with_the_shift(self) -> None:
        """The one row whose answer can be read off without any arithmetic, so
        a body standardising by the batch instead cannot reproduce it."""
        layer = predicting_layer()

        answer = layer.respond_to(
            np.array([[10.0, -4.0, 0.5], [0.0, 0.0, 0.0]]), PassPurpose.PREDICTING
        )

        assert np.allclose(answer.outputs[0], [1.0, -1.0, 0.25])

    def test_a_row_gets_the_same_answer_whatever_company_it_keeps(self) -> None:
        """The whole reason prediction cannot use the batch. Otherwise the same
        request would score differently in a batch of 32 and a batch of 64."""
        layer = predicting_layer()
        shared = np.array([1.0, 2.0, 3.0])
        first = np.array([shared, [0.0, 0.0, 0.0], [9.0, -9.0, 4.0]])
        second = np.array([[100.0, -50.0, 7.0], shared])

        first_answer = layer.respond_to(first, PassPurpose.PREDICTING).outputs[0]
        second_answer = layer.respond_to(second, PassPurpose.PREDICTING).outputs[1]

        assert np.allclose(first_answer, second_answer)

    def test_training_by_contrast_does_depend_on_the_company(self) -> None:
        """The same comparison the other way round, so that the test above is
        asserting a difference between the two passes rather than a constant."""
        layer = predicting_layer()
        shared = np.array([1.0, 2.0, 3.0])
        first = np.array([shared, [0.0, 0.0, 0.0], [9.0, -9.0, 4.0]])
        second = np.array([shared, [50.0, 50.0, 50.0], [-30.0, 2.0, 8.0]])

        first_answer = layer.respond_to(first, PassPurpose.TRAINING).outputs[0]
        second_answer = layer.respond_to(second, PassPurpose.TRAINING).outputs[0]

        assert not np.allclose(first_answer, second_answer)

    def test_a_single_row_can_be_predicted_on(self) -> None:
        """It has no spread of its own, which is exactly why it borrows one."""
        layer = predicting_layer()

        answer = layer.respond_to(np.array([[12.0, -1.0, 1.0]]), PassPurpose.PREDICTING)

        assert np.all(np.isfinite(answer.outputs))
        assert not np.allclose(answer.outputs, layer.shift)

    def test_a_prediction_pass_carries_no_statistics(self) -> None:
        """It standardised by different numbers, so a backward step taken from
        it would be the gradient of a different function."""
        layer = predicting_layer()

        response = layer.respond_to(AWKWARD_BLOCK, PassPurpose.PREDICTING)

        assert isinstance(response, LayerResponse)
        assert not isinstance(response, NormalisationResponse)

    def test_predicting_is_what_a_caller_who_says_nothing_gets(self) -> None:
        """Forgetting to say training costs stale statistics; forgetting to say
        predicting would make every answer depend on the request it arrived in."""
        layer = predicting_layer()

        defaulted = layer.respond_to(AWKWARD_BLOCK)
        stated = layer.respond_to(AWKWARD_BLOCK, PassPurpose.PREDICTING)

        assert not isinstance(defaulted, NormalisationResponse)
        assert np.allclose(defaulted.outputs, stated.outputs)

    def test_an_untrained_layer_predicts_close_to_the_identity(self) -> None:
        """Zeros and ones are what it believes before it has seen anything."""
        layer = BatchNormalization(n_features=3)

        answer = layer.respond_to(AWKWARD_BLOCK, PassPurpose.PREDICTING)

        assert np.allclose(answer.outputs, AWKWARD_BLOCK, atol=1e-4)


class TestRefusedBlocks:
    def test_a_block_of_the_wrong_width_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            BatchNormalization(n_features=3).respond_to(np.zeros((4, 5)))

    def test_a_block_of_no_rows_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            BatchNormalization(n_features=3).respond_to(np.zeros((0, 3)))

    def test_a_non_finite_entry_is_refused_where_it_enters(self) -> None:
        """One ``nan`` would otherwise poison a whole column's mean and travel
        to every row that shares it."""
        poisoned = np.array(AWKWARD_BLOCK, copy=True)
        poisoned[2, 1] = np.nan

        with pytest.raises(InvalidValuesError):
            BatchNormalization(n_features=3).respond_to(poisoned, PassPurpose.TRAINING)

    def test_a_block_that_is_not_numbers_at_all_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            BatchNormalization(n_features=3).respond_to("a batch")  # type: ignore[arg-type]


def recorded_response(batch_mean: Sequence[float]) -> NormalisationResponse:
    """A response over one fixed block, standardised by the stated mean.

    Built directly rather than through a forward pass, because the claim under
    test is about what equality reads and not about what a pass computes. Two
    responses that agree on every block and disagree on the statistics are the
    case worth constructing, and no single input block produces both.
    """
    block = np.array([[1.0, 2.0], [3.0, 4.0]])
    return NormalisationResponse.recording(
        inputs=block.copy(),
        normalised=block.copy(),
        outputs=block.copy(),
        batch_mean=np.array(batch_mean, dtype=np.float64),
        batch_variance=np.ones(2),
        deviation=np.ones(2),
    )


class TestTwoResponsesAreTheSameOnlyIfTheyStandardisedAlike:
    """The statistics are the whole reason this class exists, so equality reads them.

    A response carries what it centred and scaled by, and two responses holding
    identical blocks while having standardised by different batches are not the
    same response. What makes this worth a test rather than an assertion is the
    reflected call in the third one: a subclass comparing against a plain
    response answers ``NotImplemented``, Python retries the other way round,
    and a base class that compared by ``isinstance`` would find matching
    outputs and say True over the subclass's objection.
    """

    def test_the_same_statistics_and_the_same_blocks_are_equal(self) -> None:
        assert recorded_response([0.0, 0.0]) == recorded_response([0.0, 0.0])

    def test_the_same_blocks_under_a_different_mean_are_not(self) -> None:
        assert recorded_response([0.0, 0.0]) != recorded_response([5.0, 5.0])

    def test_it_is_not_equal_to_a_plain_response_with_the_same_blocks(
        self,
    ) -> None:
        block = np.array([[1.0, 2.0], [3.0, 4.0]])
        plain = LayerResponse(inputs=block, scores=block, outputs=block)
        carrying = recorded_response([0.0, 0.0])

        assert carrying != plain
        assert plain != carrying

    def test_it_defers_to_anything_that_is_not_a_response(self) -> None:
        assert recorded_response([0.0, 0.0]).__eq__("response") is NotImplemented


class TestTheTrainingResponse:
    def test_a_training_pass_carries_its_statistics(self) -> None:
        response = BatchNormalization(n_features=3).respond_to(
            AWKWARD_BLOCK, PassPurpose.TRAINING
        )

        assert isinstance(response, NormalisationResponse)

    def test_the_normalised_block_is_the_response_s_scores(self) -> None:
        """Here the affine plays the part an activation plays elsewhere, so the
        pre-affine block is what a score has always been."""
        response = BatchNormalization(n_features=3).respond_to(
            AWKWARD_BLOCK, PassPurpose.TRAINING
        )
        assert isinstance(response, NormalisationResponse)

        assert response.normalised is response.scores

    def test_the_statistics_are_the_ones_it_standardised_by(self) -> None:
        response = BatchNormalization(n_features=3).respond_to(
            AWKWARD_BLOCK, PassPurpose.TRAINING
        )
        assert isinstance(response, NormalisationResponse)

        assert np.allclose(response.batch_mean, column_means_by_hand(AWKWARD_BLOCK))
        assert np.allclose(
            response.batch_variance, column_variances_by_hand(AWKWARD_BLOCK)
        )

    def test_the_deviation_is_the_root_of_the_variance_plus_epsilon(self) -> None:
        """Carried rather than recomputed, so the two passes cannot disagree
        about which epsilon went in."""
        layer = BatchNormalization(n_features=3, epsilon=0.25)

        response = layer.respond_to(AWKWARD_BLOCK, PassPurpose.TRAINING)
        assert isinstance(response, NormalisationResponse)

        assert np.allclose(response.deviation, np.sqrt(response.batch_variance + 0.25))

    def test_the_statistics_are_one_value_per_feature(self) -> None:
        response = BatchNormalization(n_features=3).respond_to(
            AWKWARD_BLOCK, PassPurpose.TRAINING
        )
        assert isinstance(response, NormalisationResponse)

        assert response.batch_mean.shape == (3,)
        assert response.batch_variance.shape == (3,)
        assert response.deviation.shape == (3,)

    @pytest.mark.parametrize(
        "read",
        [
            lambda response: response.inputs,
            lambda response: response.scores,
            lambda response: response.outputs,
            lambda response: response.normalised,
            lambda response: response.batch_mean,
            lambda response: response.batch_variance,
            lambda response: response.deviation,
        ],
        ids=[
            "inputs",
            "scores",
            "outputs",
            "normalised",
            "batch_mean",
            "batch_variance",
            "deviation",
        ],
    )
    def test_every_block_it_hands_out_is_frozen(
        self, read: Callable[[NormalisationResponse], np.ndarray]
    ) -> None:
        """A caller who edited one would change what the backward pass believes
        happened on a forward pass that has already run."""
        response = BatchNormalization(n_features=3).respond_to(
            AWKWARD_BLOCK, PassPurpose.TRAINING
        )
        assert isinstance(response, NormalisationResponse)

        with pytest.raises(ValueError):
            read(response)[0] = 1.0


class TestTheGradientCheck:
    """Nudge one number, watch the summary, compare against the claim.

    Run over the inputs, the scale and the shift in turn. The inputs are the one
    that matters, because the naive reading of the backward pass leaves the
    other two bit-identical and would sail through a spec that checked only
    those. See the module docstring for the measured disagreement.
    """

    def measured_input_gradient(
        self, layer: BatchNormalization, rows: np.ndarray, weights: np.ndarray
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

    def measured_scale_gradient(
        self, layer: BatchNormalization, rows: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        measured = np.zeros_like(layer.scale)
        for feature in range(layer.n_features):
            moved = []
            for direction in (+NUDGE, -NUDGE):
                scale = np.array(layer.scale, copy=True)
                scale[feature] += direction
                moved.append(
                    summary_of(layer_carrying(layer, scale, layer.shift), rows, weights)
                )
            measured[feature] = (moved[0] - moved[1]) / (2.0 * NUDGE)
        return measured

    def measured_shift_gradient(
        self, layer: BatchNormalization, rows: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        measured = np.zeros_like(layer.shift)
        for feature in range(layer.n_features):
            moved = []
            for direction in (+NUDGE, -NUDGE):
                shift = np.array(layer.shift, copy=True)
                shift[feature] += direction
                moved.append(
                    summary_of(layer_carrying(layer, layer.scale, shift), rows, weights)
                )
            measured[feature] = (moved[0] - moved[1]) / (2.0 * NUDGE)
        return measured

    def test_every_input_slope_matches_a_finite_difference(self) -> None:
        """The three routes, collected. This is the test the module is about."""
        layer = bent_layer()
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        correction = layer.correction_for(
            layer.respond_to(rows, PassPurpose.TRAINING), weights
        )
        measured = self.measured_input_gradient(layer, rows, weights)

        assert np.allclose(correction.passed_down, measured, atol=1e-7)

    def test_every_scale_slope_matches_a_finite_difference(self) -> None:
        layer = bent_layer()
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        gradient = statistics_for(layer, rows, weights)
        measured = self.measured_scale_gradient(layer, rows, weights)

        assert np.allclose(gradient.weights[:, 0], measured, atol=1e-7)

    def test_every_shift_slope_matches_a_finite_difference(self) -> None:
        layer = bent_layer()
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        gradient = statistics_for(layer, rows, weights)
        measured = self.measured_shift_gradient(layer, rows, weights)

        assert np.allclose(gradient.biases, measured, atol=1e-7)

    def test_the_input_slopes_hold_on_a_wider_batch(self) -> None:
        """Nine rows and four features, so the ``1 / n`` in the two subtracted
        terms is a different number than in the fixture above."""
        generator = np.random.default_rng(31)
        layer = BatchNormalization(
            n_features=4, scale=generator.normal(size=4), shift=generator.normal(size=4)
        )
        rows = (
            np.random.default_rng(32).normal(size=(9, 4))
            * np.array([1.0, 2.0, 0.3, 5.0])
            + 1.0
        )
        weights = np.random.default_rng(33).normal(size=(9, 4))

        correction = layer.correction_for(
            layer.respond_to(rows, PassPurpose.TRAINING), weights
        )
        measured = self.measured_input_gradient(layer, rows, weights)

        assert np.allclose(correction.passed_down, measured, atol=1e-7)

    def test_the_input_slopes_hold_through_a_constant_feature(self) -> None:
        """Where the deviation is epsilon alone. A larger epsilon keeps the
        slopes near one rather than near ``1 / sqrt(1e-5)``, so a genuine
        disagreement is not hidden inside a relative tolerance."""
        layer = BatchNormalization(
            n_features=3,
            epsilon=0.25,
            scale=np.array([1.5, -0.8, 2.0]),
            shift=np.array([0.1, 0.2, 0.3]),
        )
        rows = np.array(
            [
                [1.0, 4.0, -2.0],
                [2.0, 4.0, 0.5],
                [3.0, 4.0, 1.5],
                [4.5, 4.0, -0.25],
            ]
        )
        weights = np.random.default_rng(34).normal(size=(4, 3))

        correction = layer.correction_for(
            layer.respond_to(rows, PassPurpose.TRAINING), weights
        )
        measured = self.measured_input_gradient(layer, rows, weights)

        assert np.allclose(correction.passed_down, measured, atol=1e-7)

    def test_the_blame_sums_to_zero_down_every_column(self) -> None:
        """The structural signature of the three-term form, and the cheapest
        way to see the naive one fail. Shifting a whole feature by a constant
        cannot change this layer's answer, so the slopes down a column must
        cancel exactly. Measured, the naive reading gives 0.601 here.
        """
        layer = bent_layer()
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        correction = layer.correction_for(
            layer.respond_to(rows, PassPurpose.TRAINING), weights
        )

        assert np.allclose(correction.passed_down.sum(axis=0), 0.0, atol=1e-10)


class TestTheGradientObject:
    def test_the_gradient_is_a_batch_statistics(self) -> None:
        """A plain ``LayerGradient`` would leave the running figures unable to
        move, so prediction would keep using whatever it was last told."""
        layer = bent_layer()

        correction = layer.correction_for(
            layer.respond_to(gradient_check_rows(), PassPurpose.TRAINING),
            gradient_check_weights(),
        )

        assert isinstance(correction.gradient, BatchStatistics)

    def test_it_carries_the_statistics_the_forward_pass_measured(self) -> None:
        layer = bent_layer()
        rows = gradient_check_rows()

        gradient = statistics_for(layer, rows, gradient_check_weights())

        assert np.allclose(gradient.batch_mean, column_means_by_hand(rows))
        assert np.allclose(gradient.batch_variance, column_variances_by_hand(rows))

    def test_the_weight_block_is_one_row_per_feature(self) -> None:
        """``(n_features, 1)``, not the transposed ``(1, n_features)``.

        Each feature is a unit reading a single number, since the answer for
        feature ``j`` is ``scale_j * normalised_j + shift_j``, so this is
        ``LayerGradient``'s ordinary ``(n_neurons, n_inputs)`` rather than a
        shape bent to fit. The width is three deliberately, because at a width
        of one the two readings agree and the mistake survives.
        """
        layer = bent_layer()

        gradient = statistics_for(
            layer, gradient_check_rows(), gradient_check_weights()
        )

        assert gradient.weights.shape == (3, 1)
        assert gradient.biases.shape == (3,)

    def test_the_blame_passed_down_has_the_shape_of_what_was_read(self) -> None:
        layer = bent_layer()
        rows = gradient_check_rows()

        correction = layer.correction_for(
            layer.respond_to(rows, PassPurpose.TRAINING), gradient_check_weights()
        )

        assert correction.passed_down.shape == rows.shape

    def test_the_shift_slope_is_the_blame_summed_down_the_rows(self) -> None:
        """One shift serves every row, so every row it answered contributes."""
        layer = BatchNormalization(n_features=3)
        rows = AWKWARD_BLOCK

        gradient = statistics_for(layer, rows, np.ones_like(rows))

        assert np.allclose(gradient.biases, rows.shape[0])


class TestRefusedBackwardSteps:
    def test_a_prediction_pass_response_is_refused(self) -> None:
        """It standardised by the running figures, so the gradient taken from
        it would be the gradient of a different function."""
        layer = bent_layer()
        rows = gradient_check_rows()
        response = layer.respond_to(rows, PassPurpose.PREDICTING)

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, gradient_check_weights())

    def test_an_arriving_block_of_the_wrong_width_is_refused(self) -> None:
        layer = bent_layer()
        response = layer.respond_to(gradient_check_rows(), PassPurpose.TRAINING)

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((5, 4)))

    def test_an_arriving_block_of_the_wrong_row_count_is_refused(self) -> None:
        layer = bent_layer()
        response = layer.respond_to(gradient_check_rows(), PassPurpose.TRAINING)

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((4, 3)))

    def test_a_response_from_a_differently_shaped_layer_is_refused(self) -> None:
        """It carries real numbers of a plausible shape, so nothing else would
        catch it and the blame would be routed to arbitrary places."""
        layer = bent_layer()
        narrower = BatchNormalization(n_features=2)
        response = narrower.respond_to(
            np.array([[1.0, 2.0], [3.0, 5.0], [0.5, -1.0], [2.0, 2.5], [1.0, 0.0]]),
            PassPurpose.TRAINING,
        )

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((5, 3)))

    def test_an_arriving_block_that_is_not_numbers_is_refused(self) -> None:
        layer = bent_layer()
        response = layer.respond_to(gradient_check_rows(), PassPurpose.TRAINING)

        with pytest.raises(InvalidValuesError):
            layer.correction_for(response, "blame")  # type: ignore[arg-type]


class TestBatchStatistics:
    def statistics_of_width(self, n_features: int) -> BatchStatistics:
        return BatchStatistics(
            weights=np.zeros((n_features, 1)),
            biases=np.zeros(n_features),
            batch_mean=np.zeros(n_features),
            batch_variance=np.ones(n_features),
        )

    def test_it_carries_both_statistics(self) -> None:
        statistics = BatchStatistics(
            weights=np.array([[1.0], [2.0], [3.0]]),
            biases=np.array([4.0, 5.0, 6.0]),
            batch_mean=np.array([0.5, -0.5, 2.0]),
            batch_variance=np.array([1.0, 4.0, 0.0]),
        )

        assert np.allclose(statistics.batch_mean, [0.5, -0.5, 2.0])
        assert np.allclose(statistics.batch_variance, [1.0, 4.0, 0.0])

    def test_it_is_a_layer_gradient(self) -> None:
        """So a stack that knows nothing about this layer can still thread it."""
        assert isinstance(self.statistics_of_width(3), LayerGradient)

    def test_it_builds_at_a_width_above_one(self) -> None:
        """The transposed weight block cannot construct at all here, which is
        ``LayerGradient``'s unit-count check doing its job."""
        assert self.statistics_of_width(4).weights.shape == (4, 1)

    def test_the_transposed_weight_block_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            BatchStatistics(
                weights=np.zeros((1, 3)),
                biases=np.zeros(3),
                batch_mean=np.zeros(3),
                batch_variance=np.ones(3),
            )

    def test_a_negative_batch_variance_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            BatchStatistics(
                weights=np.zeros((3, 1)),
                biases=np.zeros(3),
                batch_mean=np.zeros(3),
                batch_variance=np.array([1.0, -0.5, 1.0]),
            )

    @pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("statistic", ["batch_mean", "batch_variance"])
    def test_a_non_finite_statistic_is_refused(
        self, poison: float, statistic: str
    ) -> None:
        values = np.ones(3)
        values[1] = poison
        blocks = {"batch_mean": np.zeros(3), "batch_variance": np.ones(3)}
        blocks[statistic] = values

        with pytest.raises(InvalidValuesError):
            BatchStatistics(
                weights=np.zeros((3, 1)),
                biases=np.zeros(3),
                batch_mean=blocks["batch_mean"],
                batch_variance=blocks["batch_variance"],
            )

    @pytest.mark.parametrize("length", [2, 4])
    def test_a_statistic_of_the_wrong_length_is_refused(self, length: int) -> None:
        with pytest.raises(ShapeMismatchError):
            BatchStatistics(
                weights=np.zeros((3, 1)),
                biases=np.zeros(3),
                batch_mean=np.zeros(length),
                batch_variance=np.ones(3),
            )

    def test_a_statistic_that_is_not_one_dimensional_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            BatchStatistics(
                weights=np.zeros((3, 1)),
                biases=np.zeros(3),
                batch_mean=np.zeros(3),
                batch_variance=np.ones((3, 1)),
            )

    @pytest.mark.parametrize(
        "read",
        [
            lambda statistics: statistics.batch_mean,
            lambda statistics: statistics.batch_variance,
        ],
        ids=["batch_mean", "batch_variance"],
    )
    def test_the_statistics_are_frozen(
        self, read: Callable[[BatchStatistics], np.ndarray]
    ) -> None:
        statistics = self.statistics_of_width(3)

        with pytest.raises(ValueError):
            read(statistics)[0] = 1.0


class TestStepping:
    def stepped(
        self, layer: BatchNormalization, learning_rate: float = 0.1
    ) -> BatchNormalization:
        gradient = statistics_for(
            layer, gradient_check_rows(), gradient_check_weights()
        )
        stepped = layer.stepped_by(gradient, learning_rate)
        assert isinstance(stepped, BatchNormalization)
        return stepped

    def test_the_scale_and_shift_move_against_their_slopes(self) -> None:
        layer = bent_layer()
        gradient = statistics_for(
            layer, gradient_check_rows(), gradient_check_weights()
        )

        stepped = layer.stepped_by(gradient, learning_rate=0.1)
        assert isinstance(stepped, BatchNormalization)

        assert np.allclose(stepped.scale, layer.scale - 0.1 * gradient.weights[:, 0])
        assert np.allclose(stepped.shift, layer.shift - 0.1 * gradient.biases)

    def test_the_running_figures_move_toward_the_batch(self) -> None:
        """An exponential moving average, so this one adds where the parameters
        subtract. Two kinds of update sharing one method, which is the
        awkwardness this layer has rather than one to be smoothed over."""
        layer = bent_layer()
        gradient = statistics_for(
            layer, gradient_check_rows(), gradient_check_weights()
        )

        stepped = layer.stepped_by(gradient, learning_rate=0.1)
        assert isinstance(stepped, BatchNormalization)

        keep = layer.momentum
        assert np.allclose(
            stepped.running_mean,
            keep * layer.running_mean + (1.0 - keep) * gradient.batch_mean,
        )
        assert np.allclose(
            stepped.running_variance,
            keep * layer.running_variance + (1.0 - keep) * gradient.batch_variance,
        )

    def test_a_momentum_of_zero_replaces_the_running_figures_outright(self) -> None:
        """Legal, and it makes prediction depend on whichever batch came last."""
        layer = BatchNormalization(
            n_features=3,
            momentum=0.0,
            scale=np.array([1.3, -0.7, 0.4]),
            shift=np.array([0.2, 0.9, -0.5]),
            running_mean=np.array([50.0, 50.0, 50.0]),
            running_variance=np.array([50.0, 50.0, 50.0]),
        )
        rows = gradient_check_rows()
        gradient = statistics_for(layer, rows, gradient_check_weights())

        stepped = layer.stepped_by(gradient, learning_rate=0.1)
        assert isinstance(stepped, BatchNormalization)

        assert np.allclose(stepped.running_mean, column_means_by_hand(rows))
        assert np.allclose(stepped.running_variance, column_variances_by_hand(rows))

    def test_the_step_keeps_the_configuration_it_was_not_asked_to_change(
        self,
    ) -> None:
        layer = BatchNormalization(
            n_features=3,
            momentum=0.25,
            epsilon=0.5,
            scale=np.array([1.3, -0.7, 0.4]),
            shift=np.array([0.2, 0.9, -0.5]),
        )

        stepped = self.stepped(layer)

        assert stepped.momentum == pytest.approx(0.25)
        assert stepped.epsilon == pytest.approx(0.5)
        assert stepped.n_features == 3
        assert stepped.shape == layer.shape

    def test_the_original_layer_is_untouched(self) -> None:
        """Training does not mutate, which is what keeps every block frozen."""
        layer = bent_layer()
        scale_before = np.array(layer.scale, copy=True)
        shift_before = np.array(layer.shift, copy=True)
        running_mean_before = np.array(layer.running_mean, copy=True)
        running_variance_before = np.array(layer.running_variance, copy=True)

        self.stepped(layer, learning_rate=0.5)

        assert np.array_equal(layer.scale, scale_before)
        assert np.array_equal(layer.shift, shift_before)
        assert np.array_equal(layer.running_mean, running_mean_before)
        assert np.array_equal(layer.running_variance, running_variance_before)

    def test_the_stepped_layer_is_a_new_object(self) -> None:
        layer = bent_layer()

        assert self.stepped(layer) is not layer

    def test_stepping_by_nothing_is_refused(self) -> None:
        """This layer has parameters, so a missing gradient is a mistake."""
        with pytest.raises(ShapeMismatchError):
            BatchNormalization(n_features=3).stepped_by(None, learning_rate=0.1)

    def test_stepping_by_a_plain_layer_gradient_is_refused(self) -> None:
        """It carries the parameter slopes and not the batch's statistics, so
        the running figures could not move and prediction would keep using
        whatever it was last told."""
        with pytest.raises(ShapeMismatchError):
            BatchNormalization(n_features=3).stepped_by(
                LayerGradient(weights=np.zeros((3, 1)), biases=np.zeros(3)),
                learning_rate=0.1,
            )

    @pytest.mark.parametrize("width", [2, 4])
    def test_a_gradient_of_the_wrong_width_is_refused(self, width: int) -> None:
        with pytest.raises(ShapeMismatchError):
            BatchNormalization(n_features=3).stepped_by(
                BatchStatistics(
                    weights=np.zeros((width, 1)),
                    biases=np.zeros(width),
                    batch_mean=np.zeros(width),
                    batch_variance=np.ones(width),
                ),
                learning_rate=0.1,
            )

    def test_a_step_lowers_the_summary_it_was_asked_to_lower(self) -> None:
        """The point of the whole exercise, on one small step."""
        layer = bent_layer()
        rows = gradient_check_rows()
        weights = gradient_check_weights()

        stepped = self.stepped(layer, learning_rate=0.05)

        assert summary_of(stepped, rows, weights) < summary_of(layer, rows, weights)


class TestInAStack:
    def stack(self, seed: int = 4) -> LayerStack:
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
        return LayerStack([hidden, BatchNormalization(n_features=3), output])

    def rows(self, seed: int = 5) -> np.ndarray:
        return np.random.default_rng(seed).normal(size=(8, 4))

    def targets(self, seed: int = 6) -> np.ndarray:
        return np.random.default_rng(seed).normal(size=(8, 2))

    def test_it_moves_no_join(self) -> None:
        """Same width on both sides, so it can be inserted into a sound stack
        without making it unsound."""
        stack = self.stack()

        assert stack.shape == LayerShape(n_inputs=4, n_outputs=2)

    def test_it_can_sit_at_the_bottom_of_a_stack_too(self) -> None:
        hidden = DenseLayer(
            [
                Neuron(np.ones(4) * 0.1, bias=0.0, activation=Identity())
                for _ in range(2)
            ]
        )

        stack = LayerStack([BatchNormalization(n_features=4), hidden])

        assert stack.shape == LayerShape(n_inputs=4, n_outputs=2)

    def test_a_backward_pass_produces_its_gradient_in_layer_order(self) -> None:
        stack = self.stack()

        backward = stack.backward_pass(self.rows(), self.targets(), SquaredError())

        assert len(backward) == 3
        assert isinstance(backward[1], BatchStatistics)

    def test_repeated_steps_lower_the_loss(self) -> None:
        stack = self.stack()
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

    def test_training_moves_the_running_statistics(self) -> None:
        """They start at zeros and ones and are the only thing a forward pass
        alone would never change, since the update lives in the step."""
        stack = self.stack()
        rows = self.rows()
        targets = self.targets()
        loss = SquaredError()

        for _ in range(5):
            stack = stack.stepped_by(
                stack.backward_pass(rows, targets, loss), learning_rate=0.2
            )

        trained = stack[1]
        assert isinstance(trained, BatchNormalization)
        assert not np.allclose(trained.running_mean, 0.0)
        assert not np.allclose(trained.running_variance, 1.0)

    def test_a_forward_pass_alone_leaves_the_running_statistics_where_they_were(
        self,
    ) -> None:
        """The cost of putting the update in the step rather than in the
        forward pass, stated rather than hidden. A batch that was never learned
        from should not shape what the model believes about its inputs."""
        stack = self.stack()

        stack.respond_to(self.rows(), PassPurpose.TRAINING)

        untouched = stack[1]
        assert isinstance(untouched, BatchNormalization)
        assert np.allclose(untouched.running_mean, 0.0)
        assert np.allclose(untouched.running_variance, 1.0)

    def test_a_trained_stack_still_predicts_row_by_row(self) -> None:
        """The reason the running figures are kept at all. One row on its own
        has no spread, and a prediction has to work anyway."""
        stack = self.stack()
        rows = self.rows()
        targets = self.targets()
        loss = SquaredError()

        for _ in range(5):
            stack = stack.stepped_by(
                stack.backward_pass(rows, targets, loss), learning_rate=0.2
            )

        answer = stack.respond_to(rows[:1], PassPurpose.PREDICTING).outputs

        assert answer.shape == (1, 2)
        assert np.all(np.isfinite(answer))
