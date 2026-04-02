"""Spec for the layer -- red until ``_response_for`` lands.

Three of these are worth reading rather than skimming.

The first is the agreement with the neurons themselves. A layer is defined as
its neurons all reading the same row, so the block it answers with has to equal
what those neurons say one at a time, and the test builds the expected block by
asking each neuron individually.

Be honest about what that pins. The body it specifies also calls
``neuron.respond_to`` in a loop, so the *arithmetic* halves of the two agree by
construction and an error inside that shared call is invisible here. What it
does pin is placement, which is where a layer can actually go wrong: neuron j
in column j, row i in row i, both blocks filled rather than one. A reversed
iteration, a transposed fill, or a flat buffer reshaped in visiting order all
fail it. The arithmetic itself is pinned by the hand-worked numbers below,
which were computed on paper and owe the implementation nothing.

The second is column order. Neuron ``j``'s answer has to land in column ``j``,
every time, and the fixture is built so that each neuron answers a
distinguishable number. Transposing a block or filling it in visiting order
rather than neuron order both survive a test that only checks shapes and sums,
and both make the next layer's weights meaningless.

The third is that the scores and the outputs must disagree. A layer whose bend
was never applied returns the scores twice and passes any test that only looks
at one of the two blocks, so the rectified fixture is chosen to have negative
scores whose outputs are zero.
"""

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    ShapeMismatchError,
)
from oop_ml.core.network.activation import Identity, RectifiedLinear, Sigmoid
from oop_ml.core.network.layer import DenseLayer, LayerResponse
from oop_ml.core.network.neuron import Neuron
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray

TWO_ROWS = np.array([[1.0, 2.0], [3.0, 4.0]])


def counting_layer() -> DenseLayer:
    """Three neurons over two inputs, each answering a distinguishable number.

    The first reads only the first input, the second only the second, and the
    third sums them, so a transposed or misordered block cannot pass unnoticed.
    """
    return DenseLayer(
        [
            Neuron([1.0, 0.0], bias=0.0, activation=Identity()),
            Neuron([0.0, 1.0], bias=0.0, activation=Identity()),
            Neuron([1.0, 1.0], bias=0.0, activation=Identity()),
        ]
    )


class TestConstruction:
    def test_its_shape_comes_from_the_neurons(self) -> None:
        layer = counting_layer()

        assert layer.shape == LayerShape(n_inputs=2, n_outputs=3)

    def test_the_output_width_is_the_neuron_count(self) -> None:
        assert len(counting_layer()) == 3

    def test_a_layer_with_no_neurons_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            DenseLayer([])

    def test_neurons_disagreeing_on_width_are_refused(self) -> None:
        """They are handed the identical row, so one of them is unfeedable."""
        with pytest.raises(ShapeMismatchError):
            DenseLayer(
                [
                    Neuron([1.0, 2.0], bias=0.0, activation=Identity()),
                    Neuron([1.0, 2.0, 3.0], bias=0.0, activation=Identity()),
                ]
            )

    def test_the_output_width_owes_nothing_to_the_input_width(self) -> None:
        wide = DenseLayer([Neuron([1.0] * 40, bias=0.0, activation=Identity())])

        assert wide.shape == LayerShape(n_inputs=40, n_outputs=1)

    def test_mixed_activations_are_allowed(self) -> None:
        """Nothing in the mathematics forbids it, so nothing here does."""
        layer = DenseLayer(
            [
                Neuron([1.0], bias=0.0, activation=Identity()),
                Neuron([1.0], bias=0.0, activation=Sigmoid()),
            ]
        )

        assert len(layer) == 2

    def test_it_is_iterable_rather_than_handing_out_its_container(self) -> None:
        layer = counting_layer()

        assert not hasattr(layer, "neurons")
        assert [neuron.n_inputs for neuron in layer] == [2, 2, 2]

    def test_mutating_the_caller_s_sequence_does_not_reorder_the_layer(self) -> None:
        """Reordering the neurons transposes the meaning of every column the
        layer answers with, so the order has to stop being the caller's."""
        supplied = [
            Neuron([1.0, 0.0], bias=0.0, activation=Identity()),
            Neuron([0.0, 1.0], bias=0.0, activation=Identity()),
        ]
        layer = DenseLayer(supplied)

        supplied.reverse()

        assert np.allclose(layer.respond_to(TWO_ROWS).outputs, [[1.0, 2.0], [3.0, 4.0]])


class TestRespondingShapes:
    def test_the_blocks_are_rows_by_neurons(self) -> None:
        response = counting_layer().respond_to(TWO_ROWS)

        assert response.scores.shape == (2, 3)
        assert response.outputs.shape == (2, 3)
        assert response.n_rows == 2
        assert response.n_neurons == 3

    def test_a_block_of_the_wrong_width_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            counting_layer().respond_to(np.array([[1.0, 2.0, 3.0]]))

    def test_a_one_dimensional_block_is_refused(self) -> None:
        """A single row is ``(1, n)``, and guessing which axis is which is how
        a transposed block reaches the arithmetic."""
        with pytest.raises(ShapeMismatchError):
            counting_layer().respond_to(np.array([1.0, 2.0]))

    def test_a_block_with_no_rows_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            counting_layer().respond_to(np.empty((0, 2)))

    def test_one_row_is_allowed(self) -> None:
        response = counting_layer().respond_to(np.array([[1.0, 2.0]]))

        assert response.scores.shape == (1, 3)


class TestRespondingValues:
    def test_it_agrees_with_the_neurons_asked_one_at_a_time(self) -> None:
        """The oracle is the definition of a layer, built from the units."""
        layer = counting_layer()
        response = layer.respond_to(TWO_ROWS)

        for row_index, row in enumerate(TWO_ROWS):
            for neuron_index, neuron in enumerate(layer):
                alone = neuron.respond_to(row)
                assert response.scores[row_index][neuron_index] == pytest.approx(
                    alone.score
                )
                assert response.outputs[row_index][neuron_index] == pytest.approx(
                    alone.output
                )

    def test_each_neuron_answers_in_its_own_column(self) -> None:
        """Worked by hand on the counting layer, which cannot be transposed."""
        response = counting_layer().respond_to(TWO_ROWS)

        assert np.allclose(response.outputs, [[1.0, 2.0, 3.0], [3.0, 4.0, 7.0]])

    def test_the_bend_is_actually_applied(self) -> None:
        """Scores and outputs have to differ, or the layer skipped its bends."""
        layer = DenseLayer(
            [
                Neuron([1.0, 1.0], bias=-10.0, activation=RectifiedLinear()),
                Neuron([1.0, 1.0], bias=0.0, activation=RectifiedLinear()),
            ]
        )
        response = layer.respond_to(TWO_ROWS)

        assert np.allclose(response.scores, [[-7.0, 3.0], [-3.0, 7.0]])
        assert np.allclose(response.outputs, [[0.0, 3.0], [0.0, 7.0]])

    def test_a_row_is_answered_the_same_whatever_it_travels_with(self) -> None:
        """Rows are independent, so batching cannot change any single answer."""
        layer = counting_layer()
        alone = layer.respond_to(np.array([[3.0, 4.0]]))
        together = layer.respond_to(TWO_ROWS)

        assert np.allclose(alone.outputs[0], together.outputs[1])


class TestLayerResponse:
    def test_it_refuses_blocks_that_disagree(self) -> None:
        with pytest.raises(ShapeMismatchError):
            LayerResponse(
                inputs=np.zeros((2, 2)),
                scores=np.zeros((2, 3)),
                outputs=np.zeros((2, 4)),
            )

    def test_it_refuses_a_one_dimensional_block(self) -> None:
        with pytest.raises(ShapeMismatchError):
            LayerResponse(
                inputs=np.zeros((1, 2)), scores=np.zeros(3), outputs=np.zeros(3)
            )

    def test_its_blocks_cannot_be_written_through(self) -> None:
        response = LayerResponse(
            inputs=np.zeros((2, 2)), scores=np.zeros((2, 2)), outputs=np.zeros((2, 2))
        )

        with pytest.raises(ValueError):
            response.scores[0][0] = 1.0
        with pytest.raises(ValueError):
            response.outputs[0][0] = 1.0

    def test_mutating_the_source_does_not_reach_the_response(self) -> None:
        scores = np.zeros((2, 2))
        response = LayerResponse(
            inputs=np.zeros((2, 2)), scores=scores, outputs=np.zeros((2, 2))
        )

        scores[0][0] = 99.0

        assert response.scores[0][0] == 0.0

    def test_two_alike_are_equal(self) -> None:
        first = LayerResponse(
            inputs=np.zeros((1, 2)), scores=np.zeros((1, 2)), outputs=np.ones((1, 2))
        )
        second = LayerResponse(
            inputs=np.zeros((1, 2)), scores=np.zeros((1, 2)), outputs=np.ones((1, 2))
        )

        assert first == second

    def test_matching_scores_are_not_enough(self) -> None:
        """An __eq__ reading only the scores calls a bent layer equal to an
        unbent one, which is the difference that matters most here."""
        bent = LayerResponse(
            inputs=np.zeros((1, 2)),
            scores=np.full((1, 2), -1.0),
            outputs=np.zeros((1, 2)),
        )
        unbent = LayerResponse(
            inputs=np.zeros((1, 2)),
            scores=np.full((1, 2), -1.0),
            outputs=np.full((1, 2), -1.0),
        )

        assert bent != unbent

    def test_mutating_the_source_outputs_does_not_reach_the_response(self) -> None:
        outputs = np.zeros((2, 2))
        response = LayerResponse(
            inputs=np.zeros((2, 2)), scores=np.zeros((2, 2)), outputs=outputs
        )

        outputs[0][0] = 99.0

        assert response.outputs[0][0] == 0.0
        assert outputs.flags.writeable

    def test_its_counts_are_python_integers(self) -> None:
        response = LayerResponse(
            inputs=np.zeros((2, 2)), scores=np.zeros((2, 3)), outputs=np.zeros((2, 3))
        )

        assert type(response.n_rows) is int
        assert type(response.n_neurons) is int

    def test_it_defers_to_anything_that_is_not_a_response(self) -> None:
        response = LayerResponse(
            inputs=np.zeros((1, 2)), scores=np.zeros((1, 1)), outputs=np.zeros((1, 1))
        )

        assert response.__eq__("response") is NotImplemented


class CarryingResponse(LayerResponse):
    """A response that carries one extra thing, standing in for a real subclass.

    Written here rather than borrowed from
    :class:`~oop_ml.core.network.dropout.DropoutResponse` on purpose. The claim
    under test belongs to ``LayerResponse``, so a test that reached into a layer
    two packages away for a witness would be testing the rule through something
    that could change for its own reasons. A subclass carrying one array is all
    the rule is about.
    """

    __slots__ = ("_carried",)

    def __init__(self, block: FloatArray, carried: FloatArray) -> None:
        super().__init__(inputs=block, scores=block, outputs=block)
        self._carried = np.array(carried, dtype=np.float64, copy=True)

    def __eq__(self, other: object) -> bool:
        # Written the way the real subclasses are, including the guard against
        # coercing a NotImplemented from the base, which is what makes the
        # two-different-subclasses test below a test of the pattern rather than
        # a test of one class.
        if not isinstance(other, CarryingResponse):
            return NotImplemented
        alike = super().__eq__(other)
        if alike is NotImplemented:
            return NotImplemented
        return bool(alike) and bool(np.array_equal(self._carried, other._carried))


class SeparatelyCarryingResponse(CarryingResponse):
    """A second subclass, so that two of them can be compared with each other."""

    __slots__ = ()


class TestAResponseIsNotEqualToASubclassOfItself:
    """A subclass exists because it carries something more, and equality has to
    know that.

    ``LayerResponse`` compares scores and outputs, which is right for two plain
    responses and wrong the moment a subclass appears. Two subclass responses
    carrying different extra state can hold identical answers, and calling them
    the same response is a claim that the thing distinguishing them does not
    count.

    The subtlety worth pinning is that a subclass cannot fix this alone. Under
    an ``isinstance`` check on the base, a subclass comparing against a plain
    response answers ``NotImplemented``, Python retries the other way round,
    the base says the outputs match, and the answer comes back True regardless
    of what the subclass wanted. So the rule lives on the base as
    ``type(self) is type(other)``, and both directions are tested here because
    only one of them exercises the reflected call.
    """

    def test_a_plain_response_still_equals_a_plain_response(self) -> None:
        block = np.array([[1.0, 2.0]])
        first = LayerResponse(inputs=block, scores=block, outputs=block)
        second = LayerResponse(inputs=block, scores=block, outputs=block)

        assert first == second

    def test_two_subclass_responses_carrying_alike_are_equal(self) -> None:
        block = np.array([[1.0, 2.0]])

        first = CarryingResponse(block, carried=np.ones((1, 2)))
        second = CarryingResponse(block, carried=np.ones((1, 2)))

        assert first == second

    def test_the_same_answers_carrying_differently_are_not(self) -> None:
        block = np.array([[1.0, 2.0]])

        first = CarryingResponse(block, carried=np.ones((1, 2)))
        second = CarryingResponse(block, carried=np.zeros((1, 2)))

        assert first != second

    def test_a_subclass_is_not_equal_to_the_base(self) -> None:
        block = np.array([[1.0, 2.0]])
        plain = LayerResponse(inputs=block, scores=block, outputs=block)
        carrying = CarryingResponse(block, carried=np.ones((1, 2)))

        assert carrying != plain

    def test_the_base_is_not_equal_to_a_subclass_either(self) -> None:
        """The reflected call, which is where an isinstance rule would leak."""
        block = np.array([[1.0, 2.0]])
        plain = LayerResponse(inputs=block, scores=block, outputs=block)
        carrying = CarryingResponse(block, carried=np.ones((1, 2)))

        assert plain != carrying

    def test_two_different_subclasses_are_not_equal(self) -> None:
        block = np.array([[1.0, 2.0]])
        carried = np.ones((1, 2))

        first = CarryingResponse(block, carried=carried)
        second = SeparatelyCarryingResponse(block, carried=carried)

        assert first != second


class TestLayersChain:
    """Two layers joined, which is the point of the shape vocabulary."""

    def test_one_layer_feeds_the_next_when_the_shapes_follow(self) -> None:
        first = counting_layer()
        second = DenseLayer([Neuron([1.0, 1.0, 1.0], bias=0.0, activation=Identity())])

        assert second.shape.follows(first.shape)

        hidden = first.respond_to(TWO_ROWS).outputs
        answer = second.respond_to(hidden)

        assert answer.outputs.shape == (2, 1)
        assert np.allclose(answer.outputs, [[6.0], [14.0]])

    def test_a_mismatched_join_is_refused_by_the_shapes_before_any_data(
        self,
    ) -> None:
        """The whole goal, stated as a test. No rows are involved."""
        first = counting_layer()
        second = DenseLayer([Neuron([1.0, 1.0], bias=0.0, activation=Identity())])

        assert not second.shape.follows(first.shape)

    def test_a_mismatched_join_also_refuses_at_the_block(self) -> None:
        first = counting_layer()
        second = DenseLayer([Neuron([1.0, 1.0], bias=0.0, activation=Identity())])

        with pytest.raises(ShapeMismatchError):
            second.respond_to(first.respond_to(TWO_ROWS).outputs)


class TestTheTwoRoutesAgree:
    """The fast matrix multiply against the neuron-at-a-time reading.

    A fast path and a slow path with nothing between them are two
    implementations rather than one calculation read two ways, so this is the
    test that justifies the matrix existing at all. It is ``approx`` and not
    exact on purpose: BLAS sums the products in a different order than a Python
    loop does, so the two routes differ in the last bits.
    """

    def test_the_blocks_equal_the_neurons_asked_one_at_a_time(self) -> None:
        layer = counting_layer()
        block = layer.respond_to(TWO_ROWS)

        for row_index, row in enumerate(TWO_ROWS):
            one_by_one = layer.neuron_responses(row)
            for neuron_index, answer in enumerate(one_by_one):
                assert block.scores[row_index][neuron_index] == pytest.approx(
                    answer.score
                )
                assert block.outputs[row_index][neuron_index] == pytest.approx(
                    answer.output
                )

    def test_they_agree_on_a_wide_layer_of_mixed_signs(self) -> None:
        """A size where the two routes genuinely reassociate differently."""
        generator = np.random.default_rng(11)
        layer = DenseLayer(
            [
                Neuron(
                    generator.normal(size=12),
                    bias=float(generator.normal()),
                    activation=RectifiedLinear(),
                )
                for _ in range(9)
            ]
        )
        rows = generator.normal(size=(7, 12))
        block = layer.respond_to(rows)

        for row_index, row in enumerate(rows):
            for neuron_index, answer in enumerate(layer.neuron_responses(row)):
                assert block.scores[row_index][neuron_index] == pytest.approx(
                    answer.score
                )

    def test_the_observed_route_answers_one_response_per_neuron_in_order(
        self,
    ) -> None:
        answers = counting_layer().neuron_responses([1.0, 2.0])

        assert len(answers) == 3
        assert [answer.output for answer in answers] == pytest.approx([1.0, 2.0, 3.0])

    def test_the_observed_route_still_refuses_a_bad_row(self) -> None:
        with pytest.raises(NonEqualArrayLengthError):
            counting_layer().neuron_responses([1.0])


class TestMixedActivations:
    """The column-by-column branch, which a uniform layer never reaches."""

    def test_each_neuron_bends_with_its_own_activation(self) -> None:
        layer = DenseLayer(
            [
                Neuron([1.0, 1.0], bias=-10.0, activation=Identity()),
                Neuron([1.0, 1.0], bias=-10.0, activation=RectifiedLinear()),
            ]
        )
        response = layer.respond_to(TWO_ROWS)

        assert np.allclose(response.scores, [[-7.0, -7.0], [-3.0, -3.0]])
        assert np.allclose(response.outputs, [[-7.0, 0.0], [-3.0, 0.0]])

    def test_a_mixed_layer_agrees_with_its_neurons(self) -> None:
        layer = DenseLayer(
            [
                Neuron([1.0, 1.0], bias=0.0, activation=Sigmoid()),
                Neuron([1.0, 1.0], bias=0.0, activation=Identity()),
            ]
        )
        block = layer.respond_to(TWO_ROWS)

        for row_index, row in enumerate(TWO_ROWS):
            for neuron_index, answer in enumerate(layer.neuron_responses(row)):
                assert block.outputs[row_index][neuron_index] == pytest.approx(
                    answer.output
                )


class TestTheGuardTheMatmulRemoved:
    """A non-finite entry used to be refused by a neuron, incidentally."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_entry_is_refused_at_the_layer(self, bad: float) -> None:
        with pytest.raises(InvalidValuesError):
            counting_layer().respond_to(np.array([[1.0, bad]]))

    def test_a_poisoned_row_never_reaches_the_answer(self) -> None:
        """Without the guard this returned a block of nan and raised nothing."""
        with pytest.raises(InvalidValuesError):
            counting_layer().respond_to(np.array([[1.0, 2.0], [3.0, float("nan")]]))


class TestAlreadyChecked:
    def test_it_freezes_without_copying(self) -> None:
        scores = np.zeros((2, 2))
        response = LayerResponse.already_checked(
            inputs=np.zeros((2, 2)), scores=scores, outputs=np.ones((2, 2))
        )

        assert response.scores is scores
        assert not response.scores.flags.writeable

    def test_the_checking_constructor_still_copies(self) -> None:
        scores = np.zeros((2, 2))
        response = LayerResponse(
            inputs=np.zeros((2, 2)), scores=scores, outputs=np.ones((2, 2))
        )

        assert response.scores is not scores
        assert scores.flags.writeable


class TestPositionalAccess:
    """Position is the key here, so the collections answer by it.

    The library's rule is that a collection does not hand out its container,
    not that it refuses to be asked about its members. ``Coefficients`` indexes
    by feature name for exactly this reason; a layer's neurons and a stack's
    layers have no names, and their position is what identifies them.
    """

    def test_a_layer_answers_for_a_neuron_by_position(self) -> None:
        layer = counting_layer()

        assert layer[0].n_inputs == 2
        assert np.allclose(layer[2].weights.values, [1.0, 1.0])

    def test_the_position_is_the_column_that_neuron_answers_in(self) -> None:
        layer = counting_layer()
        outputs = layer.respond_to(TWO_ROWS).outputs

        for index in range(len(layer)):
            alone = layer[index].respond_to(TWO_ROWS[0])
            assert outputs[0][index] == pytest.approx(alone.output)

    def test_it_counts_from_the_end_too(self) -> None:
        layer = counting_layer()

        assert layer[-1] is layer[2]

    def test_it_still_hands_out_no_container(self) -> None:
        assert not hasattr(counting_layer(), "neurons")


class TestAResponseKnowsWhatItsLayerRead:
    """The block a layer read belongs to the record of what it did.

    Without it a backward pass has to rebuild the list of what each layer was
    handed, by shifting the outputs down one and pushing the caller's block on
    the front. That shift reconstructs something the forward pass knew and
    discarded, which is why it lives here instead.
    """

    def test_the_first_layer_read_the_caller_s_block(self) -> None:
        response = counting_layer().respond_to(TWO_ROWS)

        assert np.allclose(response.inputs, TWO_ROWS)

    def test_a_later_layer_read_what_the_one_below_answered(self) -> None:
        from oop_ml.core.network.stack import LayerStack

        first = counting_layer()
        second = DenseLayer([Neuron([1.0, 1.0, 1.0], bias=0.0, activation=Identity())])
        forward = LayerStack([first, second]).respond_to(TWO_ROWS)

        assert np.allclose(forward[1].inputs, forward[0].outputs)

    def test_it_is_frozen_like_the_rest(self) -> None:
        response = counting_layer().respond_to(TWO_ROWS)

        with pytest.raises(ValueError):
            response.inputs[0][0] = 99.0

    def test_a_block_whose_rows_disagree_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            LayerResponse(
                inputs=np.zeros((3, 2)),
                scores=np.zeros((2, 2)),
                outputs=np.zeros((2, 2)),
            )

    def test_the_walk_needs_no_shifted_list(self) -> None:
        """Layers and responses pair up directly, both reversed."""
        from oop_ml.core.network.stack import LayerStack

        stack = LayerStack(
            [
                counting_layer(),
                DenseLayer([Neuron([1.0, 1.0, 1.0], bias=0.0, activation=Identity())]),
            ]
        )
        forward = stack.respond_to(TWO_ROWS)

        paired = list(zip(reversed(stack), reversed(forward), strict=True))

        assert len(paired) == 2
        for layer, response in paired:
            assert response.inputs.shape[1] == layer.shape.n_inputs
            assert response.scores.shape[1] == layer.shape.n_outputs
