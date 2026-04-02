"""Spec for LayerShape -- green from the start, since none of it is stubbed.

The one worth reading is the chain test. A network's whole shape agreement is
``follows`` asked once per join, so the test walks a three-layer chain and
asserts both that the real one holds and that widening a middle layer breaks it
at exactly one place. That is the property the whole vocabulary exists for, and
it is the reason a shape error can be refused at construction rather than
discovered part-way through a training run.
"""

import pytest

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.network.shape import LayerShape


class TestConstruction:
    def test_it_keeps_both_widths(self) -> None:
        shape = LayerShape(n_inputs=4, n_outputs=8)

        assert shape.n_inputs == 4
        assert shape.n_outputs == 8

    @pytest.mark.parametrize("width", [0, -1, -10])
    def test_a_width_below_one_is_refused(self, width: int) -> None:
        """A width of zero is an absent layer, not a degenerate one."""
        with pytest.raises(InvalidValuesError):
            LayerShape(n_inputs=width, n_outputs=3)

        with pytest.raises(InvalidValuesError):
            LayerShape(n_inputs=3, n_outputs=width)

    @pytest.mark.parametrize("width", [2.0, "4", None])
    def test_a_width_that_is_not_a_whole_number_is_refused(self, width: object) -> None:
        with pytest.raises(InvalidValuesError):
            LayerShape(n_inputs=width, n_outputs=3)  # type: ignore[arg-type]

    def test_a_boolean_is_not_a_width(self) -> None:
        """``True`` is an ``int`` in Python, and a width of one by accident."""
        with pytest.raises(InvalidValuesError):
            LayerShape(n_inputs=True, n_outputs=3)


class TestChaining:
    def test_a_shape_follows_one_that_answers_its_input_width(self) -> None:
        first = LayerShape(n_inputs=4, n_outputs=8)
        second = LayerShape(n_inputs=8, n_outputs=3)

        assert second.follows(first)

    def test_a_shape_does_not_follow_a_mismatched_one(self) -> None:
        first = LayerShape(n_inputs=4, n_outputs=8)
        second = LayerShape(n_inputs=5, n_outputs=3)

        assert not second.follows(first)

    def test_following_is_not_symmetric(self) -> None:
        """The join is directional, and a shape vocabulary has to say so."""
        first = LayerShape(n_inputs=4, n_outputs=8)
        second = LayerShape(n_inputs=8, n_outputs=3)

        assert second.follows(first)
        assert not first.follows(second)

    def test_a_whole_chain_holds_join_by_join(self) -> None:
        chain = [
            LayerShape(n_inputs=4, n_outputs=8),
            LayerShape(n_inputs=8, n_outputs=8),
            LayerShape(n_inputs=8, n_outputs=3),
        ]

        assert all(
            later.follows(earlier)
            for earlier, later in zip(chain, chain[1:], strict=False)
        )

    def test_widening_one_layer_breaks_exactly_one_join(self) -> None:
        """The error is local, which is what makes it worth reporting early."""
        chain = [
            LayerShape(n_inputs=4, n_outputs=8),
            LayerShape(n_inputs=8, n_outputs=16),
            LayerShape(n_inputs=8, n_outputs=3),
        ]

        broken = [
            index
            for index, (earlier, later) in enumerate(
                zip(chain, chain[1:], strict=False)
            )
            if not later.follows(earlier)
        ]

        assert broken == [1]

    def test_the_input_width_does_not_constrain_the_output_width(self) -> None:
        narrow = LayerShape(n_inputs=400, n_outputs=3)
        wide = LayerShape(n_inputs=3, n_outputs=400)

        assert narrow.n_outputs == 3
        assert wide.n_outputs == 400


class TestEquality:
    def test_two_alike_are_equal(self) -> None:
        assert LayerShape(4, 8) == LayerShape(4, 8)
        assert hash(LayerShape(4, 8)) == hash(LayerShape(4, 8))

    def test_the_widths_are_not_interchangeable(self) -> None:
        """The reason this is an object and not a pair of loose integers."""
        assert LayerShape(4, 8) != LayerShape(8, 4)

    def test_a_difference_in_either_width_alone_is_enough(self) -> None:
        """Comparing only (4,8) against (8,4) lets an __eq__ reading one field
        pass, and n_outputs is the half the next layer reads."""
        assert LayerShape(4, 8) != LayerShape(4, 9)
        assert LayerShape(4, 8) != LayerShape(5, 8)

    def test_it_defers_to_anything_that_is_not_a_shape(self) -> None:
        assert LayerShape(4, 8).__eq__((4, 8)) is NotImplemented
