"""Spec for BootstrapSample -- red until ``draw`` lands.

A resample is the entire mechanism behind bagging, so the properties worth
pinning are structural rather than numerical: it is the same size as what it
drew from, it repeats some rows, it misses others, and drawn and missed together
account for every row exactly once.

The out-of-bag share is checked against ``1/e`` rather than against a
particular seed's answer. The convergence is the fact worth testing; which
34.5% a given seed happens to miss is not.
"""

import numpy as np
import pytest

from oop_ml.core.ensemble.bootstrap import BootstrapSample
from test.fixtures import OUT_OF_BAG_SHARE, OUT_OF_BAG_TOLERANCE

ROW_COUNTS = [1, 2, 5, 50, 200, 1000]


class TestDraw:
    """What a draw produces."""

    @pytest.mark.parametrize("n_rows", ROW_COUNTS)
    def test_draws_as_many_rows_as_it_drew_from(self, n_rows: int) -> None:
        sample = BootstrapSample.draw(n_rows, np.random.default_rng(0))

        assert len(sample) == n_rows
        assert sample.drawn.size == n_rows

    @pytest.mark.parametrize("n_rows", ROW_COUNTS)
    def test_every_drawn_position_is_a_real_row(self, n_rows: int) -> None:
        sample = BootstrapSample.draw(n_rows, np.random.default_rng(1))

        assert sample.drawn.min() >= 0
        assert sample.drawn.max() < n_rows

    def test_draws_with_replacement(self) -> None:
        """Some row must arrive twice, or nothing about bagging works.

        Sampling *without* replacement from a set to its own size returns a
        permutation, every member sees identical data, and the average is one
        model repeated. Fifty rows make the permutation outcome vanishingly
        unlikely rather than merely improbable.
        """
        sample = BootstrapSample.draw(50, np.random.default_rng(2))

        assert np.unique(sample.drawn).size < 50

    def test_the_generator_is_the_only_source_of_randomness(self) -> None:
        first = BootstrapSample.draw(100, np.random.default_rng(3))
        second = BootstrapSample.draw(100, np.random.default_rng(3))

        assert np.array_equal(first.drawn, second.drawn)

    def test_successive_draws_from_one_generator_differ(self) -> None:
        """The reason a generator is passed in rather than created inside.

        An ensemble seeds once and draws ``n_members`` times from that stream.
        A ``draw`` that seeded itself would hand every member the same rows.
        """
        generator = np.random.default_rng(4)
        first = BootstrapSample.draw(100, generator)
        second = BootstrapSample.draw(100, generator)

        assert not np.array_equal(first.drawn, second.drawn)


class TestBags:
    """What was drawn, and what was left behind."""

    @pytest.mark.parametrize("n_rows", ROW_COUNTS)
    def test_in_bag_and_out_of_bag_partition_the_rows(self, n_rows: int) -> None:
        sample = BootstrapSample.draw(n_rows, np.random.default_rng(5))

        assert sample.in_bag.size == n_rows
        assert sample.in_bag.sum() + sample.out_of_bag.size == n_rows

    def test_out_of_bag_holds_exactly_the_rows_never_drawn(self) -> None:
        sample = BootstrapSample.draw(200, np.random.default_rng(6))

        assert set(sample.out_of_bag.tolist()).isdisjoint(sample.drawn.tolist())

    def test_in_bag_holds_exactly_the_rows_drawn(self) -> None:
        sample = BootstrapSample.draw(200, np.random.default_rng(7))

        assert set(np.flatnonzero(sample.in_bag).tolist()) == set(sample.drawn.tolist())

    @pytest.mark.parametrize("seed", range(8))
    def test_out_of_bag_share_converges_on_one_over_e(self, seed: int) -> None:
        sample = BootstrapSample.draw(1000, np.random.default_rng(seed))

        assert sample.out_of_bag_share == pytest.approx(
            OUT_OF_BAG_SHARE, abs=OUT_OF_BAG_TOLERANCE
        )

    def test_a_single_row_is_always_drawn(self) -> None:
        """The degenerate case, where the estimate has nothing to say."""
        sample = BootstrapSample.draw(1, np.random.default_rng(9))

        assert sample.out_of_bag.size == 0
        assert sample.out_of_bag_share == 0.0
