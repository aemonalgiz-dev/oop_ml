"""Spec for UnigramSampler -- where word2vec's negatives come from.

The shares for counts ``1000, 10, 1`` at the three-quarters exponent are
``1000 ** 0.75 = 177.828``, ``10 ** 0.75 = 5.623`` and ``1`` over their sum
``184.451``: ``0.964``, ``0.030``, ``0.005``. The sampling module's docstring
gives the first as ``0.947``, which cannot be right, since the three must sum
to one; the number pinned here was computed, and the other two agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.embeddings.prediction.sampling import (
    UnigramSampler,
)

DOCSTRING_COUNTS = [1000, 10, 1]


class TestProbabilities:
    def test_the_docstring_counts_at_three_quarters(self) -> None:
        sampler = UnigramSampler(DOCSTRING_COUNTS, 0.75)

        assert np.allclose(sampler.probabilities, [0.964, 0.030, 0.005], atol=1e-3)

    def test_exponent_one_is_raw_frequency(self) -> None:
        sampler = UnigramSampler(DOCSTRING_COUNTS, 1.0)

        assert np.allclose(
            sampler.probabilities, [1000 / 1011, 10 / 1011, 1 / 1011], atol=1e-12
        )

    def test_flattening_lifts_the_rarest_word_about_five_times(self) -> None:
        """``0.00542 / 0.000989`` is 5.48: the docstring's "five times as often"."""
        flattened = UnigramSampler(DOCSTRING_COUNTS, 0.75).probability_of(2)
        raw = UnigramSampler(DOCSTRING_COUNTS, 1.0).probability_of(2)

        assert 5.0 < flattened / raw < 6.0

    def test_flattening_lowers_the_commonest_word_a_little(self) -> None:
        flattened = UnigramSampler(DOCSTRING_COUNTS, 0.75).probability_of(0)
        raw = UnigramSampler(DOCSTRING_COUNTS, 1.0).probability_of(0)

        assert 0.95 < flattened < raw

    def test_equal_counts_give_equal_shares_at_any_exponent(self) -> None:
        for exponent in (0.25, 0.75, 1.0):
            assert np.allclose(
                UnigramSampler([4, 4, 4, 4], exponent).probabilities, 0.25
            )

    def test_probabilities_sum_to_one_and_are_frozen(self) -> None:
        sampler = UnigramSampler([3, 1, 4, 1, 5], 0.75)

        assert sampler.probabilities.sum() == pytest.approx(1.0)
        assert not sampler.probabilities.flags.writeable

    def test_probability_of_reads_one_word(self) -> None:
        sampler = UnigramSampler([1, 3], 1.0)

        assert sampler.probability_of(0) == pytest.approx(0.25)
        assert sampler.probability_of(1) == pytest.approx(0.75)

    def test_n_words_counts_the_ids(self) -> None:
        assert UnigramSampler([3, 1, 4], 0.75).n_words == 3

    def test_the_array_protocol_answers_the_probabilities_without_aliasing(
        self,
    ) -> None:
        sampler = UnigramSampler([1, 3], 1.0)

        assert np.array_equal(np.asarray(sampler), sampler.probabilities)
        assert not np.shares_memory(np.array(sampler), sampler.probabilities)

    def test_repr_names_the_size(self) -> None:
        assert repr(UnigramSampler([3, 1, 4], 0.75)) == "UnigramSampler(n_words=3)"


class TestDraw:
    def test_draws_the_requested_count_of_valid_ids(self) -> None:
        sampler = UnigramSampler([3, 1, 4, 1, 5], 0.75)
        drawn = sampler.draw(np.random.default_rng(0), 50)

        assert drawn.shape == (50,)
        assert drawn.dtype == np.intp
        assert np.all((drawn >= 0) & (drawn < 5))

    def test_never_returns_the_excluded_id(self) -> None:
        """The excluded word takes 97% of the draws, so a collision is near-certain
        on every attempt and the redraw has to work."""
        sampler = UnigramSampler([1000, 1, 1], 0.75)
        drawn = sampler.draw(np.random.default_rng(1), 2000, excluding=0)

        assert not np.any(drawn == 0)
        assert drawn.shape == (2000,)

    def test_excluding_a_word_leaves_the_others_in_proportion(self) -> None:
        """Excluding word 0 from ``50, 30, 20`` at exponent 1 leaves 30 : 20."""
        sampler = UnigramSampler([50, 30, 20], 1.0)
        drawn = sampler.draw(np.random.default_rng(2), 20000, excluding=0)

        share_of_one = float(np.mean(drawn == 1))

        assert not np.any(drawn == 0)
        assert share_of_one == pytest.approx(0.6, abs=0.02)

    def test_empirical_frequencies_are_within_two_hundredths(self) -> None:
        sampler = UnigramSampler([50, 30, 20], 1.0)
        drawn = sampler.draw(np.random.default_rng(3), 20000)

        empirical = np.bincount(drawn, minlength=3) / 20000

        assert np.allclose(empirical, [0.5, 0.3, 0.2], atol=0.02)

    def test_empirical_frequencies_follow_the_flattened_shares(self) -> None:
        sampler = UnigramSampler(DOCSTRING_COUNTS, 0.75)
        drawn = sampler.draw(np.random.default_rng(4), 20000)

        empirical = np.bincount(drawn, minlength=3) / 20000

        assert np.allclose(empirical, sampler.probabilities, atol=0.02)

    def test_a_seed_reproduces_the_draws(self) -> None:
        sampler = UnigramSampler([3, 1, 4, 1, 5], 0.75)

        assert np.array_equal(
            sampler.draw(np.random.default_rng(7), 100, excluding=2),
            sampler.draw(np.random.default_rng(7), 100, excluding=2),
        )

    def test_a_single_word_is_always_drawn_when_nothing_is_excluded(self) -> None:
        drawn = UnigramSampler([5], 0.75).draw(np.random.default_rng(0), 10)

        assert np.all(drawn == 0)


class TestRefusals:
    def test_no_counts_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            UnigramSampler([], 0.75)

    @pytest.mark.parametrize("counts", [[0, 1], [3, -2]])
    def test_a_count_below_one_is_refused(self, counts: list[int]) -> None:
        with pytest.raises(InvalidValuesError):
            UnigramSampler(counts, 0.75)

    @pytest.mark.parametrize("exponent", [0.0, -0.5, 1.5])
    def test_an_exponent_outside_zero_one_is_refused(self, exponent: float) -> None:
        with pytest.raises(InvalidValuesError):
            UnigramSampler([1, 2], exponent)

    def test_the_default_exponent_is_word2vecs(self) -> None:
        assert np.array_equal(
            UnigramSampler([1000, 10, 1]).probabilities,
            UnigramSampler([1000, 10, 1], 0.75).probabilities,
        )

    @pytest.mark.parametrize("n_draws", [0, -1])
    def test_drawing_fewer_than_one_is_refused(self, n_draws: int) -> None:
        with pytest.raises(InvalidValuesError):
            UnigramSampler([1, 2], 0.75).draw(np.random.default_rng(0), n_draws)

    def test_excluding_the_only_word_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            UnigramSampler([5], 0.75).draw(np.random.default_rng(0), 3, excluding=0)
