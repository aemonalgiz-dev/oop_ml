"""Spec for the impurity measures -- red until the three formulas land.

Three of these are worth reading rather than skimming.

The first is the concavity test. It is not checking a formula, it is checking
the property that makes splitting work at all: a strictly concave measure sits
above its chords, so any split into unequal children must show a positive gain.
An implementation that returned, say, ``max(p)`` would satisfy several of the
tests below and fail that one.

The second is the pair of splits that misclassification rate cannot tell apart.
Gini has to prefer the split producing a pure child, and the numbers are chosen
so that accuracy scores both at exactly 0.25.

The third is that ``0 * log2(0)`` has to be zero and not ``nan``. numpy will say
``nan``, a single ``nan`` anywhere makes every comparison against it false, and
a split search comparing gains would then silently never choose that candidate.
"""

import numpy as np
import pytest

from oop_ml.core.tree.impurity import (
    EntropyImpurity,
    GiniImpurity,
    Impurity,
    VarianceImpurity,
)

CLASSIFICATION_MEASURES = [GiniImpurity(), EntropyImpurity()]
EVERY_MEASURE = [*CLASSIFICATION_MEASURES, VarianceImpurity()]


def labels(zeros: int, ones: int) -> np.ndarray:
    """A class column holding this many of each class."""
    return np.concatenate([np.zeros(zeros), np.ones(ones)])


class TestGini:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([0, 0, 1, 1], 0.5),
            ([0, 0, 0, 1], 0.375),
            ([0, 0, 0, 0], 0.0),
            ([0, 1, 2], 1.0 - 3 * (1 / 3) ** 2),
        ],
    )
    def test_matches_the_formula(self, values, expected):
        assert GiniImpurity().of(np.array(values, dtype=float)) == (
            pytest.approx(expected)
        )

    def test_it_is_the_chance_two_draws_disagree(self):
        # The reading that explains the formula: with three quarters class 0,
        # two independent draws match with probability 0.75^2 + 0.25^2.
        values = labels(30, 10)

        assert GiniImpurity().of(values) == pytest.approx(1.0 - (0.75**2 + 0.25**2))

    def test_the_maximum_is_at_a_uniform_split(self):
        uniform = GiniImpurity().of(labels(50, 50))

        for zeros in (10, 30, 49, 51, 70, 90):
            assert GiniImpurity().of(labels(zeros, 100 - zeros)) <= uniform


class TestEntropy:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([0, 0, 1, 1], 1.0),
            ([0, 0, 0, 1], 0.8112781244591328),
            ([0, 0, 0, 0], 0.0),
            ([0, 1, 2, 3], 2.0),
        ],
    )
    def test_matches_the_formula(self, values, expected):
        assert EntropyImpurity().of(np.array(values, dtype=float)) == (
            pytest.approx(expected)
        )

    def test_a_class_with_no_rows_contributes_zero_not_nan(self):
        # 0 * log2(0) is the limit 0, and numpy says nan. A single nan makes
        # every gain comparison against it false, so the candidate is silently
        # never chosen and the failure looks like a bad tree rather than a bug.
        values = np.array([0.0, 0.0, 2.0, 2.0])

        result = EntropyImpurity().of(values)

        assert not np.isnan(result)
        assert result == pytest.approx(1.0)

    def test_a_pure_node_needs_no_bits(self):
        assert EntropyImpurity().of(labels(9, 0)) == pytest.approx(0.0)


class TestVariance:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([1.0, 2.0, 3.0, 4.0], 1.25),
            ([5.0, 5.0, 5.0], 0.0),
            ([10.0, 50.0], 400.0),
        ],
    )
    def test_matches_the_formula(self, values, expected):
        assert VarianceImpurity().of(np.array(values)) == pytest.approx(expected)

    def test_the_mean_is_what_minimises_it(self):
        # Why a regression leaf predicts a mean: no other constant does better.
        values = np.array([2.0, 4.0, 9.0, 13.0])
        at_the_mean = VarianceImpurity().of(values)

        for guess in (0.0, 3.0, 6.0, 7.5, 12.0):
            elsewhere = float(((values - guess) ** 2).mean())
            assert elsewhere >= at_the_mean - 1e-12


class TestEveryMeasure:
    @pytest.mark.parametrize("measure", EVERY_MEASURE)
    def test_a_pure_node_is_zero(self, measure):
        assert measure.of(np.full(7, 3.0)) == pytest.approx(0.0)

    @pytest.mark.parametrize("measure", EVERY_MEASURE)
    def test_an_empty_node_is_zero(self, measure):
        # Nothing in it to disagree. Defined rather than incidental, because a
        # nan here would travel straight into a gain comparison.
        assert measure.of(np.empty(0)) == pytest.approx(0.0)

    @pytest.mark.parametrize("measure", EVERY_MEASURE)
    def test_it_is_never_negative(self, measure):
        generator = np.random.default_rng(0)

        for _ in range(20):
            values = generator.integers(0, 4, size=12).astype(float)
            assert measure.of(values) >= 0.0

    @pytest.mark.parametrize("measure", EVERY_MEASURE)
    def test_it_returns_a_python_float(self, measure):
        # numpy reductions hand back float64, which is duck-compatible enough
        # that nothing complains until the value has travelled through a gain
        # and into a leaf. The base coerces once so no measure has to.
        assert type(measure.of(np.array([1.0, 2.0, 3.0, 4.0]))) is float
        assert type(measure.of(np.empty(0))) is float

    @pytest.mark.parametrize("measure", CLASSIFICATION_MEASURES)
    def test_it_ignores_how_the_rows_are_ordered(self, measure):
        values = labels(4, 6)
        shuffled = np.random.default_rng(1).permutation(values)

        assert measure.of(values) == pytest.approx(measure.of(shuffled))


class TestGain:
    @pytest.mark.parametrize("measure", CLASSIFICATION_MEASURES)
    def test_a_split_into_identical_children_buys_nothing(self, measure):
        parent = labels(40, 40)
        left, right = labels(20, 20), labels(20, 20)

        assert measure.gain(parent, left, right) == pytest.approx(0.0)

    @pytest.mark.parametrize("measure", CLASSIFICATION_MEASURES)
    def test_a_perfect_split_removes_all_of_it(self, measure):
        parent = labels(40, 40)

        assert measure.gain(parent, labels(40, 0), labels(0, 40)) == (
            pytest.approx(measure.of(parent))
        )

    def test_the_children_are_weighted_by_size(self):
        # Without the weighting, peeling one row into its own pure child would
        # score perfectly and the tree would grow one leaf per row.
        parent = labels(99, 1)
        peeled_left, peeled_right = labels(99, 0), labels(0, 1)

        gain = GiniImpurity().gain(parent, peeled_left, peeled_right)

        assert gain == pytest.approx(GiniImpurity().of(parent))
        assert gain < 0.02

    def test_gini_prefers_the_split_accuracy_cannot_see(self):
        # 400/400 at the parent. Both splits leave misclassification at 0.25,
        # so accuracy calls them equal; only split B produces a pure child.
        parent = labels(400, 400)

        split_a = GiniImpurity().gain(parent, labels(300, 100), labels(100, 300))
        split_b = GiniImpurity().gain(parent, labels(200, 400), labels(200, 0))

        assert split_a == pytest.approx(0.125)
        assert split_b == pytest.approx(1.0 / 6.0)
        assert split_b > split_a

    def test_a_split_accuracy_scores_at_zero_still_has_gain(self):
        # Parent 30% class 1, children at 10% and 40%. Misclassification rate
        # is 0.30 before and 0.30 after -- exactly nothing -- because both
        # children sit on the same straight piece of it.
        parent = labels(210, 90)
        left, right = labels(90, 10), labels(120, 80)

        assert GiniImpurity().gain(parent, left, right) == pytest.approx(0.04)

    @pytest.mark.parametrize("measure", EVERY_MEASURE)
    def test_gain_is_never_negative(self, measure):
        # The concavity property, stated as a test. A measure that failed this
        # would make splitting able to increase impurity.
        generator = np.random.default_rng(2)

        for _ in range(50):
            parent = generator.integers(0, 3, size=30).astype(float)
            cut = int(generator.integers(1, 29))
            assert measure.gain(parent, parent[:cut], parent[cut:]) >= -1e-12


class TestTheHierarchy:
    @pytest.mark.parametrize("measure", EVERY_MEASURE)
    def test_every_measure_is_an_impurity(self, measure):
        assert isinstance(measure, Impurity)

    def test_the_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Impurity()  # pyright: ignore[reportAbstractUsage]
