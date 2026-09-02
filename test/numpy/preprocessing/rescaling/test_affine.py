"""Spec for the affine scaling family, where the differences carry the file.

Four scalers are specified here and they share almost everything. Each learns a
centre and a spread per column and then answers ``(value - centre) / spread``,
so the fit, the by-name matching, the round trip and every refusal are one
implementation with four readings of two numbers laid over it. That shared half
is asserted of every member at once in
:class:`TestTheAffineFamilyContract`, because a claim about a family that is
only ever made of whichever member was written first is a claim nobody is
keeping.

The shared half is also the easy half. A round trip through a division and a
multiplication is hard to get wrong and, having got it right once, impossible to
get wrong for only one subclass. What is genuinely hard is choosing between the
four, and nothing in the shared contract helps with that at all, since all four
satisfy it completely while behaving entirely differently on the data anyone
would reach for them about.

So the second half of this file is the differences, and every one of them is a
measurement rather than an argument.

What was measured
-----------------
**Outliers.** The column ``[1, 2, 3, 4, 5]`` against ``[1, 2, 3, 4, 500]``, and
the question is how far the four honest values move when the fifth turns to
nonsense. Furthest movement is 1.1996 for ``Standardizer``, 0.7440 for
``MinMaxScaler``, and exactly 0.0 for ``RobustScaler``. The last of those is not
a small number, it is the number zero. A median and an interquartile range are
computed from the middle of the column and the wild value is not in the middle,
so it contributes nothing rather than contributing little.

**Structural zeros.** On ``[0, 0, 0, 4, 0, 8]``, ``MaxAbsScaler`` leaves all
four zeros at exactly ``0.0`` and ``Standardizer`` leaves none of them at zero,
turning every one into ``-0.6547``. That is the entire sparsity argument in one
fixture. Subtracting a mean from a column whose zeros are structural writes a
number into every absent value.

**Constant columns.** The family does not agree about these, which was a
surprise worth pinning. A constant column of sevens is refused by
``MinMaxScaler`` and ``RobustScaler`` and *accepted* by ``MaxAbsScaler`` and
``RootMeanSquareScaler``, both of which answer all-ones, because neither centres
and seven really is that column's magnitude. The column every member refuses is
the all-zero one, which has no magnitude either. So the family claim below is
made about zeros and the constant-sevens split is tested as a difference.

**Root mean square against standardizing.** On an already-centred column the two
agree to the last bit, because the root mean square of a zero-mean column is its
standard deviation by definition. On ``[1, 2, 3, 4, 5]`` they share not one
value, the root mean square scaler answering ``0.3015`` where standardizing
answers ``-1.4142``, and the whole of that difference is the level that one
removes and the other keeps.

**Sign against magnitude.** Every column above is non-negative, and on a
non-negative column a value and its magnitude are the same number, so four
readings coincide with four others there. Measured, a ``MinMaxScaler`` centring
on the smallest *magnitude* rather than the smallest value passed all 147 of the
claims this file first made, without one failure, and so did one dividing by the
range of the magnitudes. ``[-10, -2, 1, 2, 4]`` separates them: the centre is
-10.0 against a smallest magnitude of 1.0, the spread is 14.0 against a range of
magnitudes of 9.0, the largest magnitude is 10.0 and belongs to the negative
value where the largest value is 4.0, and the mean of -1.0 sits on the far side
of zero from the median of 1.0.

**Where a quartile falls between two values.** Both fixture columns hold five
values, so the quartile positions are 1 and 3 exactly and every interpolation
rule agrees on them, and the median lands on a value rather than between two.
Measured, a robust scaler using numpy's ``midpoint`` rule instead of the linear
one passed both readings. ``[2, 4, 5, 9, 10, 20]`` puts the positions at 1.25
and 3.75, so linear gives 4.25 and 9.75 against midpoint's 4.5 and 9.5, and the
median is 7.0, which the column does not contain.

**The robust zero-spread case.** A column can vary and still have an
interquartile range of zero, which is the failure mode no other member has. Both
quartiles land inside a single repeated run once about three quarters of the
column is that run, and three quarters is measured rather than assumed. At a
length of eight, six identical values leave quartiles of 5.0 and 5.25 and are
accepted while seven leave 5.0 and 5.0 and are refused. At a length of a hundred
the boundary is seventy six. ``MinMaxScaler`` fits the same eight-value column
happily, which is the cleanest statement of what robustness costs.

The oracles
-----------
Every expected number here is written from the definition. The centres and
spreads for the fixture column are hand arithmetic recorded in the
parametrization, the quantile oracle is the linear-interpolation rule
implemented in plain Python over a sorted list, and the standard deviation and
root mean square are sums over a comprehension. Nothing calls the
implementation to find out what the implementation should answer, and nothing
calls the numpy function the implementation calls.

That this spec discriminates, measured rather than assumed
----------------------------------------------------------
Breaks were installed over the real classes from outside the repository and the
whole file was run against each. Failing tests out of 169, none of them passing
clean. The two that once did are the reason two of the fixtures exist, and they
are the more useful entries in the table.

``MinMaxScaler`` centring on ``min(|value|)``, 3, **and 0 before
``TestColumnsThatCrossZero`` was added**. Dividing by
``max(|value|) - min(|value|)``, the same, 3 and **0** before. Both were wholly
invisible, because every column the file fitted was non-negative and on such a
column a value and its magnitude are the same number. Neither break touches a
round trip, a refusal, a name or a bound, so nothing structural could ever have
seen them; only a stated reading on a column that crosses zero does.

``RobustScaler`` reading its quartiles by numpy's ``midpoint`` rule, 2, and by
its ``lower`` rule, 3. Before the two tests added to ``TestTheRobustReadings``
those were 1 and 2, and in both cases the failure was the zero-spread boundary
rather than the reading test that claims to check the interpolation -- which it
could not, since the fixture column's quartile positions are 1 and 3 exactly and
every rule agrees there.

``RobustScaler`` centring on the mean instead of the median, 12, of which 8 are
older than the two fixtures above. Three of those eight are the outlier
measurements, two are the readings checked against the plain Python definitions,
and three fall out of the shared contract because the hand-worked centre of 21.0
is parametrized over the whole family. Every claim about the round trip, the
by-name matching and the refusals passes under it, because a mean is a perfectly
good centre for an invertible affine map. Nothing structural can see this break
at all.

``MaxAbsScaler`` centring on the mean, 12, of which 9 are older. It stops keeping
structural zeros at zero, stops preserving sign, stops reaching either bound of
``[-1, 1]``, and starts refusing the constant column of sevens it is supposed to
accept.

``MinMaxScaler`` dividing by the maximum rather than the range, 11, of which 8
are older. The largest value no longer lands on 1.0, the constant column of
sevens is no longer refused since its maximum is 7.0 rather than a range of zero,
and the outlier movement falls from 0.7440 to 0.5940. That last one is caught by
the measurement and *not* by the ordering assertion, which still holds at
0.0 < 0.5940 < 1.1996. Stating the number is what catches it and ranking the
three is not.

``AffineScaling.restore`` multiplying and adding in the wrong order, 4. That is
the quietest break here rather than the loudest, and the reason is worth keeping.
``(scaled + centre) * spread`` and ``scaled * spread + centre`` are the same
expression whenever the centre is zero, and two of the four members deliberately
have a centre of zero. So the round trip catches this break for ``MinMaxScaler``
and ``RobustScaler`` and is blind to it for ``MaxAbsScaler`` and
``RootMeanSquareScaler``, correctly, because for those two it is not a break. A
family claim discriminates only over the members whose numbers differ.

Six more were installed and all six are caught, which is recorded because a table
of only the breaks that were interesting is a table chosen after the fact.
``AffineScaling.scale`` dividing before centring, 15. ``FeatureScaler.fit``
committing one column at a time, 12. ``transform`` recomputing the statistics
from the column handed to it, 12. ``transform`` pairing by position, 8. A zero
spread substituted with a one, which is what established libraries do, 25.
``RootMeanSquareScaler`` centring before measuring magnitude, and the same class
copying ``MaxAbsScaler``'s reading, 8 each.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    AllSameValuesError,
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
    NotFittedError,
)
from oop_ml.numpy.preprocessing.rescaling.affine import (
    AffineScaling,
    AffineScalings,
    FeatureScaler,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    RootMeanSquareScaler,
)
from oop_ml.numpy.preprocessing.standardization.standardizer import Standardizer

# Two columns chosen so that every reading of them is exact arithmetic a person
# can check on paper. Odd lengths put the median on a value rather than between
# two, and the quartile positions land on values as well, so the interpolation
# never has to be reasoned about while the readings are being checked.
#
# TEMPERATURES: mean 22.0, smallest 18.0, range 10.0, largest magnitude 28.0,
# median 21.0, quartiles 20.0 and 23.0 so an interquartile range of 3.0, and a
# root mean square of sqrt(2478 / 5) = 22.262075.
TEMPERATURES = [18.0, 20.0, 21.0, 23.0, 28.0]

# HUMIDITIES: mean 50.0, smallest 30.0, range 40.0, largest magnitude 70.0,
# median 50.0, quartiles 45.0 and 55.0 so an interquartile range of 10.0, and a
# root mean square of sqrt(13350 / 5) = 51.672043.
HUMIDITIES = [30.0, 45.0, 50.0, 55.0, 70.0]

# A column that crosses zero, because both of the above are non-negative and on a
# non-negative column a value and its magnitude are the same number. Sorted it is
# [-10, -2, 1, 2, 4]: mean -1.0, smallest -10.0, range 14.0, largest magnitude
# 10.0 and it is the negative one, median 1.0, quartiles -2.0 and 2.0 so an
# interquartile range of 4.0, and a root mean square of sqrt(125 / 5) = 5.0.
#
# Every one of those differs from the same reading taken over the magnitudes,
# which is the whole reason the column is here. The smallest magnitude is 1.0
# rather than -10.0, the range of the magnitudes is 9.0 rather than 14.0, and the
# largest *value* is 4.0 rather than the largest magnitude of 10.0.
CROSSES_ZERO = [-10.0, -2.0, 1.0, 2.0, 4.0]

# Every concrete affine scaler. A claim about the family is parametrized over
# this rather than asserted of whichever member happened to be written first,
# and the ids are the class names so a failure names the member that broke it.
SCALERS = [
    pytest.param(MinMaxScaler, id="MinMaxScaler"),
    pytest.param(MaxAbsScaler, id="MaxAbsScaler"),
    pytest.param(RobustScaler, id="RobustScaler"),
    pytest.param(RootMeanSquareScaler, id="RootMeanSquareScaler"),
]

# The same four members paired with the centre and spread each one must read off
# TEMPERATURES. Hand arithmetic, recorded above where the fixture is defined,
# and deliberately not computed here from anything the implementation uses.
HAND_WORKED_READINGS = [
    pytest.param(MinMaxScaler, 18.0, 10.0, id="MinMaxScaler"),
    pytest.param(MaxAbsScaler, 0.0, 28.0, id="MaxAbsScaler"),
    pytest.param(RobustScaler, 21.0, 3.0, id="RobustScaler"),
    pytest.param(RootMeanSquareScaler, 0.0, 22.262075, id="RootMeanSquareScaler"),
]

# The same four paired with what each must read off CROSSES_ZERO. All four are
# exact integers, and none of them coincides with the reading of that column's
# magnitudes, which is what TEMPERATURES cannot say.
SIGNED_READINGS = [
    pytest.param(MinMaxScaler, -10.0, 14.0, id="MinMaxScaler"),
    pytest.param(MaxAbsScaler, 0.0, 10.0, id="MaxAbsScaler"),
    pytest.param(RobustScaler, 1.0, 4.0, id="RobustScaler"),
    pytest.param(RootMeanSquareScaler, 0.0, 5.0, id="RootMeanSquareScaler"),
]


def mean_of(values: Sequence[float]) -> float:
    """The arithmetic mean, from the definition and not from numpy."""
    return sum(values) / len(values)


def population_standard_deviation_of(values: Sequence[float]) -> float:
    """Root mean squared deviation about the mean, dividing by ``n``.

    The population reading rather than the sample one, which is what
    ``Column.standard_deviation`` computes and therefore what ``Standardizer``
    scales by.
    """
    average = mean_of(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def quantile_of(values: Sequence[float], fraction: float) -> float:
    """The linearly interpolated quantile, written out over a sorted list.

    This is the rule numpy documents as its default, implemented here from that
    description so that the robust scaler's quartiles are checked against
    something other than the call it makes itself.
    """
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    below = math.floor(position)
    above = math.ceil(position)

    return ordered[below] + (position - below) * (ordered[above] - ordered[below])


def scaled_by_hand(
    values: Sequence[float], centre: float, spread: float
) -> list[float]:
    """``(value - centre) / spread``, one value at a time in plain Python."""
    return [(value - centre) / spread for value in values]


def fitted(scaler: type[FeatureScaler]) -> FeatureScaler:
    """One of the four, fitted on both fixture columns."""
    return scaler().fit(training_features())


def training_features() -> list[Feature]:
    """The two fixture columns as features, in a fixed order."""
    return [Feature("temperature", TEMPERATURES), Feature("humidity", HUMIDITIES)]


def furthest_movement(
    scaler: Callable[[], FeatureScaler | Standardizer],
    honest: Sequence[float],
    corrupted: Sequence[float],
) -> float:
    """How far the honest values move when a wild value is appended.

    Both columns are fitted and transformed independently, and the comparison is
    over the leading positions the two share, which are the values that did not
    change. Whatever moves there is the outlier reaching a value it has nothing
    to do with.
    """
    before = scaler().fit_transform([Feature("reading", honest)])[0].values
    after = scaler().fit_transform([Feature("reading", corrupted)])[0].values

    shared = len(honest) - 1

    return float(np.max(np.abs(before[:shared] - after[:shared])))


class TestTheAffineFamilyContract:
    """What every affine scaler must do, asserted of every affine scaler.

    :class:`~oop_ml.numpy.preprocessing.rescaling.affine.FeatureScaler` owns the fit,
    the by-name matching, the subset rule, the round trip and every refusal, and
    a subclass supplies two functions of one column. So these claims belong to
    the base and are made here of all four subclasses at once, which is what
    stops the next scaler inheriting a contract nobody checks it against.
    """

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_scalings_raise_before_fit(self, scaler: type[FeatureScaler]) -> None:
        """Reading what was learned before anything was learned is the error."""
        with pytest.raises(NotFittedError):
            _ = scaler().scalings

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_transform_raises_before_fit(self, scaler: type[FeatureScaler]) -> None:
        """There is no centre and no spread yet, so there is nothing to apply."""
        with pytest.raises(NotFittedError):
            scaler().transform(training_features())

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_inverse_transform_raises_before_fit(
        self, scaler: type[FeatureScaler]
    ) -> None:
        """The same reason in the other direction."""
        with pytest.raises(NotFittedError):
            scaler().inverse_transform(training_features())

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_is_not_fitted_before_fit(self, scaler: type[FeatureScaler]) -> None:
        assert scaler().is_fitted is False

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_is_fitted_after_fit(self, scaler: type[FeatureScaler]) -> None:
        assert fitted(scaler).is_fitted is True

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_fit_returns_self(self, scaler: type[FeatureScaler]) -> None:
        """So that calls can chain, which every other fittable here allows."""
        instance = scaler()

        assert instance.fit(training_features()) is instance

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_learns_one_scaling_per_feature(self, scaler: type[FeatureScaler]) -> None:
        scalings = fitted(scaler).scalings

        assert scalings.n_features == 2
        assert scalings.names == ("temperature", "humidity")

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_the_scalings_are_iterable_and_addressable(
        self, scaler: type[FeatureScaler]
    ) -> None:
        """The collection is walked rather than handing out its container."""
        scalings = fitted(scaler).scalings

        assert [scaling.name for scaling in scalings] == ["temperature", "humidity"]
        assert scalings["humidity"].name == "humidity"
        assert "temperature" in scalings

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_the_round_trip_recovers_the_original_column(
        self, scaler: type[FeatureScaler]
    ) -> None:
        """Which is the one claim the whole family exists to keep.

        Exact to floating point rather than to the last bit, since it divides by
        a number and then multiplies by the same one.
        """
        instance = fitted(scaler)

        recovered = instance.inverse_transform(instance.transform(training_features()))

        np.testing.assert_allclose(recovered[0].values, TEMPERATURES)
        np.testing.assert_allclose(recovered[1].values, HUMIDITIES)

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_the_round_trip_holds_for_values_the_fit_never_saw(
        self, scaler: type[FeatureScaler]
    ) -> None:
        """The map is affine, so it inverts everywhere and not only on training rows."""
        instance = fitted(scaler)
        held_out = [Feature("temperature", [-40.0, 0.5, 999.0])]

        recovered = instance.inverse_transform(instance.transform(held_out))

        np.testing.assert_allclose(recovered[0].values, [-40.0, 0.5, 999.0])

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_names_are_preserved_through_transform(
        self, scaler: type[FeatureScaler]
    ) -> None:
        rescaled = fitted(scaler).transform(training_features())

        assert [feature.name for feature in rescaled] == ["temperature", "humidity"]

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_matches_features_by_name_not_position(
        self, scaler: type[FeatureScaler]
    ) -> None:
        """A caller may hand the columns over in any order.

        The failure this rules out is silent. Matching positionally would scale
        the temperatures by the humidity's numbers and answer a column of
        entirely plausible floats.
        """
        instance = fitted(scaler)

        in_order = instance.transform(training_features())
        reversed_order = instance.transform(list(reversed(training_features())))

        assert [feature.name for feature in reversed_order] == [
            "humidity",
            "temperature",
        ]
        np.testing.assert_allclose(reversed_order[1].values, in_order[0].values)
        np.testing.assert_allclose(reversed_order[0].values, in_order[1].values)

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_a_strict_subset_of_the_fitted_features_is_allowed(
        self, scaler: type[FeatureScaler]
    ) -> None:
        """Unlike predict, rescaling one column of a held-out set is legitimate."""
        rescaled = fitted(scaler).transform([Feature("humidity", HUMIDITIES)])

        assert [feature.name for feature in rescaled] == ["humidity"]

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_an_unknown_feature_is_refused(self, scaler: type[FeatureScaler]) -> None:
        """A column the fit never saw has no centre and no spread to apply."""
        with pytest.raises(InvalidValuesError):
            fitted(scaler).transform([Feature("pressure", TEMPERATURES)])

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_an_unknown_feature_is_refused_on_the_way_back(
        self, scaler: type[FeatureScaler]
    ) -> None:
        with pytest.raises(InvalidValuesError):
            fitted(scaler).inverse_transform([Feature("pressure", TEMPERATURES)])

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_duplicate_names_are_refused_by_transform_too(
        self, scaler: type[FeatureScaler]
    ) -> None:
        """The refusal ``transform`` documents, which ``fit`` documents separately.

        Two columns under one name would both be scaled by the one scaling that
        name has, and the answer would be a list with a repeated name in it.
        """
        with pytest.raises(NonUniqueFeaturesError):
            fitted(scaler).transform(
                [
                    Feature("temperature", TEMPERATURES),
                    Feature("temperature", HUMIDITIES),
                ]
            )

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_uses_the_training_statistics_and_not_the_new_column(
        self, scaler: type[FeatureScaler]
    ) -> None:
        """The heart of keeping fit and transform apart.

        Held-out rows are rescaled by the numbers the training column produced.
        Recomputing them here would be leakage, and the way to see it is that
        fitting this held-out column on its own gives a different answer.
        """
        held_out = [Feature("temperature", [100.0, 200.0, 300.0])]

        reused = fitted(scaler).transform(held_out)[0].values
        refitted = scaler().fit_transform(held_out)[0].values

        assert not np.allclose(reused, refitted)

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_a_column_of_zeros_is_refused(self, scaler: type[FeatureScaler]) -> None:
        """No centre and no reading of spread makes this column divisible.

        The constant column of *sevens* is deliberately not the fixture here,
        because the four members disagree about it. See
        :class:`TestWhatAConstantColumnSeparates`.
        """
        with pytest.raises(AllSameValuesError):
            scaler().fit([Feature("flat", [0.0, 0.0, 0.0])])

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_no_features_at_all_is_refused(self, scaler: type[FeatureScaler]) -> None:
        with pytest.raises(EmptyValuesError):
            scaler().fit([])

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_duplicate_names_are_refused(self, scaler: type[FeatureScaler]) -> None:
        """Two scalings under one name would leave the second silently unused."""
        with pytest.raises(NonUniqueFeaturesError):
            scaler().fit(
                [Feature("temperature", [1.0, 2.0]), Feature("temperature", [3.0, 4.0])]
            )

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_misaligned_lengths_are_refused(self, scaler: type[FeatureScaler]) -> None:
        with pytest.raises(NonEqualArrayLengthError):
            scaler().fit(
                [
                    Feature("temperature", [1.0, 2.0, 3.0]),
                    Feature("humidity", [4.0, 5.0]),
                ]
            )

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_fit_transform_matches_fit_then_transform(
        self, scaler: type[FeatureScaler]
    ) -> None:
        combined = scaler().fit_transform(training_features())
        separate = fitted(scaler).transform(training_features())

        for from_combined, from_separate in zip(combined, separate, strict=True):
            np.testing.assert_allclose(from_combined.values, from_separate.values)

    @pytest.mark.parametrize(
        ("scaler", "expected_centre", "expected_spread"), HAND_WORKED_READINGS
    )
    def test_reads_the_centre_and_spread_worked_out_by_hand(
        self,
        scaler: type[FeatureScaler],
        expected_centre: float,
        expected_spread: float,
    ) -> None:
        """One row of the family table, checked against arithmetic done on paper."""
        scaling = fitted(scaler).scalings["temperature"]

        assert scaling.centre == pytest.approx(expected_centre)
        assert scaling.spread == pytest.approx(expected_spread, abs=1e-6)

    @pytest.mark.parametrize(
        ("scaler", "expected_centre", "expected_spread"), HAND_WORKED_READINGS
    )
    def test_applies_the_expression_the_family_shares(
        self,
        scaler: type[FeatureScaler],
        expected_centre: float,
        expected_spread: float,
    ) -> None:
        """Every member answers ``(value - centre) / spread`` and nothing else."""
        rescaled = fitted(scaler).transform(training_features())

        np.testing.assert_allclose(
            rescaled[0].values,
            scaled_by_hand(TEMPERATURES, expected_centre, expected_spread),
            atol=1e-6,
        )


class TestTheScalingsThemselves:
    """The two value objects, which the family contract reaches only indirectly.

    Everything above goes through a fitted scaler, so a scaling is only ever
    seen through the numbers it answers with. These are the claims the objects
    make on their own account, including the one refusal
    :class:`~oop_ml.numpy.preprocessing.rescaling.affine.AffineScalings` documents and
    no fit can reach, since a duplicate name is refused a layer earlier.
    """

    def test_asking_for_a_name_that_was_never_learned_raises(self) -> None:
        """The ``KeyError`` the collection's docstring promises."""
        scalings = fitted(MinMaxScaler).scalings

        with pytest.raises(KeyError):
            scalings.scaling_for("pressure")

    def test_two_scalings_under_one_name_are_refused(self) -> None:
        """Unreachable through ``fit``, and still the collection's own rule.

        The second would shadow the first, so the collection would quietly
        describe one column twice and the other not at all.
        """
        with pytest.raises(NonUniqueFeaturesError):
            AffineScalings([AffineScaling("a", 0.0, 1.0), AffineScaling("a", 5.0, 2.0)])

    def test_a_spread_that_is_negative_is_refused_like_a_spread_of_zero(self) -> None:
        """Nothing here produces one, and the constructor is the guarantee.

        A subclass reading its spread backwards would hand over a negative
        number, and dividing by it would flip the column over rather than fail.
        """
        with pytest.raises(AllSameValuesError):
            AffineScaling("backwards", 0.0, -3.0)

    def test_a_scaling_compares_on_all_three_of_its_parts(self) -> None:
        one = AffineScaling("temperature", 18.0, 10.0)

        assert one == AffineScaling("temperature", 18.0, 10.0)
        assert one != AffineScaling("humidity", 18.0, 10.0)
        assert one != AffineScaling("temperature", 19.0, 10.0)
        assert one != AffineScaling("temperature", 18.0, 11.0)
        assert one != "temperature"

    def test_equal_scalings_hash_alike(self) -> None:
        """So a set of them collapses duplicates rather than keeping both."""
        one = AffineScaling("temperature", 18.0, 10.0)

        assert len({one, AffineScaling("temperature", 18.0, 10.0)}) == 1


class TestAFailedFitLeavesTheOldOneIntact:
    """Nothing is committed until every scaling has been built.

    The pattern the serving audit established, and it needs a real test because
    the broken version still runs. A scaler that assigned its scalings one
    column at a time would come back from a failed refit holding the new reading
    of the first column and the old reading of the second, which is a fitted
    model whose parts disagree about which data they came from.
    """

    def refit_that_fails(self, scaler: type[FeatureScaler]) -> FeatureScaler:
        """A fitted scaler asked to relearn from a column it must refuse."""
        instance = fitted(scaler)

        with pytest.raises(AllSameValuesError):
            instance.fit(
                [
                    Feature("temperature", [500.0, 600.0, 700.0]),
                    Feature("humidity", [0.0, 0.0, 0.0]),
                ]
            )

        return instance

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_it_is_still_fitted(self, scaler: type[FeatureScaler]) -> None:
        assert self.refit_that_fails(scaler).is_fitted is True

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_it_still_knows_both_columns(self, scaler: type[FeatureScaler]) -> None:
        assert self.refit_that_fails(scaler).scalings.names == (
            "temperature",
            "humidity",
        )

    @pytest.mark.parametrize(
        ("scaler", "expected_centre", "expected_spread"), HAND_WORKED_READINGS
    )
    def test_the_first_column_kept_its_original_reading(
        self,
        scaler: type[FeatureScaler],
        expected_centre: float,
        expected_spread: float,
    ) -> None:
        """The one the half-committing version would have overwritten.

        Temperature comes first, so a fit assigning as it goes would already
        have replaced its scaling by the time humidity raised.
        """
        scaling = self.refit_that_fails(scaler).scalings["temperature"]

        assert scaling.centre == pytest.approx(expected_centre)
        assert scaling.spread == pytest.approx(expected_spread, abs=1e-6)

    @pytest.mark.parametrize("scaler", SCALERS)
    def test_it_still_answers_what_it_answered_before(
        self, scaler: type[FeatureScaler]
    ) -> None:
        before = fitted(scaler).transform(training_features())
        after = self.refit_that_fails(scaler).transform(training_features())

        for earlier, later in zip(before, after, strict=True):
            np.testing.assert_allclose(earlier.values, later.values)


class TestOutlierSensitivity:
    """How far the honest values move when one reading turns to nonsense.

    The fixture is ``[1, 2, 3, 4, 5]`` against ``[1, 2, 3, 4, 500]``, and the
    number reported is the furthest any of the four surviving values travels.
    Measured, that is 1.1996 for ``Standardizer``, 0.7440 for ``MinMaxScaler``,
    and exactly 0.0 for ``RobustScaler``.

    Nothing about the corrupted result looks wrong under the first two. Every
    value is finite, every value is in range, and the whole honest column has
    been squeezed into a corner of it.
    """

    HONEST = [1.0, 2.0, 3.0, 4.0, 5.0]
    CORRUPTED = [1.0, 2.0, 3.0, 4.0, 500.0]

    def movement_under(
        self, scaler: Callable[[], FeatureScaler | Standardizer]
    ) -> float:
        return furthest_movement(scaler, self.HONEST, self.CORRUPTED)

    def test_the_robust_scaler_does_not_move_at_all(self) -> None:
        """Exactly zero, which is a stronger claim than nearly zero.

        The median and both quartiles of the corrupted column are the same three
        numbers as before, because a single value at the top of a sorted list of
        five is at neither the middle nor either quarter. So the scaling learned
        is bit-identical and the four honest values are bit-identical too.
        """
        assert self.movement_under(RobustScaler) == 0.0

    def test_the_min_max_scaler_moves_by_three_quarters_of_the_interval(self) -> None:
        """0.7440, which is nearly the whole of ``[0, 1]``.

        The value that had been at 0.75 is now at 0.006. One reading a hundred
        times too large has compressed the entire honest column into the first
        hundredth of the range.
        """
        assert self.movement_under(MinMaxScaler) == pytest.approx(0.7440, abs=1e-4)

    def test_standardizing_moves_furthest_of_the_three(self) -> None:
        """1.1996, and it moves further than min-max because it also recentres.

        Both the mean and the standard deviation of the corrupted column are
        wrong, so the honest values are shifted and squeezed rather than only
        squeezed.
        """
        assert self.movement_under(Standardizer) == pytest.approx(1.1996, abs=1e-4)

    def test_the_three_are_ordered_robust_then_min_max_then_standardizing(
        self,
    ) -> None:
        """The ordering is the point and it is what a choice between them rests on."""
        robust = self.movement_under(RobustScaler)
        min_max = self.movement_under(MinMaxScaler)
        standardizing = self.movement_under(Standardizer)

        assert robust < min_max < standardizing

    def test_the_outlier_itself_still_lands_where_it_belongs(self) -> None:
        """Robustness is about the other values and not about hiding this one.

        Under the robust scaling the wild reading comes back at 248.5, far
        outside anything the honest values reach, which is exactly what a caller
        wanting to find it would want.
        """
        rescaled = (
            RobustScaler().fit_transform([Feature("reading", self.CORRUPTED)])[0].values
        )

        assert float(rescaled[-1]) == pytest.approx(248.5)


class TestStructuralZeros:
    """Why a scaler that does not centre is worth having.

    On ``[0, 0, 0, 4, 0, 8]`` the zeros are not measurements, they are absences.
    ``MaxAbsScaler`` divides without centring so all four stay at exactly 0.0,
    and ``Standardizer`` subtracts a mean of 2.0 so all four become -0.6547.

    That is a memory problem and a modelling claim at once. A sparse matrix that
    has had a mean subtracted from it is no longer sparse, and every absent value
    now asserts something about itself.
    """

    SPARSE = [0.0, 0.0, 0.0, 4.0, 0.0, 8.0]

    def test_max_abs_keeps_every_zero_at_exactly_zero(self) -> None:
        rescaled = (
            MaxAbsScaler().fit_transform([Feature("counts", self.SPARSE)])[0].values
        )

        kept = [float(value) for value in rescaled if value == 0.0]

        assert len(kept) == 4

    def test_standardizing_keeps_none_of_them(self) -> None:
        """All four become the same non-zero number, which is -0.6547 here."""
        rescaled = (
            Standardizer().fit_transform([Feature("counts", self.SPARSE)])[0].values
        )

        assert not np.any(rescaled == 0.0)
        assert float(rescaled[0]) == pytest.approx(-0.6547, abs=1e-4)

    def test_the_root_mean_square_scaler_keeps_them_too(self) -> None:
        """The other member that does not centre, and for the same reason."""
        rescaled = (
            RootMeanSquareScaler()
            .fit_transform([Feature("counts", self.SPARSE)])[0]
            .values
        )

        assert len([value for value in rescaled if value == 0.0]) == 4

    def test_the_non_zero_values_are_divided_by_the_largest_magnitude(self) -> None:
        """Which is 8.0 here, so 4.0 becomes 0.5 and 8.0 becomes 1.0."""
        rescaled = (
            MaxAbsScaler().fit_transform([Feature("counts", self.SPARSE)])[0].values
        )

        np.testing.assert_allclose(rescaled, [0.0, 0.0, 0.0, 0.5, 0.0, 1.0])


class TestTheRanges:
    """What each scaler promises about where its answers land.

    ``MinMaxScaler`` lands exactly on 0.0 and 1.0 and nothing outside them.
    ``MaxAbsScaler`` lands inside ``[-1, 1]`` and touches one of the bounds, and
    it preserves sign, which is the property that makes it the one to reach for
    when the sign carries meaning of its own.
    """

    SIGNED = [-4.0, -1.0, 0.0, 2.0, 8.0]

    def test_min_max_puts_the_smallest_value_exactly_on_zero(self) -> None:
        """Exactly, since the smallest value is what gets subtracted from itself."""
        rescaled = MinMaxScaler().fit_transform(training_features())[0].values

        assert float(np.min(rescaled)) == 0.0

    def test_min_max_puts_the_largest_value_exactly_on_one(self) -> None:
        """Exactly, since the range divides itself."""
        rescaled = MinMaxScaler().fit_transform(training_features())[0].values

        assert float(np.max(rescaled)) == 1.0

    def test_min_max_puts_everything_else_between_them(self) -> None:
        rescaled = MinMaxScaler().fit_transform(training_features())[0].values

        assert np.all(rescaled >= 0.0)
        assert np.all(rescaled <= 1.0)

    def test_min_max_reproduces_the_hand_worked_positions(self) -> None:
        """``(value - 18) / 10`` on the temperature column."""
        rescaled = MinMaxScaler().fit_transform(training_features())[0].values

        np.testing.assert_allclose(rescaled, [0.0, 0.2, 0.3, 0.5, 1.0])

    def test_max_abs_lands_inside_minus_one_to_one(self) -> None:
        rescaled = MaxAbsScaler().fit_transform([Feature("signed", self.SIGNED)])[0]

        assert np.all(rescaled.values >= -1.0)
        assert np.all(rescaled.values <= 1.0)

    def test_max_abs_reaches_the_bound(self) -> None:
        """The extreme value lands exactly on 1.0 or on -1.0 and never short.

        Here the largest magnitude is 8.0 and it is positive, so the bound
        touched is 1.0.
        """
        rescaled = MaxAbsScaler().fit_transform([Feature("signed", self.SIGNED)])[0]

        assert float(np.max(np.abs(rescaled.values))) == 1.0

    def test_max_abs_reaches_the_lower_bound_when_the_extreme_is_negative(self) -> None:
        """The same claim on a column whose largest magnitude is below zero."""
        rescaled = MaxAbsScaler().fit_transform([Feature("signed", [-20.0, 1.0, 3.0])])[
            0
        ]

        assert float(np.min(rescaled.values)) == -1.0

    def test_max_abs_preserves_sign(self) -> None:
        """Dividing by a positive number cannot move a value across zero."""
        rescaled = MaxAbsScaler().fit_transform([Feature("signed", self.SIGNED)])[0]

        assert [float(np.sign(value)) for value in rescaled.values] == [
            float(np.sign(value)) for value in self.SIGNED
        ]

    def test_standardizing_does_not_promise_a_range_at_all(self) -> None:
        """Which is the distinction being drawn, so it is worth asserting.

        A standardized column reaches wherever the data reaches. On the
        temperature column the largest value comes back at 1.7617, well outside
        anything either bounded member would answer.
        """
        rescaled = Standardizer().fit_transform(training_features())[0].values

        assert float(np.max(rescaled)) > 1.0


class TestColumnsThatCrossZero:
    """Every reading again, on a column where sign and magnitude disagree.

    Every other column fitted in this file is non-negative, and on a
    non-negative column a value and its magnitude are the same number. That
    makes four readings indistinguishable from four others: the smallest value
    from the smallest magnitude, the range from the range of the magnitudes, the
    largest value from the largest magnitude, and the median from the median of
    the magnitudes.

    Measured rather than argued. A ``MinMaxScaler`` centring on the smallest
    *magnitude* instead of the smallest value, and one dividing by the range of
    the magnitudes instead of the range, each passed every other claim in this
    file without a single failure. Both are wrong by 10.0 and by 5.0 on the
    column below.

    So this column crosses zero, and its largest magnitude is the negative one
    so that ``MaxAbsScaler`` is separated from a scaler reading the largest
    value. Its mean and its median differ too, which separates the robust centre
    a second way.
    """

    @pytest.mark.parametrize(
        ("scaler", "expected_centre", "expected_spread"), SIGNED_READINGS
    )
    def test_reads_the_centre_and_spread_worked_out_by_hand(
        self,
        scaler: type[FeatureScaler],
        expected_centre: float,
        expected_spread: float,
    ) -> None:
        """The family table again, on the column that can tell sign from size."""
        instance = scaler().fit([Feature("signed", CROSSES_ZERO)])

        scaling = instance.scalings["signed"]

        assert scaling.centre == pytest.approx(expected_centre)
        assert scaling.spread == pytest.approx(expected_spread)

    @pytest.mark.parametrize(
        ("scaler", "expected_centre", "expected_spread"), SIGNED_READINGS
    )
    def test_applies_the_expression_the_family_shares(
        self,
        scaler: type[FeatureScaler],
        expected_centre: float,
        expected_spread: float,
    ) -> None:
        rescaled = scaler().fit_transform([Feature("signed", CROSSES_ZERO)])[0].values

        np.testing.assert_allclose(
            rescaled, scaled_by_hand(CROSSES_ZERO, expected_centre, expected_spread)
        )

    def test_min_max_still_lands_on_both_bounds(self) -> None:
        """The promise is ``[0, 1]``, and a negative value does not weaken it.

        The smallest value is -10.0 and it lands on 0.0, which a scaler
        subtracting the smallest magnitude of 1.0 could not manage.
        """
        rescaled = (
            MinMaxScaler().fit_transform([Feature("signed", CROSSES_ZERO)])[0].values
        )

        assert float(np.min(rescaled)) == 0.0
        assert float(np.max(rescaled)) == 1.0

    def test_max_abs_reaches_the_bound_through_the_negative_extreme(self) -> None:
        """The largest magnitude here is -10.0, so the bound touched is -1.0.

        A scaler dividing by the largest *value* would divide by 4.0 and answer
        -2.5, which is outside the range the class promises.
        """
        rescaled = (
            MaxAbsScaler().fit_transform([Feature("signed", CROSSES_ZERO)])[0].values
        )

        assert float(np.min(rescaled)) == -1.0
        assert np.all(rescaled >= -1.0)
        assert np.all(rescaled <= 1.0)

    def test_the_robust_centre_is_the_median_and_not_the_mean(self) -> None:
        """They differ in sign here, which is as separated as they get."""
        assert mean_of(CROSSES_ZERO) == pytest.approx(-1.0)

        centre = (
            RobustScaler()
            .fit([Feature("signed", CROSSES_ZERO)])
            .scalings["signed"]
            .centre
        )

        assert centre == pytest.approx(1.0)


class TestRootMeanSquareAgainstStandardizing:
    """Where the two agree completely and where they share nothing.

    The root mean square of a zero-mean column is its population standard
    deviation, by definition and not by coincidence, since both are the square
    root of the mean squared deviation from the same number. So on an
    already-centred column the two scalers are the same scaler.

    On anything else they differ by exactly the level, which one removes and the
    other treats as information.
    """

    CENTRED = [-2.0, -1.0, 0.0, 1.0, 2.0]

    def test_the_two_readings_of_spread_coincide_on_a_centred_column(self) -> None:
        """Checked against both definitions written out in plain Python."""
        root_mean_square = math.sqrt(
            sum(value**2 for value in self.CENTRED) / len(self.CENTRED)
        )

        assert population_standard_deviation_of(self.CENTRED) == pytest.approx(
            root_mean_square
        )

    def test_they_answer_identically_on_a_centred_column(self) -> None:
        """To the last bit, since the arithmetic performed is the same arithmetic."""
        by_magnitude = (
            RootMeanSquareScaler()
            .fit_transform([Feature("centred", self.CENTRED)])[0]
            .values
        )
        by_standardizing = (
            Standardizer().fit_transform([Feature("centred", self.CENTRED)])[0].values
        )

        np.testing.assert_array_equal(by_magnitude, by_standardizing)

    def test_they_share_no_value_on_a_column_with_a_level(self) -> None:
        """On ``[1, 2, 3, 4, 5]`` the two disagree everywhere.

        The root mean square scaler answers 0.3015 where standardizing answers
        -1.4142, and the gap is the mean of 3.0 that one subtracted.
        """
        uncentred = [1.0, 2.0, 3.0, 4.0, 5.0]

        by_magnitude = (
            RootMeanSquareScaler().fit_transform([Feature("raw", uncentred)])[0].values
        )
        by_standardizing = (
            Standardizer().fit_transform([Feature("raw", uncentred)])[0].values
        )

        assert not np.any(np.isclose(by_magnitude, by_standardizing))
        assert float(by_magnitude[0]) == pytest.approx(0.3015, abs=1e-4)
        assert float(by_standardizing[0]) == pytest.approx(-1.4142, abs=1e-4)

    def test_the_root_mean_square_scaler_keeps_the_level(self) -> None:
        """Its answers stay positive on a positive column, which is the whole point.

        Standardizing puts half of any column below zero by construction. This
        one does not, because it never moved anything.
        """
        rescaled = (
            RootMeanSquareScaler()
            .fit_transform([Feature("raw", [1.0, 2.0, 3.0, 4.0, 5.0])])[0]
            .values
        )

        assert np.all(rescaled > 0.0)

    def test_it_divides_by_the_hand_worked_root_mean_square(self) -> None:
        """sqrt(2478 / 5) = 22.262075 on the temperature column."""
        by_hand = math.sqrt(sum(value**2 for value in TEMPERATURES) / len(TEMPERATURES))

        spread = fitted(RootMeanSquareScaler).scalings["temperature"].spread

        assert spread == pytest.approx(by_hand)
        assert spread == pytest.approx(22.262075, abs=1e-6)


class TestWhatAConstantColumnSeparates:
    """The four do not agree about a constant column, which was worth pinning.

    A column of sevens has no range and no interquartile range, so
    ``MinMaxScaler`` and ``RobustScaler`` refuse it, as ``Standardizer`` does.
    It has a largest magnitude of 7.0 and a root mean square of 7.0, so the two
    members that do not centre accept it and answer all-ones. That is correct
    rather than a gap, since seven really is that column's magnitude and dividing
    by it really does say something.

    The column every member refuses is the all-zero one, which is why the family
    claim above uses that and not this.
    """

    CONSTANT = [7.0, 7.0, 7.0]

    @pytest.mark.parametrize(
        "scaler",
        [
            pytest.param(MinMaxScaler, id="MinMaxScaler"),
            pytest.param(RobustScaler, id="RobustScaler"),
        ],
    )
    def test_the_centring_members_refuse_it(self, scaler: type[FeatureScaler]) -> None:
        with pytest.raises(AllSameValuesError):
            scaler().fit([Feature("flat", self.CONSTANT)])

    def test_standardizing_refuses_it_too(self) -> None:
        """The fifth member of the table behaves like the other centring ones."""
        with pytest.raises(AllSameValuesError):
            Standardizer().fit([Feature("flat", self.CONSTANT)])

    @pytest.mark.parametrize(
        "scaler",
        [
            pytest.param(MaxAbsScaler, id="MaxAbsScaler"),
            pytest.param(RootMeanSquareScaler, id="RootMeanSquareScaler"),
        ],
    )
    def test_the_uncentred_members_accept_it_and_answer_all_ones(
        self, scaler: type[FeatureScaler]
    ) -> None:
        rescaled = scaler().fit_transform([Feature("flat", self.CONSTANT)])[0].values

        np.testing.assert_allclose(rescaled, [1.0, 1.0, 1.0])


class TestTheRobustZeroSpreadCase:
    """A column that varies and still has an interquartile range of zero.

    This is the failure mode no other member has, and how lopsided the column
    must be was measured rather than reasoned about. Both quartiles have to land
    inside a single repeated run, and at a length of eight that takes seven of
    the eight values. Six leaves quartiles of 5.0 and 5.25 and is accepted, seven
    leaves 5.0 and 5.0 and is refused, and the boundary at a length of a hundred
    is seventy six.

    The same eight-value column is fitted happily by ``MinMaxScaler``, which is
    the clearest statement of what robustness costs. Where one scaler sees a
    range of 1.0 the other sees no spread at all, and both are right about the
    question they are asking.
    """

    SEVEN_IDENTICAL = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0]
    SIX_IDENTICAL = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0]

    def test_the_refused_column_is_not_constant(self) -> None:
        """Which is what makes this case worth a test of its own.

        Every other zero-spread refusal in the library is a column with one
        distinct value. This column has two, and it varies by a fifth of its own
        magnitude.
        """
        assert len(set(self.SEVEN_IDENTICAL)) == 2

    def test_seven_identical_values_out_of_eight_are_refused(self) -> None:
        with pytest.raises(AllSameValuesError):
            RobustScaler().fit([Feature("lopsided", self.SEVEN_IDENTICAL)])

    def test_six_identical_values_out_of_eight_are_accepted(self) -> None:
        """The boundary, and it is one value wide."""
        scaler = RobustScaler().fit([Feature("lopsided", self.SIX_IDENTICAL)])

        assert scaler.scalings["lopsided"].spread == pytest.approx(0.25)

    @pytest.mark.parametrize(
        ("column", "expected_first", "expected_third"),
        [
            pytest.param(SEVEN_IDENTICAL, 5.0, 5.0, id="seven identical"),
            pytest.param(SIX_IDENTICAL, 5.0, 5.25, id="six identical"),
        ],
    )
    def test_the_quartiles_are_where_the_interpolation_rule_puts_them(
        self, column: list[float], expected_first: float, expected_third: float
    ) -> None:
        """Worked from the linear-interpolation definition over a sorted list.

        At a length of eight the quartile positions are 1.75 and 5.25. Seven
        identical values put both of those inside the run and leave a spread of
        zero, six put the upper one a quarter of the way from 5.0 to 6.0.
        """
        assert quantile_of(column, 0.25) == pytest.approx(expected_first)
        assert quantile_of(column, 0.75) == pytest.approx(expected_third)

    def test_min_max_accepts_the_column_the_robust_one_refuses(self) -> None:
        """Because a range is a question about the ends and there is one there."""
        rescaled = (
            MinMaxScaler()
            .fit_transform([Feature("lopsided", self.SEVEN_IDENTICAL)])[0]
            .values
        )

        np.testing.assert_allclose(rescaled, [0.0] * 7 + [1.0])

    def test_the_boundary_at_a_hundred_is_seventy_six(self) -> None:
        """Which is three quarters and not the half a first guess suggests.

        Seventy five repeated values leave quartiles of 5.0 and 5.25 and are
        accepted, seventy six leave 5.0 and 5.0 and are refused.
        """
        accepted = [5.0] * 75 + [6.0 + index for index in range(25)]
        refused = [5.0] * 76 + [6.0 + index for index in range(24)]

        RobustScaler().fit([Feature("lopsided", accepted)])

        with pytest.raises(AllSameValuesError):
            RobustScaler().fit([Feature("lopsided", refused)])

    def test_the_refusal_names_the_feature(self) -> None:
        """So a caller reading the message knows which column to look at."""
        with pytest.raises(AllSameValuesError, match="lopsided"):
            RobustScaler().fit([Feature("lopsided", self.SEVEN_IDENTICAL)])


class TestTheRobustReadings:
    """That the robust scaler really uses the median and the quartiles.

    Checked against the definitions written out in plain Python rather than
    against the numpy calls the implementation makes.
    """

    # Six values rather than five, because the fixture column is deliberately
    # arranged so that the quartile positions land on values and the median
    # lands on a value. That makes it silent about the two rules the robust
    # reading actually rests on: how a quartile between two values is
    # interpolated, and what the median of an even-length column is. Here the
    # positions are 1.25 and 3.75, giving 4.25 and 9.75, and the median is the
    # average of 5.0 and 9.0, which is a number the column does not contain.
    EVEN_LENGTH = [2.0, 4.0, 5.0, 9.0, 10.0, 20.0]

    def test_the_centre_is_the_median(self) -> None:
        """21.0 on the temperature column, which is its middle value."""
        by_hand = sorted(TEMPERATURES)[len(TEMPERATURES) // 2]

        assert fitted(RobustScaler).scalings["temperature"].centre == pytest.approx(
            by_hand
        )

    def test_the_spread_is_the_distance_between_the_quartiles(self) -> None:
        """23.0 minus 20.0, so 3.0."""
        by_hand = quantile_of(TEMPERATURES, 0.75) - quantile_of(TEMPERATURES, 0.25)

        assert fitted(RobustScaler).scalings["temperature"].spread == pytest.approx(
            by_hand
        )
        assert by_hand == pytest.approx(3.0)

    def test_the_centre_is_not_the_mean(self) -> None:
        """The distinction the whole class exists for.

        On this column the mean is 22.0 and the median is 21.0, so a scaler that
        reached for the wrong one is visible here as well as under an outlier.
        """
        assert mean_of(TEMPERATURES) == pytest.approx(22.0)

        assert fitted(RobustScaler).scalings["temperature"].centre == pytest.approx(
            21.0
        )

    def test_the_quartiles_interpolate_when_they_fall_between_two_values(self) -> None:
        """Which the fixture column deliberately cannot say.

        Measured: a robust scaler reaching for numpy's ``midpoint`` rule rather
        than the linear one passes every reading checked above, because at five
        values the quartile positions are 1 and 3 exactly and every rule agrees.
        At six the positions are 1.25 and 3.75 and the rules part, linear giving
        4.25 and 9.75 against midpoint's 4.5 and 9.5, so 5.5 against 5.0.
        """
        first = quantile_of(self.EVEN_LENGTH, 0.25)
        third = quantile_of(self.EVEN_LENGTH, 0.75)

        assert first == pytest.approx(4.25)
        assert third == pytest.approx(9.75)

        spread = (
            RobustScaler()
            .fit([Feature("uneven", self.EVEN_LENGTH)])
            .scalings["uneven"]
            .spread
        )

        assert spread == pytest.approx(third - first)
        assert spread == pytest.approx(5.5)

    def test_the_median_of_an_even_length_column_is_the_middle_pair(self) -> None:
        """7.0 here, and it is a number the column does not contain.

        Both fixture columns have an odd length, so their median lands on a
        value and a scaler taking either middle element instead of averaging the
        two would answer correctly on both.
        """
        centre = (
            RobustScaler()
            .fit([Feature("uneven", self.EVEN_LENGTH)])
            .scalings["uneven"]
            .centre
        )

        assert centre == pytest.approx(7.0)
        assert 7.0 not in self.EVEN_LENGTH
