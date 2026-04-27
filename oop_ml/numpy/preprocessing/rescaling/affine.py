"""Rescaling a column by a centre and a spread, which is every affine scaler.

The family, in one table
------------------------
Scaling input data looks like several unrelated recipes and is one. Every affine
scaler learns two numbers per feature and applies the same expression::

    scaled = (value - centre) / spread

What differs is only how those two are read off the training column:

===========================  ==================  ==========================
scaler                       centre              spread
===========================  ==================  ==========================
``Standardizer``             the mean            the standard deviation
``MinMaxScaler``             the smallest value  the range
``MaxAbsScaler``             zero                the largest magnitude
``RobustScaler``             the median          the interquartile range
``RootMeanSquareScaler``     zero                the root mean square
===========================  ==================  ==========================

So a subclass here supplies :meth:`FeatureScaler.centre_of` and
:meth:`FeatureScaler.spread_of` and nothing else. Everything around them, the
fit, the by-name matching, the round trip and every refusal, is written once.

Why ``Standardizer`` is in the table and not in this module
------------------------------------------------------------
It predates this base and is woven through the persistence format, the pipeline
steps, the benchmark and five examples. Folding it in would rename
``FeatureScaling.mean`` and ``FeatureScaling.standard_deviation`` to a centre and
a spread, and those names are written into saved documents. Changing a stored
format to remove a small duplication is the wrong trade, so it stays where it
is and this docstring records that the two vocabularies describe one idea.

What each one is actually for
------------------------------
They are not interchangeable and the choice is about the data rather than taste.

**Standardizing** assumes the spread is meaningfully summarised by a standard
deviation, which a heavy tail breaks. **Min-max** puts everything in ``[0, 1]``,
which is what an image pipeline or a bounded activation wants, and it is the
most outlier-sensitive of the family, since one wild value sets the range for
every other. **Max-abs** divides without centring, so a zero stays a zero and a
sparse column stays sparse, which matters when the zeros are structural rather
than measured. **Robust** uses the median and the interquartile range, so a
tenth of the column can be nonsense without moving either number. **Root mean
square** divides by magnitude without centring at all, which is the input-side
twin of :class:`~oop_ml.core.network.row_normalisation.RMSNormalization`.

Why a zero spread is refused rather than patched
--------------------------------------------------
A column whose spread is zero cannot be scaled, because the expression divides by
it. Established libraries substitute a spread of one and carry on, which turns a
column carrying no information into a column of zeros or of some constant, and
says nothing. This refuses, with :class:`~oop_ml.core.exceptions.AllSameValuesError`
and a message naming the number.

One case there is worth knowing, because it is not the obvious one.
:class:`MinMaxScaler` and :class:`Standardizer` only see a zero spread on a
genuinely constant column. :class:`RobustScaler` can see one on a column that is
not constant at all, because both quartiles can land inside a single repeated
value while the column still varies elsewhere.

How lopsided that has to be was measured rather than guessed, and the first
guess was wrong. "More than half identical" is nowhere near enough: a column of
eight needs seven of them, a column of twelve needs ten, and a column of a
hundred needs seventy six. It converges on three quarters, because that is when
the third quartile is still inside the repeated run. Six identical values out of
eight leaves quartiles of 5.0 and 5.25 and is accepted; seven leaves 5.0 and 5.0
and is refused.

That is a real data shape rather than a degenerate one, and the refusal is still
right, because there is no robust spread there to divide by.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np

from oop_ml.core.base.estimator import Transformer
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.exceptions import (
    InvalidValuesError,
    NotFittedError,
)
from oop_ml.core.preprocessing.affine_scalings import (
    AffineScaling,
    AffineScalings,
)
from oop_ml.core.types import FloatArray


class FeatureScaler(Transformer[Sequence[Feature]]):
    """Learn a centre and a spread per column, then rescale by them.

    A subclass supplies :meth:`centre_of` and :meth:`spread_of`, both reading one
    column and answering one number. Nothing else varies across the family.

    Notes
    -----
    ``transform`` accepts a *subset* of the fitted features and rejects unknown
    ones, which is the rule
    :class:`~oop_ml.numpy.preprocessing.standardization.standardizer.Standardizer`
    follows and for the same reason. Scaling a column the fit never saw would
    have to invent a centre and a spread for it.

    Matching is by name and never by position, so features may be supplied in
    any order.
    """

    LEARNED_STATE: ClassVar[tuple[str, ...]] = ("_scalings",)
    """The centre and spread per column, which is the whole of the fit.

    Declared on the family rather than on each of the four, since what a
    scaler learns is the same pair whichever rule chose it.
    """

    _scalings: AffineScalings | None = None

    def model_post_init(self, context: object) -> None:
        """Start with nothing learned, which ``_check_fitted`` already enforces."""
        self._scalings = None

    @property
    def scalings(self) -> AffineScalings:
        """What the fit learned, one scaling per column.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        if self._scalings is None:
            raise NotFittedError(f"{type(self).__name__} has learned nothing")
        return self._scalings

    @staticmethod
    @abstractmethod
    def centre_of(values: FloatArray) -> float:
        """What to subtract from this column before dividing.

        Zero for the scalers that deliberately do not centre, which is what
        keeps a structural zero at zero and a sparse column sparse.
        """

    @staticmethod
    @abstractmethod
    def spread_of(values: FloatArray) -> float:
        """What to divide this column's centred values by.

        Must be strictly positive for a column that can be scaled at all;
        :class:`AffineScaling` refuses anything else, by name.
        """

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Learn one centre and one spread per feature.

        A :class:`~oop_ml.core.data.feature_set.FeatureSet` is built first,
        because its constructor already refuses duplicate names and misaligned
        lengths. Constant columns are *not* refused there, since zero variance is
        a fitting rule rather than a structural one, so they are refused here by
        :class:`AffineScaling` instead, which is the object that knows a spread
        of zero cannot be divided by.

        Nothing is committed until every scaling has been built, so a fit that
        raises partway leaves the previous one intact.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If the features are not all the same length.
        AllSameValuesError
            If any feature has no spread under this scaler's reading of spread.
        """
        feature_set = FeatureSet(input_values)

        learned = AffineScalings(
            [
                AffineScaling(
                    name=feature.name,
                    centre=self.centre_of(feature.values),
                    spread=self.spread_of(feature.values),
                )
                for feature in feature_set
            ]
        )

        self._scalings = learned
        self._mark_fitted()

        return self

    def transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Rescale each supplied feature by what the fit learned for it.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonUniqueFeaturesError
            If two features share a name.
        InvalidValuesError
            If a feature was not among the fitted ones. A column the fit never
            saw has no centre and no spread, and guessing one would answer a
            question nobody asked.
        """
        scalings = self.scalings
        supplied = FeatureSet(input_values)

        unknown = [feature.name for feature in supplied if feature.name not in scalings]
        if unknown:
            raise InvalidValuesError(
                f"this scaler learned {list(scalings.names)} and was handed "
                f"{unknown}, which it has no scaling for"
            )

        return [
            Feature(feature.name, scalings[feature.name].scale(feature.values))
            for feature in supplied
        ]

    def inverse_transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Undo :meth:`transform`, so an answer can be read in original units.

        The round trip is exact to floating point rather than to the last bit,
        since it divides and then multiplies by the same number.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If a feature was not among the fitted ones.
        """
        scalings = self.scalings
        supplied = FeatureSet(input_values)

        unknown = [feature.name for feature in supplied if feature.name not in scalings]
        if unknown:
            raise InvalidValuesError(
                f"this scaler learned {list(scalings.names)} and was handed "
                f"{unknown}, which it has no scaling for"
            )

        return [
            Feature(feature.name, scalings[feature.name].restore(feature.values))
            for feature in supplied
        ]

    def __repr__(self) -> str:
        fitted = self._scalings.n_features if self._scalings is not None else None
        return f"{type(self).__name__}(n_features={fitted!r})"


class MinMaxScaler(FeatureScaler):
    """Squash each column into ``[0, 1]``.

    What an image pipeline wants, and what a bounded activation wants, because
    both have a range in mind rather than a spread.

    It is the most outlier-sensitive member of the family and that is worth
    stating plainly. One wild value sets the range for every other, so a single
    reading a thousand times too large compresses the entire remaining column
    into the first thousandth of the interval. Nothing about the result looks
    wrong.
    """

    @staticmethod
    def centre_of(values: FloatArray) -> float:
        """The smallest value, so that it lands on zero."""
        return float(np.min(values))

    @staticmethod
    def spread_of(values: FloatArray) -> float:
        """The range, so that the largest value lands on one."""
        return float(np.max(values) - np.min(values))


class MaxAbsScaler(FeatureScaler):
    """Divide each column by its largest magnitude, into ``[-1, 1]``.

    It does not centre, and that is the whole reason to reach for it. A zero
    stays a zero, so a column whose zeros are *structural* rather than measured
    keeps its meaning, and a sparse column stays sparse. Subtracting a mean from
    such a column replaces every absent value with a number, which is both a
    memory problem and a modelling lie.

    Sign is preserved too, which matters wherever the sign carries meaning of its
    own.
    """

    @staticmethod
    def centre_of(values: FloatArray) -> float:
        """Zero. This scaler deliberately does not move anything."""
        return 0.0

    @staticmethod
    def spread_of(values: FloatArray) -> float:
        """The largest magnitude, so the extreme value lands on one or minus one."""
        return float(np.max(np.abs(values)))


class RobustScaler(FeatureScaler):
    """Centre on the median and divide by the interquartile range.

    Both statistics ignore the tails entirely, so a tenth of the column can be
    nonsense without moving either number. That is the property to want whenever
    the data has genuine outliers you do not wish to delete, which is most
    measured data.

    The failure mode it does *not* share with the others is worth naming. A
    column can vary and still have an interquartile range of zero, once enough
    of it is a single repeated value for both quartiles to fall inside the run.
    Measured, that is about three quarters of the column and not merely half:
    seven identical values out of eight, or seventy six out of a hundred. This
    scaler refuses such a column where the others accept it, which is correct
    rather than a limitation, since there is no robust spread there to divide
    by.
    """

    @staticmethod
    def centre_of(values: FloatArray) -> float:
        """The median, which a wild value cannot move."""
        return float(np.median(values))

    @staticmethod
    def spread_of(values: FloatArray) -> float:
        """The interquartile range, the distance between the quartiles."""
        first, third = np.quantile(values, [0.25, 0.75])
        return float(third - first)


class RootMeanSquareScaler(FeatureScaler):
    """Divide each column by its root mean square, without centring it.

    The input-side twin of
    :class:`~oop_ml.core.network.row_normalisation.RMSNormalization`, and the
    argument is the same one. Where standardizing removes the level and then the
    magnitude, this removes only the magnitude, on the view that the level is
    information rather than nuisance.

    It agrees with standardizing exactly on a column that is already centred,
    since the root mean square of a zero-mean column *is* its standard
    deviation. On anything else the two differ, and the difference is precisely
    the level that this one keeps.
    """

    @staticmethod
    def centre_of(values: FloatArray) -> float:
        """Zero. The level is kept rather than removed."""
        return 0.0

    @staticmethod
    def spread_of(values: FloatArray) -> float:
        """The root mean square, which is magnitude measured about zero."""
        return float(np.sqrt(np.mean(values**2)))
