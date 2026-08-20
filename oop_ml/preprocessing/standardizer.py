"""Rescale every feature to mean 0 and standard deviation 1.

Theory
------
Least squares does not care about units. Measure a predictor in metres or in
kilometres and OLS simply divides the coefficient by a thousand; the fitted
plane, the residuals, and R^2 are identical. That property is called scale
equivariance, and it is why nothing so far in this library has needed scaling.

Every method that goes *beyond* plain least squares loses it.

* **Ridge and lasso** penalise the size of the coefficients, and coefficient
  size depends on units. The same data with one column rescaled gives a
  genuinely different answer::

      x2 in original units   ->  b = (1.4537, 2.0093)
      x2 multiplied by 1000  ->  b = (1.6364, 2.8182)

  After standardizing, that change of unit becomes invisible and both give
  ``(1.5384, 1.7176)``. Without it, the penalty falls hardest on whichever
  feature happens to be measured in small units, which is an artifact of
  bookkeeping rather than of the data.

* **Gradient descent** takes one step size in every direction. Mixed scales turn
  the error bowl into a long narrow valley, and the step must be small enough
  for the steepest direction while the shallow one crawls. For the usual fixture
  the condition number of ``X.T X`` goes from 48.7 to 1.7e7 when a single column
  is scaled by a thousand, which is roughly the factor by which convergence
  slows down.

* **Comparing coefficients** to each other only means anything on a common
  scale. On standardized features a coefficient answers "how much does the
  target move per standard deviation of this predictor", which is comparable
  across predictors; in raw units it is not.

Mathematics
-----------
For each column independently::

    z = (x - mean(x)) / standard_deviation(x)

The result has mean 0 and standard deviation exactly 1, provided the spread we
use is the population one, dividing by ``n`` rather than by ``n - 1``, which is
what ``Column.standard_deviation`` computes.

Fit and transform are separate on purpose
-----------------------------------------
The statistics belong to the *training* data. Scoring on held-out data must
reuse those numbers, not recompute them:

* recomputing means the test set is centred using its own mean, which is
  information the model would never have had at training time. That is data
  leakage, and it silently flatters every score you compute afterwards;
* the columns would also be rescaled differently from training, so the fitted
  coefficients would no longer correspond to the numbers being fed in.

Hence ``fit`` learns and stores, ``transform`` applies what was stored, and
``fit_transform`` is a convenience for the training set alone.

Interpreting coefficients afterwards
------------------------------------
A model fitted on standardized features has standardized coefficients. Back on
the original scale::

    b_original_j = b_standardized_j / standard_deviation_j

    intercept_original = intercept_standardized
                         - sum_j (b_standardized_j * mean_j / standard_deviation_j)

Worth internalizing rather than memorizing: dividing by the spread undoes the
rescaling, and the intercept absorbs the recentring of every column at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from pydantic import PrivateAttr

from oop_ml.core.base import Transformer
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.feature import Feature
from oop_ml.core.feature_set import FeatureSet
from oop_ml.preprocessing.scaling import FeatureScaling, FeatureScalings


class Standardizer(Transformer[Sequence[Feature]]):
    """Centre each feature at zero and rescale it to unit standard deviation.

    Learns one mean and one standard deviation per feature during ``fit``, and
    applies exactly those to every later ``transform``.
    """

    _scalings: FeatureScalings | None = PrivateAttr(default=None)

    @property
    def scalings_(self) -> FeatureScalings:
        """The statistics learned per feature (available after ``fit``).

        You can read one by name, as in ``standardizer.scalings_["age"].mean``,
        or iterate the
        :class:`~oop_ml.preprocessing.scaling.FeatureScaling` objects directly.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._scalings is not None
        return self._scalings

    @staticmethod
    def _compute_scaling(feature: Feature) -> FeatureScaling:
        """The centre and spread learned from one feature.

        The statistics come off the feature's :class:`~oop_ml.core.column.Column`, which
        already owns them, so the population against sample choice gets made in
        one place rather than being re-decided by every caller of ``np.std``.

        Raises
        ------
        AllSameValuesError
            If the feature is constant, via ``FeatureScaling``: a zero spread
            has nothing to rescale by and would divide by zero.
        """
        return FeatureScaling(
            feature.name,
            feature.column.mean,
            feature.column.standard_deviation,
        )

    @staticmethod
    def _validated_feature_set(input_values: Sequence[Feature]) -> FeatureSet:
        """Return the inputs as a ``FeatureSet``, which enforces every rule.

        The constructor rejects an empty sequence, duplicate names and
        misaligned lengths, while ``check_columns_vary`` adds the rule that is
        specific to fitting, since a constant column has zero spread and cannot
        be standardized at all.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_columns_vary()

        return feature_set

    def _check_names_were_learned(self, input_values: Sequence[Feature]) -> None:
        """Raise unless every supplied feature has a scaling from ``fit``.

        Extra features are the error; a subset is fine, since transforming one
        column of a held-out set is legitimate.

        Raises
        ------
        InvalidValuesError
            If a supplied feature was never seen during ``fit``.
        """
        learned_names = {scaling.name for scaling in self.scalings_}
        unknown_names = {feature.name for feature in input_values} - learned_names

        if unknown_names:
            raise InvalidValuesError(
                f"never learned a scaling for {', '.join(sorted(unknown_names))}; "
                f"this standardizer knows {', '.join(sorted(learned_names))}"
            )

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Learn each feature's mean and standard deviation.

        Build a :class:`~oop_ml.core.feature_set.FeatureSet` first, because its
        constructor rejects duplicate names, misaligned lengths, and constant
        columns, and a constant column is exactly the one that cannot be
        standardized, its spread being zero. Then pair each feature's name with
        the statistics its :class:`~oop_ml.core.column.Column` already knows how
        to compute, and store the result as a
        :class:`~oop_ml.preprocessing.scaling.FeatureScalings`.

        Returns
        -------
        Self
            This standardizer, so calls can chain.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If the features are not all the same length.
        AllSameValuesError
            If any feature is constant.
        """
        feature_set = self._validated_feature_set(input_values)

        self._scalings = FeatureScalings(
            [self._compute_scaling(feature) for feature in feature_set]
        )

        self._mark_fitted()

        return self

    def transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Rescale each feature using the statistics learned during ``fit``.

        Match features to scalings by name and never by position, so that the
        caller may pass them in any order, which is the same contract
        ``predict`` follows. Ask
        each :class:`~oop_ml.preprocessing.scaling.FeatureScaling` to do the arithmetic
        rather than repeating the formula here, and return new
        :class:`~oop_ml.core.feature.Feature` objects keeping their original names.

        Returns
        -------
        list[Feature]
            One standardized feature per input, in the order supplied.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If a supplied feature was not seen during ``fit``.
        """
        self._check_fitted()
        self._check_names_were_learned(input_values)

        return [
            Feature(
                feature.name, self.scalings_[feature.name].standardize(feature.values)
            )
            for feature in input_values
        ]
