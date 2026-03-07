"""Expand features into their powers and products.

Theory
------
This is not a new model. "Linear" in linear regression has always meant linear
in the coefficients rather than in the predictors, so ``x ** 2`` and ``x1 * x2``
are perfectly ordinary columns, and fitting::

    y = b0 + b1*x + b2*x**2 + b3*x**3

is still solving ``X.T X b = X.T y``. Nothing about the estimator changes. What
changes is the design matrix handed to it, which is why this belongs in
preprocessing beside :class:`~oop_ml.preprocessing.standardizer.Standardizer` rather
than in ``regression``.

That single observation buys curves out of a straight-line model, and it
generalises: any transform of the inputs is fair game. Polynomials are simply
the most common one.

Interaction terms
-----------------
With more than one predictor the expansion includes cross terms such as
``x1*x2``. These are worth more than the powers in many problems, because they
are the only way a linear model can express "the effect of x1 depends on the
level of x2". Without one, the model insists every predictor's effect is the
same regardless of the others, which is exactly what a partial effect means and
is sometimes exactly wrong.

Degree ``d`` over ``p`` features produces ``C(p + d, d) - 1`` columns (the minus
one drops the constant term, which is the model's intercept). Two features at
degree 2 gives five; three features at degree 3 gives nineteen. The count grows
fast, which is the practical reason to keep the degree low.

What goes wrong, measured
-------------------------
Ten noisy points from a gentle curve, fitted at rising degree, scored against
the true curve::

    degree  terms   train R2     test R2
         1      2    0.77178      0.7868
         2      3    0.94430      0.9907      <- the truth is quadratic
         3      4    0.95849      0.9774
         5      6    0.97672      0.9517
         7      8    0.99508      0.8854
         9     10    1.00000      0.8583

Train R^2 rises monotonically and reaches *exactly* 1.0 at degree 9, where ten
coefficients fit ten points perfectly. The curve then passes through every
training point and is useless between them: test R^2 has fallen from 0.9907 to
0.8583. The training score cannot see any of this happening, since it improves
the whole way down, and that is the entire argument for held-out evaluation.

Degree is therefore a complexity dial, and turning it up always improves the fit
you can measure while eventually destroying the one you care about. Pair it with
:class:`~oop_ml.regression.ridge_regression.RidgeRegression` or
:class:`~oop_ml.regression.lasso_regression.LassoRegression` to keep the extra
freedom in check, and with
:class:`~oop_ml.preprocessing.standardizer.Standardizer` because ``x ** 9`` on raw
units produces columns whose scales differ by orders of magnitude.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import combinations_with_replacement
from typing import Self

from pydantic import Field, PrivateAttr

from oop_ml.core.base.estimator import Transformer
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.preprocessing.polynomial import PolynomialTerm, PolynomialTerms


class PolynomialFeatures(Transformer[Sequence[Feature]]):
    """Expand features into every power and product up to ``degree``.

    Parameters
    ----------
    degree:
        Highest total degree to generate. Degree 1 returns the original features
        unchanged, which makes it a useful no-op baseline.
    include_interactions:
        Whether to generate cross terms such as ``x1*x2`` (default), or only
        pure powers such as ``x1^2``.
    """

    degree: int = Field(default=2, ge=1)
    include_interactions: bool = True

    _terms: PolynomialTerms | None = PrivateAttr(default=None)

    @property
    def terms(self) -> PolynomialTerms:
        """The expansion learned during ``fit`` (available after ``fit``).

        Fixing the terms at fit time is what makes ``transform`` reproducible,
        so that held-out data yields the same columns in the same order with the
        same names, rather than whatever the new data happens to suggest.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._terms is not None
        return self._terms

    @staticmethod
    def _validated_feature_set(input_values: Sequence[Feature]) -> FeatureSet:
        """Return the inputs as a ``FeatureSet``, which enforces every rule.

        The constructor rejects an empty sequence, duplicate names and
        misaligned lengths; ``check_columns_vary`` adds the fitting rule, since
        a constant column expands into constant columns that are collinear with
        the intercept.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_columns_vary()

        return feature_set

    def _check_names_were_learned(self, input_values: Sequence[Feature]) -> None:
        """Raise unless every feature the expansion needs was supplied.

        Unlike ``Standardizer``, a subset is *not* acceptable: a term such as
        ``x1*x2`` cannot be computed without both columns, so the expansion is
        all-or-nothing.

        Raises
        ------
        InvalidValuesError
            If a feature seen during ``fit`` is missing.
        """
        supplied_names = {feature.name for feature in input_values}
        missing_names = set(self.terms.source_feature_names) - supplied_names

        if missing_names:
            raise InvalidValuesError(
                f"the expansion needs {', '.join(sorted(missing_names))}; "
                f"got {', '.join(sorted(supplied_names)) or 'none'}"
            )

    @staticmethod
    def _term_for(repeated_names: Sequence[str]) -> PolynomialTerm:
        """One term from a multiset of feature names.

        A name appearing ``k`` times means that feature raised to the ``k``-th
        power, so counting occurrences *is* reading off the exponents::

            ("x1", "x1", "x2")  ->  {"x1": 2, "x2": 1}  ->  x1^2*x2

        ``collections.Counter`` does the counting and is already a
        ``Mapping[str, int]``, so it can be handed straight to
        :class:`~oop_ml.preprocessing.polynomial.PolynomialTerm`.
        """
        return PolynomialTerm(Counter(repeated_names))

    def _is_wanted(self, repeated_names: Sequence[str]) -> bool:
        """Whether this multiset should become a column.

        Everything is wanted when ``include_interactions`` is set. Otherwise
        only pure powers survive, meaning a multiset built from a single
        distinct name, which ``len(set(repeated_names)) == 1`` tests for.
        """
        if self.include_interactions:
            return True

        return len(set(repeated_names)) == 1

    def _build_terms(self, feature_names: Sequence[str]) -> PolynomialTerms:
        """Generate every term up to ``degree``, in column order.

        Ordered by total degree first, then by the order the features were
        given, so degree 2 over ``[x1, x2]`` yields::

            x1, x2, x1^2, x1*x2, x2^2

        The ordering needs no sorting at all, since it falls out of the shape
        of the loop:

        1. Outer loop over ``total_degree in range(1, self.degree + 1)``, which
           gives ascending degree.
        2. Inner loop over
           ``combinations_with_replacement(feature_names, total_degree)``, whose
           natural order is feature order. Each item is a multiset such as
           ``("x1", "x1", "x2")``; it never emits the same multiset twice in a
           different arrangement, so no term is duplicated.
        3. Keep the ones :meth:`_is_wanted` accepts, turn each into a term with
           :meth:`_term_for`, and hand the list to
           :class:`~oop_ml.preprocessing.polynomial.PolynomialTerms`.
        """
        terms_to_build = []

        for total_degree in range(1, self.degree + 1):
            for names in combinations_with_replacement(feature_names, total_degree):
                if self._is_wanted(names):
                    terms_to_build.append(self._term_for(names))

        return PolynomialTerms(terms_to_build)

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Work out which terms the expansion will produce.

        Nothing is learned from the data here, since the terms depend only on
        the feature names and on the degree. What ``fit`` pins down is the
        column set and its order, so that every later ``transform`` agrees with
        it.

        Returns
        -------
        Self
            This transformer, so calls can chain.

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

        self._terms = self._build_terms(
            [feature.name for feature in feature_set],
        )

        self._mark_fitted()

        return self

    def transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Compute the expanded columns for the supplied features.

        Matches by name, never by position, and returns the columns in the order
        fixed at ``fit``.

        Returns
        -------
        list[Feature]
            One feature per term, named for the term that produced it.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If a feature the expansion needs was not supplied.
        """
        self._check_fitted()
        self._check_names_were_learned(input_values)

        return self.terms.expand(input_values)
