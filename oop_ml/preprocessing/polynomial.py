"""One term of a polynomial expansion, and the ordered group of them.

A term is a product of feature powers, so ``x1``, ``x1^2``, ``x1*x2``, and
``x1^2*x2`` are all terms. Represented as a bare mapping of name to exponent it
would be another collection of pairs for the caller to unpack; represented as an
object it can own the two things anyone actually wants from it, which are what
to call the column it produces and how to compute that column from the input
features.

:class:`PolynomialTerms` is the ordered group of them, and the order genuinely
matters here, because it fixes the column order that ``transform`` reproduces on
every later call.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from oop_ml.data.feature import Feature
from oop_ml.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
)
from oop_ml.types import FloatArray


class FeaturePowers:
    """The source columns, with each power computed at most once.

    Evaluating an expansion term by term asks for the same ``x_j ** k`` over and
    over, because every term that contains ``x1^2`` needs exactly that column.
    Recomputing it each time is wasted work, and looking the feature up by
    scanning the supplied sequence is a second helping of the same, so this
    holds the columns in a dict and memoises each power as it is first asked
    for.

    Parameters
    ----------
    input_values:
        The source features an expansion is being computed from.
    """

    __slots__ = ("_columns", "_raised")

    def __init__(self, input_values: Sequence[Feature]) -> None:
        self._columns = {feature.name: feature.values for feature in input_values}
        self._raised: dict[tuple[str, int], FloatArray] = {}

    def raised(self, feature_name: str, exponent: int) -> FloatArray:
        """``feature_name`` raised to ``exponent``, computed at most once.

        Raises
        ------
        InvalidValuesError
            If the named feature was not among the supplied inputs.
        """
        cached = self._raised.get((feature_name, exponent))
        if cached is not None:
            return cached

        column = self._columns.get(feature_name)
        if column is None:
            raise InvalidValuesError(
                f"{feature_name!r} is needed by a polynomial term but was not supplied"
            )

        # An exponent of one is the column itself, and squaring is cheaper as a
        # multiply than as a general power.
        if exponent == 1:
            result = column
        elif exponent == 2:
            result = column * column
        else:
            result = column**exponent

        self._raised[(feature_name, exponent)] = result

        return result

    def __repr__(self) -> str:
        return f"FeaturePowers({', '.join(self._columns)})"


class PolynomialTerm:
    """A product of feature powers, such as ``x1^2 * x2``.

    Parameters
    ----------
    powers:
        Exponent per feature name. Exponents have to be positive, since a
        feature raised to the zeroth power contributes nothing but a constant,
        and that constant is the model's intercept rather than a feature.

    Raises
    ------
    EmptyValuesError
        If no powers are supplied.
    InvalidValuesError
        If any exponent is not a positive integer.
    """

    __slots__ = ("_powers",)

    def __init__(self, powers: Mapping[str, int]) -> None:
        if not powers:
            raise EmptyValuesError("a polynomial term needs at least one feature")

        for feature_name, exponent in powers.items():
            if not isinstance(exponent, int) or exponent < 1:
                raise InvalidValuesError(
                    f"exponent for {feature_name!r} must be a positive integer, "
                    f"got {exponent!r}"
                )

        self._powers = dict(powers)

    @property
    def name(self) -> str:
        """The column name this term produces, e.g. ``"x1^2*x2"``.

        Written in the order the features were given, with ``^`` for a power
        above one and ``*`` between factors. Fixed here in one place so the
        format is a property of the class rather than something each caller
        invents.
        """
        return "*".join(
            feature_name if exponent == 1 else f"{feature_name}^{exponent}"
            for feature_name, exponent in self._powers.items()
        )

    @property
    def total_degree(self) -> int:
        """The sum of the exponents, which is the degree of this term."""
        return sum(self._powers.values())

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The features this term is built from, in order."""
        return tuple(self._powers)

    def exponent_for(self, feature_name: str) -> int:
        """The power this term raises ``feature_name`` to, or zero if absent."""
        return self._powers.get(feature_name, 0)

    def evaluate(self, input_values: Sequence[Feature]) -> Feature:
        """Compute this term's column from the supplied features.

        Multiplies each required feature raised to its exponent, matching by
        name so the caller may supply them in any order.

        Raises
        ------
        InvalidValuesError
            If a feature this term needs was not supplied.
        """
        return self.evaluate_with(FeaturePowers(input_values))

    def evaluate_with(self, powers: FeaturePowers) -> Feature:
        """Compute this term's column from an existing :class:`FeaturePowers`.

        A whole expansion asks for the same ``x_j ** k`` many times over, since
        every term containing ``x1^2`` needs that identical column, so sharing
        one :class:`FeaturePowers` across the terms computes each power once
        instead of once per term. :meth:`evaluate` is the same thing for a
        caller holding a single term and nothing to share.

        Raises
        ------
        InvalidValuesError
            If a feature this term needs was not supplied.
        """
        product = None

        for feature_name, exponent in self._powers.items():
            raised = powers.raised(feature_name, exponent)
            product = raised if product is None else product * raised

        assert product is not None

        # Feature copies on the way in, so handing it a cached array cannot let
        # a later term mutate a column that has already been produced.
        return Feature(self.name, product)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolynomialTerm):
            return NotImplemented
        return self._powers == other._powers

    def __hash__(self) -> int:
        return hash(frozenset(self._powers.items()))

    def __repr__(self) -> str:
        return f"PolynomialTerm({self.name!r})"


class PolynomialTerms:
    """The expansion's terms, in the order they become columns.

    Parameters
    ----------
    terms:
        One :class:`PolynomialTerm` per generated column. Names must be unique.

    Raises
    ------
    EmptyValuesError
        If no terms are supplied.
    NonUniqueFeaturesError
        If two terms would produce the same column name.
    """

    __slots__ = ("_terms",)

    def __init__(self, terms: Sequence[PolynomialTerm]) -> None:
        if not terms:
            raise EmptyValuesError("at least one polynomial term is required")

        seen_names: set[str] = set()
        for term in terms:
            if term.name in seen_names:
                raise NonUniqueFeaturesError(f"duplicate polynomial term {term.name!r}")
            seen_names.add(term.name)

        self._terms = tuple(terms)

    @property
    def names(self) -> tuple[str, ...]:
        """The column names, in column order."""
        return tuple(term.name for term in self._terms)

    @property
    def n_terms(self) -> int:
        """How many columns the expansion produces."""
        return len(self._terms)

    @property
    def source_feature_names(self) -> tuple[str, ...]:
        """Every original feature any term is built from, without repeats."""
        ordered: dict[str, None] = {}
        for term in self._terms:
            for feature_name in term.feature_names:
                ordered[feature_name] = None

        return tuple(ordered)

    def expand(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Compute every term's column, in term order.

        One :class:`FeaturePowers` is built for the whole expansion so that a
        power needed by twenty terms is computed once rather than twenty times.
        """
        powers = FeaturePowers(input_values)

        return [term.evaluate_with(powers) for term in self._terms]

    def __iter__(self) -> Iterator[PolynomialTerm]:
        return iter(self._terms)

    def __len__(self) -> int:
        return self.n_terms

    def __repr__(self) -> str:
        return f"PolynomialTerms({', '.join(self.names)})"
