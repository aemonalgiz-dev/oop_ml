"""Datasets shared by the regression and preprocessing specs.

Four test modules were each carrying their own copy of the same five points and
the same expected answers. When a fixture is duplicated, the expected values
drift out of step with it silently -- nothing fails, the two files simply stop
testing the same thing.

Each dataset here is a :class:`LinearFixture`: the columns *and* the answer they
are known to produce, kept together so a reader never has to hunt for where a
magic number came from. Every figure was verified against a reference
computation before being written down.
"""

from __future__ import annotations

from oop_ml.core.feature import Feature
from oop_ml.core.types import NumericValues

FIRST_PREDICTOR: NumericValues = [1, 1, 2, 0, 3]
SECOND_PREDICTOR: NumericValues = [1, 2, 2, 1, 0]


class LinearFixture:
    """A small dataset paired with the least-squares answer it is known to give.

    Parameters
    ----------
    first_predictor, second_predictor:
        The two feature columns, named ``x1`` and ``x2``.
    target:
        The response column, named ``y``.
    expected_intercept, expected_first_weight, expected_second_weight:
        The ordinary-least-squares solution for this data, computed by hand or
        verified against a reference implementation.
    """

    __slots__ = (
        "_expected_first_weight",
        "_expected_intercept",
        "_expected_second_weight",
        "_first_predictor",
        "_second_predictor",
        "_target",
    )

    def __init__(
        self,
        first_predictor: NumericValues,
        second_predictor: NumericValues,
        target: NumericValues,
        expected_intercept: float,
        expected_first_weight: float,
        expected_second_weight: float,
    ) -> None:
        self._first_predictor = first_predictor
        self._second_predictor = second_predictor
        self._target = target
        self._expected_intercept = expected_intercept
        self._expected_first_weight = expected_first_weight
        self._expected_second_weight = expected_second_weight

    @property
    def input_features(self) -> list[Feature]:
        """Fresh feature objects, in column order."""
        return [
            Feature("x1", self._first_predictor),
            Feature("x2", self._second_predictor),
        ]

    @property
    def target_feature(self) -> Feature:
        """A fresh target feature."""
        return Feature("y", self._target)

    @property
    def target_values(self) -> NumericValues:
        """The raw target values."""
        return self._target

    @property
    def expected_intercept(self) -> float:
        """The intercept least squares recovers from this data."""
        return self._expected_intercept

    @property
    def expected_first_weight(self) -> float:
        """The weight least squares recovers for ``x1``."""
        return self._expected_first_weight

    @property
    def expected_second_weight(self) -> float:
        """The weight least squares recovers for ``x2``."""
        return self._expected_second_weight

    def shifted_target(self, offset: float) -> Feature:
        """The target with ``offset`` added to every value.

        Shifting the target moves the plane bodily, so the intercept must absorb
        the whole change and the slopes must not move.
        """
        return Feature("y", [value + offset for value in self._target])


EXACT_PLANE = LinearFixture(
    FIRST_PREDICTOR,
    SECOND_PREDICTOR,
    # y = 1 + 2*x1 + 3*x2 exactly, so every residual is zero.
    [6, 9, 11, 4, 7],
    expected_intercept=1.0,
    expected_first_weight=2.0,
    expected_second_weight=3.0,
)

DISPLACED_PLANE = LinearFixture(
    FIRST_PREDICTOR,
    SECOND_PREDICTOR,
    # The exact target displaced by e = (-1, -1, 1, 1, 0), chosen so X.T @ e == 0.
    # Orthogonal to every column, so the projection -- and therefore the
    # solution -- is unchanged, while the residuals become exactly e.
    [5, 8, 12, 5, 7],
    expected_intercept=1.0,
    expected_first_weight=2.0,
    expected_second_weight=3.0,
)
"""Imperfect fit with hand-checkable metrics: RSS 4, TSS 33.2, R^2 = 73/83."""

DISPLACED_PLANE_RESIDUALS = [-1.0, -1.0, 1.0, 1.0, 0.0]
DISPLACED_PLANE_RESIDUAL_SUM_OF_SQUARES = 4.0
DISPLACED_PLANE_TOTAL_SUM_OF_SQUARES = 33.2

ORIGIN_PLANE = LinearFixture(
    [1, 0, 2, 1],
    [0, 1, 1, 2],
    # y = 2*x1 + 3*x2 exactly, no bias, for the fit_intercept=False path.
    [2, 3, 7, 8],
    expected_intercept=0.0,
    expected_first_weight=2.0,
    expected_second_weight=3.0,
)

# (x1 - 1.4) / 1.019804, where 1.019804 = sqrt(5.2 / 5) is the *population* spread.
STANDARDIZED_FIRST_PREDICTOR = [
    -0.392232,
    -0.392232,
    0.588348,
    -1.372813,
    1.568929,
]
FIRST_PREDICTOR_MEAN = 1.4
FIRST_PREDICTOR_STANDARD_DEVIATION = 1.019804
