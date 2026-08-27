"""Datasets shared by the regression and preprocessing specs.

Four test modules were each carrying their own copy of the same five points and
the same expected answers. When a fixture is duplicated the expected values drift
out of step with it silently, so nothing fails and the two files simply stop
testing the same thing.

Each dataset here pairs the columns with the answer they are known to produce,
kept together so a reader never has to hunt for where a magic number came from.
Every figure was verified against a reference computation before being written
down.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from oop_ml.core.data.feature import Feature
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


class LabelledFixture:
    """A classification dataset paired with the boundary it is known to give.

    Parameters
    ----------
    predictor:
        The single feature column, named ``hours``.
    labels:
        The 0/1 target, named ``passed``.
    expected_intercept, expected_weight:
        The unpenalised maximum-likelihood solution, verified against
        scikit-learn to within 1.6e-09.
    """

    __slots__ = (
        "_expected_intercept",
        "_expected_weight",
        "_labels",
        "_predictor",
    )

    def __init__(
        self,
        predictor: NumericValues,
        labels: NumericValues,
        expected_intercept: float,
        expected_weight: float,
    ) -> None:
        self._predictor = predictor
        self._labels = labels
        self._expected_intercept = expected_intercept
        self._expected_weight = expected_weight

    @property
    def input_features(self) -> list[Feature]:
        """Fresh feature objects, in column order."""
        return [Feature("hours", self._predictor)]

    @property
    def target_feature(self) -> Feature:
        """A fresh 0/1 target feature."""
        return Feature("passed", self._labels)

    @property
    def label_values(self) -> NumericValues:
        """The raw labels."""
        return self._labels

    @property
    def expected_intercept(self) -> float:
        """The intercept maximum likelihood recovers from this data."""
        return self._expected_intercept

    @property
    def expected_weight(self) -> float:
        """The weight maximum likelihood recovers for ``hours``."""
        return self._expected_weight


OVERLAPPING_LABELS = LabelledFixture(
    [0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0],
    # Deliberately not separable: one student passes on 2 hours and one fails on
    # 4, which is what gives the likelihood a maximum to actually reach.
    [0, 0, 1, 0, 1, 0, 1, 1],
    expected_intercept=-2.4383,
    expected_weight=0.8637,
)
"""Boundary at hours = 2.8231, odds multiplier exp(0.8637) = 2.3719, 6 of 8 right."""

OVERLAPPING_LABELS_BOUNDARY = 2.8231
OVERLAPPING_LABELS_ODDS_MULTIPLIER = 2.3719

SEPARABLE_LABELS = LabelledFixture(
    [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    # Perfectly separable, so no finite maximum likelihood estimate exists and
    # the coefficients climb without limit. Any fit here must report
    # converged = False rather than pretending it found something.
    [0, 0, 0, 1, 1, 1],
    expected_intercept=float("nan"),
    expected_weight=float("nan"),
)
"""No finite solution. Used to pin the separation behaviour, never an answer."""


class MultiClassFixture:
    """A three-class dataset paired with the softmax solution it is known to give.

    Parameters
    ----------
    first, second:
        The two predictor columns, named ``first`` and ``second``.
    classes:
        The target, named ``outcome``, holding whole class positions 0, 1, 2.
    expected_weights:
        The reference-class maximum likelihood solution, one row per learned
        class in the order ``(intercept, first, second)``. Class 0 is the
        reference and is not listed because it is zero by construction.
    """

    __slots__ = ("_classes", "_expected_weights", "_first", "_second")

    def __init__(
        self,
        first: Sequence[float],
        second: Sequence[float],
        classes: Sequence[int],
        expected_weights: Sequence[Sequence[float]],
    ) -> None:
        self._first = list(first)
        self._second = list(second)
        self._classes = list(classes)
        self._expected_weights = [list(row) for row in expected_weights]

    @property
    def input_features(self) -> list[Feature]:
        """The two predictors."""
        return [Feature("first", self._first), Feature("second", self._second)]

    @property
    def target_feature(self) -> Feature:
        """The class positions."""
        return Feature("outcome", [float(value) for value in self._classes])

    @property
    def class_values(self) -> list[int]:
        """The classes as plain integers, for building expected tables by hand."""
        return list(self._classes)

    @property
    def n_classes(self) -> int:
        """How many classes the fixture spans."""
        return len(set(self._classes))

    @property
    def expected_weights(self) -> list[list[float]]:
        """The learned classes' weights: ``(intercept, first, second)`` each."""
        return [list(row) for row in self._expected_weights]


THREE_CLASSES = MultiClassFixture(
    [
        2.5829,
        0.0501,
        3.6526,
        0.3855,
        3.4657,
        -1.8207,
        2.1585,
        2.9271,
        1.2587,
        0.5089,
        2.0575,
        2.3023,
        -0.159,
        -1.3553,
        -0.2526,
        0.6366,
        4.7236,
        0.5183,
        -1.2959,
        2.428,
        0.4565,
        -1.1956,
        2.7278,
        3.2783,
        -0.1199,
        -1.05,
        0.4067,
        -0.6217,
        0.6105,
        0.2705,
        2.5055,
        1.4226,
        -0.4719,
        -1.5237,
        0.4265,
        5.5203,
    ],
    [
        -3.378,
        0.4393,
        3.4796,
        -3.2911,
        1.6466,
        3.04,
        0.2084,
        -1.7078,
        2.3597,
        -0.3528,
        1.7067,
        0.312,
        -2.2101,
        1.0514,
        2.2669,
        0.071,
        1.7615,
        -2.3433,
        0.917,
        -0.1782,
        1.914,
        -0.9272,
        0.1553,
        1.9767,
        -1.1199,
        -1.4911,
        0.4628,
        -0.0878,
        2.8605,
        0.8031,
        3.9973,
        3.1061,
        0.7988,
        -2.5163,
        -2.1749,
        3.6196,
    ],
    # Twelve of each class, deliberately overlapping: every class is reachable
    # from the others, so the likelihood has a finite maximum. A cleaner
    # separation would give no answer at all to compare against.
    [
        1,
        1,
        2,
        0,
        2,
        0,
        0,
        1,
        2,
        1,
        2,
        2,
        0,
        0,
        2,
        1,
        1,
        0,
        1,
        1,
        2,
        0,
        2,
        1,
        0,
        0,
        1,
        1,
        2,
        2,
        2,
        0,
        0,
        0,
        1,
        2,
    ],
    expected_weights=[
        [-0.355987, 0.919134, 0.114455],
        [-1.409842, 0.992247, 0.995121],
    ],
)
"""Reached by Newton in 7 iterations; 25 of 36 right, so accuracy 0.694444."""

THREE_CLASSES_ACCURACY = 0.694444


class NeighbourFixture:
    """A three-by-three grid, small enough to check the neighbours by eye.

    Two predictors on identical scales, holding both a quantity and a class per
    row, so one fixture serves the regressor and the classifier and any
    disagreement between them is about combining rather than about finding.

    Parameters
    ----------
    first, second:
        Grid coordinates, named ``first`` and ``second``.
    quantity:
        A continuous target, named ``quantity``.
    classes:
        A three-class target, named ``outcome``.
    """

    __slots__ = ("_classes", "_first", "_quantity", "_second")

    def __init__(
        self,
        first: Sequence[float],
        second: Sequence[float],
        quantity: Sequence[float],
        classes: Sequence[int],
    ) -> None:
        self._first = list(first)
        self._second = list(second)
        self._quantity = list(quantity)
        self._classes = list(classes)

    @property
    def input_features(self) -> list[Feature]:
        """The two grid coordinates."""
        return [Feature("first", self._first), Feature("second", self._second)]

    @property
    def quantity_feature(self) -> Feature:
        """The continuous target."""
        return Feature("quantity", self._quantity)

    @property
    def class_feature(self) -> Feature:
        """The three-class target."""
        return Feature("outcome", [float(value) for value in self._classes])

    @property
    def quantity_values(self) -> list[float]:
        """The continuous target as plain floats."""
        return list(self._quantity)

    @property
    def class_values(self) -> list[int]:
        """The classes as plain integers."""
        return list(self._classes)

    @property
    def n_samples(self) -> int:
        """How many rows the grid holds."""
        return len(self._first)


NEIGHBOUR_GRID = NeighbourFixture(
    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
    # Three classes laid out in bands, so a query in the middle draws
    # neighbours from more than one of them.
    [0, 0, 1, 0, 1, 1, 2, 2, 2],
)
"""Verified by hand: from (0.4, 0.4) the three nearest are rows 0, 1 and 3."""

# From (0.4, 0.4), on either metric, the nearest three are rows 0, 1, 3.
NEIGHBOUR_QUERY = (0.4, 0.4)
NEIGHBOUR_QUERY_NEAREST_THREE = (0, 1, 3)
NEIGHBOUR_QUERY_MEAN_OF_THREE = 23.333333333333332

# From (1.0, 1.0) with k=5 the classes tie 2-2-1, which is the tie-break case.
NEIGHBOUR_TIE_QUERY = (1.0, 1.0)
NEIGHBOUR_TIE_SHARES = (0.4, 0.4, 0.2)


class TreeFixture:
    """Fifteen rows whose greedy tree was worked out by exhaustive search.

    Two predictors and a binary outcome, sized so the whole tree can be checked
    by hand. The story is an interaction: sleeping enough is the first gate, and
    studying only matters for rows that cleared it -- which is exactly the shape
    a tree represents natively and a linear model cannot without being handed a
    product column.

    One row is deliberately unreachable. Row 5 slept 7.5 hours, studied 3.5, and
    passed anyway; it lands in a leaf holding three fails, so the tree gets it
    wrong and stays wrong. That is what stops the fixture from rewarding an
    implementation that grows until every leaf is pure.

    Parameters
    ----------
    studied, slept:
        Hours, named ``studied`` and ``slept``.
    passed:
        The outcome, named ``passed``, as 0/1 class positions.
    """

    __slots__ = ("_passed", "_slept", "_studied")

    def __init__(
        self,
        studied: Sequence[float],
        slept: Sequence[float],
        passed: Sequence[int],
    ) -> None:
        self._studied = list(studied)
        self._slept = list(slept)
        self._passed = list(passed)

    @property
    def input_features(self) -> list[Feature]:
        """The two predictors, in the order the fit sees them."""
        return [Feature("studied", self._studied), Feature("slept", self._slept)]

    @property
    def class_feature(self) -> Feature:
        """The binary outcome."""
        return Feature("passed", [float(value) for value in self._passed])

    @property
    def class_values(self) -> list[int]:
        """The outcome as plain integers."""
        return list(self._passed)

    @property
    def n_samples(self) -> int:
        """How many rows."""
        return len(self._studied)


EXAM_OUTCOMES = TreeFixture(
    [1.0, 2.5, 3.0, 4.0, 2.0, 3.5, 5.0, 6.0, 7.0, 5.5, 5.0, 6.5, 7.5, 8.0, 9.0],
    [8.0, 7.0, 5.0, 8.5, 6.0, 7.5, 4.0, 5.5, 5.0, 3.5, 7.0, 8.0, 7.5, 6.5, 8.0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
)
"""Verified by exhaustive search over every candidate threshold, both criteria."""

# Both criteria choose the same two splits here, which is the usual case and
# worth pinning: if an implementation ever made them disagree on this data, the
# disagreement would be a bug rather than a finding.
# The canonical tree below is the one grown with this rule in force. Left at
# the defaults the recursion does not stop there: the four-row node splits, and
# then its two-row child splits again, until every leaf is pure and all fifteen
# rows are reproduced exactly. That is the memorisation failure rather than the
# model worth studying, so the fixture pins the stopped tree and a separate test
# pins the unstopped one.
EXAM_MIN_SAMPLES_SPLIT = 5

EXAM_ROOT_SPLIT = ("slept", 6.25)
EXAM_SECOND_SPLIT = ("studied", 4.5)

EXAM_ROOT_GINI = 0.48
EXAM_ROOT_GINI_GAIN = 0.2133333333333333
EXAM_ROOT_ENTROPY = 0.9709505944546686
EXAM_ROOT_ENTROPY_GAIN = 0.4199730940219749

# leaf 1 (slept < 6.25), leaf 2 (slept >= 6.25, studied < 4.5), leaf 3 (rest)
EXAM_LEAF_SIZES = (6, 4, 5)
EXAM_LEAF_GINI = (0.0, 0.375, 0.0)
EXAM_LEAF_SHARES = ((1.0, 0.0), (0.75, 0.25), (0.0, 1.0))
EXAM_LEAF_PREDICTIONS = (0.0, 0.0, 1.0)

# Grown to the default, every leaf ends up pure and nothing is left wrong.
EXAM_UNSTOPPED_LEAF_COUNT = 5
EXAM_UNSTOPPED_ACCURACY = 1.0

# Depth 2, three leaves, and one row it cannot reach.
EXAM_TREE_DEPTH = 2
EXAM_TREE_LEAF_COUNT = 3
EXAM_MISCLASSIFIED_ROW = 5
EXAM_TREE_ACCURACY = 14.0 / 15.0


class StepFixture:
    """A single predictor and a target that jumps once, with no noise at all.

    The smallest dataset on which a regression tree has exactly one right
    answer: one split at 3.5, two leaves predicting 10 and 50, and a gain equal
    to the whole of the parent variance because both children are constant.

    Also the fixture that shows what a tree cannot do. The target is a step, so
    the tree reproduces it perfectly; make it a straight line instead and the
    same model can only approximate it with a staircase.
    """

    __slots__ = ("_position", "_quantity")

    def __init__(self, position: Sequence[float], quantity: Sequence[float]) -> None:
        self._position = list(position)
        self._quantity = list(quantity)

    @property
    def input_features(self) -> list[Feature]:
        """The single predictor."""
        return [Feature("position", self._position)]

    @property
    def target_feature(self) -> Feature:
        """The step target."""
        return Feature("quantity", self._quantity)

    @property
    def n_samples(self) -> int:
        """How many rows."""
        return len(self._position)


STEP_FUNCTION = StepFixture(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    [10.0, 10.0, 10.0, 10.0, 50.0, 50.0, 50.0, 50.0],
)
"""Verified by hand: variance 400 at the root, all of it removed by one split."""

STEP_SPLIT = ("position", 3.5)
STEP_ROOT_VARIANCE = 400.0
STEP_ROOT_GAIN = 400.0
STEP_LEAF_MEANS = (10.0, 50.0)


# Drawing n from n with replacement misses (1 - 1/n)^n of the rows, which is
# 1/e in the limit and close to it well before two hundred. The band is wide
# enough that no particular seed is being pinned.
OUT_OF_BAG_SHARE = 1.0 / np.e
OUT_OF_BAG_TOLERANCE = 0.06
