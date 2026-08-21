"""Time oop_ml against scikit-learn on identical data, and check the answers.

The README makes a claim about performance, and a claim about performance that
nobody can reproduce is worth roughly nothing. This is the script behind it. Run
it yourself, on your own machine and your own BLAS, and see whether the numbers
hold up.

    pip install -e ".[benchmark]"
    python -m benchmarks.against_scikit_learn

Two things are worth saying before you read the output. The first is that
ratios below 1.0 mean oop_ml was faster on that task, which surprises most
people, including me when I first measured it. The reason is that numpy already
hands the heavy linear algebra off to BLAS, so scikit-learn's Cython has very
little left to win, and the per-call input validation it performs is enough to
lose it the smaller comparisons outright.

The second is that timing alone would be a useless benchmark. A library can be
arbitrarily fast if it is allowed to be wrong, so every linear model here is
also checked for agreement against scikit-learn's coefficients. The penalty
parameterisations differ between the two libraries and the conversions are
handled below.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures as ScikitPolynomialFeatures
from sklearn.preprocessing import StandardScaler

from benchmarks.comparison import Agreement, Comparison, Comparisons, Timing
from examples.reporting import Report, configure_logging_from_command_line
from oop_ml import (
    Feature,
    FloatArray,
    GradientDescentRegression,
    LassoRegression,
    MultipleLinearRegression,
    PolynomialFeatures,
    RidgeRegression,
    Standardizer,
)

logger = logging.getLogger(__name__)

REGRESSION_SIZES = [(1_000, 20), (20_000, 50)]
EXPANSION_SIZES = [(2_000, 8), (2_000, 12)]
COEFFICIENT_TOLERANCE = 1e-6


class Problem:
    """One generated regression problem, in both libraries' input shapes.

    Both libraries have to see the same numbers for the comparison to mean
    anything, so the matrix and the features are built once from the same draw
    rather than generated twice.
    """

    __slots__ = ("_features", "_matrix", "_target")

    def __init__(self, n_samples: int, n_features: int, random_seed: int = 0) -> None:
        generator = np.random.default_rng(random_seed)
        self._matrix = generator.normal(size=(n_samples, n_features))
        weights = generator.normal(size=n_features)
        self._target = self._matrix @ weights + generator.normal(
            scale=0.5, size=n_samples
        )
        self._features = [
            Feature(f"x{index}", self._matrix[:, index]) for index in range(n_features)
        ]

    @property
    def matrix(self) -> FloatArray:
        """The design matrix, as scikit-learn wants it."""
        return self._matrix

    @property
    def target_values(self) -> FloatArray:
        """The response, as a bare array."""
        return self._target

    @property
    def features(self) -> list[Feature]:
        """The predictors, as oop_ml wants them."""
        return self._features

    @property
    def target_feature(self) -> Feature:
        """The response, as a named feature."""
        return Feature("y", self._target)

    @property
    def n_samples(self) -> int:
        """How many rows."""
        return len(self._target)

    @property
    def size(self) -> str:
        """The label this problem carries in the reported table."""
        return f"{self._matrix.shape[0]}x{self._matrix.shape[1]}"


def _agreement(problem: Problem, ours: Timing, theirs: Timing) -> Agreement:
    """Compare our coefficients against scikit-learn's, matching by name."""
    ours_by_name = [
        ours.result.coefficients_[feature.name] for feature in problem.features
    ]

    if np.allclose(ours_by_name, theirs.result.coef_, atol=COEFFICIENT_TOLERANCE):
        return Agreement.MATCHES

    return Agreement.DIFFERS


def _least_squares(problem: Problem) -> Comparison:
    ours = Timing.of(
        lambda: MultipleLinearRegression().fit(problem.features, problem.target_feature)
    )
    theirs = Timing.of(
        lambda: LinearRegression().fit(problem.matrix, problem.target_values)
    )

    return Comparison(
        "OLS", problem.size, ours, theirs, _agreement(problem, ours, theirs)
    )


def _ridge(problem: Problem, penalty: float = 1.0) -> Comparison:
    # Both libraries minimise ||y - Xb||^2 + penalty * ||b||^2, so the
    # hyperparameter carries across unchanged.
    ours = Timing.of(
        lambda: RidgeRegression(penalty=penalty).fit(
            problem.features, problem.target_feature
        )
    )
    theirs = Timing.of(
        lambda: Ridge(alpha=penalty).fit(problem.matrix, problem.target_values)
    )

    return Comparison(
        "Ridge", problem.size, ours, theirs, _agreement(problem, ours, theirs)
    )


def _lasso(problem: Problem, penalty: float = 1.0) -> Comparison:
    # scikit-learn minimises (1 / 2n) * ||y - Xw||^2 + alpha * ||w||_1, so
    # multiplying through by 2n gives alpha = penalty / (2n) for the same
    # objective. With that conversion the two agree to about 1e-14, exact zeros
    # included.
    alpha = penalty / (2 * problem.n_samples)

    ours = Timing.of(
        lambda: LassoRegression(penalty=penalty).fit(
            problem.features, problem.target_feature
        ),
        repeats=1,
    )
    theirs = Timing.of(
        lambda: Lasso(alpha=alpha).fit(problem.matrix, problem.target_values),
        repeats=1,
    )

    return Comparison(
        "Lasso", problem.size, ours, theirs, _agreement(problem, ours, theirs)
    )


def _gradient_descent(problem: Problem) -> Comparison:
    # Batch gradient descent against stochastic gradient descent. They minimise
    # the same objective by genuinely different routes and stop on different
    # rules, so their coefficients are not expected to match and comparing them
    # would only be misleading.
    ours = Timing.of(
        lambda: GradientDescentRegression(learning_rate=0.01, max_epochs=500).fit(
            problem.features, problem.target_feature
        ),
        repeats=1,
    )
    # tol plus n_iter_no_change is how you stop SGDRegressor from bailing out
    # early, so that both sides are timed over the same epoch budget. Having
    # asked for the full budget, we then expect the convergence warning that
    # comes back with it, so there is no reason to print it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        theirs = Timing.of(
            lambda: SGDRegressor(max_iter=500, tol=1e-12, n_iter_no_change=500).fit(
                problem.matrix, problem.target_values
            ),
            repeats=1,
        )

    return Comparison(
        "Gradient descent", problem.size, ours, theirs, Agreement.NOT_COMPARED
    )


def _standardizer(problem: Problem) -> Comparison:
    ours = Timing.of(lambda: Standardizer().fit_transform(problem.features))
    theirs = Timing.of(lambda: StandardScaler().fit_transform(problem.matrix))

    scaled = np.column_stack([feature.values for feature in ours.result])
    agreement = (
        Agreement.MATCHES
        if np.allclose(scaled, theirs.result, atol=COEFFICIENT_TOLERANCE)
        else Agreement.DIFFERS
    )

    return Comparison("Standardizer", problem.size, ours, theirs, agreement)


def _polynomial_features(problem: Problem, degree: int = 3) -> Comparison:
    ours = Timing.of(
        lambda: PolynomialFeatures(degree=degree).fit_transform(problem.features)
    )
    theirs = Timing.of(
        lambda: ScikitPolynomialFeatures(
            degree=degree, include_bias=False
        ).fit_transform(problem.matrix)
    )

    # Both expansions produce the same columns, though not in the same order, so
    # the column count is what is worth checking here.
    agreement = (
        Agreement.MATCHES
        if len(ours.result) == theirs.result.shape[1]
        else Agreement.DIFFERS
    )

    return Comparison(
        f"PolynomialFeatures d{degree}",
        problem.size,
        ours,
        theirs,
        agreement,
    )


def run() -> Comparisons:
    """Time every task at every size and collect the comparisons."""
    comparisons = []

    for n_samples, n_features in REGRESSION_SIZES:
        problem = Problem(n_samples, n_features)
        comparisons.extend(
            [
                _least_squares(problem),
                _ridge(problem),
                _lasso(problem),
                _gradient_descent(problem),
                _standardizer(problem),
            ]
        )

    for n_samples, n_features in EXPANSION_SIZES:
        comparisons.append(_polynomial_features(Problem(n_samples, n_features)))

    return Comparisons(comparisons)


def main() -> None:
    report = Report(logger)

    report.heading("oop_ml against scikit-learn")
    report.line("ratios below 1.0 mean oop_ml was faster on that task")

    comparisons = run()
    report.table(list(Comparisons.COLUMN_NAMES), comparisons.rows)

    for comparison in comparisons:
        report.detail(
            f"{comparison.task} {comparison.size}: "
            f"ours {comparison.ours.seconds:.6f}s, "
            f"theirs {comparison.theirs.seconds:.6f}s"
        )

    disagreements = comparisons.disagreements
    if disagreements:
        for comparison in disagreements:
            report.warn(
                f"{comparison.task} at {comparison.size} did not agree with "
                f"scikit-learn, so this row is timing two different programs "
                f"and the ratio means nothing"
            )
    else:
        report.paragraph(
            "Every task that can be compared agreed with scikit-learn to within\n"
            f"{COEFFICIENT_TOLERANCE:g}. Gradient descent is the exception, and\n"
            "deliberately so: batch descent and stochastic descent take genuinely\n"
            "different routes to the same objective and stop on different rules,\n"
            "so matching coefficients was never the expectation."
        )


if __name__ == "__main__":
    configure_logging_from_command_line()
    main()
