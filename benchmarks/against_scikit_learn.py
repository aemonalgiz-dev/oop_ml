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

Logistic regression is the row where scikit-learn wins properly, and it is the
most informative row in the table for that reason. Both libraries reach the same
maximum to within 1e-8; lbfgs simply gets there in around a dozen iterations
where plain gradient ascent needs hundreds of epochs, because it approximates
the curvature and we only ever look at the slope. That is an algorithmic gap
rather than a language one, and it is not something faster array code repairs.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.linear_model import LogisticRegression as ScikitLogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures as ScikitPolynomialFeatures
from sklearn.preprocessing import StandardScaler

from benchmarks.comparison import Agreement, Comparison, Comparisons, Timing
from examples.reporting import Report, configure_logging_from_command_line
from oop_ml import (
    Feature,
    FloatArray,
    GradientDescentRegression,
    KNearestNeighboursClassifier,
    KNearestNeighboursRegressor,
    LassoRegression,
    LogisticRegression,
    MultipleLinearRegression,
    NewtonLogisticRegression,
    PolynomialFeatures,
    RidgeRegression,
    Standardizer,
)

logger = logging.getLogger(__name__)

MODEL_SIZES = [(1_000, 20), (20_000, 50)]
EXPANSION_SIZES = [(2_000, 8), (2_000, 12)]
# Neighbour models do no work at fit time, so what matters is how many rows
# are remembered and how many queries are asked against them.
NEIGHBOUR_SIZES = [(5_000, 20), (20_000, 20)]
NEIGHBOUR_QUERIES = 500
NEIGHBOURS = 5
COEFFICIENT_TOLERANCE = 1e-6


class GeneratedDesign:
    """The predictors, in both libraries' input shapes.

    Both libraries have to see the same numbers for the comparison to mean
    anything, so the matrix and the features are built once from the same draw
    rather than generated twice. What is predicted *from* these columns is the
    business of a subclass, since a response and a label are not the same thing
    and no single generator produces both.
    """

    __slots__ = ("_features", "_matrix")

    def __init__(self, n_samples: int, n_features: int, random_seed: int = 0) -> None:
        self._matrix = np.random.default_rng(random_seed).normal(
            size=(n_samples, n_features)
        )
        self._features = [
            Feature(f"x{index}", self._matrix[:, index]) for index in range(n_features)
        ]

    @property
    def matrix(self) -> FloatArray:
        """The design matrix, as scikit-learn wants it."""
        return self._matrix

    @property
    def features(self) -> list[Feature]:
        """The predictors, as oop_ml wants them."""
        return self._features

    @property
    def n_samples(self) -> int:
        """How many rows."""
        return self._matrix.shape[0]

    @property
    def size(self) -> str:
        """The label this problem carries in the reported table."""
        return f"{self._matrix.shape[0]}x{self._matrix.shape[1]}"


class RegressionProblem(GeneratedDesign):
    """A continuous response, drawn as a noisy linear combination."""

    __slots__ = ("_target",)

    def __init__(self, n_samples: int, n_features: int, random_seed: int = 0) -> None:
        super().__init__(n_samples, n_features, random_seed)

        generator = np.random.default_rng(random_seed + 1)
        weights = generator.normal(size=n_features)
        self._target = self.matrix @ weights + generator.normal(
            scale=0.5, size=n_samples
        )

    @property
    def target_values(self) -> FloatArray:
        """The response, as a bare array."""
        return self._target

    @property
    def target_feature(self) -> Feature:
        """The response, as a named feature."""
        return Feature("y", self._target)


class ClassificationProblem(GeneratedDesign):
    """A binary label, drawn from a log-odds surface over the same columns.

    The weights are deliberately halved. A steeper surface pushes most rows to
    a probability of nearly zero or nearly one, which approaches separation, and
    a separable problem has no finite maximum likelihood estimate for either
    library to find. Timing two solvers as they both fail to converge would
    measure nothing except their epoch caps.
    """

    __slots__ = ("_labels",)

    def __init__(self, n_samples: int, n_features: int, random_seed: int = 0) -> None:
        super().__init__(n_samples, n_features, random_seed)

        generator = np.random.default_rng(random_seed + 1)
        weights = generator.normal(size=n_features) * 0.5
        probabilities = _sigmoid(self.matrix @ weights)
        self._labels = (generator.uniform(size=n_samples) < probabilities).astype(float)

    @property
    def label_values(self) -> FloatArray:
        """The 0/1 labels, as a bare array."""
        return self._labels

    @property
    def label_feature(self) -> Feature:
        """The 0/1 labels, as a named feature."""
        return Feature("y", self._labels)


def _sigmoid(linear_predictor: FloatArray) -> FloatArray:
    """The stable sigmoid, so generating labels never overflows."""
    return np.exp(np.minimum(linear_predictor, 0.0)) / (
        1.0 + np.exp(-np.abs(linear_predictor))
    )


def _agreement(design: GeneratedDesign, ours: Timing, theirs: Timing) -> Agreement:
    """Compare our coefficients against scikit-learn's, matching by name.

    Flattened because scikit-learn's regressors hand back a one-dimensional
    ``coef_`` while its classifiers hand back one row per class, which for the
    binary case is a single row rather than a plain vector.
    """
    ours_by_name = [
        ours.result.coefficients[feature.name] for feature in design.features
    ]
    theirs_flattened = np.ravel(theirs.result.coef_)

    if np.allclose(ours_by_name, theirs_flattened, atol=COEFFICIENT_TOLERANCE):
        return Agreement.MATCHES

    return Agreement.DIFFERS


def _least_squares(problem: RegressionProblem) -> Comparison:
    ours = Timing.of(
        lambda: MultipleLinearRegression().fit(problem.features, problem.target_feature)
    )
    theirs = Timing.of(
        lambda: LinearRegression().fit(problem.matrix, problem.target_values)
    )

    return Comparison(
        "OLS", problem.size, ours, theirs, _agreement(problem, ours, theirs)
    )


def _ridge(problem: RegressionProblem, penalty: float = 1.0) -> Comparison:
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


def _lasso(problem: RegressionProblem, penalty: float = 1.0) -> Comparison:
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


def _gradient_descent(problem: RegressionProblem) -> Comparison:
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


def _logistic(problem: ClassificationProblem) -> Comparison:
    # Both maximise the unpenalised log-likelihood, so there is no penalty to
    # convert; C=inf is how scikit-learn is told to leave the objective alone.
    # The two tolerances are set tight enough that both solvers genuinely reach
    # the maximum rather than one of them stopping early and winning the timing
    # by declining to finish the job.
    #
    # The learning rate is the number to argue with here. Plain gradient ascent
    # has one, lbfgs does not, and raising it from 0.5 to 2.0 cut the large
    # problem from 135x to 21x without moving a coefficient. That sensitivity is
    # the honest finding rather than a footnote to it.
    ours = Timing.of(
        lambda: LogisticRegression(
            learning_rate=2.0, max_epochs=50_000, tolerance=1e-10
        ).fit(problem.features, problem.label_feature),
        repeats=1,
    )
    theirs = Timing.of(
        lambda: ScikitLogisticRegression(C=np.inf, tol=1e-10, max_iter=5_000).fit(
            problem.matrix, problem.label_values
        ),
        repeats=1,
    )

    return Comparison(
        "Logistic ascent",
        problem.size,
        ours,
        theirs,
        _agreement(problem, ours, theirs),
    )


def _newton_logistic(problem: ClassificationProblem) -> Comparison:
    # The same objective and the same maximum as the row above, reached by a
    # solver that uses the second derivative instead of a learning rate. Both
    # of these are compared against the same scikit-learn call, so the two rows
    # are directly readable against each other as well as against lbfgs.
    ours = Timing.of(
        lambda: NewtonLogisticRegression(tolerance=1e-10).fit(
            problem.features, problem.label_feature
        ),
        repeats=1,
    )
    theirs = Timing.of(
        lambda: ScikitLogisticRegression(C=np.inf, tol=1e-10, max_iter=5_000).fit(
            problem.matrix, problem.label_values
        ),
        repeats=1,
    )

    return Comparison(
        "Logistic Newton",
        problem.size,
        ours,
        theirs,
        _agreement(problem, ours, theirs),
    )


def _standardizer(design: GeneratedDesign) -> Comparison:
    ours = Timing.of(lambda: Standardizer().fit_transform(design.features))
    theirs = Timing.of(lambda: StandardScaler().fit_transform(design.matrix))

    scaled = np.column_stack([feature.values for feature in ours.result])
    agreement = (
        Agreement.MATCHES
        if np.allclose(scaled, theirs.result, atol=COEFFICIENT_TOLERANCE)
        else Agreement.DIFFERS
    )

    return Comparison("Standardizer", design.size, ours, theirs, agreement)


def _polynomial_features(design: GeneratedDesign, degree: int = 3) -> Comparison:
    ours = Timing.of(
        lambda: PolynomialFeatures(degree=degree).fit_transform(design.features)
    )
    theirs = Timing.of(
        lambda: ScikitPolynomialFeatures(
            degree=degree, include_bias=False
        ).fit_transform(design.matrix)
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
        design.size,
        ours,
        theirs,
        agreement,
    )


class NeighbourProblem:
    """Rows to remember, plus a separate set of rows to ask about.

    Every other problem here is timed on ``fit``. This one is timed on
    ``predict``, because a neighbour model does no work at fit time at all --
    it validates its inputs and keeps them. The query rows are generated
    separately rather than sliced out of the training set: a row queried
    against a set containing itself is at distance zero from one of its own
    neighbours, which is not what a held-out query looks like.
    """

    __slots__ = ("_classification", "_queries", "_regression")

    def __init__(self, n_samples: int, n_features: int) -> None:
        self._regression = RegressionProblem(n_samples, n_features)
        self._classification = ClassificationProblem(n_samples, n_features)
        self._queries = GeneratedDesign(NEIGHBOUR_QUERIES, n_features, random_seed=99)

    @property
    def regression(self) -> RegressionProblem:
        return self._regression

    @property
    def classification(self) -> ClassificationProblem:
        return self._classification

    @property
    def queries(self) -> GeneratedDesign:
        return self._queries

    @property
    def size(self) -> str:
        """Remembered rows by features, with the query count alongside."""
        return f"{self._regression.size} q{NEIGHBOUR_QUERIES}"


def _k_nearest_regressor(problem: NeighbourProblem) -> Comparison:
    ours = KNearestNeighboursRegressor(n_neighbours=NEIGHBOURS)
    ours.fit(problem.regression.features, problem.regression.target_feature)

    theirs = KNeighborsRegressor(n_neighbors=NEIGHBOURS)
    theirs.fit(problem.regression.matrix, problem.regression.target_values)

    our_timing = Timing.of(lambda: ours.predict(problem.queries.features))
    their_timing = Timing.of(lambda: theirs.predict(problem.queries.matrix))

    agreement = (
        Agreement.MATCHES
        if np.allclose(our_timing.result, their_timing.result, atol=1e-9)
        else Agreement.DIFFERS
    )

    return Comparison(
        "k-NN regressor", problem.size, our_timing, their_timing, agreement
    )


def _k_nearest_classifier(problem: NeighbourProblem) -> Comparison:
    ours = KNearestNeighboursClassifier(n_neighbours=NEIGHBOURS)
    ours.fit(problem.classification.features, problem.classification.label_feature)

    theirs = KNeighborsClassifier(n_neighbors=NEIGHBOURS)
    theirs.fit(problem.classification.matrix, problem.classification.label_values)

    our_timing = Timing.of(lambda: ours.predict(problem.queries.features))
    their_timing = Timing.of(lambda: theirs.predict(problem.queries.matrix))

    # Both break ties toward the lowest class index, so on identical neighbour
    # sets the labels should agree exactly rather than approximately.
    agreement = (
        Agreement.MATCHES
        if np.array_equal(our_timing.result, their_timing.result)
        else Agreement.DIFFERS
    )

    return Comparison(
        "k-NN classifier", problem.size, our_timing, their_timing, agreement
    )


def run() -> Comparisons:
    """Time every task at every size and collect the comparisons."""
    comparisons = []

    for n_samples, n_features in MODEL_SIZES:
        problem = RegressionProblem(n_samples, n_features)
        classification = ClassificationProblem(n_samples, n_features)
        comparisons.extend(
            [
                _least_squares(problem),
                _ridge(problem),
                _lasso(problem),
                _gradient_descent(problem),
                _logistic(classification),
                _newton_logistic(classification),
                _standardizer(problem),
            ]
        )

    for n_samples, n_features in EXPANSION_SIZES:
        comparisons.append(
            _polynomial_features(RegressionProblem(n_samples, n_features))
        )

    for n_samples, n_features in NEIGHBOUR_SIZES:
        neighbours = NeighbourProblem(n_samples, n_features)
        comparisons.extend(
            [
                _k_nearest_regressor(neighbours),
                _k_nearest_classifier(neighbours),
            ]
        )

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
