"""The regression family, with scikit-learn doing the arithmetic.

Every class here has a namesake in ``oop_ml.numpy.regression`` with the same
name, the same pydantic fields, the same base class and the same learned
properties, so that a caller can swap one backend for the other at the import
line and change nothing else. What differs is who does the sums. The numpy
version solves its own normal equations and grows its own trees; this one
hands the validated arrays to a scikit-learn estimator and reads the answer
back into the library's own vocabulary, so ``coefficients["area"]`` is still a
:class:`~oop_ml.core.data.coefficients.Coefficients` and a fitted tree is still
a :class:`~oop_ml.core.tree.node.TreeNode` you can walk.

Where the two backends keep the same base class the shared frame does most of
the work, and that is the point of having the frames in ``core``. A linear
wrapper inherits :class:`~oop_ml.core.base.linear_model.LinearModel`, so the
design matrix, the intercept split, the by-name prediction and every guard on
the way in are the numpy backend's own, and the wrapper supplies only the
solve. A tree wrapper inherits :class:`~oop_ml.core.base.tree_model.TreeModel`,
converts the engine's fitted tree into the library's nodes once, and then
``depth``, ``n_leaves``, ``describe`` and ``feature_importances`` come from the
frame unchanged. The ensembles inherit the two ensemble frames and take their
members and their bootstrap samples from the engine, which is what lets the
out-of-bag estimate and the averaged importances work here without a second
implementation.

Hyperparameters keep this library's names and are translated at the boundary.
Every translation is written down on the wrapper that makes it, and the ones
that carry a scale factor are the ones worth reading, since the same number
means a different thing on the two sides of the boundary.

Where a numpy model exposes something the engine cannot supply, an observed
route such as ``solver_path`` or a diagnostic the engine does not record, the
member is omitted here rather than stubbed or imitated, and each wrapper lists
what it leaves out under "Not mirrored from the numpy backend".

The chores every wrapper shares, the array conversion, the name matching,
the tree walk and the engine's vocabulary for this library's metrics and
kernels, live in :mod:`oop_ml.scikit.plumbing` and are used unchanged by the
classification family.

Who holds an engine, and who does not
--------------------------------------
A wrapper keeps its fitted scikit-learn estimator only where prediction needs
it. A neighbour model, a tree, a kernel model and an ensemble all predict by
asking the engine. A linear model does not, since its learned state is the
coefficients and the intercept, and once those are read out of the engine the
frame's own ``intercept + sum(coefficients[name] * values)`` is the prediction.
Keeping an estimator that nothing would read again is the kind of dead field
the serving audit removed elsewhere.

Three bodies here are arithmetic this backend never runs.
``KNearestNeighboursRegressor._combine`` is the mean spelled out,
``DecisionTreeRegressor._leaf`` is the leaf and
``GradientBoostingRegressor._residuals`` is the negative gradient, and the
engine averages the neighbours, grows the tree and takes the round's step, so
none of the three is reached. All three stay because the frames in ``core``
declare them abstract. What is recorded instead is the reachability. Replacing
all three with a ``raise`` leaves ``test/contract`` and ``test/scikit`` exactly
as they were, where the same three substitutions on the numpy backend's
namesakes fail 48 tests in ``test/contract`` alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr, model_validator
from sklearn.ensemble import BaggingRegressor as EngineBaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor as EngineGradientBoosting
from sklearn.ensemble import RandomForestRegressor as EngineRandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor as EngineDecisionTreeRegressor

from oop_ml.core.base.ensemble import (
    AveragingEnsemble,
    AveragingMember,
    BoostingEnsemble,
    BoostingMember,
)
from oop_ml.core.base.estimator import Regressor
from oop_ml.core.base.linear_model import LinearModel
from oop_ml.core.base.neighbour_model import NeighbourModel
from oop_ml.core.base.tree_model import TreeModel
from oop_ml.core.data.coefficients import Coefficient, Coefficients
from oop_ml.core.data.column import Column
from oop_ml.core.data.dataset import Dataset
from oop_ml.core.data.design_matrix import DesignMatrix
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.ensemble.bootstrap import BootstrapSample
from oop_ml.core.ensemble.member_predictions import MemberPredictions
from oop_ml.core.evaluation.regression import RegressionEvaluation
from oop_ml.core.exceptions import (
    CollinearFeaturesError,
    InvalidValuesError,
    TooFewValuesError,
)
from oop_ml.core.kernel.functions import Kernel, LinearKernel
from oop_ml.core.tree.criterion import RegressionCriterion
from oop_ml.core.tree.impurity import Impurity
from oop_ml.core.tree.node import LeafNode
from oop_ml.core.types import FloatArray, NumericInput
from oop_ml.core.validation import ValueRole
from oop_ml.scikit.plumbing import (
    EngineMember,
    configuration_of,
    converted_tree,
    engine_kernel_parameters,
    fit_watching_convergence,
    matched_matrix,
    matrix_of,
    neighbour_engine_parameters,
    node_row_count,
    predictor_columns,
    scalar_intercept_of,
    solution_of,
)

MINIMUM_SIMPLE_REGRESSION_SAMPLES = 2
"""Two points determine a line, and anything fewer cannot pin down a slope."""

RANK_THRESHOLD = float(np.finfo(np.float64).eps)
"""How small a singular value has to be before it names no direction at all.

Machine epsilon, relative to the largest singular value, which is the tightest
threshold float64 admits. The engine's own default is 1e-6, ten orders looser,
and :class:`MultipleLinearRegression` records what that costs.
"""

ENGINE_CRITERION_NAMES: dict[RegressionCriterion, str] = {
    RegressionCriterion.SQUARED_ERROR: "squared_error",
}
"""Each regression criterion under the engine's name for it."""


class LinearEngineRegressor(
    LinearModel, EngineMember, Regressor[Sequence[Feature], Feature]
):
    """A hyperplane over named features, solved by a scikit-learn engine.

    The counterpart of the numpy backend's ``LinearFeatureRegressor``. The
    frame, :class:`~oop_ml.core.base.linear_model.LinearModel`, already
    validates the features, builds the design matrix, splits the intercept off
    the solution, pairs the weights with their names and evaluates the
    hyperplane by name. A concrete wrapper supplies only the engine.

    Parameters
    ----------
    fit_intercept:
        Inherited. Passed to the engine, which learns its intercept by
        centring rather than through a ones column, and the two are the same
        fit.
    """

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Fit the hyperplane, with the solve itself done by the engine.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        AllSameValuesError
            If any predictor is constant.
        TooFewValuesError
            If there are fewer observations than parameters to estimate.
        """
        return self._fit_linear_model(input_values, target_values)

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """Evaluate the fitted hyperplane, matching features by name.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return Predictions.already_checked(self._linear_predictor(input_values))

    def _solve(self, design_matrix: DesignMatrix, target_column: Column) -> FloatArray:
        """Fit the engine on the feature columns and read the weights back."""
        engine = self._engine_prototype(target_column.n_samples)
        engine.fit(predictor_columns(design_matrix), target_column.values)
        self._read_diagnostics(engine)

        return solution_of(engine, design_matrix)

    def _read_diagnostics(self, engine: Any) -> None:
        """What a fitted engine says about the fit beyond its weights.

        Nothing, for a closed-form solve. Lasso reads its sweep count here
        and multiple regression checks the rank.
        """

    def _adopting(self, engine: Any, training: Dataset) -> Self:
        feature_names = [feature.name for feature in training.input_features]
        weights = np.asarray(engine.coef_, dtype=np.float64).ravel()

        adopted = type(self)(**configuration_of(self))
        adopted._read_diagnostics(engine)
        adopted._intercept = scalar_intercept_of(engine)
        adopted._coefficients = Coefficients(
            [
                Coefficient(name, weight)
                for name, weight in zip(feature_names, weights, strict=True)
            ]
        )
        adopted._mark_fitted()

        return adopted


class MultipleLinearRegression(LinearEngineRegressor):
    """Ordinary least squares, solved by scikit-learn's ``LinearRegression``.

    Translation
    -----------
    ``fit_intercept`` is passed through under the same name. There are no
    other hyperparameters, and one engine field this library does not expose
    is set rather than left where the engine leaves it. ``tol`` is pinned at
    :data:`RANK_THRESHOLD`, machine epsilon, because the engine hands it to
    ``scipy.linalg.lstsq`` as ``cond`` and every singular value below ``tol``
    times the largest is then dropped from the rank *and* from the solution.
    At the engine's own default of 1e-6 that is a threshold on how far apart
    the columns' spreads are rather than on whether one is a combination of
    the others, so a plain change of units crosses it.

    Measured on forty rows of two uncorrelated columns, correlation -0.0197,
    with only the second column rescaled, and at the engine's default, the
    wrapper answered at factors 1e0 through 1e-6 and refused at 1e-7 through
    1e-12, where the numpy backend fitted throughout and scored 1.000000 at
    every factor. Pinned at epsilon the wrapper answers at all thirteen. The
    truncation is not only a refusal either. On a height in millimetres
    beside a trace mass in grams, correlation -0.0591, the engine at its
    default answers ``coef_`` ``[8.907e-03, -9.334e-11]`` at rank 1, where
    the least-squares solution both backends reach at epsilon is
    ``[1.139e-02, 2.371e+05]``, so reading a tighter rank off ``singular_``
    while leaving the engine at its default would turn a false refusal into a
    quiet wrong answer.

    ``tol`` reached ``LinearRegression`` in scikit-learn 1.7 and began to
    govern dense data in 1.9.

    The engine solves by ``lstsq`` rather than by the normal equations, so a
    rank-deficient design does not raise inside it; it returns the
    minimum-norm solution and reports the rank. The wrapper reads ``rank_``
    and refuses a deficient one, since the numpy backend raises
    :class:`~oop_ml.core.exceptions.CollinearFeaturesError` there and a
    caller who swapped backends must not find that a fit which used to be
    refused now quietly answers. The rank read back is the rank at the pinned
    threshold, so the refusal and the coefficients describe one system rather
    than two. A duplicated column still reports rank 1 across 2 and a column
    that is the total of two others still reports rank 2 across 3, which are
    the refusals worth keeping.

    Where the backends disagree
    ---------------------------
    A column small enough to underflow ``X.T X`` is refused there and
    answered here, and the pinned threshold cannot close that half. The numpy
    backend forms the moment matrix, and a value of 1e-170 squares to
    something float64 has no room for, so on the five-row plane fixture two
    independent columns both scaled by that leave the whole predictor block
    of ``X.T X`` exactly zero and its condition number infinite, and the
    numpy backend raises
    :class:`~oop_ml.core.exceptions.CollinearFeaturesError`. The engine never
    forms that matrix, fits, and scores 1.000000.

    That refusal is the end of a slope rather than a cliff, which is the half
    worth knowing. Measured on that fixture, both backends score 1.000000 at
    every factor down to 1e-150. From 1e-155 to 1e-162 the moment matrix's
    entries have gone subnormal but not to zero, so ``numpy.linalg.solve``
    still returns and the numpy backend answers, scoring 0.409829, 0.410308
    and 0.891534 at 1e-155, 1e-160 and 1e-162 where the engine still scores
    1.000000. Only at 1e-165 does the block reach exact zero and the refusal
    arrive. The disagreement is the numpy solver's floor rather than this
    wrapper's rule, and its degraded band is wider than its refusing one.

    Not mirrored from the numpy backend
    -----------------------------------
    ``normal_equations``, the observed route. The engine never forms
    ``X.T X`` and so has no condition number to report.
    """

    def _engine_prototype(self, n_rows: int) -> LinearRegression:
        return LinearRegression(fit_intercept=self.fit_intercept, tol=RANK_THRESHOLD)

    def _read_diagnostics(self, engine: Any) -> None:
        """Refuse a design the engine could only solve by choosing among many.

        Raises
        ------
        CollinearFeaturesError
            If the rank of the feature matrix is below its width.
        """
        n_columns = int(np.asarray(engine.coef_).size)

        if int(engine.rank_) < n_columns:
            raise CollinearFeaturesError(
                f"the features have rank {int(engine.rank_)} across {n_columns} "
                "columns, so one is a linear combination of the others and no "
                "unique least-squares solution exists"
            )


class RidgeRegression(LinearEngineRegressor):
    """Least squares with an L2 penalty, solved by scikit-learn's ``Ridge``.

    Translation
    -----------
    ``penalty`` is the engine's ``alpha``, and no scale factor is needed. Both
    minimise ``||y - X b||^2 + penalty * ||b||^2`` with the intercept left out
    of the norm, the numpy backend by exempting column zero of the penalty
    matrix and the engine by centring the data before it solves. On a centred
    fixture the two agree to floating point.

    ``fit_intercept`` is passed through under the same name.

    Not mirrored from the numpy backend
    -----------------------------------
    ``normal_equations``, the observed route. The engine solves the same
    system without exposing the matrices it built.
    """

    penalty: float = Field(default=1.0, ge=0.0)

    def _engine_prototype(self, n_rows: int) -> Ridge:
        return Ridge(alpha=self.penalty, fit_intercept=self.fit_intercept)


class LassoRegression(LinearEngineRegressor):
    """Least squares with an L1 penalty, solved by scikit-learn's ``Lasso``.

    Translation
    -----------
    ``penalty`` is **not** the engine's ``alpha`` unchanged, and this is the
    translation that carries a scale factor. The numpy backend minimises

        ``||y - X b||^2 + penalty * sum(abs(b_j))``

    while the engine minimises

        ``(1 / (2 n)) * ||y - X w||^2 + alpha * sum(abs(w_j))``

    Multiplying the engine's objective by ``2 n`` gives the numpy one with
    ``penalty = 2 n alpha``, so the wrapper passes ``alpha = penalty / (2 n)``
    where ``n`` is the number of training rows. That means the engine is built
    inside ``fit``, since the translation needs a number only the data can
    supply. On the numpy backend's own worked fixture, ``penalty = 12`` zeroes
    the second coefficient and ``penalty = 16`` zeroes both, and the engine
    reproduces both under this translation.

    ``max_iterations`` is the engine's ``max_iter``, a sweep for a sweep.
    ``tolerance`` is passed to ``tol`` unchanged, though the two measure
    different things. The numpy backend stops when no coefficient moved more
    than the tolerance in a sweep; the engine stops when the duality gap falls
    below it, scaled by ``||y||^2``. Both are a threshold on being finished,
    and the default here is the numpy one.

    At ``penalty = 0`` the engine warns that coordinate descent converges
    poorly and recommends ``LinearRegression``. The warning is let through,
    since it is true, and the fit still completes.

    ``fit_intercept`` is passed through under the same name; the engine
    centres and leaves the intercept unpenalised, as the numpy backend does.

    ``iterations_run`` is the engine's ``n_iter_``. ``converged`` is read
    from whether the engine issued a ``ConvergenceWarning``, which is the
    only signal it gives; ``n_iter_`` alone cannot separate a fit that
    settled on the last permitted sweep from one that ran out. A member
    adopted from a bagging engine has no such signal and reads ``converged``
    as ``n_iter_ < max_iterations`` instead.

    Not mirrored from the numpy backend
    -----------------------------------
    ``solver_path``, the observed route. The engine's coordinate descent
    keeps no record of its sweeps.
    """

    penalty: float = Field(default=1.0, ge=0.0)
    max_iterations: int = Field(default=1_000, gt=0)
    tolerance: float = Field(default=1e-10, gt=0.0)

    _iterations_run: int | None = PrivateAttr(default=None)
    _converged: bool | None = PrivateAttr(default=None)

    @property
    def iterations_run(self) -> int:
        """How many full sweeps the last fit took.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._iterations_run is not None
        return self._iterations_run

    @property
    def converged(self) -> bool:
        """Whether the last fit stopped on ``tolerance`` rather than the cap.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._converged is not None
        return self._converged

    def _engine_alpha(self, n_rows: int) -> float:
        """``penalty / (2 n)``, the scale factor the two objectives differ by."""
        return self.penalty / (2.0 * n_rows)

    def _engine_prototype(self, n_rows: int) -> Lasso:
        """The engine at the alpha this many rows call for."""
        return Lasso(
            alpha=self._engine_alpha(n_rows),
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iterations,
            tol=self.tolerance,
        )

    def _solve(self, design_matrix: DesignMatrix, target_column: Column) -> FloatArray:
        """Fit the engine at the alpha this many rows call for.

        The convergence warning is caught rather than shown, through
        :func:`~oop_ml.scikit.plumbing.fit_watching_convergence`, because it
        is the engine's only way of saying the cap was reached and
        ``converged`` is where this library says that.
        """
        engine: Any = self._engine_prototype(target_column.n_samples)
        reached_the_cap = fit_watching_convergence(
            engine, predictor_columns(design_matrix), target_column.values
        )

        self._iterations_run = int(engine.n_iter_)
        self._converged = not reached_the_cap

        return solution_of(engine, design_matrix)

    def _read_diagnostics(self, engine: Any) -> None:
        """Read the sweep count off an engine an ensemble fitted.

        Without the warning to go on, a fit that used every permitted sweep
        is read as unconverged, which is right in every case but the one
        where it settled on the very last sweep.
        """
        self._iterations_run = int(engine.n_iter_)
        self._converged = int(engine.n_iter_) < self.max_iterations


class SimpleLinearRegression(Regressor[NumericInput, NumericInput]):
    """Least-squares line over a single predictor, by ``LinearRegression``.

    Predicts ``intercept + slope * input_value`` for each observation. The
    engine sees the predictor as a one-column matrix, and ``slope`` is the one
    coefficient it learns.

    Translation
    -----------
    There are no hyperparameters. The engine always fits an intercept, which
    is what the numpy model does.

    Not mirrored from the numpy backend
    -----------------------------------
    ``least_squares_line``, the observed route. The engine does not expose
    the covariation and variation the slope is the ratio of.
    """

    _slope: float | None = PrivateAttr(default=None)
    _intercept: float | None = PrivateAttr(default=None)

    @property
    def slope(self) -> float:
        """Learned slope (available after ``fit``)."""
        self._check_fitted()
        assert self._slope is not None
        return self._slope

    @property
    def intercept(self) -> float:
        """Learned intercept (available after ``fit``)."""
        self._check_fitted()
        assert self._intercept is not None
        return self._intercept

    def fit(self, input_values: NumericInput, target_values: NumericInput) -> Self:
        """Learn ``slope`` and ``intercept`` from the training pairs.

        Raises
        ------
        NonEqualArrayLengthError
            If the two columns differ in length.
        TooFewValuesError
            If there are fewer than two rows.
        AllSameValuesError
            If the predictor does not vary.
        """
        input_column = Column.of(input_values, ValueRole.INPUT_VALUES)
        target_column = Column.of(target_values, ValueRole.TARGET_VALUES)

        input_column.check_equal_length(target_column)
        input_column.check_min_length(MINIMUM_SIMPLE_REGRESSION_SAMPLES)
        input_column.check_has_variance()

        engine = LinearRegression().fit(
            input_column.values[:, None], target_column.values
        )

        self._slope = float(np.asarray(engine.coef_).ravel()[0])
        self._intercept = float(engine.intercept_)
        self._mark_fitted()

        return self

    def predict(self, input_values: NumericInput) -> Predictions:
        """Evaluate the fitted line at each input value.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        self._check_fitted()

        input_column = Column.of(input_values, ValueRole.INPUT_VALUES)

        return Predictions.already_checked(
            input_column.values * self.slope + self.intercept
        )


class KernelRidgeRegression(Regressor[Sequence[Feature], Feature]):
    """Ridge regression through the Gram matrix, by ``KernelRidge``.

    Translation
    -----------
    ``penalty`` is the engine's ``alpha`` unchanged. Both add it to every
    entry of the Gram matrix's diagonal and solve ``(K + penalty I) a = y``.

    ``kernel`` is one of this library's :class:`~oop_ml.core.kernel.functions.Kernel`
    objects and is translated by
    :func:`~oop_ml.scikit.plumbing.engine_kernel_parameters` into the
    engine's kernel name and parameters; ``constant`` becomes ``coef0`` and
    everything else keeps its name.

    The engine does not centre anything, and the numpy backend centres both
    the features and the target, which is what makes the linear-kernel case
    agree with :class:`RidgeRegression` exactly rather than approximately.
    So this wrapper centres before it calls the engine and adds the target's
    mean back when it predicts, the same arithmetic in the same place. The
    learned state is therefore identical across the two backends: one dual
    weight per training row, the centred rows, the feature means and the
    target mean.

    Where the backends disagree
    ---------------------------
    Wherever ``K + penalty I`` comes out singular, the numpy backend raises
    :class:`~oop_ml.core.exceptions.InvalidValuesError` and the engine warns,
    falls back to a least-squares solve and answers. What comes back here is
    therefore a least-squares solution rather than the dual solve, and it is
    a number the fallback chose among many.

    The condition is the conditioning of that system and not the family the
    kernel belongs to, which is worth saying because the numpy backend's own
    message names Mercer's condition and a reader can carry that over. A
    kernel that fails Mercer's condition does reach it, and so does a
    perfectly valid one on badly scaled data. Measured on five rows and two
    features at penalty 1.0, a sigmoid kernel at ``constant=-5.0`` is refused
    there and answers 2.999086 here, and the default
    :class:`~oop_ml.core.kernel.functions.LinearKernel` on the same rows
    multiplied by 1e15 is refused there and answers 1.000000 here. At
    magnitude 1 that same kernel and those same rows give 1.265583 on both
    backends, so it is the scaling that parts them and not the kernel.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    kernel: Kernel = LinearKernel()
    penalty: float = Field(default=1.0, gt=0.0)

    _engine: KernelRidge | None = PrivateAttr(default=None)
    _training_rows: RowBlock | None = PrivateAttr(default=None)
    _feature_means: FloatArray | None = PrivateAttr(default=None)
    _target_mean: float = PrivateAttr(default=0.0)

    @property
    def dual_weights(self) -> FloatArray:
        """One weight per training row, which is what this model learned.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._engine is not None
        return np.asarray(self._engine.dual_coef_, dtype=np.float64).ravel().copy()

    @property
    def n_training_rows(self) -> int:
        """How many rows this model has to keep in order to predict.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._training_rows is not None
        return self._training_rows.n_rows

    @property
    def training_rows(self) -> RowBlock:
        """The centred rows this model kept, as a copy.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._training_rows is not None
        return rows_of(
            self._training_rows.values.copy(), self._training_rows.feature_names
        )

    @property
    def target_mean(self) -> float:
        """The mean subtracted before fitting, added back when predicting.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        return self._target_mean

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Centre, hand the engine the rows, keep what prediction needs.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonEqualArrayLengthError
            If the features and target are different lengths.
        NonUniqueFeaturesError
            If two features share a name.
        InvalidValuesError
            If the kernel is one the engine cannot be handed.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_aligned_with(target_values)

        names = tuple(feature.name for feature in feature_set)
        raw_rows = matrix_of(feature_set)
        target_column = Column.of(target_values, ValueRole.TARGET_VALUES)

        feature_means = np.mean(raw_rows, axis=0)
        rows = rows_of(raw_rows - feature_means, names)
        target_mean = float(np.mean(target_column.values))

        # scikit-learn is untyped, and pyright reads each unannotated default
        # as the parameter's type, so ``alpha=1`` would refuse a float.
        engine_type: Any = KernelRidge
        engine = engine_type(
            alpha=self.penalty, **engine_kernel_parameters(self.kernel)
        )
        engine.fit(rows.values, target_column.values - target_mean)

        self._feature_means = feature_means
        self._training_rows = rows
        self._target_mean = target_mean
        self._engine = engine
        self._mark_fitted()

        return self

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The engine's answer on the centred queries, with the mean restored.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        self._check_fitted()
        assert self._training_rows is not None
        assert self._feature_means is not None
        assert self._engine is not None

        queries = matched_matrix(self._training_rows.feature_names, input_values)
        centred = queries - self._feature_means

        return Predictions.already_checked(
            np.asarray(self._engine.predict(centred), dtype=np.float64)
            + self._target_mean
        )

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"KernelRidgeRegression({self.kernel!r}, unfitted)"

        return (
            f"KernelRidgeRegression({self.kernel!r}, penalty={self.penalty}, "
            f"n_training_rows={self.n_training_rows})"
        )


class KNearestNeighboursRegressor(
    NeighbourModel, EngineMember, Regressor[Sequence[Feature], Feature]
):
    """The mean of the nearest neighbours' targets, by ``KNeighborsRegressor``.

    Translation
    -----------
    ``n_neighbours`` is the engine's ``n_neighbors`` and ``metric`` is
    translated by
    :func:`~oop_ml.scikit.plumbing.neighbour_engine_parameters`: the six
    named metrics by name, a Minkowski distance by its order, and any other
    :class:`~oop_ml.core.distance.calculations.Distance` as a callable, which
    works and is slow.

    The engine breaks a tie between equally distant rows by its own rule, and
    the numpy backend by the order the rows were remembered in. Where the
    ``k``-th and ``k + 1``-th nearest rows sit at exactly the same distance
    the two backends can average a different neighbour.

    ``neighbour_search`` is inherited from the frame and answers with the
    library's own distance calculation over the remembered rows, not with the
    engine's search. The two find the same neighbours except on the ties just
    described.
    """

    _engine: KNeighborsRegressor | None = PrivateAttr(default=None)

    def _combine(self, neighbour_targets: FloatArray) -> FloatArray:
        """The mean of each query's neighbours' targets.

        The frame requires it. The engine finds the neighbours and averages
        them, so nothing here calls it; it is what this model's answer is.
        Measured, replacing this body with a raise leaves test/contract and
        test/scikit unchanged, where the same substitution on the numpy
        namesake takes tests down with it.
        """
        return neighbour_targets.mean(axis=1)

    def _engine_prototype(self, n_rows: int) -> KNeighborsRegressor:
        # The untyped engine's defaults read as its parameter types, and
        # ``p`` and ``metric`` take more than those.
        engine_type: Any = KNeighborsRegressor

        return engine_type(
            **neighbour_engine_parameters(self.n_neighbours, self.metric)
        )

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Remember the rows and hand them to the engine.

        The same guards as the frame's ``_remember``, in the same order, but
        with nothing assigned until the engine has fitted, so a failure inside
        it cannot leave a model that says it is fitted and has no engine.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        TooFewValuesError
            If there are fewer rows than ``n_neighbours``.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_aligned_with(target_values)

        if feature_set.n_samples < self.n_neighbours:
            raise TooFewValuesError(
                f"{self.n_neighbours} neighbours were asked for and only "
                f"{feature_set.n_samples} rows were supplied"
            )

        names = tuple(feature.name for feature in feature_set)
        rows = rows_of(feature_set.feature_matrix, names)
        targets = target_values.column.values

        engine = self._engine_prototype(feature_set.n_samples).fit(rows.values, targets)
        self._absorb(engine, names, rows, targets)

        return self

    def _absorb(
        self,
        engine: KNeighborsRegressor,
        names: tuple[str, ...],
        rows: RowBlock,
        targets: FloatArray,
    ) -> None:
        self._feature_names = names
        self._remembered_rows = rows
        self._remembered_targets = targets
        self._engine = engine
        self._mark_fitted()

    def _adopting(self, engine: Any, training: Dataset) -> Self:
        feature_set = FeatureSet(training.input_features)
        names = tuple(feature.name for feature in feature_set)

        adopted = type(self)(**configuration_of(self))
        adopted._absorb(
            engine,
            names,
            rows_of(feature_set.feature_matrix, names),
            training.target_feature.column.values,
        )

        return adopted

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The engine's mean over each query's nearest rows.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        query_rows = self._matched_rows(input_values)
        assert self._engine is not None

        return Predictions.already_checked(
            np.asarray(self._engine.predict(query_rows.values), dtype=np.float64)
        )


class DecisionTreeRegressor(
    TreeModel, EngineMember, Regressor[Sequence[Feature], Feature]
):
    """A regression tree grown by scikit-learn's ``DecisionTreeRegressor``.

    The engine grows the tree, which is the expensive half and the half
    written in Cython. The fitted tree is then converted once into the
    library's own nodes, so ``root``, ``depth``, ``n_leaves``, ``describe``
    and ``feature_importances`` all come from the
    :class:`~oop_ml.core.base.tree_model.TreeModel` frame reading a tree it
    did not grow, and ``split_search`` runs the frame's own search on any
    node handed to it.

    Translation
    -----------
    ``criterion`` is translated by name through :data:`ENGINE_CRITERION_NAMES`.
    ``max_depth``, ``min_samples_split`` and ``min_samples_leaf`` pass through
    unchanged, as does ``max_features`` as a count. ``random_seed`` is the
    engine's ``random_state``.

    ``min_impurity_decrease`` passes through under its own name and does not
    mean quite the same thing. The numpy backend compares a split's gain to it
    directly; the engine first scales the gain by the node's share of the
    training rows, so at any value above zero the engine stops growing deep
    nodes sooner than the numpy backend does. At the default of zero the two
    agree.

    The engine also draws on ``random_state`` when ``max_features`` is
    ``None``, to permute the features it scans, and that decides which of two
    equally good splits it keeps. The numpy backend keeps the earlier feature.
    Where a fixture has an exact tie the two can grow different trees of the
    same quality.

    Where the backends disagree
    ---------------------------
    A row exactly at a threshold goes left in the engine and right in this
    library's own routing; see
    :func:`~oop_ml.scikit.plumbing.converted_tree`. Predictions here follow
    the engine.
    """

    criterion: RegressionCriterion = RegressionCriterion.SQUARED_ERROR

    _engine: EngineDecisionTreeRegressor | None = PrivateAttr(default=None)

    @property
    def _impurity(self) -> Impurity:
        return self.criterion.impurity

    def _leaf(self, target_values: Column) -> LeafNode:
        """A leaf predicting the mean of these targets.

        The frame requires it. The engine grows the tree, so nothing here
        calls it during a fit; it is what a leaf of this tree is.
        """
        return LeafNode(
            prediction=float(np.mean(target_values.values)),
            n_samples=target_values.n_samples,
            impurity=self._impurity.of(target_values),
        )

    @staticmethod
    def _engine_leaf(tree: Any, node_id: int) -> LeafNode:
        """A leaf of the engine's tree, which stores the mean in ``value``."""
        return LeafNode(
            prediction=float(tree.value[node_id, 0, 0]),
            n_samples=node_row_count(tree, node_id),
            impurity=float(tree.impurity[node_id]),
        )

    def _engine_prototype(self, n_rows: int) -> EngineDecisionTreeRegressor:
        return EngineDecisionTreeRegressor(
            criterion=ENGINE_CRITERION_NAMES[self.criterion],
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            random_state=self.random_seed,
        )

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Have the engine grow the tree, then convert it for reading.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_aligned_with(target_values)

        names = tuple(feature.name for feature in feature_set)
        engine = self._engine_prototype(feature_set.n_samples).fit(
            matrix_of(feature_set), self._validated_target(target_values).values
        )
        self._absorb(engine, names)

        return self

    def _absorb(
        self, engine: EngineDecisionTreeRegressor, names: tuple[str, ...]
    ) -> None:
        """Commit a fitted engine and its converted tree together."""
        root = converted_tree(engine, names, self._engine_leaf)

        self._feature_names = names
        self._root = root
        self._engine = engine
        self._mark_fitted()

    def _adopting(self, engine: Any, training: Dataset) -> Self:
        """Wrap a tree an ensemble grew, carrying the seed the engine gave it.

        A bagging engine seeds every member itself, so the adopted wrapper
        reads its ``random_seed`` from the engine rather than from this
        prototype, which is what the numpy ensembles do by offsetting the
        seed per member.
        """
        configuration = configuration_of(self)
        engine_seed = getattr(engine, "random_state", None)
        if isinstance(engine_seed, int):
            configuration["random_seed"] = engine_seed

        adopted = type(self)(**configuration)
        adopted._absorb(
            engine, tuple(feature.name for feature in training.input_features)
        )

        return adopted

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The mean of the box each row falls in, as the engine routes it.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        rows = self._matched_rows(input_values)
        assert self._engine is not None

        return Predictions.already_checked(
            np.asarray(self._engine.predict(rows.values), dtype=np.float64)
        )


class BaggingRegressor(AveragingEnsemble, Regressor[Sequence[Feature], Feature]):
    """Bootstrap-averaged regressors, by scikit-learn's ``BaggingRegressor``.

    The engine draws the resamples, fits a member on each and averages them.
    The wrapper reads the fitted members and the rows each one drew back out
    of it, so ``members``, ``samples``, ``feature_importances`` and the whole
    out-of-bag estimate are the
    :class:`~oop_ml.core.base.ensemble.AveragingEnsemble` frame's, reading
    state the engine produced.

    Parameters
    ----------
    base_model:
        The prototype every member is built from. It has to be a model from
        this backend that can hand the engine a prototype and wrap a fitted
        one back, which is every regressor here except kernel ridge; a numpy
        model is refused at construction rather than silently replaced.
        Defaults to an unpruned tree.
    n_members, random_seed:
        Inherited from the frame.

    Translation
    -----------
    ``n_members`` is the engine's ``n_estimators`` and ``random_seed`` its
    ``random_state``. The engine's ``bootstrap``, ``max_samples`` and
    ``max_features`` are left at the defaults that make it draw ``n`` rows
    with replacement over every feature, which is what the numpy backend
    does. The engine seeds each member from its own stream, where the numpy
    backend offsets the member's own seed by position; a member read back
    carries the seed the engine gave it.

    A tree member's ``min_samples_split`` and ``min_samples_leaf`` do not
    keep their meaning inside the resample, and it is the one field of a
    member that this backend makes stricter. The engine does not fit a member
    on the drawn rows; it fits it on every row carrying a weight equal to how
    often the resample drew it, and its splitter counts only the positively
    weighted ones, so both fields are compared against the count of
    *distinct* drawn rows. The numpy backend resamples outright, so the same
    two fields count rows with their repeats. A bootstrap of ``n`` rows holds
    about ``0.632 n`` distinct rows, which makes the same number roughly 1.6
    times stricter here. Measured on 200 rows and four features at 6 members
    and seed 3, the six bootstraps drew 129, 128, 125, 122, 118 and 129
    distinct rows, and at ``min_samples_leaf=20`` the numpy members' smallest
    leaves hold 20, 20, 21, 24, 20 and 20 resampled rows while the members
    here hold 30, 27, 30, 33, 35 and 27, whose distinct counts are 20, 21,
    20, 20, 20 and 20. The constraint is binding on the distinct count
    exactly.

    The shift reaches only the tree members. A bagged
    :class:`KNearestNeighboursRegressor` is fitted on indexed rows rather
    than weighted ones, and :class:`GradientBoostingRegressor` does not
    resample at all, so neither moves. The counts a member reports are the
    weighted ones, for the reason
    :func:`~oop_ml.scikit.plumbing.node_row_count` gives; this is the other
    consequence of the same fact.

    Raises
    ------
    InvalidValuesError
        If ``base_model`` is not an engine-backed model of this backend.
    """

    base_model: Regressor = DecisionTreeRegressor()

    _engine: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _refuse_a_member_without_an_engine(self) -> Self:
        if not isinstance(self.base_model, EngineMember):
            raise InvalidValuesError(
                f"{type(self.base_model).__name__} cannot be a member here: a "
                "scikit-learn bagging engine needs a member that can hand it a "
                "scikit-learn prototype, so base_model must be a regressor "
                "from oop_ml.scikit"
            )

        return self

    def _prototype(self, position: int) -> AveragingMember:
        """The configured member. The engine seeds each copy itself."""
        return self.base_model

    def _member_answer(
        self, member: AveragingMember, input_values: Sequence[Feature]
    ) -> FloatArray:
        assert isinstance(member, Regressor)
        return member.predict(input_values).values

    def _combine(self, member_predictions: MemberPredictions) -> FloatArray:
        """The mean of what the members said, every member counting equally."""
        return member_predictions.values.mean(axis=0)

    def _ensemble_engine(self, n_rows: int) -> Any:
        """The unfitted engine, configured from this model's fields.

        Parameters
        ----------
        n_rows:
            How many rows each member will be fitted on, which for a
            bootstrap is the training set's own count.
        """
        assert isinstance(self.base_model, EngineMember)

        return EngineBaggingRegressor(
            estimator=self.base_model._engine_prototype(n_rows),
            n_estimators=self.n_members,
            random_state=self.random_seed,
        )

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Have the engine fit every member, then read them back as models.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        """
        dataset = Dataset(input_values, target_values)
        self._validated_target(target_values)
        names = tuple(feature.name for feature in dataset.input_features)

        engine = self._ensemble_engine(dataset.n_samples).fit(
            matrix_of(FeatureSet(dataset.input_features)),
            dataset.target_feature.values,
        )

        samples = tuple(
            BootstrapSample(np.asarray(drawn, dtype=np.intp), dataset.n_samples)
            for drawn in engine.estimators_samples_
        )
        members: list[AveragingMember] = []
        for position, (fitted, sample) in enumerate(
            zip(engine.estimators_, samples, strict=True)
        ):
            prototype = self._prototype(position)
            assert isinstance(prototype, EngineMember)
            members.append(
                prototype._adopting(fitted, dataset.select_rows(sample.drawn))
            )

        self._feature_names = names
        self._samples = samples
        self._members = tuple(members)
        self._training = dataset
        self._engine = engine
        self._mark_fitted()

        return self

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The engine's mean over its members, one value per row.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        matched = self._matched_rows(input_values)
        assert self._engine is not None

        return Predictions.already_checked(
            np.asarray(self._engine.predict(matrix_of(matched)), dtype=np.float64)
        )

    def out_of_bag_evaluate(self) -> RegressionEvaluation:
        """Score the fit against rows each member never drew.

        Computed by the frame from the samples the engine reported, rather
        than read from the engine's own ``oob_prediction_``, which fills a
        row no member missed with zero. Zero is a number a caller can average
        by accident, and the frame marks such a row uncovered instead.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        estimate = self.out_of_bag_estimate()
        assert self._training is not None
        actual = self._training.target_feature.values[estimate.covered]

        return RegressionEvaluation(actual, estimate.covered_predictions)

    def out_of_bag_score(self) -> float:
        """R^2 against the rows each member never drew.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        return self.out_of_bag_evaluate().r2_score


class RandomForestRegressor(BaggingRegressor):
    """Bagged trees with a per-node feature restriction, by ``RandomForestRegressor``.

    Parameters
    ----------
    max_features:
        How many features each node may consider. ``None`` restricts nothing.
    max_depth, min_samples_split, min_samples_leaf:
        Passed to every member.
    n_members, random_seed:
        Inherited.

    Translation
    -----------
    ``n_members`` is the engine's ``n_estimators``; the four tree fields pass
    through to the engine under their own names, ``max_features`` as a count
    and ``None`` as the engine's "every feature"; ``random_seed`` is
    ``random_state``. The engine's ``bootstrap`` is left on, which is what
    makes this bagging.

    Passing through to the engine is not the same as meaning the same thing
    inside a resampled tree, and ``min_samples_split`` and
    ``min_samples_leaf`` are the two that part company, for the reason
    :class:`BaggingRegressor` sets out. The engine weights the rows where the
    numpy backend resamples them, so both are compared against the count of
    distinct drawn rows and are roughly 1.6 times stricter here. Measured on
    200 rows and four features at 6 members, seed 3 and
    ``min_samples_split=40``, the numpy members' smallest split nodes hold
    40, 41, 40, 52, 40 and 43 resampled rows while the members here hold 63,
    76, 68, 71, 75 and 72, whose distinct counts are 40, 48, 44, 46, 46 and
    49.

    The engine exposes which rows each tree drew, so ``samples`` and the
    out-of-bag estimate are the frame's, as for :class:`BaggingRegressor`.

    Raises
    ------
    InvalidValuesError
        If ``base_model`` is configured. A forest builds its own trees.
    """

    max_features: int | None = Field(default=None, ge=1)
    max_depth: int | None = Field(default=None, ge=1)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _refuse_a_configured_base_model(self) -> Self:
        """Raise if a caller configured the one field a forest ignores.

        The default passes, so a search rebuilding candidates field-by-field
        is unaffected.
        """
        if self.base_model != type(self).model_fields["base_model"].default:
            raise InvalidValuesError(
                "a forest builds its own trees and ignores base_model; "
                "configure max_depth, min_samples_split, min_samples_leaf and "
                "max_features on the forest itself"
            )

        return self

    def _prototype(self, position: int) -> AveragingMember:
        """A tree configured to restrict its features.

        The seed offset by position is what the numpy backend does; here the
        engine seeds every tree itself and the adopted member reads that
        seed, so the offset only describes the prototype.
        """
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_seed=(
                None if self.random_seed is None else self.random_seed + position
            ),
        )

    def _ensemble_engine(self, n_rows: int) -> Any:
        # The untyped engine reads ``max_features=1.0`` as float, refusing None.
        engine_type: Any = EngineRandomForestRegressor

        return engine_type(
            n_estimators=self.n_members,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=True,
            random_state=self.random_seed,
        )


class GradientBoostingRegressor(
    BoostingEnsemble, Regressor[Sequence[Feature], Feature]
):
    """A constant plus a sum of shrunken trees, by ``GradientBoostingRegressor``.

    Parameters
    ----------
    max_depth:
        How deep each round's tree may go. Three, by default, for the reason
        the numpy backend gives.
    min_samples_split, min_samples_leaf:
        Passed to every member.
    n_rounds, learning_rate:
        Inherited from the frame.

    Translation
    -----------
    ``n_rounds`` is the engine's ``n_estimators``; ``learning_rate`` and the
    three tree fields pass through under their own names. The loss is fixed
    at squared error, which is the only one the numpy backend fits, and the
    engine's initial estimate is left at its default, which for squared
    error is the target's mean and is what ``initial_prediction`` reports.

    The engine's ``subsample`` is left at one, so every round sees every row,
    and its ``random_state`` is left unset. The numpy backend has no seed
    because nothing in plain boosting is drawn; the engine still permutes
    the features each tree scans, so on an exact tie between two splits two
    fits here can differ.

    ``members`` are the engine's trees wrapped as
    :class:`DecisionTreeRegressor`, one per round in order. Each was fitted
    on that round's residuals, and the wrapper takes only the feature names
    from the training set it is handed.
    """

    max_depth: int | None = Field(default=3, ge=1)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)

    _engine: Any = PrivateAttr(default=None)

    def _prototype(self, round_number: int) -> BoostingMember:
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )

    def _residuals(
        self, target_values: FloatArray, predictions: FloatArray
    ) -> FloatArray:
        """What squared error leaves unexplained, the plain difference.

        The frame requires it. The engine runs the rounds and takes its own
        residual, so nothing here calls it, the same as the ``_leaf`` and
        ``_combine`` bodies the other wrappers keep for their frames.
        """
        return target_values - predictions

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Have the engine fit the rounds, then read them back as trees.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        """
        dataset = Dataset(input_values, target_values)
        self._validated_target(target_values)
        names = tuple(feature.name for feature in dataset.input_features)
        matrix = matrix_of(FeatureSet(dataset.input_features))

        # The untyped engine reads ``max_depth=3`` as int, refusing None.
        engine_type: Any = EngineGradientBoosting
        engine = engine_type(
            loss="squared_error",
            n_estimators=self.n_rounds,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        ).fit(matrix, dataset.target_feature.values)

        members: list[BoostingMember] = []
        for round_number, fitted in enumerate(engine.estimators_[:, 0], start=1):
            prototype = self._prototype(round_number)
            assert isinstance(prototype, EngineMember)
            members.append(prototype._adopting(fitted, dataset))

        initial_prediction = float(np.asarray(engine.init_.predict(matrix[:1]))[0])

        self._feature_names = names
        self._members = tuple(members)
        self._initial_prediction = initial_prediction
        self._engine = engine
        self._mark_fitted()

        return self

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The engine's constant plus every round's shrunken contribution.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        matched = self._matched_rows(input_values)
        assert self._engine is not None

        return Predictions.already_checked(
            np.asarray(self._engine.predict(matrix_of(matched)), dtype=np.float64)
        )


__all__ = [
    "BaggingRegressor",
    "DecisionTreeRegressor",
    "GradientBoostingRegressor",
    "KNearestNeighboursRegressor",
    "KernelRidgeRegression",
    "LassoRegression",
    "MultipleLinearRegression",
    "RandomForestRegressor",
    "RidgeRegression",
    "SimpleLinearRegression",
]
