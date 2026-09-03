"""The classification family, with scikit-learn doing the arithmetic.

Every class here has a namesake in ``oop_ml.numpy.classification`` with the
same name, the same pydantic fields, the same base class and the same learned
properties, so a caller swaps backends at the import line and changes nothing
else. What differs is who does the sums. The numpy version climbs its own
likelihood and grows its own trees; this one hands the validated arrays to a
scikit-learn estimator and reads the answer back into the library's own
vocabulary, so a binary model's ``coefficients["hours"]`` is still a
:class:`~oop_ml.core.data.coefficients.Coefficients`, a tree is still a
:class:`~oop_ml.core.tree.node.TreeNode` you can walk, and a support vector
machine still answers with the numpy backend's ``SupportVectors``.

Two distinctions the numpy backend draws are kept exactly. A binary
classifier tightens its target to 0/1, refuses anything else with
:class:`~oop_ml.core.exceptions.NonBinaryLabelsError`, and answers
``predict_probability`` with :class:`~oop_ml.core.data.probabilities.Probabilities`.
A multi-class one answers ``predict_probabilities`` with a
:class:`~oop_ml.core.data.probabilities.ProbabilityMatrix` whose rows sum to
one, except :class:`OneVsRestClassifier`, whose K models were never asked to
agree and which therefore returns the weaker
:class:`~oop_ml.core.data.probabilities.ClassScores`.

Where the frame is shared, it does the work. The binary wrappers inherit the
numpy backend's
:class:`~oop_ml.numpy.classification.linear_classifier.LinearClassifier`, which
is written entirely against ``core`` and carries the threshold rule,
the by-name prediction, ``decision_boundary_at`` and ``odds_multiplier_for``;
a wrapper supplies only the solve. The tree, neighbour and ensemble wrappers
inherit the same frames the regression family does, and the chores every
wrapper shares live in :mod:`oop_ml.scikit.plumbing`.

A field the engine cannot read is refused, not ignored
------------------------------------------------------
Three numpy models walk with a ``learning_rate``: the two gradient-ascent
logistic regressions and the support vector machine's projected ascent. Their
engines choose their own step, so the field has no meaning here. Accepting it
silently is the failure ``extra="forbid"`` exists to stop, a configured value
that changes nothing, so each wrapper keeps the field for the signature and
refuses any value but the default at construction, the way the forests refuse
a configured ``base_model``. The default passes, so a search rebuilding
candidates field by field is unaffected.

Where a numpy model exposes something the engine cannot supply, an observed
route or a solver diagnostic, the member is omitted rather than stubbed, and
each wrapper lists what it leaves out under "Not mirrored from the numpy
backend".

Who holds an engine, and who does not
-------------------------------------
A wrapper keeps its fitted scikit-learn estimator only where a later call
needs it. The neighbour model, the tree, both ensembles and the support
vector machine all answer by asking the engine, so those five keep theirs.
The four linear wrappers do not. For the two binary logistic models and
:class:`MultinomialLogisticRegression` the whole learned state is the
coefficients and the intercept, and once those are read out of the engine the
prediction is the frame's own threshold rule, or the shared
:func:`~oop_ml.core.logistic.stable_softmax` over ``X b``;
:class:`OneVsRestClassifier` holds K such models and so keeps no estimator
either. Keeping one that nothing would read again is the kind of dead field
the serving audit removed elsewhere.

Two bodies here are arithmetic this backend never runs.
``KNearestNeighboursClassifier._combine`` is the vote spelled out and
``DecisionTreeClassifier._leaf`` is the leaf, and the engine seats the voters
and grows the tree, so neither is reached. Both stay because the frames in
``core`` declare them abstract, and a class missing either cannot be
constructed, which was checked rather than assumed. What is recorded instead
is the reachability. Replacing both with a ``raise`` leaves ``test/contract``
and ``test/scikit`` exactly as they were, where the same two substitutions on
the numpy backend's namesakes fail 40 tests in ``test/contract`` alone.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr, model_validator
from sklearn.ensemble import BaggingClassifier as EngineBaggingClassifier
from sklearn.ensemble import RandomForestClassifier as EngineRandomForestClassifier
from sklearn.linear_model import LogisticRegression as EngineLogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as EngineDecisionTreeClassifier

from oop_ml.core.base.ensemble import AveragingEnsemble, AveragingMember
from oop_ml.core.base.estimator import Classifier, Fittable, MultiClassClassifier
from oop_ml.core.base.neighbour_model import NeighbourModel
from oop_ml.core.base.tree_model import TreeModel
from oop_ml.core.data.coefficients import Coefficient, Coefficients
from oop_ml.core.data.column import Column
from oop_ml.core.data.dataset import Dataset
from oop_ml.core.data.design_matrix import DesignMatrix
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.probabilities import (
    ClassScores,
    Probabilities,
    ProbabilityMatrix,
)
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.ensemble.bootstrap import BootstrapSample
from oop_ml.core.ensemble.member_predictions import MemberPredictions
from oop_ml.core.evaluation.multiclass import MultiClassEvaluation
from oop_ml.core.exceptions import InvalidValuesError, TooFewValuesError
from oop_ml.core.kernel.functions import Kernel, LinearKernel
from oop_ml.core.kernel.matrix import KernelMatrix
from oop_ml.core.logistic import stable_logistic, stable_softmax
from oop_ml.core.tree.criterion import ClassificationCriterion
from oop_ml.core.tree.impurity import Impurity
from oop_ml.core.tree.node import ClassificationLeaf, LeafNode
from oop_ml.core.types import FloatArray
from oop_ml.core.validation import ValueRole
from oop_ml.numpy.classification.kernels.support_vector_classifier import (
    SUPPORT_VECTOR_THRESHOLD,
    SupportVector,
    SupportVectors,
)
from oop_ml.numpy.classification.linear_classifier import LinearClassifier
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

UNPENALISED = np.inf
"""The engine's ``C`` for a logistic regression with no penalty at all.

``C`` is the inverse of the penalty strength, so infinity is none, and it is
the spelling scikit-learn 1.8 chose when it deprecated ``penalty=None``. The
numpy backend maximises the plain likelihood, so this is the only setting
under which the two fit the same objective.
"""

ENGINE_CRITERION_NAMES: dict[ClassificationCriterion, str] = {
    ClassificationCriterion.GINI: "gini",
    ClassificationCriterion.ENTROPY: "entropy",
}
"""Each classification criterion under the engine's name for it."""

DECISION_VALUE_CLIP = 500.0
"""Beyond this a decision value's logistic is 0 or 1 to the last bit anyway."""


def refuse_a_configured_step_size(model: Fittable, engine_name: str) -> None:
    """Raise if ``learning_rate`` was set on a model whose engine has no step.

    The default passes, so a candidate rebuilt field by field survives.

    Raises
    ------
    InvalidValuesError
        If ``learning_rate`` is anything but its default.
    """
    default = type(model).model_fields["learning_rate"].default

    if model.learning_rate != default:  # type: ignore[attr-defined]
        raise InvalidValuesError(
            f"{engine_name} chooses its own step size, so learning_rate is never "
            f"read by this backend; leave it at its default of {default}, or "
            "use the numpy backend where it means something"
        )


def full_width_scores(
    narrow: FloatArray, engine_classes: Any, n_classes: int
) -> FloatArray:
    """The engine's per-class columns placed into a matrix of the stated width.

    An engine only knows the classes its fit saw, and a member fitted on a
    resample can miss one. Its ``predict_proba`` is then narrower than the
    problem, and a column of zeros is the honest reading for a class it never
    met: the engine assigns it no chance, rather than an unknown one.

    Raises
    ------
    InvalidValuesError
        If the engine names a class at or beyond the stated width.
    """
    positions = np.asarray(engine_classes, dtype=np.float64).astype(np.int64)

    if positions.size and int(positions.max()) >= n_classes:
        raise InvalidValuesError(
            f"the engine saw class {int(positions.max())} in a problem stated "
            f"to have {n_classes} classes"
        )

    full = np.zeros((narrow.shape[0], n_classes), dtype=np.float64)
    full[:, positions] = narrow

    return full


def first_iteration_count(engine: Any) -> int:
    """``n_iter_`` as one number, whatever shape the engine reports it in.

    A logistic engine reports one count per class in an array; a support
    vector engine one per class pair. For the binary and the multinomial
    cases there is exactly one, and this reads it.
    """
    return int(np.asarray(engine.n_iter_).ravel()[0])


class LinearEngineClassifier(LinearClassifier, EngineMember):
    """A linear boundary over named features, solved by a scikit-learn engine.

    The frame, :class:`~oop_ml.numpy.classification.linear_classifier.LinearClassifier`,
    validates the features, insists the target is 0/1 with both classes, builds
    the design matrix, splits the intercept off the solution, pairs the weights
    with their names and thresholds the sigmoid. It imports nothing from the
    numpy backend's arithmetic, which is why this backend inherits it rather
    than copying it. A concrete wrapper supplies only the engine and the cap
    it reports its iterations against.

    Parameters
    ----------
    fit_intercept:
        Inherited. Passed to the engine, which learns its intercept by
        centring rather than through a ones column, and the two are the same
        fit.
    """

    _iterations_run: int | None = PrivateAttr(default=None)
    _converged: bool | None = PrivateAttr(default=None)

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

    @property
    @abstractmethod
    def _pass_limit(self) -> int:
        """The cap on engine iterations, under whichever name the model gives it."""

    @staticmethod
    def _sigmoid(linear_predictor: FloatArray) -> Probabilities:
        """The overflow-safe logistic from ``core``, carrying its bound."""
        return Probabilities(stable_logistic(linear_predictor))

    def _solve(self, design_matrix: DesignMatrix, target_column: Column) -> FloatArray:
        """Fit the engine on the feature columns and read the weights back."""
        engine = self._engine_prototype(target_column.n_samples)
        reached_the_cap = fit_watching_convergence(
            engine, predictor_columns(design_matrix), target_column.values
        )
        self._read_diagnostics(engine, reached_the_cap)

        return solution_of(engine, design_matrix)

    def _read_diagnostics(self, engine: Any, reached_the_cap: bool | None) -> None:
        """Read the iteration count and whether the engine settled.

        A fit this wrapper ran knows from the engine's warning whether the cap
        was reached. A member adopted from an ensemble has no such signal and
        reads ``converged`` as ``n_iter_ < cap`` instead, which is right in
        every case but the one where it settled on the very last iteration.
        """
        iterations_run = first_iteration_count(engine)

        self._iterations_run = iterations_run
        self._converged = (
            iterations_run < self._pass_limit
            if reached_the_cap is None
            else not reached_the_cap
        )

    def _adopting(self, engine: Any, training: Dataset) -> Self:
        feature_names = [feature.name for feature in training.input_features]
        weights = np.asarray(engine.coef_, dtype=np.float64).ravel()

        adopted = type(self)(**configuration_of(self))
        adopted._read_diagnostics(engine, None)
        adopted._intercept = scalar_intercept_of(engine)
        adopted._coefficients = Coefficients(
            [
                Coefficient(name, weight)
                for name, weight in zip(feature_names, weights, strict=True)
            ]
        )
        adopted._mark_fitted()

        return adopted


class LogisticRegression(LinearEngineClassifier):
    """A linear boundary by maximum likelihood, solved by the engine's L-BFGS.

    Translation
    -----------
    The numpy model maximises the unpenalised log-likelihood by gradient
    ascent. The engine is asked for the same objective with ``C`` at
    :data:`UNPENALISED` and its ``lbfgs`` solver, which reaches the same
    maximum by a quasi-Newton walk. Measured on the numpy backend's
    eight-student fixture, both models at their defaults, the two backends
    agree to 3.4e-06, and what accounts for that is the ascent stopping
    short rather than the engine being loose. Against the maximum both
    Newton routes reach, the engine's L-BFGS answer sits 1.6e-09 away and
    the numpy ascent's sits 3.4e-06 away, which is the stronger evidence
    that ``C`` at infinity really is the unpenalised objective.

    ``max_epochs`` is the engine's ``max_iter``, a cap for a cap, though an
    L-BFGS iteration is a curvature-informed step and an epoch is a fixed
    one, so the engine needs far fewer. Measured on the same fixture at the
    defaults, 12 iterations here where the ascent runs 4495 epochs.
    ``tolerance`` is passed to ``tol`` unchanged and measures a different
    thing, the projected gradient's norm rather than the coefficients'
    movement. ``threshold`` and ``fit_intercept`` are the frame's own and
    never reach the engine.

    ``learning_rate`` is refused at any value but its default; see the module
    docstring. ``epochs_run`` is the engine's ``n_iter_`` and ``converged``
    is read from whether it issued a ``ConvergenceWarning``.

    Where the backends disagree
    ---------------------------
    On separable classes the likelihood has no maximum. The numpy backend
    reports ``converged=False`` with the coefficients still climbing; the
    engine's L-BFGS stops when the gradient is flat enough, which on such
    data it reaches at very large coefficients, and it may report that as
    settled.

    Not mirrored from the numpy backend
    -----------------------------------
    ``solver_path``, the observed route. The engine keeps no record of its
    iterations.
    """

    learning_rate: float = Field(default=0.1, gt=0.0)
    max_epochs: int = Field(default=10_000, gt=0)
    tolerance: float = Field(default=1e-8, gt=0.0)
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _refuse_a_configured_learning_rate(self) -> Self:
        refuse_a_configured_step_size(self, "the engine's L-BFGS solver")
        return self

    @property
    def _pass_limit(self) -> int:
        return self.max_epochs

    @property
    def epochs_run(self) -> int:
        """How many L-BFGS iterations the fit took.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._iterations_run is not None
        return self._iterations_run

    def _engine_prototype(self, n_rows: int) -> EngineLogisticRegression:
        # The untyped engine reads ``C=1.0`` as float and refuses numpy's inf
        # only by pyright's reading; naming the constructor as Any keeps the
        # call honest without a cast.
        engine_type: Any = EngineLogisticRegression

        return engine_type(
            C=UNPENALISED,
            solver="lbfgs",
            max_iter=self.max_epochs,
            tol=self.tolerance,
            fit_intercept=self.fit_intercept,
        )


class NewtonLogisticRegression(LinearEngineClassifier):
    """Logistic regression by Newton's method, solved by ``newton-cholesky``.

    Translation
    -----------
    The numpy model is iteratively reweighted least squares: each step solves
    ``X.T W X`` against the gradient. The engine's ``newton-cholesky`` solver
    is that same step, a Cholesky solve of the same Hessian, with ``C`` at
    :data:`UNPENALISED` so the objective matches. On the eight-student
    fixture both settle on the same coefficients, agreeing to 4.4e-16.

    ``max_iterations`` is the engine's ``max_iter``, an iteration for an
    iteration. ``tolerance`` is passed to ``tol`` unchanged and measures the
    gradient's norm rather than the coefficients' movement. ``threshold`` and
    ``fit_intercept`` are the frame's own. ``iterations_run`` is the engine's
    ``n_iter_`` and ``converged`` is read from its ``ConvergenceWarning``.

    Where the backends disagree
    ---------------------------
    The count of steps, by one. On the eight-student fixture the numpy
    backend reports six iterations and the engine five. The step is the same
    either way and only the stopping rule differs. Measured, the fifth step
    leaves the gradient's largest component at 2.2e-16, which is inside the
    shared default tolerance of 1e-10, so the engine stops on it; the numpy
    rule tests the movement of the step just taken, which was 2.7e-08, so it
    takes a sixth step of 9.2e-16 to observe that nothing moved.

    Separation. The numpy backend raises
    :class:`~oop_ml.core.exceptions.SingularHessianError` once every
    ``p (1 - p)`` weight collapses, or reports ``converged=False`` before
    that. Measured, the engine ran 25 iterations on the separable fixture to
    a coefficient of 88.86 and returned without a warning, so a fit that is
    refused there completes here with enormous coefficients.

    Not mirrored from the numpy backend
    -----------------------------------
    ``solver_path``, the observed route. The engine keeps no record of its
    iterations.
    """

    max_iterations: int = Field(default=100, gt=0)
    tolerance: float = Field(default=1e-10, gt=0.0)
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)

    @property
    def _pass_limit(self) -> int:
        return self.max_iterations

    @property
    def iterations_run(self) -> int:
        """How many Newton steps the fit took.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._iterations_run is not None
        return self._iterations_run

    def _engine_prototype(self, n_rows: int) -> EngineLogisticRegression:
        engine_type: Any = EngineLogisticRegression

        return engine_type(
            C=UNPENALISED,
            solver="newton-cholesky",
            max_iter=self.max_iterations,
            tol=self.tolerance,
            fit_intercept=self.fit_intercept,
        )


class MultinomialLogisticRegression(MultiClassClassifier[Sequence[Feature], Feature]):
    """A softmax classifier with class 0 as the reference, solved by L-BFGS.

    Translation
    -----------
    The engine fits the multinomial objective with one weight vector per
    class and nothing pinned, so its parameters sit anywhere along the ridge
    the numpy module docstring describes. The numpy backend pins class 0 at
    zero. The two are the same model: subtracting class 0's row from every
    row moves the engine's answer onto the pinned parametrisation without
    changing a single probability, and that is what ``coefficients_for`` and
    ``intercepts`` report. With exactly two classes the engine fits a binary
    model instead, whose one weight vector is already ``b_1 - b_0``.

    ``max_epochs`` is the engine's ``max_iter`` and ``tolerance`` its ``tol``,
    with the caveats :class:`LogisticRegression` gives; ``fit_intercept``
    passes through under its own name; ``C`` is :data:`UNPENALISED`.
    ``learning_rate`` is refused at any value but its default; see the module
    docstring. ``epochs_run`` is the engine's ``n_iter_`` and ``converged``
    is read from its ``ConvergenceWarning``.

    Prediction does not consult the engine. The learned state is the
    coefficients and the intercepts, and once those are read out the softmax
    over ``X b`` is the prediction, computed with the same
    :func:`~oop_ml.core.logistic.stable_softmax` the numpy backend uses.

    Not mirrored from the numpy backend
    -----------------------------------
    ``solver_path``, the observed route. The engine keeps no record of its
    iterations.
    """

    learning_rate: float = Field(default=0.1, gt=0.0)
    max_epochs: int = Field(default=10_000, gt=0)
    tolerance: float = Field(default=1e-8, gt=0.0)
    fit_intercept: bool = True

    _n_classes: int | None = PrivateAttr(default=None)
    _coefficients: tuple[Coefficients, ...] | None = PrivateAttr(default=None)
    _intercepts: FloatArray | None = PrivateAttr(default=None)
    _epochs_run: int | None = PrivateAttr(default=None)
    _converged: bool | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _refuse_a_configured_learning_rate(self) -> Self:
        refuse_a_configured_step_size(self, "the engine's L-BFGS solver")
        return self

    @property
    def n_classes(self) -> int:
        """How many classes the fit saw.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._n_classes is not None
        return self._n_classes

    @property
    def epochs_run(self) -> int:
        """How many L-BFGS iterations the fit took.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._epochs_run is not None
        return self._epochs_run

    @property
    def converged(self) -> bool:
        """Whether the engine settled rather than exhausting ``max_epochs``.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._converged is not None
        return self._converged

    @property
    def intercepts(self) -> FloatArray:
        """One bias per class, class 0's held at zero as the reference.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._intercepts is not None
        return self._intercepts.copy()

    def coefficients_for(self, class_index: int) -> Coefficients:
        """The weights for one class against class 0, keyed by feature name.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        InvalidValuesError
            If ``class_index`` is not a class of this problem.
        """
        self._check_fitted()
        assert self._coefficients is not None

        if not 0 <= class_index < self.n_classes:
            raise InvalidValuesError(
                f"class {class_index} is outside a problem with "
                f"{self.n_classes} classes"
            )

        return self._coefficients[class_index]

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Fit the engine and read its weights back against the reference.

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
        NonBinaryLabelsError
            If the target holds a negative or fractional value.
        SingleClassError
            If the target holds fewer than two classes, or leaves a gap in the
            run from zero.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_columns_vary()
        feature_set.check_aligned_with(target_values)

        target_column = target_values.column
        target_column.check_is_label_encoded()

        n_classes = target_column.n_classes
        feature_set.check_supports_parameter_count(
            feature_set.n_features + (1 if self.fit_intercept else 0)
        )

        engine_type: Any = EngineLogisticRegression
        engine = engine_type(
            C=UNPENALISED,
            solver="lbfgs",
            max_iter=self.max_epochs,
            tol=self.tolerance,
            fit_intercept=self.fit_intercept,
        )
        reached_the_cap = fit_watching_convergence(
            engine, matrix_of(feature_set), target_column.values
        )

        weights, intercepts = self._against_the_reference(engine, n_classes)
        names = [feature.name for feature in feature_set]

        self._n_classes = n_classes
        self._intercepts = intercepts
        self._coefficients = tuple(
            Coefficients(
                [
                    Coefficient(name, float(weight))
                    for name, weight in zip(names, row, strict=True)
                ]
            )
            for row in weights
        )
        self._epochs_run = first_iteration_count(engine)
        self._converged = not reached_the_cap
        self._mark_fitted()

        return self

    def _against_the_reference(
        self, engine: Any, n_classes: int
    ) -> tuple[FloatArray, FloatArray]:
        """The engine's weights with class 0's subtracted from every class.

        Returns ``(n_classes, n_features)`` weights and ``(n_classes,)``
        intercepts, class 0's row and entry exactly zero. A tuple of two
        *like* things, both halves of one parametrisation.
        """
        coefficients = np.asarray(engine.coef_, dtype=np.float64)
        intercepts = np.asarray(engine.intercept_, dtype=np.float64).ravel()

        if n_classes == 2:
            # A two-class engine fits one binary model, whose single weight
            # vector is already the difference against class 0.
            weights = np.vstack([np.zeros_like(coefficients[0]), coefficients[0]])
            biases = np.array([0.0, intercepts[0]])
        else:
            weights = coefficients - coefficients[0]
            biases = intercepts - intercepts[0]

        if not self.fit_intercept:
            biases = np.zeros(n_classes)

        return weights, biases

    @property
    def _fitted_feature_names(self) -> tuple[str, ...]:
        assert self._coefficients is not None

        return tuple(coefficient.name for coefficient in self._coefficients[0])

    def _probability_matrix(self, input_values: Sequence[Feature]) -> ProbabilityMatrix:
        self._check_fitted()
        assert self._coefficients is not None
        assert self._intercepts is not None

        matrix = matched_matrix(self._fitted_feature_names, input_values)
        weights = np.array(
            [
                [coefficient.value for coefficient in class_coefficients]
                for class_coefficients in self._coefficients
            ]
        )
        scores = matrix @ weights.T + self._intercepts

        return ProbabilityMatrix(stable_softmax(scores))

    def predict_probabilities(
        self, input_values: Sequence[Feature]
    ) -> ProbabilityMatrix:
        """P(class is k) for every row and class, rows summing to 1.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return self._probability_matrix(input_values)

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The most probable class per row, as ``0.0 .. K-1``; ties go lower.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return Predictions.already_checked(
            self._probability_matrix(input_values).most_likely
        )


class OneVsRestClassifier(MultiClassClassifier[Sequence[Feature], Feature]):
    """Multi-class by one binary wrapper per class, each fitted by its engine.

    Parameters
    ----------
    binary_model:
        The classifier to clone once per class. It has to be a binary model
        from this backend, so that every fit here is the engine's; a numpy
        model is refused at construction rather than quietly run.

    Translation
    -----------
    The engine's own ``OneVsRestClassifier`` is deliberately not used. With
    two classes it fits a single estimator, where this library fits two and
    promises ``model_for`` for each, and with more it would only hand back
    fitted engines that still have to be wrapped one by one. Fitting K deep
    copies of the wrapper against K recoded targets is the same loop the
    numpy backend runs, with every solve inside it the engine's.

    Not mirrored from the numpy backend
    -----------------------------------
    ``one_vs_rest_fits``, the observed route. Its vocabulary, ``ClassFit``
    and ``OneVsRestFits``, belongs to the numpy backend.

    Raises
    ------
    InvalidValuesError
        If ``binary_model`` is not an engine-backed model of this backend.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    binary_model: LinearClassifier

    _n_classes: int | None = PrivateAttr(default=None)
    _fitted_models: tuple[LinearClassifier, ...] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _refuse_a_model_without_an_engine(self) -> Self:
        if not isinstance(self.binary_model, EngineMember):
            raise InvalidValuesError(
                f"{type(self.binary_model).__name__} cannot be the binary model "
                "here: every fit in this backend is a scikit-learn engine's, so "
                "binary_model must be a classifier from oop_ml.scikit"
            )

        return self

    @property
    def n_classes(self) -> int:
        """How many classes the fit saw.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._n_classes is not None
        return self._n_classes

    def model_for(self, class_index: int) -> LinearClassifier:
        """The binary model fitted for one class against all the others.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        InvalidValuesError
            If ``class_index`` is not a class of this problem.
        """
        self._check_fitted()
        assert self._fitted_models is not None

        if not 0 <= class_index < self.n_classes:
            raise InvalidValuesError(
                f"class {class_index} is outside a problem with "
                f"{self.n_classes} classes"
            )

        return self._fitted_models[class_index]

    @staticmethod
    def _binary_target(target_values: Feature, class_index: int) -> Feature:
        """``target_values`` recoded as "is it this class" 1/0, named for it."""
        return Feature(
            f"{target_values.name}=={class_index}",
            (target_values.values == float(class_index)).astype(np.float64),
        )

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Fit one binary model per class, each against all the others.

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
        NonBinaryLabelsError
            If the target holds a negative or fractional value.
        SingleClassError
            If the target holds fewer than two classes, or leaves a gap in the
            run from zero.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_columns_vary()
        feature_set.check_aligned_with(target_values)

        target_column = target_values.column
        target_column.check_is_label_encoded()

        n_classes = target_column.n_classes
        fitted_models = tuple(
            self.binary_model.model_copy(deep=True).fit(
                input_values, self._binary_target(target_values, class_index)
            )
            for class_index in range(n_classes)
        )

        self._n_classes = n_classes
        self._fitted_models = fitted_models
        self._mark_fitted()

        return self

    def predict_probabilities(self, input_values: Sequence[Feature]) -> ClassScores:
        """Each class's own probability, ``(n_samples, n_classes)``.

        **These rows do not sum to one**, for the reason the numpy backend
        gives: column ``k`` is the ``k``-th model's own answer and the K
        models were never introduced.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        self._check_fitted()
        assert self._fitted_models is not None

        return ClassScores(
            np.column_stack(
                [
                    model.predict_probability(input_values).values
                    for model in self._fitted_models
                ]
            )
        )

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The class whose own model was most confident; ties go lower.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return Predictions.already_checked(
            self.predict_probabilities(input_values).most_likely
        )


class KNearestNeighboursClassifier(
    NeighbourModel, EngineMember, MultiClassClassifier[Sequence[Feature], Feature]
):
    """A majority vote among the nearest rows, by ``KNeighborsClassifier``.

    Translation
    -----------
    ``n_neighbours`` is the engine's ``n_neighbors`` and ``metric`` is
    translated by
    :func:`~oop_ml.scikit.plumbing.neighbour_engine_parameters`, exactly as
    for the neighbour regressor. The engine's ``weights`` is left uniform,
    which is the plain vote.

    A tie between classes goes to the lowest index in both backends, since
    both take the first maximum. A tie between equally distant rows is broken
    by the engine's rule here and by remembered order in the numpy backend,
    so on such a tie the two can seat a different voter.

    ``neighbour_search`` is inherited from the frame and answers with the
    library's own distance calculation over the remembered rows.
    """

    _engine: KNeighborsClassifier | None = PrivateAttr(default=None)
    _n_classes: int | None = PrivateAttr(default=None)

    @property
    def n_classes(self) -> int:
        """How many classes the remembered rows span.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._n_classes is not None
        return self._n_classes

    def _validated_target(self, target_values: Feature) -> Column:
        """The target, insisted upon as whole class positions ``0 .. K-1``.

        Raises
        ------
        NonBinaryLabelsError
            If the target holds a negative or fractional value.
        SingleClassError
            If fewer than two classes are present, or they leave a gap.
        """
        target_column = super()._validated_target(target_values)
        target_column.check_is_label_encoded()

        return target_column

    def _combine(self, neighbour_targets: FloatArray) -> FloatArray:
        """The most common class among each query's neighbours; ties go lower.

        The frame requires it. Prediction asks the engine, so nothing here
        calls it during a fit; it is what a vote is.
        """
        assert self._n_classes is not None
        n_queries, n_voters = neighbour_targets.shape

        cells = neighbour_targets.astype(np.int64).ravel() + np.repeat(
            np.arange(n_queries) * self._n_classes, n_voters
        )
        counts = np.bincount(cells, minlength=n_queries * self._n_classes)

        return (
            counts.reshape(n_queries, self._n_classes).argmax(axis=1).astype(np.float64)
        )

    def _engine_prototype(self, n_rows: int) -> KNeighborsClassifier:
        engine_type: Any = KNeighborsClassifier

        return engine_type(
            **neighbour_engine_parameters(self.n_neighbours, self.metric)
        )

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Remember the rows, count the classes, hand the rows to the engine.

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
        NonBinaryLabelsError
            If the target holds a negative or fractional value.
        SingleClassError
            If the target holds fewer than two classes, or leaves a gap.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_aligned_with(target_values)

        if feature_set.n_samples < self.n_neighbours:
            raise TooFewValuesError(
                f"{self.n_neighbours} neighbours were asked for and only "
                f"{feature_set.n_samples} rows were supplied"
            )

        target_column = self._validated_target(target_values)
        names = tuple(feature.name for feature in feature_set)
        rows = rows_of(feature_set.feature_matrix, names)

        engine = self._engine_prototype(feature_set.n_samples).fit(
            rows.values, target_column.values
        )
        self._absorb(engine, names, rows, target_column.values, target_column.n_classes)

        return self

    def _absorb(
        self,
        engine: KNeighborsClassifier,
        names: tuple[str, ...],
        rows: RowBlock,
        targets: FloatArray,
        n_classes: int,
    ) -> None:
        self._feature_names = names
        self._remembered_rows = rows
        self._remembered_targets = targets
        self._n_classes = n_classes
        self._engine = engine
        self._mark_fitted()

    def _adopting(self, engine: Any, training: Dataset) -> Self:
        """Wrap a fitted engine; the width is the classes the engine saw.

        A member has no way to be told a wider width, so a resample missing
        a class leaves it narrower than its siblings, exactly as the numpy
        backend's neighbour member is.
        """
        feature_set = FeatureSet(training.input_features)
        names = tuple(feature.name for feature in feature_set)

        adopted = type(self)(**configuration_of(self))
        adopted._absorb(
            engine,
            names,
            rows_of(feature_set.feature_matrix, names),
            training.target_feature.column.values,
            int(np.asarray(engine.classes_).size),
        )

        return adopted

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The engine's majority class per query, as ``0.0 .. K-1``.

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

    def predict_probabilities(
        self, input_values: Sequence[Feature]
    ) -> ProbabilityMatrix:
        """Each class's share of the nearest neighbours, ``(n_queries, K)``.

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
        assert self._n_classes is not None

        return ProbabilityMatrix(
            full_width_scores(
                np.asarray(self._engine.predict_proba(query_rows.values)),
                self._engine.classes_,
                self._n_classes,
            )
        )


class DecisionTreeClassifier(
    TreeModel, EngineMember, MultiClassClassifier[Sequence[Feature], Feature]
):
    """A classification tree grown by scikit-learn's ``DecisionTreeClassifier``.

    The engine grows the tree and the fitted tree is converted once into the
    library's own nodes, each leaf a
    :class:`~oop_ml.core.tree.node.ClassificationLeaf` carrying the class
    shares the engine stored, so ``root``, ``depth``, ``n_leaves``,
    ``describe`` and ``feature_importances`` are the frame's.

    Parameters
    ----------
    criterion:
        How splits are scored, translated by name through
        :data:`ENGINE_CRITERION_NAMES`. Both backends measure entropy in
        bits, so a converted node reports the same impurity under either
        criterion. Measured on the exam fixture at ``ENTROPY``, both roots
        answer 0.9709505944546686, the binary entropy of 0.4 in bits.
    n_known_classes:
        The class width, stated by the caller, with the same rule the numpy
        backend gives: stated, the target need only hold whole positions
        inside it, and every probability row is that wide whichever classes
        the fit met.
    max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease,
    max_features, random_seed:
        As for the regression tree wrapper, including the caveat on
        ``min_impurity_decrease``, which the engine scales by the node's
        share of the rows.

    Where the backends disagree
    ---------------------------
    A row exactly at a threshold goes left in the engine and right in this
    library's own routing; predictions here follow the engine.
    """

    criterion: ClassificationCriterion = ClassificationCriterion.GINI
    n_known_classes: int | None = Field(default=None, ge=2)

    _engine: EngineDecisionTreeClassifier | None = PrivateAttr(default=None)
    _n_classes: int | None = PrivateAttr(default=None)

    @property
    def n_classes(self) -> int:
        """How many classes the fit saw.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._n_classes is not None
        return self._n_classes

    @property
    def _impurity(self) -> Impurity:
        return self.criterion.impurity

    def _validated_target(self, target_values: Feature) -> Column:
        """The target as whole class positions, dense unless a width is stated.

        Raises
        ------
        NonBinaryLabelsError
            If the target holds a negative or fractional value.
        SingleClassError
            If the width is being inferred and fewer than two classes are
            present or they leave a gap.
        InvalidValuesError
            If ``n_known_classes`` is stated and the target names a class at
            or beyond it.
        """
        target_column = super()._validated_target(target_values)

        if self.n_known_classes is None:
            target_column.check_is_label_encoded()
        else:
            target_column.check_are_class_positions(self.n_known_classes)

        return target_column

    def _leaf(self, target_values: Column) -> LeafNode:
        """A leaf answering with the most common class among these targets.

        The frame requires it. The engine grows the tree, so nothing here
        calls it during a fit; it is what a leaf of this tree is.
        """
        assert self._n_classes is not None
        counts = np.bincount(
            target_values.values.astype(np.int64), minlength=self._n_classes
        )

        return ClassificationLeaf(
            prediction=float(np.argmax(counts)),
            class_shares=counts / target_values.n_samples,
            n_samples=target_values.n_samples,
            impurity=self._impurity.of(target_values),
        )

    def _engine_prototype(self, n_rows: int) -> EngineDecisionTreeClassifier:
        return EngineDecisionTreeClassifier(
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
        NonBinaryLabelsError
            If the target holds a negative or fractional value.
        SingleClassError
            If the width is inferred and the target holds fewer than two
            classes or leaves a gap.
        InvalidValuesError
            If ``n_known_classes`` is stated and the target names a class at
            or beyond it.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_aligned_with(target_values)

        target_column = self._validated_target(target_values)
        n_classes = (
            target_column.n_classes
            if self.n_known_classes is None
            else self.n_known_classes
        )
        names = tuple(feature.name for feature in feature_set)

        engine = self._engine_prototype(feature_set.n_samples).fit(
            matrix_of(feature_set), target_column.values
        )
        self._absorb(engine, names, n_classes)

        return self

    @staticmethod
    def _leaf_reader(engine: Any, n_classes: int) -> Callable[[Any, int], LeafNode]:
        """How to read one of this engine's leaves at the stated width.

        The engine stores each leaf's class shares over the classes *it* saw,
        so the reader places them into a row of the problem's width.
        """
        classes = engine.classes_

        def leaf_of(tree: Any, node_id: int) -> LeafNode:
            shares = full_width_scores(
                np.asarray(tree.value[node_id, 0, :])[None, :], classes, n_classes
            )[0]

            return ClassificationLeaf(
                prediction=float(np.argmax(shares)),
                class_shares=shares,
                n_samples=node_row_count(tree, node_id),
                impurity=float(tree.impurity[node_id]),
            )

        return leaf_of

    def _absorb(
        self,
        engine: EngineDecisionTreeClassifier,
        names: tuple[str, ...],
        n_classes: int,
    ) -> None:
        """Commit a fitted engine, its class width and its converted tree together."""
        root = converted_tree(engine, names, self._leaf_reader(engine, n_classes))

        self._feature_names = names
        self._root = root
        self._n_classes = n_classes
        self._engine = engine
        self._mark_fitted()

    def _adopting(self, engine: Any, training: Dataset) -> Self:
        """Wrap a tree an ensemble grew, at the stated width if one was stated.

        The engine seeds every member itself, so the adopted wrapper reads
        its ``random_seed`` from the engine. Without a stated width the
        classes are the ones the engine saw.
        """
        configuration = configuration_of(self)
        engine_seed = getattr(engine, "random_state", None)
        if isinstance(engine_seed, int):
            configuration["random_seed"] = engine_seed

        n_classes = (
            int(np.asarray(engine.classes_).size)
            if self.n_known_classes is None
            else self.n_known_classes
        )

        adopted = type(self)(**configuration)
        adopted._absorb(
            engine,
            tuple(feature.name for feature in training.input_features),
            n_classes,
        )

        return adopted

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The majority class of the box each row falls in, as the engine routes it.

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

    def predict_probabilities(
        self, input_values: Sequence[Feature]
    ) -> ProbabilityMatrix:
        """Each class's share of the leaf's rows, ``(n_queries, K)``.

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
        assert self._n_classes is not None

        return ProbabilityMatrix(
            full_width_scores(
                np.asarray(self._engine.predict_proba(rows.values)),
                self._engine.classes_,
                self._n_classes,
            )
        )


class BaggingClassifier(
    AveragingEnsemble, MultiClassClassifier[Sequence[Feature], Feature]
):
    """Bootstrap-averaged classifiers, by scikit-learn's ``BaggingClassifier``.

    The engine draws the resamples, fits a member on each and averages their
    probabilities, which is the combination the numpy backend chose and the
    engine's own. The wrapper reads the fitted members and the rows each drew
    back out of it, so ``members``, ``samples``, ``feature_importances`` and
    the out-of-bag estimate are the frame's, reading state the engine
    produced.

    Parameters
    ----------
    base_model:
        The prototype every member is built from, which has to be a
        multi-class model from this backend. Defaults to an unpruned tree.
        A member whose class declares ``n_known_classes`` is told the width
        before the engine sees it, for the reason the numpy backend gives.
    n_members, random_seed:
        Inherited from the frame.

    Translation
    -----------
    ``n_members`` is the engine's ``n_estimators`` and ``random_seed`` its
    ``random_state``. ``bootstrap``, ``max_samples`` and ``max_features``
    are left at the defaults that draw ``n`` rows with replacement over every
    feature. The engine seeds each member from its own stream; a member read
    back carries the seed the engine gave it.

    Two of a tree member's stopping rules change meaning in here, and the
    field names deny it, so it is stated. The engine does not fit a member on
    the resampled rows. It hands every training row to the member weighted by
    how often the resample drew it, and the splitter keeps only positively
    weighted rows, so ``min_samples_split`` and ``min_samples_leaf`` are
    compared against the count of *distinct* drawn rows where the numpy
    backend compares them against the resampled rows with their repeats.
    Measured, a bootstrap of 200 rows holds about 127 distinct ones, so the
    same number is roughly 1.6 times stricter here. On six members at
    ``min_samples_leaf=20`` the numpy members' smallest leaf holds 20 or 21
    resampled rows, while the engine's holds exactly 20 distinct drawn rows,
    which :func:`~oop_ml.scikit.plumbing.node_row_count` reports as 27 to 29
    because it counts the draws. A k-nearest-neighbours member is fitted on
    indexed rows rather than weighted ones and does not shift.

    Raises
    ------
    InvalidValuesError
        If ``base_model`` is not an engine-backed model of this backend.
    """

    base_model: MultiClassClassifier = DecisionTreeClassifier()

    _n_classes: int | None = PrivateAttr(default=None)
    _engine: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _refuse_a_member_without_an_engine(self) -> Self:
        if not isinstance(self.base_model, EngineMember):
            raise InvalidValuesError(
                f"{type(self.base_model).__name__} cannot be a member here: a "
                "scikit-learn bagging engine needs a member that can hand it a "
                "scikit-learn prototype, so base_model must be a classifier "
                "from oop_ml.scikit"
            )

        return self

    @property
    def n_classes(self) -> int:
        """How many classes the fit saw.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._n_classes is not None
        return self._n_classes

    def _member_prototype(self, position: int, n_classes: int) -> EngineMember:
        """The configured member, told the class width when it can be told.

        Matched by field rather than by type, as the numpy backend does: any
        member whose class declares ``n_known_classes`` is rebuilt with the
        width filled in.
        """
        member_fields = type(self.base_model).model_fields

        if (
            "n_known_classes" in member_fields
            and getattr(self.base_model, "n_known_classes", None) is None  # noqa: B009
        ):
            prototype = type(self.base_model)(
                **{**configuration_of(self.base_model), "n_known_classes": n_classes}
            )
        else:
            prototype = self.base_model

        assert isinstance(prototype, EngineMember)
        return prototype

    def _prototype(self, position: int) -> AveragingMember:
        assert self._n_classes is not None
        prototype = self._member_prototype(position, self._n_classes)
        assert isinstance(prototype, MultiClassClassifier)

        return prototype

    def _validated_target(self, target_values: Feature) -> Column:
        """The target, insisted upon as whole class positions ``0 .. K-1``.

        Raises
        ------
        NonBinaryLabelsError
            If the target holds a negative or fractional value.
        SingleClassError
            If fewer than two classes are present, or they leave a gap.
        """
        target_column = super()._validated_target(target_values)
        target_column.check_is_label_encoded()

        return target_column

    def _member_answer(
        self, member: AveragingMember, input_values: Sequence[Feature]
    ) -> FloatArray:
        assert isinstance(member, MultiClassClassifier)
        return member.predict_probabilities(input_values).values

    def _combine(self, member_predictions: MemberPredictions) -> FloatArray:
        """The class with the highest averaged probability; ties go lower."""
        return member_predictions.values.mean(axis=0).argmax(-1).astype(np.float64)

    def _ensemble_engine(self, n_rows: int, n_classes: int) -> Any:
        """The unfitted engine, configured from this model's fields."""
        return EngineBaggingClassifier(
            estimator=self._member_prototype(0, n_classes)._engine_prototype(n_rows),
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
        NonBinaryLabelsError
            If the target holds a negative or fractional value.
        SingleClassError
            If the target holds fewer than two classes, or leaves a gap.
        """
        dataset = Dataset(input_values, target_values)
        n_classes = self._validated_target(target_values).n_classes
        names = tuple(feature.name for feature in dataset.input_features)

        engine = self._ensemble_engine(dataset.n_samples, n_classes).fit(
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
            adopted = self._member_prototype(position, n_classes)._adopting(
                fitted, dataset.select_rows(sample.drawn)
            )
            assert isinstance(adopted, MultiClassClassifier)
            members.append(adopted)

        self._feature_names = names
        self._samples = samples
        self._members = tuple(members)
        self._training = dataset
        self._n_classes = n_classes
        self._engine = engine
        self._mark_fitted()

        return self

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The engine's most probable class per row, averaged across members.

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

    def predict_probabilities(
        self, input_values: Sequence[Feature]
    ) -> ProbabilityMatrix:
        """The members' mean probability matrix, ``(n_queries, n_classes)``.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        matched = self._matched_rows(input_values)
        assert self._engine is not None
        assert self._n_classes is not None

        return ProbabilityMatrix(
            full_width_scores(
                np.asarray(self._engine.predict_proba(matrix_of(matched))),
                self._engine.classes_,
                self._n_classes,
            )
        )

    def out_of_bag_evaluate(self) -> MultiClassEvaluation:
        """Score the fit against rows each member never drew.

        Computed by the frame from the samples the engine reported, with the
        class count taken from the fitted model rather than the covered rows.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        estimate = self.out_of_bag_estimate()
        assert self._training is not None
        actual = self._training.target_feature.values[estimate.covered]

        return MultiClassEvaluation(
            actual, estimate.covered_predictions, self.n_classes
        )

    def out_of_bag_score(self) -> float:
        """Accuracy against the rows each member never drew.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        return self.out_of_bag_evaluate().accuracy


class RandomForestClassifier(BaggingClassifier):
    """Bagged classification trees with a per-node feature restriction.

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
    through under their own names, ``max_features`` as a count and ``None``
    as the engine's "every feature", where the engine's own default would be
    the square root; ``random_seed`` is ``random_state``. ``bootstrap`` is
    left on, which is what makes this bagging.

    ``min_samples_split`` and ``min_samples_leaf`` therefore carry the shift
    :class:`BaggingClassifier` describes, since the engine resamples the same
    way and weights the rows rather than selecting them. Measured on six
    members at ``min_samples_split=40``, the numpy members' smallest split
    node holds 40 to 49 resampled rows and the engine's holds 40 to 47
    distinct drawn rows, which this library reports as 60 to 88.

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
        """Raise if a caller configured the one field a forest ignores."""
        if self.base_model != type(self).model_fields["base_model"].default:
            raise InvalidValuesError(
                "a forest builds its own trees and ignores base_model; "
                "configure max_depth, min_samples_split, min_samples_leaf and "
                "max_features on the forest itself"
            )

        return self

    def _member_prototype(self, position: int, n_classes: int) -> EngineMember:
        """A tree at the stated width, configured to restrict its features.

        The seed offset by position is what the numpy backend does; here the
        engine seeds every tree itself and the adopted member reads that
        seed, so the offset only describes the prototype.
        """
        return DecisionTreeClassifier(
            n_known_classes=n_classes,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_seed=(
                None if self.random_seed is None else self.random_seed + position
            ),
        )

    def _ensemble_engine(self, n_rows: int, n_classes: int) -> Any:
        # The untyped engine reads ``max_features="sqrt"`` as str, refusing None.
        engine_type: Any = EngineRandomForestClassifier

        return engine_type(
            n_estimators=self.n_members,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=True,
            random_state=self.random_seed,
        )


class SupportVectorClassifier(Classifier[Sequence[Feature], Feature]):
    """The widest corridor the kernel allows, solved by scikit-learn's ``SVC``.

    Translation
    -----------
    ``capacity`` is the engine's ``C`` unchanged: both cap every multiplier
    at it. ``kernel`` is one of this library's
    :class:`~oop_ml.core.kernel.functions.Kernel` objects, translated by
    :func:`~oop_ml.scikit.plumbing.engine_kernel_parameters`.

    ``max_epochs`` is the engine's ``max_iter``, a cap on solver steps for a
    cap on solver steps, and the unit differs: an epoch of the numpy
    backend's projected ascent moves every multiplier, an iteration of the
    engine's sequential minimal optimisation moves two, so the cap binds
    sooner here. ``tolerance`` is passed to ``tol`` unchanged and is the
    engine's own stopping rule rather than a bound on how far a multiplier
    moved. ``learning_rate`` is refused at any value but its default; see the
    module docstring. ``epochs_run`` is the engine's ``n_iter_``.

    ``multipliers`` and ``support_vectors`` are read from the engine's
    ``dual_coef_``, which stores ``a_i y_i`` for the support vectors alone,
    so the multiplier is its magnitude and the label its sign, and every
    other row's multiplier is zero.

    ``predict_probability`` is the same logistic squash of the decision value
    the numpy backend uses, and is not a probability for the same reason. The
    engine's ``probability=True`` would run Platt scaling inside the model,
    which this library deliberately keeps a separate step.

    Where the backends disagree
    ---------------------------
    The engine learns a free intercept, where the numpy backend absorbs its
    offset into the kernel and shrinks it with the weights. At a small
    ``capacity`` the engine's boundary therefore sits at the data's midpoint
    where the numpy one is held nearer the origin, and the two decision
    values are on different scales: only their sign is promised to agree.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    kernel: Kernel = LinearKernel()
    capacity: float = Field(default=1.0, gt=0.0)
    learning_rate: float = Field(default=0.001, gt=0.0)
    max_epochs: int = Field(default=1000, ge=1)
    tolerance: float = Field(default=1e-06, gt=0.0)

    _engine: SVC | None = PrivateAttr(default=None)
    _multipliers: FloatArray | None = PrivateAttr(default=None)
    _signed_labels: FloatArray | None = PrivateAttr(default=None)
    _training_rows: RowBlock | None = PrivateAttr(default=None)
    _epochs_run: int | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _refuse_a_configured_learning_rate(self) -> Self:
        refuse_a_configured_step_size(
            self, "the engine's sequential minimal optimisation"
        )
        return self

    @property
    def support_vectors(self) -> SupportVectors:
        """The rows the boundary depends on.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._multipliers is not None
        assert self._signed_labels is not None
        assert self._training_rows is not None

        return SupportVectors(
            [
                SupportVector(position, float(multiplier), float(label))
                for position, (multiplier, label) in enumerate(
                    zip(self._multipliers, self._signed_labels, strict=True)
                )
                if multiplier > SUPPORT_VECTOR_THRESHOLD
            ],
            self._training_rows.n_rows,
        )

    @property
    def multipliers(self) -> FloatArray:
        """Every dual variable, including the zeros.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._multipliers is not None
        return self._multipliers.copy()

    @property
    def signed_labels(self) -> FloatArray:
        """The training target as -1 and +1.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._signed_labels is not None
        return self._signed_labels.copy()

    @property
    def epochs_run(self) -> int:
        """Solver iterations the engine took.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._epochs_run is not None
        return self._epochs_run

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Hand the engine the rows, then read the multipliers back.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonEqualArrayLengthError
            If the features and target are different lengths.
        NonUniqueFeaturesError
            If two features share a name.
        NonBinaryLabelsError
            If the target holds anything besides 0 and 1.
        SingleClassError
            If the target does not hold both classes.
        InvalidValuesError
            If the kernel is one the engine cannot be handed.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_aligned_with(target_values)

        target_column = Column.of(target_values, ValueRole.TARGET_VALUES)
        target_column.check_is_binary()
        target_column.check_has_both_classes()

        names = tuple(feature.name for feature in feature_set)
        rows = rows_of(matrix_of(feature_set), names)

        engine_type: Any = SVC
        engine = engine_type(
            C=self.capacity,
            tol=self.tolerance,
            max_iter=self.max_epochs,
            **engine_kernel_parameters(self.kernel),
        )
        engine.fit(rows.values, target_column.values)

        multipliers = np.zeros(rows.n_rows, dtype=np.float64)
        multipliers[np.asarray(engine.support_, dtype=np.intp)] = np.abs(
            np.asarray(engine.dual_coef_, dtype=np.float64).ravel()
        )

        self._training_rows = rows
        self._signed_labels = np.where(target_column.values > 0.5, 1.0, -1.0)
        self._multipliers = multipliers
        self._epochs_run = first_iteration_count(engine)
        self._engine = engine
        self._mark_fitted()

        return self

    def query_matrix(self, input_values: Sequence[Feature]) -> KernelMatrix:
        """The kernel matrix pairing query rows against the training rows.

        Computed with this library's own kernel over the rows the fit kept,
        so it is the same table the numpy backend builds.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        return self.kernel.between(self._matched_rows(input_values), self._rows)

    @property
    def _rows(self) -> RowBlock:
        self._check_fitted()
        assert self._training_rows is not None
        return self._training_rows

    def _matched_rows(self, input_values: Sequence[Feature]) -> RowBlock:
        names = self._rows.feature_names

        return rows_of(matched_matrix(names, input_values), names)

    def decision_values(self, input_values: Sequence[Feature]) -> FloatArray:
        """The engine's signed distance from the boundary; positive is class 1.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        queries = self._matched_rows(input_values)
        assert self._engine is not None

        return np.asarray(
            self._engine.decision_function(queries.values), dtype=np.float64
        )

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """Which side of the boundary each row falls on, as 0 or 1.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        return Predictions.already_checked(
            np.where(self.decision_values(input_values) >= 0.0, 1.0, 0.0)
        )

    def predict_probability(self, input_values: Sequence[Feature]) -> Probabilities:
        """A bounded score per row, and **not** a calibrated probability.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        clipped = np.clip(
            self.decision_values(input_values),
            -DECISION_VALUE_CLIP,
            DECISION_VALUE_CLIP,
        )

        return Probabilities(1.0 / (1.0 + np.exp(-clipped)))

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"SupportVectorClassifier({self.kernel!r}, unfitted)"

        return (
            f"SupportVectorClassifier({self.kernel!r}, capacity={self.capacity}, "
            f"{self.support_vectors!r})"
        )


__all__ = [
    "BaggingClassifier",
    "DecisionTreeClassifier",
    "KNearestNeighboursClassifier",
    "LogisticRegression",
    "MultinomialLogisticRegression",
    "NewtonLogisticRegression",
    "OneVsRestClassifier",
    "RandomForestClassifier",
    "SupportVectorClassifier",
]
