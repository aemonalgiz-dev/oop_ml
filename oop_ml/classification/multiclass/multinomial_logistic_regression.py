"""Multi-class logistic regression by softmax, fitted by gradient ascent.

Theory
------
The sigmoid returns *a* probability. With ``K`` classes you need ``K`` of them,
and they have to be a distribution: non-negative and summing to one. Nothing
about the sigmoid arranges that.

Give each class its own weight vector instead. Each produces a score
``z_k = X b_k``, and the scores are normalised across the classes::

    p_k  =  exp(z_k) / sum_j exp(z_j)

Exponentiate to force positivity, divide by the total to force the sum to one.
It is the same "model something unbounded, then map it back" move as the
log-odds, done ``K`` ways at once.

The gradient does not change shape
----------------------------------
Differentiating the log-likelihood ``sum_i log p_{y_i}`` with respect to one
class's weights gives::

    dLL/db_k  =  X.T ( 1[y = k] - p_k )

which is ``X.T (y - p)`` again, with the 0/1 label replaced by a 0/1 *indicator*
of "is this row class k". Checked against central differences over every entry,
the largest disagreement was 2.8e-08.

As in the binary model the implementation divides that by the sample count, so
that one ``learning_rate`` means the same thing across datasets of different
sizes -- and, more usefully here, the same thing across the binary model and
this one. Left unaveraged on a 36-row fixture it takes steps 36 times too large
and diverges at any learning rate above about 0.1.

It is not an analogy either. With two classes this collapses exactly onto the
binary model::

    max | softmax_class1 - sigmoid(X (b1 - b0)) |  =  2.2e-16

Two softmax weight vectors reduce to one sigmoid applied to their difference,
so :class:`~oop_ml.classification.binary.logistic_regression.LogisticRegression` is
the ``K = 2`` special case of this and not a separate idea.

The parameters are not unique
-----------------------------
This is the trap with no binary counterpart. Add the same constant to every
class's weight for a given feature and nothing observable changes::

    log-likelihood before: -358.4280949215
    log-likelihood after : -358.4280949215
    probabilities identical: True

The constant appears in every ``exp(z_k)`` and cancels top and bottom. That is
one redundant degree of freedom per feature, so the likelihood has a flat ridge
running through it, there is no unique maximum, and the Hessian is singular
along the ridge. A second-order solver would fail outright rather than slowly.

Two ways out. Pin one class as a reference by holding ``b_0`` at zero, which is
what makes ``K = 2`` collapse to a single vector; or keep all ``K`` and add a
penalty, which selects the smallest-norm point on the ridge. This class takes
the first, because it is the one that leaves the model identifiable rather than
merely regularised, and because it makes the connection to the binary case
exact instead of approximate.

Class 0 is therefore the reference: its scores are held at zero and its
coefficients are not learned. Every other class's weights read as "against
class 0", which is also how their odds ratios read.

Why gradient ascent rather than Newton
--------------------------------------
The Hessian's diagonal blocks are ``-X.T diag(p_k (1 - p_k)) X``, exactly the
binary curvature, but the off-diagonal blocks are ``+X.T diag(p_k p_m) X``:
the classes compete for a fixed unit of probability, so moving one class's
weights changes another's optimum. The whole thing is
``(K-1)p x (K-1)p`` and has to be rebuilt every iteration. At ``K = 10`` and
``p = 50`` that is a 450x450 matrix per step. IRLS was a clear win at ``K = 2``
and is a judgement call here, so this starts with the walk that has no such
cost and leaves the second-order version as a later question.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import Field, PrivateAttr

from oop_ml.classification.logistic import softmax
from oop_ml.core.base.estimator import MultiClassClassifier
from oop_ml.core.data.coefficients import Coefficient, Coefficients
from oop_ml.core.data.column import Column
from oop_ml.core.data.design_matrix import DesignMatrix
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.probabilities import ProbabilityMatrix
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.solving.path import SolverPath, SolverStep, SolverStop
from oop_ml.core.types import FloatArray


class MultinomialLogisticRegression(MultiClassClassifier[Sequence[Feature], Feature]):
    """A softmax classifier over named features, with class 0 as the reference.

    Parameters
    ----------
    learning_rate:
        Step size for the ascent. As with the binary model this stands in for
        curvature the method does not compute, and too large a value diverges.
    max_epochs:
        Cap on passes over the data.
    tolerance:
        Stop once no coefficient moved further than this in a whole epoch.
    fit_intercept:
        Learn a bias per class, which is the default.
    """

    learning_rate: float = Field(default=0.1, gt=0.0)
    max_epochs: int = Field(default=10_000, gt=0)
    tolerance: float = Field(default=1e-8, gt=0.0)
    fit_intercept: bool = True

    _n_classes: int | None = PrivateAttr(default=None)
    # One Coefficients per class, in class order, class 0's all zero. The names
    # live here rather than in a parallel tuple: Coefficients already is the
    # pairing of a weight with the feature it came from, which is what every
    # other model in the library relies on to match features by name.
    _coefficients: tuple[Coefficients, ...] | None = PrivateAttr(default=None)
    _intercepts: FloatArray | None = PrivateAttr(default=None)
    _epochs_run: int | None = PrivateAttr(default=None)
    _converged: bool | None = PrivateAttr(default=None)

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
        """How many epochs the ascent took.

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
        """Whether the walk settled rather than exhausting ``max_epochs``.

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

        Length ``n_classes``. Zeros throughout when ``fit_intercept`` is off.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._intercepts is not None
        return self._intercepts.copy()

    @staticmethod
    def _probabilities(
        design_matrix: DesignMatrix, learned: FloatArray
    ) -> ProbabilityMatrix:
        return softmax(design_matrix.values @ learned.T)

    def coefficients_for(self, class_index: int) -> Coefficients:
        """The weights for one class, keyed by feature name.

        Class 0 is the reference and its weights are all zero by construction,
        not by fitting. Every other class's weights read as "against class 0":
        a coefficient of 0.7 for class 2 means that raising this feature by one
        unit multiplies the odds of class 2 *relative to class 0* by exp(0.7).

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

    def _design_matrix(self, feature_set: FeatureSet) -> DesignMatrix:
        """``X``, with a leading ones column when the bias is being learned.

        Column-major for the same reason every other linear model here wants
        it: the ascent reaches for ``X.T @ something`` once an epoch.
        """
        names = [feature.name for feature in feature_set]

        if not self.fit_intercept:
            return DesignMatrix(feature_set.feature_matrix, names, False)

        values = np.empty(
            (feature_set.n_samples, feature_set.n_features + 1), order="F"
        )
        values[:, 0] = 1.0
        values[:, 1:] = feature_set.feature_matrix

        return DesignMatrix(values, names, True)

    @staticmethod
    def _indicator_matrix(target_column: Column, n_classes: int) -> ProbabilityMatrix:
        """One-hot the target: ``(n_samples, n_classes)``, one 1 per row.

        This is the ``1[y = k]`` of the gradient, built once rather than per
        epoch since it does not depend on the weights.
        """
        indicator = np.zeros((target_column.n_samples, n_classes))
        indicator[
            np.arange(target_column.n_samples), target_column.values.astype(np.int64)
        ] = 1.0

        # A one-hot row is bounded and sums to one, so it satisfies every
        # invariant a probability matrix has. It is the degenerate belief that
        # puts all its mass on the true class, which is exactly what the
        # gradient subtracts the model's own belief from.
        return ProbabilityMatrix(indicator)

    def _scores(self, design_matrix: DesignMatrix, weights: FloatArray) -> FloatArray:
        """The ``(n_samples, n_classes)`` score matrix, class 0 held at zero.

        ``weights`` carries only the learned classes, ``1 .. K-1``, because
        class 0 is the reference. This puts the zero column back so that the
        softmax sees all ``K``.

        Parameters
        ----------
        design_matrix:
            ``X``, shape ``(n_samples, parameter_count)``.
        weights:
            ``(n_classes - 1, parameter_count)``, one row per learned class.

        Returns
        -------
        FloatArray
            ``(n_samples, n_classes)``, the first column all zeros.
        """
        # Column-major, and filled in place rather than assembled by
        # column_stack. The softmax that consumes this reduces and broadcasts
        # along the class axis, which with a handful of classes and many rows
        # is strided in row-major storage and contiguous here: measured at
        # 20000x5 it is 3.63 ms row-major against 2.06 ms this way, and the
        # whole fit runs 1.57x faster for it without a coefficient moving.
        scores = np.empty((design_matrix.n_rows, weights.shape[0] + 1), order="F")
        scores[:, 0] = 0.0
        scores[:, 1:] = design_matrix.values @ weights.T

        return scores

    def _gradient(
        self,
        design_matrix: DesignMatrix,
        indicator: ProbabilityMatrix,
        probabilities: ProbabilityMatrix,
    ) -> FloatArray:
        """``X.T (1[y = k] - p_k) / n`` for every learned class at once.

        That formula is written for a *single* class: ``1[y = k]`` and ``p_k``
        are one value per row, and the result is one value per parameter. Here
        every class is done at once, so the per-class vectors have to be
        stacked, and which way round they stack is a real decision rather than
        a formatting one::

            differences.T @ design_matrix.values   ->  (n_classes, parameter_count)
            design_matrix.values.T @ differences   ->  (parameter_count, n_classes)

        The same numbers either way, transposed. Return the first: one row per
        class is how the weights are stored and what ``_solve`` adds its step
        to. Note that on a problem with as many parameters as classes both are
        square, so a shape check will not catch the wrong one -- only the
        finite-difference test will.

        Two further details are easy to miss and both bite.

        Divide by the sample count, exactly as the binary model does. It keeps
        one ``learning_rate`` meaningful across dataset sizes, and it keeps the
        two models comparable -- without it the same rate that converges here
        diverges there.

        And note which rows are wanted: class 0 is the reference and is not
        learned, so its row of the full gradient is discarded rather than
        computed and then ignored.

        Parameters
        ----------
        design_matrix:
            ``X``, shape ``(n_samples, parameter_count)``.
        indicator:
            ``(n_samples, n_classes)``, one 1 per row.
        probabilities:
            ``(n_samples, n_classes)``, each row summing to 1.

        Returns
        -------
        FloatArray
            ``(n_classes - 1, parameter_count)``, matching the learned weights.
        """
        differences = indicator.values - probabilities.values
        gradient = differences.T @ design_matrix.values / design_matrix.n_rows

        return gradient[1:]

    def _has_converged(self, step: FloatArray) -> bool:
        """Whether this epoch moved every coefficient less than ``tolerance``."""
        return bool(np.max(np.abs(step)) < self.tolerance)

    def solver_path(
        self, design_matrix: DesignMatrix, target_column: Column
    ) -> SolverPath:
        """Every epoch of the ascent, rather than only where it stopped.

        The observed route beside :meth:`_solve`. The weights here are a
        matrix -- one row per class above the reference -- and the record
        keeps them at that shape, so a step shows every class moving at once
        rather than a flattened vector nobody can read.

        Records rather than mutates, so ``epochs_run`` and ``converged`` keep
        describing the model's own fit.

        Returns
        -------
        SolverPath
            ``path.result`` is the same array :meth:`_solve` returns.
        """
        assert self._n_classes is not None

        indicator = self._indicator_matrix(target_column, self._n_classes)
        weights = np.zeros((self._n_classes - 1, design_matrix.n_columns))

        steps: list[SolverStep] = []
        stopped = SolverStop.PASS_LIMIT_REACHED

        for epoch_number in range(1, self.max_epochs + 1):
            probabilities = softmax(self._scores(design_matrix, weights))
            step = self.learning_rate * self._gradient(
                design_matrix, indicator, probabilities
            )

            steps.append(SolverStep(epoch_number, weights, step))
            weights = weights + step

            if self._has_converged(step):
                stopped = SolverStop.CONVERGED
                break

        return SolverPath(steps, weights, stopped)

    def _solve(self, design_matrix: DesignMatrix, target_column: Column) -> FloatArray:
        """Walk uphill from zero until the coefficients settle.

        Start every learned weight at zero -- which makes every class equally
        likely on the first pass, the honest starting opinion -- then each
        epoch compute the scores, the probabilities, and the gradient, scale it
        into a step, and add it. Record both exits.

        Do not set ``_fitted`` here. ``fit`` owns that.

        Parameters
        ----------
        design_matrix:
            ``X``, shape ``(n_samples, parameter_count)``.
        target_column:
            ``y``, validated as whole class positions and aligned with ``X``.

        Returns
        -------
        FloatArray
            ``(n_classes - 1, parameter_count)``, one row per learned class.
        """
        # fit sets this before calling here; the assert is what tells the type
        # checker so, matching how every other method in this class narrows.
        assert self._n_classes is not None

        indicator = self._indicator_matrix(target_column, self._n_classes)
        weights = np.zeros((self._n_classes - 1, design_matrix.n_columns))

        self._epochs_run = 0
        self._converged = False

        for _ in range(self.max_epochs):
            probabilities = softmax(self._scores(design_matrix, weights))
            step = self.learning_rate * self._gradient(
                design_matrix, indicator, probabilities
            )

            weights = weights + step
            self._epochs_run += 1

            if self._has_converged(step):
                self._converged = True
                break

        return weights

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Fit one weight vector per class beyond the reference.

        Parameters
        ----------
        input_values:
            One or more predictor columns, all the same length as the target.
        target_values:
            The classes, as whole positions running ``0 .. K - 1``.

        Returns
        -------
        Self
            This model, so calls can chain.

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

        self._n_classes = target_column.n_classes
        parameter_count = feature_set.n_features + (1 if self.fit_intercept else 0)
        feature_set.check_supports_parameter_count(parameter_count)

        solution = self._solve(self._design_matrix(feature_set), target_column)
        self._store_solution(feature_set, solution)

        self._mark_fitted()
        return self

    def _store_solution(self, feature_set: FeatureSet, solution: FloatArray) -> None:
        """Split the per-class biases off and bind the rest to their names.

        The reference class is prepended back as a row of zeros, so what is
        stored spans all ``K`` classes and a caller never has to remember that
        one of them was not fitted.
        """
        assert self._n_classes is not None

        learned = np.vstack([np.zeros((1, solution.shape[1])), solution])

        if self.fit_intercept:
            self._intercepts = learned[:, 0].copy()
            weights = learned[:, 1:]
        else:
            self._intercepts = np.zeros(self._n_classes)
            weights = learned

        names = [feature.name for feature in feature_set]
        self._coefficients = tuple(
            Coefficients(
                [
                    Coefficient(name, float(weight))
                    # strict: a mismatch here would mean the solution and the
                    # feature set had drifted out of step.
                    for name, weight in zip(names, row, strict=True)
                ]
            )
            for row in weights
        )

    def _matched_matrix(self, input_values: Sequence[Feature]) -> DesignMatrix:
        """The design matrix for ``input_values``, checked against the fit.

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

        fitted_names = self._fitted_feature_names
        by_name = {feature.name: feature for feature in input_values}

        if set(by_name) != set(fitted_names):
            raise InvalidValuesError(
                f"expected features {', '.join(sorted(fitted_names))}; "
                f"got {', '.join(sorted(by_name))}"
            )

        return self._design_matrix(FeatureSet([by_name[name] for name in fitted_names]))

    @property
    def _fitted_feature_names(self) -> tuple[str, ...]:
        """The fitted feature names, in the order the design matrix used.

        Read off the coefficients rather than kept alongside them. Every class
        carries the same names in the same order, so class 0 answers for all of
        them.
        """
        assert self._coefficients is not None

        return tuple(coefficient.name for coefficient in self._coefficients[0])

    def predict_probabilities(self, input_values: Sequence[Feature]) -> FloatArray:
        """P(class is k) for every row and class, as ``(n_samples, n_classes)``.

        Every row sums to 1 by construction, which is the property one-vs-rest
        cannot offer.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return self._probability_matrix(input_values).values

    def _probability_matrix(self, input_values: Sequence[Feature]) -> ProbabilityMatrix:
        """The same distribution, still carrying its row-sum invariant.

        ``predict_probabilities`` hands out the array because that is what a
        caller wants; ``predict`` wants the object, so the tie rule lives on
        the type instead of being written out at every call site.
        """
        design_matrix = self._matched_matrix(input_values)

        assert self._coefficients is not None
        assert self._intercepts is not None

        # Rebuilt per call rather than cached: it is O(K p) beside a matmul of
        # O(n p K), and a cache would be a second copy of the weights to keep
        # in step with the first.
        weights = np.array(
            [
                [coefficient.value for coefficient in class_coefficients]
                for class_coefficients in self._coefficients
            ]
        )
        learned = (
            np.column_stack([self._intercepts, weights])
            if self.fit_intercept
            else weights
        )

        return self._probabilities(design_matrix, learned)

    def predict(self, input_values: Sequence[Feature]) -> FloatArray:
        """The most probable class per row, as ``0.0 .. K-1``.

        Ties go to the lower class index, which is what ``argmax`` does. Exact
        ties are vanishingly rare on real data and the alternative -- refusing
        to answer -- is worse than a stated rule.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return self._probability_matrix(input_values).most_likely
