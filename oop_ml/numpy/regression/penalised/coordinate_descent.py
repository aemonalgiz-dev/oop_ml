"""Solving a penalised fit one coefficient at a time.

The search shared by every model in this package that has no closed form. It is
a different animal from gradient descent, which nudges every coefficient a
little in the direction the whole objective falls. Coordinate descent moves one
coefficient *all the way* to its own optimum, holding every other one where it
currently sits, and then moves on to the next.

Two things follow from that, and both are why the method is worth having:

- **There is no learning rate.** Each step is solved rather than tuned, so
  nothing here can diverge and ``max_iterations`` is a limit on patience rather
  than a safety rail. Compare
  :class:`~oop_ml.core.base.iterative_solver.IterativeSolver`, where choosing
  the step size badly is the whole difficulty.
- **It reaches the joint optimum anyway.** Every update is the exact minimiser
  along its own axis, so the objective can only fall, and for a convex
  objective a point that is optimal along every axis at once is optimal.

Why the partial residual is never built
----------------------------------------
The update for coefficient ``j`` is a fit of the *partial residual*, the target
with every other column's contribution already removed::

    r_j = y - X b + x_j * b_j

Materialising that means recomputing ``X b`` for every column of every sweep,
which turns an O(n p) sweep into an O(n p^2) one. It is never needed, because
only its dot product with the column is, and that expands into two quantities
already to hand::

    x_j . r_j  =  x_j . r  +  (x_j . x_j) * b_j

The full residual ``r`` is carried across the whole solve and repaired in place
each time a coefficient moves, and the column norms do not move at all, so they
are computed once before the first sweep.

What a subclass supplies
-------------------------
One method, :meth:`CoordinateDescentRegressor._penalised_optimum`, which is
where the shape of the penalty enters and is the only place the models here
differ. The sweep, the residual bookkeeping, the convergence test, the
unpenalised intercept and the observed route are all settled once, on this
frame.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import numpy as np
from pydantic import Field, PrivateAttr

from oop_ml.core.base.linear_model import LinearModel
from oop_ml.core.data.column import Column
from oop_ml.core.data.design_matrix import DesignMatrix
from oop_ml.core.solving.path import SolverPath, SolverStep, SolverStop
from oop_ml.core.types import FloatArray
from oop_ml.numpy.regression.linear_feature_regressor import LinearFeatureRegressor


class CoordinateDescentRegressor(LinearFeatureRegressor):
    """A penalised hyperplane, swept to its optimum one coefficient at a time.

    Parameters
    ----------
    penalty:
        Overall strength of the penalty. Zero reproduces ordinary least squares
        whatever shape the penalty has, since there is then no penalty to shape.
    max_iterations:
        Cap on the number of full sweeps through the coefficients, so a slow
        fit terminates.
    tolerance:
        Convergence threshold: stop once no coefficient moved more than this
        during a whole sweep.
    fit_intercept:
        Inherited. The intercept, when fitted, is never penalised.
    """

    penalty: float = Field(default=1.0, ge=0.0)
    max_iterations: int = Field(default=1_000, gt=0)
    tolerance: float = Field(default=1e-10, gt=0.0)

    LEARNED_STATE: ClassVar[tuple[str, ...]] = (
        *LinearModel.LEARNED_STATE,
        "_iterations_run",
        "_converged",
    )

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

        ``False`` means the coefficients were still moving when
        ``max_iterations`` ran out, so treat them as unfinished rather than as
        an answer.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._converged is not None
        return self._converged

    @abstractmethod
    def _penalised_optimum(self, correlation: float, column_norm: float) -> float:
        """What a penalised coefficient should be, with the others held fixed.

        The one place the models in this package differ, and the whole of the
        difference between them.

        Parameters
        ----------
        correlation:
            ``x_j . r_j``, this column against the partial residual.
        column_norm:
            ``x_j . x_j``, the denominator of the unpenalised update.

        Returns
        -------
        float
            The coefficient's own optimum under this penalty. Unpenalised the
            answer would be ``correlation / column_norm``, so a subclass is
            saying what its penalty does to that ratio.
        """

    def _is_penalised(self, column_index: int) -> bool:
        """Whether the column at this index carries the penalty.

        Every column except the leading ones column, which exists only when an
        intercept is being fitted. Without an intercept, column 0 is an ordinary
        predictor and is penalised like the rest.
        """
        return not (self.fit_intercept and column_index == 0)

    @staticmethod
    def _soft_threshold(value: float, threshold: float) -> float:
        """Move ``value`` toward zero by ``threshold``, stopping if it arrives.

        The operator every L1 penalty rests on::

            sign(value) * max(0, abs(value) - threshold)

        This returns exactly ``0.0`` whenever ``abs(value) <= threshold``, and
        that is what lets a coefficient land on zero rather than merely near it.
        """
        return float(np.sign(value) * max(0.0, abs(value) - threshold))

    @staticmethod
    def _column_norms(columns: FloatArray) -> FloatArray:
        """``sum(x_j ** 2)`` for every column, computed once for the whole solve.

        This is the denominator of every coordinate update, and it does not move
        as the weights move, so recomputing it inside the sweep is pure waste.
        The ``einsum`` form gets the column-wise sum of squares without building
        a squared copy of the matrix as a temporary.
        """
        return np.einsum("ij,ij->j", columns, columns)

    def _coordinate_optimum(
        self,
        column: FloatArray,
        residual: FloatArray,
        weight: float,
        column_norm: float,
        column_index: int,
    ) -> float:
        """The value this coefficient should take, with the others held fixed.

        Unpenalised, this is simple linear regression of the partial residual on
        this column through the origin, namely ``sum(x * r) / sum(x ** 2)``,
        which is the same ratio ``SimpleLinearRegression`` uses. What the
        penalty does to that ratio is :meth:`_penalised_optimum`, and the
        intercept never asks.

        The partial residual is never built; this module's docstring holds the
        expansion that makes one dot product and one precomputed norm enough.
        """
        correlation = float(column @ residual) + column_norm * weight

        if not self._is_penalised(column_index):
            return correlation / column_norm

        return self._penalised_optimum(correlation, column_norm)

    def solver_path(
        self, design_matrix: DesignMatrix, target_column: Column
    ) -> SolverPath:
        """Every sweep of the coordinate descent, rather than only its end.

        The observed route beside :meth:`_solve`. One recorded step per sweep,
        holding the coefficients it began with and what the whole sweep moved
        them by -- which is the level a reader wants, since a sweep is the
        unit that converges and a single coordinate move is not.

        The same :class:`~oop_ml.core.solving.path.SolverPath` the gradient
        walks produce. A sweep and an epoch are the same shape of thing: start
        somewhere, move, test whether the movement still matters. Sharing the
        record lets a coordinate descent and a gradient descent be compared
        directly.

        Records rather than mutates, so ``iterations_run`` and ``converged``
        keep describing the model's own fit.

        Returns
        -------
        SolverPath
            ``path.result`` is the same array :meth:`_solve` returns.
        """
        columns = np.asfortranarray(design_matrix.values)
        parameter_count = columns.shape[1]

        weights = np.zeros(parameter_count, dtype=np.float64)
        column_norms = self._column_norms(columns)
        residual = np.array(target_column.values, dtype=np.float64)

        steps: list[SolverStep] = []
        stopped = SolverStop.PASS_LIMIT_REACHED

        for sweep_number in range(1, self.max_iterations + 1):
            began_with = weights.copy()
            largest_change = 0.0

            for column_index in range(parameter_count):
                column = columns[:, column_index]
                previous_weight = float(weights[column_index])
                new_weight = self._coordinate_optimum(
                    column,
                    residual,
                    previous_weight,
                    float(column_norms[column_index]),
                    column_index,
                )

                change = new_weight - previous_weight
                if change != 0.0:
                    weights[column_index] = new_weight
                    residual -= column * change

                largest_change = max(largest_change, abs(change))

            steps.append(SolverStep(sweep_number, began_with, weights - began_with))

            if largest_change < self.tolerance:
                stopped = SolverStop.CONVERGED
                break

        return SolverPath(steps, weights, stopped)

    def _solve(self, design_matrix: DesignMatrix, target_column: Column) -> FloatArray:
        """Sweep the coefficients to their own optima until they settle.

        Each sweep visits every column in turn and sets it to the value that
        minimises the objective *along that one axis*, holding the others where
        they currently are. Because every update is an exact minimiser, the
        objective can only fall, and convexity makes the place it settles the
        global minimum. There is consequently no step size to choose and nothing
        here that can diverge, which makes ``max_iterations`` a limit on patience
        rather than a safety rail.

        The sweep reads ``weights`` as it revises it, so a column updated earlier
        in the pass is already reflected in the partial residual the later ones
        see. That is what makes this descent rather than a fixed-point iteration.

        Both exits are recorded: settling below ``tolerance`` (converged) or
        exhausting ``max_iterations`` (gave up). See ``converged``.
        """
        # Columns of a C-ordered matrix are strided, so every dot product below
        # would read memory with a gap between elements. One transpose up front
        # makes each column contiguous and pays for itself many sweeps over.
        columns = np.asfortranarray(design_matrix.values)
        parameter_count = columns.shape[1]

        weights = np.zeros(parameter_count, dtype=np.float64)
        column_norms = self._column_norms(columns)

        # The residual is carried across the entire solve and repaired in place
        # each time a coefficient moves, which is what keeps a sweep at O(n p).
        # Starting from all-zero weights it is simply the target itself.
        residual = np.array(target_column.values, dtype=np.float64)

        self._iterations_run = 0
        self._converged = False

        for _ in range(self.max_iterations):
            largest_change = 0.0

            for column_index in range(parameter_count):
                column = columns[:, column_index]
                previous_weight = float(weights[column_index])
                new_weight = self._coordinate_optimum(
                    column,
                    residual,
                    previous_weight,
                    float(column_norms[column_index]),
                    column_index,
                )

                change = new_weight - previous_weight
                if change != 0.0:
                    weights[column_index] = new_weight
                    # Repair the residual for this one coefficient's movement.
                    residual -= column * change

                largest_change = max(largest_change, abs(change))

            self._iterations_run += 1

            # Converged means a whole sweep in which nobody wanted to move: every
            # coefficient already optimal given the others, which for a convex
            # objective is the joint optimum.
            if largest_change < self.tolerance:
                self._converged = True
                break

        return weights
