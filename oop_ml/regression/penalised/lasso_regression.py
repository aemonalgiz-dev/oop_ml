"""Lasso regression: least squares with an L1 penalty, solved coordinate by coordinate.

Theory
------
Ridge and lasso differ by one character in the objective, and everything else
about them follows from it::

    ridge:  S(b) = || y - X b ||^2  +  penalty * sum(b_j ** 2)
    lasso:  S(b) = || y - X b ||^2  +  penalty * sum(abs(b_j))

Squares become absolute values. That looks cosmetic and is not: it costs the
closed form and buys feature selection.

Why there is no closed form
---------------------------
``abs(b)`` has no derivative at zero, since the slope is -1 coming from the left
and +1 coming from the right. The usual routine of differentiating, setting to
zero and solving is therefore unavailable to us, and no matrix formula exists.
What replaces the derivative is the subgradient, meaning the set of all slopes
of lines lying below the function::

    d(penalty * abs(b))/db  =  +penalty            b > 0
                              [-penalty, +penalty] b == 0
                              -penalty             b < 0

At zero it is an interval rather than a number, and that interval is the whole
mechanism. A minimum sits at ``b_j == 0`` whenever the data's pull on that
coefficient is weak enough to be cancelled by *some* slope inside the interval.

Contrast ridge, whose penalty has derivative ``2 * penalty * b``. That goes limp
as ``b`` approaches zero, so however weak the data's pull, there is always a tiny
non-zero ``b`` at which the penalty pushes back even more weakly than the data
pulls, so the optimum is never exactly zero. Lasso's penalty, by contrast, keeps
pushing with its full force right up to the corner, which is why ridge shrinks
forever while lasso actually arrives.

The soft-threshold operator
---------------------------
Work one coefficient at a time, holding the others fixed. Removing ``x_j``'s own
contribution from the fit leaves the *partial residual*::

    r_j = y - X b + x_j * b_j

Set the subgradient of the objective with respect to ``b_j`` to zero. For
``b_j > 0``::

    -2 * (x_j . r_j) + 2 * b_j * (x_j . x_j) + penalty = 0

    =>  b_j = (x_j . r_j - penalty / 2) / (x_j . x_j)

and symmetrically for ``b_j < 0``. Zero is optimal exactly when
``abs(x_j . r_j) <= penalty / 2``. Both cases collapse into one operator::

    soft_threshold(value, threshold) = sign(value) * max(0, abs(value) - threshold)

    b_j = soft_threshold(x_j . r_j, penalty / 2) / (x_j . x_j)

Slide the unpenalised answer toward zero by a fixed amount, and if it would
cross, stop it at zero. The ``penalty / 2`` follows from the objective above --
the 2 comes from differentiating the squared error, not from a convention.

Coordinate descent
------------------
Sweep the coefficients in order, applying that update to each. Every update is
the exact minimiser along its own axis, so the objective can only decrease, and
because the objective is convex the sweeps converge to the global minimum.

This is a different search from gradient descent, which moves every coefficient a
little along the gradient. Coordinate descent moves one coefficient all the way
to its own optimum and then moves to the next. No learning rate is involved at
all, since each step is solved rather than tuned, although the loop still needs
a convergence test and an iteration cap, and it has to report which of the two
stopped it.

The intercept is not penalised
------------------------------
As in ridge, and for the same reason: the intercept is a location parameter, not
a strength. Its coordinate update is the plain least-squares one, with no
thresholding, so the intercept is free to follow the target wherever it sits.
A consequence worth predicting before you see it: once every slope has been
driven to zero, the intercept must equal the mean of the target.

Scale sensitivity
-----------------
Worse here than for ridge. The threshold ``penalty / 2`` is compared against
``x_j . r_j``, whose size depends on the units of ``x_j``, so which features get
zeroed becomes an artifact of measurement rather than of signal. Lasso on
unstandardized columns is actively misleading; run
:class:`~oop_ml.preprocessing.standardization.standardizer.Standardizer` first.

Worked example
--------------
The usual fixture (``y = 1 + 2*x1 + 3*x2`` exactly), intercept unpenalised.
Every row was verified against a brute-force search of the objective::

    penalty = 0    ->  (1.000000, 2.000000, 3.000000)   identical to OLS
    penalty = 2    ->  (2.095238, 1.666667, 2.476190)
    penalty = 8    ->  (5.380952, 0.666667, 0.904762)
    penalty = 12   ->  (7.346154, 0.038462, 0.000000)   x2 selected out
    penalty = 16   ->  (7.400000, 0.000000, 0.000000)   both gone; intercept = mean(y)

Note ``penalty = 12``: ``x2`` is *exactly* zero while ``x1`` survives. That is
feature selection, and no ridge fit at any finite penalty can produce it.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import Field, PrivateAttr

from oop_ml.core.base.linear_model import LinearModel
from oop_ml.core.data.column import Column
from oop_ml.core.data.design_matrix import DesignMatrix
from oop_ml.core.solving.path import SolverPath, SolverStep, SolverStop
from oop_ml.core.types import FloatArray
from oop_ml.regression.linear_feature_regressor import LinearFeatureRegressor


class LassoRegression(LinearFeatureRegressor):
    """Least squares with an L1 penalty, fit by coordinate descent.

    Parameters
    ----------
    penalty:
        Strength of the L1 penalty. Zero reproduces ordinary least squares;
        large enough values drive coefficients to exactly zero.
    max_iterations:
        Cap on the number of full sweeps through the coefficients, so a slow or
        oscillating fit terminates.
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

    def _is_penalised(self, column_index: int) -> bool:
        """Whether the column at this index carries the L1 penalty.

        Every column except the leading ones column, which exists only when an
        intercept is being fitted. Without an intercept, column 0 is an ordinary
        predictor and is penalised like the rest.
        """
        return not (self.fit_intercept and column_index == 0)

    @staticmethod
    def _soft_threshold(value: float, threshold: float) -> float:
        """Move ``value`` toward zero by ``threshold``, stopping if it arrives.

        The operator the whole method rests on::

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
        which is the same ratio ``SimpleLinearRegression`` uses. The L1 penalty
        then enters by shaving the numerator toward zero before the division,
        and that is what allows the result to come out at exactly zero.

        The partial residual ``r_j = r + x_j * b_j`` is never actually built.
        Expanding the numerator shows why it does not need to be::

            x_j . (r + x_j * b_j)  =  x_j . r  +  (x_j . x_j) * b_j

        so the same number falls out of one dot product against the residual we
        are already carrying, plus a norm we precomputed. Materialising ``r_j``
        instead means recomputing ``X b`` for every column of every sweep, which
        is what turns an O(n p) sweep into an O(n p^2) one.
        """
        correlation = float(column @ residual) + column_norm * weight

        if self._is_penalised(column_index):
            correlation = self._soft_threshold(correlation, self.penalty / 2)

        return correlation / column_norm

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
        record lets a lasso and a gradient descent be compared directly.

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
