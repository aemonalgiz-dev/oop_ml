"""Least squares that stops listening once a row is far enough off.

The problem
-----------
Squared error weights a residual by itself. A row twice as far from the line as
another pulls four times as hard on the fit, and one ten times as far pulls a
hundred times as hard, so a single mistyped measurement can move the whole
answer. On this module's own fixture -- forty rows on a plane with four values
shifted by thirty -- ordinary least squares reports an intercept of 3.51 where
the truth is 1.0, and a second coefficient of -4.44 where the truth is -3.0.
Nothing is wrong with the arithmetic; it is doing exactly what squaring asks.

The repair
----------
Keep squared error where the residuals are small, since that is where it is
the right answer, and switch to absolute error once they are large::

    H(r) = 0.5 r^2                       |r| <= t
         = t (|r| - 0.5 t)               |r| >  t

The two pieces meet at ``|r| = t`` in both value and slope, so the curve has no
corner. What matters is the *derivative*, which is the pull a row exerts::

    H'(r) = r        |r| <= t
          = t sign(r) |r| >  t

Bounded. Past the threshold a row pulls with a constant force however far away
it is, so an outlier can lean on the fit but cannot drag it. That is the whole
idea, and it is the same curve
:class:`~oop_ml.core.network.loss.HuberError` uses on a network.

Why the threshold cannot be a fixed number
--------------------------------------------
``t`` is a distance in the target's units, so a threshold of 1.35 means
something quite different for a target measured in metres and the same target
measured in millimetres. A robust fit that is not scale-equivariant is barely
robust at all: too small a threshold makes every row an outlier and the fit
becomes a least-absolute-deviations one, too large and nothing is ever an
outlier and it becomes ordinary least squares.

So the scale is estimated alongside the coefficients, and ``threshold`` is in
units of *it*. What is minimised is::

    sum_i [ scale + scale * H2(r_i / scale) ]  +  penalty * ||b||^2

with ``H2(z) = z^2`` inside the threshold and ``2 t |z| - t^2`` outside, which
is the standard formulation and the one scikit-learn's ``HuberRegressor``
minimises. Measured, ignoring the scale and using a fixed threshold of 1.35 on
the fixture above misses the engine's intercept by 0.124.

How it is solved
----------------
Two steps alternated until neither moves.

The scale first, from the stationary point of the objective in ``scale``.
Differentiating and setting to zero gives::

    scale^2 = sum_{inside} r_i^2  /  (n - t^2 * n_outside)

The coefficients second, by *iteratively reweighted least squares*. Give row
``i`` the weight ``1`` if it is inside the threshold and ``t * scale / |r_i|``
if it is outside, then solve the weighted normal equations. That weight is
exactly the ratio of Huber's derivative to the squared-error one, so a
weighted least-squares step is a Newton-like step on the Huber objective, and
the fixed point of the loop is its minimum. It is the same reweighting
:class:`~oop_ml.numpy.classification.binary.newton_logistic_regression.NewtonLogisticRegression`
uses, differing only in what the weights mean.

Two edges, and neither is where it looks
------------------------------------------
The scale update divides by ``n - t^2 * n_outside``, which reaches zero once
enough rows are outside, so that is refused rather than answered. Measured, it
is also **not reachable by data**: the scale is estimated from the same
residuals, so it simply grows until most rows are inside again. A target with
no plane in it at all, drawn from noise at a spread of 100, fits happily with
six of twenty rows outside. The guard is there because the arithmetic admits
the case, not because a caller will meet it, and the spec tests it by handing
the scale step residuals directly rather than by pretending some dataset
produces it.

The edge that *is* reachable is the opposite one. A fit with as many parameters
as rows passes through every point, so every residual is zero, the sum of
squares is zero and the scale collapses to it. There is nothing left to be
robust about at that point -- the rows the model trusts are fitted exactly --
so the walk stops there and reports itself converged rather than dividing by a
zero reach.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import Field, PrivateAttr

from oop_ml.core.base.linear_model import LinearModel
from oop_ml.core.data.column import Column
from oop_ml.core.data.design_matrix import DesignMatrix
from oop_ml.core.exceptions import DivergenceError
from oop_ml.core.types import FloatArray
from oop_ml.numpy.regression.linear_feature_regressor import LinearFeatureRegressor

SMALLEST_USABLE_SCALE = 1e-300
"""Below this a residual's magnitude cannot be divided by safely."""


class HuberRegression(LinearFeatureRegressor):
    """A linear fit whose residuals stop gaining influence past a threshold.

    Parameters
    ----------
    threshold:
        How many multiples of the fitted scale a residual may reach before the
        row stops pulling any harder. Above 1 by definition, since at 1 or
        below the quadratic piece has no room to exist. The usual default of
        1.35 is the value at which the fit keeps about 95% of least squares'
        efficiency on data that really is Gaussian, which is the price paid for
        the robustness.
    penalty:
        An L2 penalty on the coefficients, the intercept exempt, for the rare
        case where the reweighting leaves a design that is nearly singular.
        Zero by default, unlike the engine this library's other backend wraps.
    max_iterations:
        Cap on the alternating passes, so a slow fit terminates.
    tolerance:
        Stop once neither the coefficients nor the scale moved more than this
        in a whole pass.
    fit_intercept:
        Inherited. The intercept carries no penalty.

    Raises
    ------
    NotFittedError
        From ``scale`` or ``n_outliers`` before ``fit``.
    """

    threshold: float = Field(default=1.35, gt=1.0)
    penalty: float = Field(default=0.0, ge=0.0)
    max_iterations: int = Field(default=1_000, gt=0)
    tolerance: float = Field(default=1e-10, gt=0.0)

    LEARNED_STATE: ClassVar[tuple[str, ...]] = (
        *LinearModel.LEARNED_STATE,
        "_scale",
        "_n_outliers",
        "_iterations_run",
        "_converged",
    )

    _scale: float | None = PrivateAttr(default=None)
    _n_outliers: int | None = PrivateAttr(default=None)
    _iterations_run: int | None = PrivateAttr(default=None)
    _converged: bool | None = PrivateAttr(default=None)

    @property
    def scale(self) -> float:
        """The spread the threshold is measured in, estimated with the fit.

        Roughly the typical size of a residual among the rows the fit took
        seriously, so ``threshold * scale`` is where a row stops gaining
        influence in the target's own units.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._scale is not None
        return self._scale

    @property
    def n_outliers(self) -> int:
        """How many training rows ended beyond ``threshold * scale``.

        The rows the fit held at arm's length. Worth reading, because it says
        whether the robustness did anything: zero means this is an ordinary
        least-squares fit wearing a different name, and a majority means the
        threshold is too small for the data rather than the data being mostly
        wrong.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._n_outliers is not None
        return self._n_outliers

    @property
    def iterations_run(self) -> int:
        """How many alternating passes the last fit took.

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

    def _solve(self, design_matrix: DesignMatrix, target_column: Column) -> FloatArray:
        """Alternate the scale and the reweighted least squares until both settle.

        Raises
        ------
        DivergenceError
            If too many rows fall outside the threshold for a scale to exist.
            See the module docstring: past ``n / threshold ** 2`` outliers the
            estimator has no majority left to be faithful to.
        """
        columns = design_matrix.values
        targets = np.asarray(target_column.values, dtype=np.float64)
        n_rows = columns.shape[0]

        weights = np.zeros(columns.shape[1], dtype=np.float64)
        # Starting from the target's own spread rather than from one, so that
        # the first pass is already in the right order of magnitude whatever
        # the units are. A constant target has no spread, and one is then as
        # good a starting guess as any.
        scale = float(np.std(targets)) or 1.0

        self._iterations_run = 0
        self._converged = False

        for _ in range(self.max_iterations):
            residual = targets - columns @ weights
            magnitude = np.abs(residual)

            settled = self._settled_scale(residual, magnitude, scale, n_rows)

            if settled <= 0.0:
                # Every trusted row is fitted exactly, so there is no spread
                # left to measure the threshold in and no reweighting that
                # could improve on an exact fit. See the module docstring.
                self._converged = True
                break

            scale = settled
            moved = self._reweighted_solution(columns, targets, magnitude, scale)

            movement = float(np.max(np.abs(moved - weights)))
            weights = moved
            self._iterations_run += 1

            if movement < self.tolerance:
                self._converged = True
                break

        self._scale = scale
        final_residual = np.abs(targets - columns @ weights)
        self._n_outliers = int(
            np.count_nonzero(final_residual > self.threshold * scale)
        )

        return weights

    def _settled_scale(
        self,
        residual: FloatArray,
        magnitude: FloatArray,
        scale: float,
        n_rows: int,
    ) -> float:
        """The scale the objective's own derivative asks for, given these residuals.

        ``scale^2 = sum(r^2 over the rows inside) / (n - t^2 * n_outside)``,
        which is what setting the joint objective's derivative in the scale to
        zero gives. The rows outside contribute a constant to that derivative
        rather than their size, which is the same bounded influence the
        coefficients see.
        """
        inside = magnitude < self.threshold * scale
        n_outside = int(np.count_nonzero(~inside))
        remaining = n_rows - self.threshold**2 * n_outside

        if remaining <= 0.0:
            raise DivergenceError(
                f"{n_outside} of {n_rows} rows fell outside a threshold of "
                f"{self.threshold}, which leaves no scale to estimate: past "
                f"{n_rows / self.threshold**2:.1f} of them the rows the fit "
                "trusts no longer outnumber the ones it does not. Raise the "
                "threshold, or accept that this data has no majority to fit"
            )

        return float(np.sqrt(float((residual[inside] ** 2).sum()) / remaining))

    def _reweighted_solution(
        self,
        columns: FloatArray,
        targets: FloatArray,
        magnitude: FloatArray,
        scale: float,
    ) -> FloatArray:
        """One weighted least-squares step, with the Huber weights.

        A row inside the threshold keeps its full weight; one outside is
        weighted down by exactly the factor that turns its squared-error pull
        into the constant pull Huber's derivative asks for.
        """
        reach = self.threshold * scale
        influence = np.where(
            magnitude <= reach,
            1.0,
            reach / np.maximum(magnitude, SMALLEST_USABLE_SCALE),
        )

        weighted = columns * influence[:, None]
        system = columns.T @ weighted + self._penalty_diagonal(columns.shape[1], scale)

        return np.linalg.solve(system, weighted.T @ targets)

    def _penalty_diagonal(self, n_columns: int, scale: float) -> FloatArray:
        """The L2 penalty as a diagonal, with the intercept's slot exempt.

        Scaled by the fitted scale, because the objective the penalty sits in
        is itself in units of it. Column zero is only the intercept when one is
        being fitted; without an intercept it is an ordinary predictor and is
        penalised like the rest, which is the trap
        :class:`~oop_ml.numpy.regression.penalised.ridge_regression.RidgeRegression`
        records.
        """
        diagonal = np.eye(n_columns, dtype=np.float64) * (self.penalty * scale)

        if self.fit_intercept:
            diagonal[0, 0] = 0.0

        return diagonal
