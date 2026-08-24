"""The walk shared by every linear model that arrives rather than jumps.

Ridge and ordinary least squares reach their answer with one call to a solver.
Gradient descent, gradient ascent, and Newton's method do not: they start
somewhere, take a step, and repeat until the steps stop mattering. That loop is
identical in all three, and for a long time it was written out three times.

What actually differs between them is one expression -- the step:

===========================  ==========================================
gradient descent             ``-learning_rate * gradient``
logistic gradient ascent     ``+learning_rate * gradient``
Newton / IRLS                ``solve(X.T W X, X.T (y - p))``
===========================  ==========================================

Everything around it is bookkeeping: start at zero, cap the passes, apply the
step, count the pass, stop when the step is smaller than the tolerance, and
record which of the two exits happened. Getting that bookkeeping subtly wrong is
easy and quiet, which is the argument for writing it once. Two of the three
copies used to increment the counter *after* the convergence break, so a fit
that settled immediately reported ``converged_ = True`` alongside zero passes
run.

Naming
------
The cap keeps its own name on each model, because the field's conventional word
differs and the word carries meaning: a pass over the whole dataset is an
*epoch* to a gradient method and an *iteration* to Newton. This class calls the
neutral thing a pass and lets each model map its own name onto it, rather than
renaming a public field to satisfy an internal one.

Coordinate descent stays out
----------------------------
``LassoRegression`` is the fourth iterative model in the library and does not
inherit from this one. A sweep of coordinate descent updates a single
coefficient at a time and repairs the residual after each, so there is no step
vector to hand back at the end of a pass -- and manufacturing one would mean
giving up the carried residual that keeps a sweep at O(n p) rather than
O(n p^2). It keeps its own loop and, for now, its own copy of the tolerance and
the two reporting properties.

That is a real remaining duplication rather than a resolved one. Closing it
would mean splitting this class in two, one half holding the state and the
reporting and the other adding the step loop on top, which is worth doing if a
fifth iterative model turns up and not obviously worth it for one.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from pydantic import Field, PrivateAttr

from oop_ml.core.column import Column
from oop_ml.core.linear_model import LinearModel
from oop_ml.core.types import FloatArray


class IterativeSolver(LinearModel):
    """A linear model whose coefficients are walked to rather than solved for.

    Parameters
    ----------
    tolerance:
        Stop once no coefficient moved further than this in a whole pass.
        Subclasses may lower the default where their convergence affords it.
    """

    tolerance: float = Field(default=1e-8, gt=0.0)

    _passes_run: int | None = PrivateAttr(default=None)
    _converged: bool | None = PrivateAttr(default=None)

    @property
    @abstractmethod
    def _pass_limit(self) -> int:
        """The cap on passes, under whatever name this model gives it."""

    @abstractmethod
    def _step(
        self,
        design_matrix: FloatArray,
        target_column: Column,
        weights: FloatArray,
    ) -> FloatArray:
        """The change to add to ``weights`` this pass.

        The one line that distinguishes one iterative solver from another.
        Signed as it will be applied, so a descent returns a negative step
        rather than relying on the caller to subtract it.

        Parameters
        ----------
        design_matrix:
            ``X``, already carrying the ones column when ``fit_intercept`` is
            set.
        target_column:
            ``y``, validated and aligned with the rows of ``X``.
        weights:
            The coefficients as they currently stand.

        Returns
        -------
        FloatArray
            One entry per column of ``design_matrix``.
        """

    @property
    def _completed_passes(self) -> int:
        """How many passes the walk took, for a model to expose under its name.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._passes_run is not None
        return self._passes_run

    @property
    def converged_(self) -> bool:
        """Whether the walk settled, rather than running out of passes.

        ``False`` means the coefficients were still moving when the cap was
        reached, and it is the attribute to check before trusting a fit. It is
        not always a near miss: where no finite answer exists the walk cannot
        settle however long it is given, and each model's docstring says what
        that looks like for the objective it maximises.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._converged is not None
        return self._converged

    def _has_converged(self, step: FloatArray) -> bool:
        """Whether this pass moved every coefficient less than ``tolerance``.

        Measured on the step rather than on the change in the objective. Near a
        maximum the objective is flat, so the improvement it reports goes as the
        *square* of the coefficient error and reaches zero in floating point
        while the coefficients are still visibly moving. The step is in the
        units the caller reads back, and needs no reference value.
        """
        return bool(np.max(np.abs(step)) < self.tolerance)

    def _solve(self, design_matrix: FloatArray, target_column: Column) -> FloatArray:
        """Walk from zero until the steps stop mattering.

        Starts every coefficient at zero, then repeats :meth:`_step` until it
        returns something smaller than ``tolerance`` or the pass limit is
        reached. Both exits are recorded.

        The step is applied *before* the convergence test, not after. It has
        already been paid for, and on a method that converges quadratically the
        error left over goes as the square of the step, so taking a final step
        of 1e-4 lands near 1e-9 rather than near 1e-4. Discarding it would give
        back several orders of magnitude for nothing.

        Does not set ``_fitted``. ``fit`` owns that, and only once the weights
        have been paired with their feature names.
        """
        weights = np.zeros(design_matrix.shape[1], dtype=np.float64)

        self._passes_run = 0
        self._converged = False

        for _ in range(self._pass_limit):
            step = self._step(design_matrix, target_column, weights)

            weights = weights + step
            self._passes_run += 1

            if self._has_converged(step):
                self._converged = True
                break

        return weights
