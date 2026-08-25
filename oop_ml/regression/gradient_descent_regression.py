"""Least squares solved by walking downhill instead of by algebra.

Theory
------
The objective is unchanged from ``MultipleLinearRegression``::

    S(b) = || y - X b ||^2

and it is still a convex bowl over the coefficient space. The closed-form solve
jumps straight to the floor of that bowl in one algebraic step. Gradient descent
starts somewhere on the wall and *walks* down, repeatedly stepping in the
steepest downhill direction until it stops moving.

Which raises the obvious question: why bother, when a closed form exists?

* **Most models have no closed form.** Logistic regression's likelihood cannot be
  solved algebraically at all, so gradient descent is not an alternative option
  there, it is the only one. Learning it here, where the right answer is already
  known, means you can check the walk against the jump.
* **It scales differently.** The closed form builds ``X.T X`` (cost grows with
  ``p^2 * n``) and inverts it (``p^3``). At large ``p`` that is prohibitive,
  while a gradient step is a single pass over the data.
* **It never forms ``X.T X``**, so the conditioning problem that plagues the
  normal equations does not arise.

Mathematics
-----------
We already derived the gradient for the closed form, where setting it to zero is
what gave us the normal equations. Here it gets used directly instead::

    dS/db = -2 X.T (y - X b)

Scaling by ``1/n`` keeps the step size independent of how many samples there
are, giving the update rule::

    gradient = (-2 / n) * X.T (y - X b)

    b <- b - learning_rate * gradient

Each step moves *against* the gradient, because the gradient points uphill.
Repeat until the coefficients stop changing.

Because ``S`` is convex there is exactly one minimum, so with a small enough
step this converges to the same answer the closed form produces. There are no
local minima to get trapped in, and the starting point does not matter.

The learning rate
-----------------
The one genuinely delicate parameter.

* **Too small**: converges, but takes far more epochs than necessary.
* **Too large**: each step overshoots the floor and lands higher up the opposite
  wall. The coefficients oscillate outward and diverge to infinity or NaN.

There is a hard threshold: for least squares, steps diverge once
``learning_rate > 2 / L``, where ``L`` is the largest eigenvalue of
``(2/n) X.T X``. Predictors on wildly different scales make that bowl a long
narrow valley, with one direction far steeper than another, and a step small
enough to be safe in the steep direction then crawls along the shallow one. This
is why gradient descent needs standardised features considerably more urgently
than the closed form ever does.

Convergence
-----------
Stop on either of two conditions, whichever comes first:

* the coefficients moved less than ``tolerance`` this epoch, so it converged;
* ``max_epochs`` steps have been taken, so it gave up.

Record which happened. A model that quietly hit its epoch cap has not converged,
and reporting its coefficients as though it had is how silently wrong numbers
get published.

Worked example
--------------
The usual fixture (``y = 1 + 2*x1 + 3*x2``, whose exact answer is
``(1, 2, 3)``), started from ``b = 0``::

    learning_rate = 0.01, 1000 epochs   ->  (1.0901, 1.9751, 2.9614)   still walking
    learning_rate = 0.01, 5000 epochs   ->  (1.0000, 2.0000, 3.0000)   arrived
    learning_rate = 0.05, 1000 epochs   ->  (1.0000, 2.0000, 3.0000)   arrived sooner

The first row is the useful one: the answer is *approached*, not landed on. Every
assertion about a gradient-descent fit is approximate by nature, which is a real
difference from the closed form.
"""

from __future__ import annotations

from pydantic import Field

from oop_ml.base.iterative_solver import IterativeSolver
from oop_ml.data.column import Column
from oop_ml.regression.linear_feature_regressor import LinearFeatureRegressor
from oop_ml.types import FloatArray


class GradientDescentRegression(IterativeSolver, LinearFeatureRegressor):
    """Least squares fit by batch gradient descent.

    Parameters
    ----------
    learning_rate:
        Step size. Too small wastes epochs; too large diverges.
    max_epochs:
        Hard cap on the number of steps, so a diverging or slow fit terminates.
    tolerance:
        Convergence threshold: stop once the largest change in any coefficient
        during an epoch falls below this.
    fit_intercept:
        Inherited. The ones column is just another coefficient to descend on.
    """

    learning_rate: float = Field(default=0.01, gt=0.0)
    max_epochs: int = Field(default=10_000, gt=0)
    tolerance: float = Field(default=1e-8, gt=0.0)

    @property
    def _pass_limit(self) -> int:
        return self.max_epochs

    @property
    def epochs_run(self) -> int:
        """How many epochs the last fit actually took."""
        return self._completed_passes

    @staticmethod
    def _compute_residuals(
        design_matrix: FloatArray, weights: FloatArray, target_column: Column
    ) -> FloatArray:
        """What the current weights still fail to explain: ``y - X @ beta``."""
        return target_column.values - design_matrix @ weights

    def _compute_gradient(
        self, design_matrix: FloatArray, weights: FloatArray, target_column: Column
    ) -> FloatArray:
        """Uphill direction of the squared error at ``weights``.

        Entry ``j`` is how strongly feature ``j`` still correlates with the
        residual, so a feature that no longer explains anything left over
        contributes nothing at all. The gradient then vanishes exactly when the
        residual is orthogonal to every column, which is the normal equations
        arriving by another route.

        The ``2 / n`` factor averages rather than totals, keeping one
        ``learning_rate`` meaningful across datasets of different sizes.
        """
        residuals = self._compute_residuals(design_matrix, weights, target_column)
        return -(2 / target_column.n_samples) * design_matrix.T @ residuals

    def _step(
        self,
        design_matrix: FloatArray,
        target_column: Column,
        weights: FloatArray,
    ) -> FloatArray:
        """One epoch of gradient descent: downhill, scaled by the rate.

        Negative, because squared error is being minimised rather than
        maximised. Returning the step already signed keeps the walk itself
        indifferent to which direction a given objective wants.
        """
        gradient = self._compute_gradient(design_matrix, weights, target_column)

        return -self.learning_rate * gradient
