"""Logistic regression fitted by Newton's method, which is IRLS.

Theory
------
:class:`~oop_ml.classification.binary.logistic_regression.LogisticRegression` walks
uphill knowing only which way is up. Ask it how far to step and it cannot say,
which is precisely what a learning rate is: a hyperparameter standing in for
information the method does not have.

Newton's method asks a second question. Slope *and* curvature are enough to fit
a parabola through the current point, and a parabola has an exact vertex, so
there is nothing left to guess about the distance::

    b  <-  b  -  H^-1 g

Both pieces come from differentiating the log-likelihood twice. Writing
``z = X b`` and ``p = sigmoid(z)``::

    LL(b)  =  sum( y z - log(1 + exp(z)) )

     dLL/db  =  X.T (y - p)
    d2LL/db2  =  -X.T W X,     W = diag(p (1 - p))

The two minus signs cancel and the step comes out as::

    b  <-  b  +  (X.T W X)^-1 X.T (y - p)

Note that ``H`` does not contain ``y`` anywhere. The curvature of the likelihood
depends only on where the model currently stands, never on the labels.

Why it is called reweighted least squares
-----------------------------------------
Define a working response ``z~ = X b + W^-1 (y - p)`` and expand::

    X.T W z~  =  X.T W X b  +  X.T (y - p)

Multiply through by ``(X.T W X)^-1`` and the Newton step reappears, so::

    b_new  =  (X.T W X)^-1 X.T W z~

which is exactly the weighted least squares fit of ``z~`` on ``X`` with weights
``W``. Every Newton step *is* a least squares fit; the weights and the response
are rebuilt from the current fit each round. Hence iteratively reweighted least
squares -- the same algorithm, named after its mechanics rather than its
calculus.

Do not implement it that way round. ``W^-1`` divides by ``p (1 - p)``, which is
the one quantity heading for zero on hard problems. The direct form above is
algebraically identical and never divides by it. Take the calculus form for the
arithmetic and keep the least squares form for the intuition.

The unification worth carrying: for ordinary least squares the log-likelihood is
*exactly* quadratic, so the parabola is not an approximation at all and Newton
lands on the answer in a single step -- and that step is the normal equations.
This runs identical machinery against a function that is only locally quadratic,
so it takes a handful of steps instead of one.

Worked, on eight students
-------------------------
``hours = 0.5, 1, 2, 2.5, 3, 4, 4.5, 5`` against ``passed = 0,0,1,0,1,0,1,1``.

Start at ``b = 0``. Every ``p`` is 0.5, so every weight is 0.25 -- all equal --
and ``z~ = (y - 0.5) / 0.25 = +/-2``. The first Newton step is therefore an
*ordinary* least squares fit of ``+/-2`` on hours, with no weighting at all,
because at the origin the model has no opinion to weight by yet::

    it   intercept       weight       ||b - b*||
     0   0.00000000   0.00000000       2.59e+00
     1  -1.97969543   0.70389171       4.86e-01
     2  -2.39781087   0.84970428       4.28e-02
     3  -2.43794406   0.86358478       3.72e-04
     4  -2.43829620   0.86370601       2.84e-08
     5  -2.43829623   0.86370602       4.58e-16

Read the error column as exponents: -1, -2, -4, -8, -16. Each step squares the
previous error, which is what quadratic convergence means and why the iteration
count is always "about six" almost regardless of the problem. Gradient ascent
on the identical data at a learning rate of 0.5 needs roughly 1000 epochs to
reach 7.51e-07 and 10000 to reach machine precision.

Traps
-----
**Separation is worse here, not better.** As the fitted probabilities saturate,
``W = p (1 - p)`` goes to zero and ``X.T W X`` collapses toward singular.
Gradient ascent crawls toward the infinity that separation implies and an epoch
cap is enough to stop it. Newton *sprints* there: on the six-row separable
fixture the weight passes 12.8 by iteration 8 and 36.8 by iteration 20, gaining
a flat 4.0 per step and never settling. Quadratic convergence toward an optimum
that does not exist is quadratic divergence, so ``converged`` carries even more
weight on this model than on the other one.

**The cost trade genuinely reverses.** A gradient epoch is ``O(n p)``; a Newton
iteration is ``O(n p^2)``, because ``X.T W X`` has to be formed. Newton is ahead
only when ``p * (Newton steps) < (gradient epochs)``, and measured against this
library's own benchmark that crossover is real rather than theoretical: at
1000x20 gradient ascent is the faster of the two, and at 20000x50 this is faster
by about 4x. "Newton is faster" is not a fact about the method; it is a fact
about the shape of your data.

**Undamped steps are safe here specifically.** Newton in general can overshoot
and wants a line search. The logistic log-likelihood is concave -- ``X.T W X``
is a Gram matrix with non-negative weights, so it is positive semi-definite by
construction -- which means no local maxima to fall into and no damping needed,
provided the design matrix has full column rank. That concavity is doing real
work, and the habit should not be carried to an objective that lacks it.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from oop_ml.classification.linear_classifier import LinearClassifier
from oop_ml.classification.logistic import sigmoid
from oop_ml.core.base.iterative_solver import IterativeSolver
from oop_ml.core.data.column import Column
from oop_ml.core.data.design_matrix import DesignMatrix
from oop_ml.core.data.probabilities import Probabilities
from oop_ml.core.exceptions import SingularHessianError
from oop_ml.core.solving.positive_definite import solve_positive_definite
from oop_ml.core.types import FloatArray


class NewtonLogisticRegression(IterativeSolver, LinearClassifier):
    """Logistic regression by Newton-Raphson, equivalently IRLS.

    Fits the same boundary as
    :class:`~oop_ml.classification.binary.logistic_regression.LogisticRegression` and
    reaches it in single-digit iterations rather than hundreds of epochs,
    because it uses the second derivative to choose the length of each step
    instead of being told.

    Parameters
    ----------
    max_iterations:
        Cap on Newton steps. The default is generous: a well-posed problem
        settles in five or six, and needing far more is itself the signal that
        something is wrong with the data rather than with the cap.
    tolerance:
        Stop once no coefficient moved further than this in a whole iteration.
        Tighter than the gradient model's default, and affordably so, since
        quadratic convergence means the last step before the tolerance is met
        is doing almost nothing.
    threshold:
        The probability above which a row is called positive. Defaults to 0.5,
        the point where the log-odds are zero.
    fit_intercept:
        Inherited from :class:`~oop_ml.core.base.linear_model.LinearModel`.
    """

    max_iterations: int = Field(default=100, gt=0)
    tolerance: float = Field(default=1e-10, gt=0.0)
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)

    @property
    def _pass_limit(self) -> int:
        return self.max_iterations

    @property
    def iterations_run(self) -> int:
        """How many Newton steps the fit actually took."""
        return self._completed_passes

    @staticmethod
    def _sigmoid(linear_predictor: FloatArray) -> Probabilities:
        """Map the linear predictor onto a probability.

        Delegates to :func:`~oop_ml.classification.logistic.sigmoid`, shared
        with the gradient-ascent model so that the overflow-safe spelling of it
        lives in exactly one place.
        """
        return sigmoid(linear_predictor)

    @staticmethod
    def _solve_newton_system(
        hessian_matrix: FloatArray, gradient: FloatArray
    ) -> FloatArray:
        """Solve ``hessian_matrix @ step = gradient`` for the step.

        Wired rather than left as a stub because the interesting part is the
        failure. ``numpy`` reports a singular system by raising ``LinAlgError``,
        which is the one exception in this library that would otherwise escape
        the :class:`~oop_ml.core.exceptions.MLLibError` hierarchy, so a caller
        could not route it with the rest.

        Parameters
        ----------
        hessian_matrix:
            ``X.T W X``, positive semi-definite and square.
        gradient:
            ``X.T (y - p)``, one entry per parameter.

        Returns
        -------
        FloatArray
            The Newton step, one entry per parameter.

        Raises
        ------
        SingularHessianError
            If the system has no unique solution. In practice this means every
            weight ``p (1 - p)`` has underflowed to zero, which is separation
            in its terminal form.
        """
        try:
            return solve_positive_definite(hessian_matrix, gradient)
        except np.linalg.LinAlgError as error:
            raise SingularHessianError(
                "the Hessian is singular, so there is no unique Newton step. "
                "Every p (1 - p) weight has collapsed to zero, which is what "
                "perfectly separable classes look like once the coefficients "
                "have grown far enough"
            ) from error

    def _gradient(
        self,
        design_matrix: DesignMatrix,
        target_values: Column,
        probabilities: Probabilities,
    ) -> FloatArray:
        """The gradient of the log-likelihood: ``X.T (y - p)``.

        Note what is *not* here. The gradient-ascent model divides by the sample
        count to keep its step size independent of how many rows there are.
        Do not do that here, or rather: do it to both this and the Hessian or to
        neither, because the step is ``H^-1 g`` and any constant scaling cancels
        between them. Leaving it out of both is the cheaper way to say that.

        Parameters
        ----------
        design_matrix:
            ``X``, already carrying the ones column when ``fit_intercept`` is
            set.
        target_values:
            ``y``, the 0/1 labels.
        probabilities:
            ``p``, the current fitted probabilities, one per observation.

        Returns
        -------
        FloatArray
            One partial derivative per parameter.
        """

        differences = target_values.values - probabilities.values
        return design_matrix.values.T @ differences

    @staticmethod
    def _variance_weights(probabilities: Probabilities) -> FloatArray:
        """The diagonal of ``W``: ``p (1 - p)``, one weight per observation.

        This is the variance of a Bernoulli trial at probability ``p``, and
        reading it that way explains the whole method. A row the model is
        unsure about (``p`` near 0.5) has variance 0.25 and carries the most
        weight in the next step. A row it is already certain about has variance
        near zero and is almost ignored. Newton is not weighting by how *wrong*
        each row is, but by how much each row still has to say.

        Parameters
        ----------
        probabilities:
            ``p``, one per observation.

        Returns
        -------
        FloatArray
            One non-negative weight per observation, at most 0.25.
        """
        chances = probabilities.values

        return chances * (1.0 - chances)

    @staticmethod
    def _hessian_matrix(
        design_matrix: DesignMatrix, variance_weights: FloatArray
    ) -> FloatArray:
        """``X.T W X``, the negated Hessian of the log-likelihood.

        Negated deliberately, so that what comes back is positive semi-definite
        and the step is ``+solve(X.T W X, gradient)`` rather than a subtraction
        of a negative. It also keeps the matrix in the form a weighted least
        squares solve would want it, which is the connection the module
        docstring draws.

        ``W`` is a diagonal matrix, so do not build it as one: an
        ``n x n`` matrix for what is a per-row scale factor would be ``n``
        times the memory of the design matrix itself. Scaling the rows of ``X``
        gives the same product.

        Parameters
        ----------
        design_matrix:
            ``X``, shape ``(n_samples, parameter_count)``.
        variance_weights:
            The diagonal of ``W``, one entry per row of ``X``.

        Returns
        -------
        FloatArray
            A symmetric ``(parameter_count, parameter_count)`` matrix.
        """
        return design_matrix.values.T @ (
            design_matrix.values * variance_weights[:, None]
        )

    def _step(
        self,
        design_matrix: DesignMatrix,
        target_column: Column,
        weights: FloatArray,
    ) -> FloatArray:
        """One Newton step: the gradient turned by the inverse curvature.

        Push the current coefficients out to the rows to get the probabilities,
        read two things off them -- how wrong each row is and how certain the
        model is about it -- and pull both back into coefficient space as a
        gradient and a curvature matrix. The step is then the small system
        those two define.

        Raises
        ------
        SingularHessianError
            If the weights collapse far enough that the system has no unique
            solution, which is separation in its terminal form.
        """
        probabilities = self._sigmoid(design_matrix.values @ weights)
        gradient = self._gradient(design_matrix, target_column, probabilities)
        hessian_matrix = self._hessian_matrix(
            design_matrix, self._variance_weights(probabilities)
        )

        return self._solve_newton_system(hessian_matrix, gradient)
