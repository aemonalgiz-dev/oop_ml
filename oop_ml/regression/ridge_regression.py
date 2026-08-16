"""Ridge regression: least squares with the coefficients penalised.

Theory
------
Ordinary least squares minimises the squared error and nothing else, so when two
predictors carry nearly the same information, it is perfectly free to put an
enormous positive weight on one and an enormous negative weight on the other.
The two almost cancel, the fit looks fine, and the coefficients are nonsense
that will swing wildly on the next sample you draw. In the exact case, where one
column is a linear combination of the others, ``X.T X`` is singular and there is
no unique answer to be had at all.

Ridge attaches a price to magnitude. The objective gains a second term::

    S(b) = || y - X b ||^2  +  penalty * || b ||^2

Minimising a sum of two things means trading them against each other; the fit
term still wants to explain ``y``, although every unit of coefficient now costs
something. Large cancelling weights stop being free, so the solution stays small
and stays stable.

This is the bias-variance tradeoff made explicit, and it is worth being clear
that we are choosing to be wrong on purpose. The penalty deliberately biases the
estimate toward zero, so ridge coefficients are not the least-squares minimum
and are not trying to be. What we get in exchange is far less variance across
resamples, and when predictors are collinear or ``n`` is barely larger than
``p``, that is usually a trade worth taking.

Mathematics
-----------
Differentiate the penalised objective and set it to zero::

    dS/db = -2 X.T y + 2 X.T X b + 2 * penalty * b = 0

    =>  (X.T X + penalty * I) b = X.T y

    =>  b_hat = inv(X.T X + penalty * I) X.T y

Only one thing has changed from the OLS normal equations, which is that
``penalty`` is added along the diagonal. That is where the name comes from; a
ridge lifted along the diagonal of ``X.T X``.

The consequence is worth dwelling on for a moment. ``X.T X`` is positive
semi-definite, so its eigenvalues are all at least zero, and a singular matrix
is precisely one with an eigenvalue of exactly zero. Adding ``penalty * I``
shifts every eigenvalue up by ``penalty``, which means that for any positive
penalty the matrix becomes positive definite and is therefore always invertible.
Ridge always has a unique solution, even in the cases where OLS has none, and
that makes it the principled answer to the collinear features that currently
cause ``MultipleLinearRegression`` to raise.

As ``penalty`` approaches zero, ridge becomes OLS exactly. As it grows without
bound, every coefficient is driven to zero.

The intercept is not penalised
------------------------------
The ``I`` above is not the plain identity, because the entry for the intercept
has to be zero::

    [ 0  0  0 ]
    [ 0  1  0 ]     for a design matrix whose first column is the ones column
    [ 0  0  1 ]

Consider what shrinking the intercept would actually do. Add 100 to every value
of ``y``, and a penalised intercept could not follow it, so every other
coefficient would have to distort itself to compensate. The fit would then
depend on where the target's origin happens to sit, which is not a property any
of us want a model to have. The intercept is a location parameter rather than a
strength, and there is nothing about it to regularise.

Scale sensitivity
-----------------
Notice that ``|| b ||^2`` is summing coefficients measured in different units, so
a predictor in metres and a predictor in kilometres end up penalised a
thousandfold differently. Ridge is only really meaningful on standardised
features, which is why a ``Standardizer`` belongs in front of this in any
pipeline that is doing real work.

Worked example
--------------
Using the usual fixture, ``y = 1 + 2*x1 + 3*x2`` exactly, with the intercept
left unpenalised::

    penalty = 0     ->  b = (1.0000, 2.0000, 3.0000)    identical to OLS
    penalty = 1     ->  b = (2.9537, 1.4537, 2.0093)
    penalty = 10    ->  b = (6.1817, 0.4528, 0.4870)

Both slopes shrink toward zero as the penalty grows, and the intercept rises to
absorb whatever they no longer explain.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from oop_ml.core.column import Column
from oop_ml.core.types import FloatArray
from oop_ml.regression.linear_feature_regressor import LinearFeatureRegressor


class RidgeRegression(LinearFeatureRegressor):
    """Least squares with an L2 penalty on the coefficients.

    Parameters
    ----------
    penalty:
        Strength of the L2 penalty, the ``lambda`` in the derivation above. Zero
        reproduces ordinary least squares exactly, and larger values shrink the
        coefficients harder. It must not be negative.
    fit_intercept:
        Inherited. The intercept, when it is fitted, is never penalised.
    """

    penalty: float = Field(default=1.0, ge=0.0)

    def _solve(self, design_matrix: FloatArray, target_column: Column) -> FloatArray:
        """Solve ``(X.T X + penalty * I) b = X.T y`` with the intercept exempt.

        Three steps. Form ``X.T X`` and ``X.T y`` exactly as the OLS case does;
        build the penalty matrix, which is ``penalty`` down the diagonal with a
        zero in the first slot whenever ``self.fit_intercept`` is set, since
        that column is the ones column and must not be shrunk; then add the two
        together and solve.
        """
        i_penalty = np.eye(design_matrix.shape[1]) * self.penalty

        # Only exempt column zero when it actually is the ones column. Without
        # an intercept it is an ordinary feature and has to be penalised along
        # with the rest, otherwise it alone escapes the shrinkage and quietly
        # gets an advantage the other predictors do not.
        if self.fit_intercept:
            i_penalty[0, 0] = 0.0

        # The single difference from ordinary least squares is the penalty
        # matrix added to X.T X. Everything past this point is the same solve.
        normal_equations_matrix = (
            self._normal_equations_matrix(design_matrix) + i_penalty
        )
        normal_equations_vector = self._normal_equations_vector(
            design_matrix, target_column
        )

        return np.linalg.solve(normal_equations_matrix, normal_equations_vector)
