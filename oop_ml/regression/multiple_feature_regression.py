"""Multiple linear regression: ordinary least squares over several predictors.

Theory
------
Simple regression asks how ``y`` moves with ``x``. Multiple regression asks the
harder question: how does ``y`` move with ``x1`` *while holding the other
predictors fixed*? That is why you cannot fit ``p`` separate simple regressions
and collect their slopes. When predictors are correlated, a simple regression on
``x1`` alone credits ``x1`` with part of what ``x2`` is doing, because ``x1`` is
acting as a proxy for its correlated partner. Fitting jointly makes each coefficient
explain only the variation unique to its own predictor. Each coefficient is a
*partial effect*: expected change in ``y`` per unit of ``xj``, others constant.

Geometrically the fit is a line in 2-D for one predictor, a plane in 3-D for
two, a hyperplane for ``p``. "Linear" means linear in the *coefficients*, not in
the predictors, so ``x**2`` and ``log(x)`` are both perfectly legal columns.

Mathematics
-----------
Stack the ``n`` observations into a design matrix ``X``. With an intercept,
prepend a column of ones, which is the trick that turns the bias into simply
another coefficient rather than a special case the solver has to know about::

        [ 1  x_11 ... x_1p ]        [ b0 ]
    X = [ 1  x_21 ... x_2p ]    b = [ b1 ]        y = X b + e
        [ .   .        .   ]        [ .. ]
        [ 1  x_n1 ... x_np ]        [ bp ]

``X`` is ``(n, p + 1)``; without an intercept it is ``(n, p)`` and ``b0 == 0``.

Least squares minimizes the squared residual norm::

    S(b) = || y - X b ||^2 = (y - X b).T (y - X b)
         = y.T y - 2 b.T X.T y + b.T X.T X b

Differentiating with respect to the vector ``b`` and setting to zero::

    dS/db = -2 X.T y + 2 X.T X b = 0

    =>  X.T X b = X.T y            <- the *normal equations*
    =>  b_hat = inv(X.T X) X.T y

``S`` is convex (``X.T X`` is positive semi-definite), so this stationary point
is the global minimum. Rewriting the condition as ``X.T (y - X b_hat) == 0``
shows why they are called "normal": the residual vector is orthogonal to every
column of ``X``. Least squares is the orthogonal projection of ``y`` onto the
column space of ``X``, which is to say the closest point it can actually reach.
Orthogonality to the ones column is then exactly why the fitted residuals sum to
zero whenever an intercept is present.

With ``p == 1`` the 2x2 system reduces to the familiar simple-regression slope,
so this is a strict generalization, not a different method.

Identifiability and conditioning
--------------------------------
``inv(X.T X)`` exists iff ``X`` has full column rank: no predictor is an exact
linear combination of the others. A duplicated column, a "total" column equal to
the sum of two others, or a full set of dummies alongside the intercept makes the
system singular, at which point infinitely many ``b`` fit equally well.
Near-collinearity is subtler and rather more common, since the matrix remains
invertible while being badly conditioned, so the coefficients swing wildly under
tiny changes in the data. You also need ``n >= p + 1``.

Numerically, do not literally invert. ``np.linalg.solve(X.T @ X, X.T @ y)`` is
better, and the QR or SVD based ``np.linalg.lstsq(X, y)`` is better still,
because forming ``X.T X`` at all squares the condition number.

Worked example
--------------
An exact plane ``y = 1 + 2*x1 + 3*x2`` over five points (the test fixture)::

    x1 = [1, 1, 2, 0, 3]
    x2 = [1, 2, 2, 1, 0]
    y  = [6, 9, 11, 4, 7]

Sums: ``n=5``, ``sum(x1)=7``, ``sum(x2)=6``, ``sum(x1^2)=15``, ``sum(x2^2)=10``,
``sum(x1*x2)=7``, ``sum(y)=37``, ``sum(x1*y)=58``, ``sum(x2*y)=50``. So::

            [ 5   7   6 ]              [ 37 ]
    X.T X = [ 7  15   7 ]     X.T y =  [ 58 ]
            [ 6   7  10 ]              [ 50 ]

Note the structure the ones column produces: symmetric, ``n`` in the corner,
column sums along the first row and column. Substituting ``b = (1, 2, 3)``::

    5(1)  + 7(2)  + 6(3)  = 5 + 14 + 18 = 37   ok
    7(1)  + 15(2) + 7(3)  = 7 + 30 + 21 = 58   ok
    6(1)  + 7(2)  + 10(3) = 6 + 14 + 30 = 50   ok

Intercept 1, coefficients 2 and 3, residuals all zero. A new point
``(x1=10, x2=0)`` predicts ``1 + 20 + 0 = 21``.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.column import Column
from oop_ml.regression.linear_feature_regressor import LinearFeatureRegressor
from oop_ml.types import FloatArray


class MultipleLinearRegression(LinearFeatureRegressor):
    """OLS solved in closed form, straight from the normal equations.

    Everything apart from the solve itself comes from
    :class:`~oop_ml.regression.linear_feature_regressor.LinearFeatureRegressor`,
    which covers validation, the design matrix, the intercept split, pairing the
    weights with their feature names, and ``predict``.
    """

    def _solve(self, design_matrix: FloatArray, target_column: Column) -> FloatArray:
        """Solve ``X.T X b = X.T y`` directly.

        This mirrors the derivation exactly, which is the point of writing it
        this way. Do note that forming ``X.T X`` squares the condition number,
        and that a rank-deficient ``X`` makes it singular outright.
        ``np.linalg.lstsq(design_matrix, y)`` avoids both problems, although it
        does so at the cost of no longer mirroring the maths on the page.
        """
        return np.linalg.solve(
            self._normal_equations_matrix(design_matrix),
            self._normal_equations_vector(design_matrix, target_column),
        )
