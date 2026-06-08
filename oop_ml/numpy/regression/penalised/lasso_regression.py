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

That single expression is the whole of this model. The sweep it sits inside,
and why sweeping to each coefficient's own optimum reaches the joint one, are
:mod:`~oop_ml.numpy.regression.penalised.coordinate_descent`, which
:class:`~oop_ml.numpy.regression.penalised.elastic_net_regression.ElasticNetRegression`
shares with this.

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
:class:`~oop_ml.numpy.preprocessing.standardization.standardizer.Standardizer` first.

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

from oop_ml.numpy.regression.penalised.coordinate_descent import (
    CoordinateDescentRegressor,
)


class LassoRegression(CoordinateDescentRegressor):
    """Least squares with an L1 penalty, fit by coordinate descent.

    Parameters
    ----------
    penalty:
        Strength of the L1 penalty. Zero reproduces ordinary least squares;
        large enough values drive coefficients to exactly zero.
    max_iterations:
        Inherited. Cap on the number of full sweeps through the coefficients,
        so a slow or oscillating fit terminates.
    tolerance:
        Inherited. Stop once no coefficient moved more than this during a
        whole sweep.
    fit_intercept:
        Inherited. The intercept, when fitted, is never penalised.
    """

    def _penalised_optimum(self, correlation: float, column_norm: float) -> float:
        """Shave the numerator toward zero, then divide as usual.

        The L1 penalty enters the coordinate update in one place and one way.
        Where an unpenalised coefficient would take ``correlation /
        column_norm``, this slides the numerator ``penalty / 2`` toward zero
        first, and lets it land there rather than pass through.

        That landing is the whole of what separates this from ridge, which
        divides by a larger number and so shrinks a coefficient without ever
        arriving at zero.
        """
        return self._soft_threshold(correlation, self.penalty / 2) / column_norm
