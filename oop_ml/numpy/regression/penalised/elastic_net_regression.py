"""Elastic net: both penalties at once, and what the mixture buys.

Ridge and lasso each have a failure the other does not, and the two failures
are on the same data: several columns measuring nearly the same thing.

- **Lasso picks one of them and drops the rest**, and which one it picks is not
  a fact about the data. Where two columns are genuinely identical the
  objective is flat along the whole segment between "all of it on the first"
  and "all of it on the second", so every split is equally optimal and the
  sweep lands on whichever corner it reaches first. Rerun on a different column
  order and a different feature is the important one.
- **Ridge keeps all of them and selects nothing.** Its penalty goes limp as a
  coefficient approaches zero, so the answer is a hundred small coefficients
  where a reader wanted to know which five mattered.

The elastic net adds both penalties and reaches for both behaviours::

    S(b) = || y - X b ||^2
           +  penalty * l1_share       * sum(abs(b_j))
           +  penalty * (1 - l1_share) * sum(b_j ** 2)

``penalty`` says how hard to push and ``l1_share`` says in what shape, so the
two questions stay separate. At ``l1_share = 1`` this *is*
:class:`~oop_ml.numpy.regression.penalised.lasso_regression.LassoRegression`
and at ``l1_share = 0`` it is
:class:`~oop_ml.numpy.regression.penalised.ridge_regression.RidgeRegression`,
at the same ``penalty``, and both are asserted rather than claimed.

The update, and where the grouping comes from
-----------------------------------------------
Setting the subgradient with respect to ``b_j`` to zero, exactly as lasso does,
and carrying the extra squared term through::

    -2 (x_j . r_j) + 2 b_j (x_j . x_j) + penalty * l1_share * sign(b_j)
                                       + 2 penalty (1 - l1_share) b_j  =  0

    =>  b_j = soft_threshold(x_j . r_j, penalty * l1_share / 2)
              / (x_j . x_j + penalty * (1 - l1_share))

The L1 part shaves the numerator, which is what can still send a coefficient to
exactly zero. The L2 part enlarges the denominator, which is what stops any one
coefficient from taking the whole of a shared effect: the more of the effect a
coefficient has already claimed, the more that division costs it, so two
columns describing the same thing settle at the same value rather than at a
corner. That single added term is the entire difference from lasso, forward and
backward, which is why both models are two lines on the shared sweep in
:mod:`~oop_ml.numpy.regression.penalised.coordinate_descent`.

Worked example
--------------
Two *identical* columns, ``x1 = x2 = 1 .. 8``, with ``y = 6 * x1`` exactly, at
``penalty = 4``. The whole of the effect is six, and the question is only how
the two columns divide it::

    l1_share = 1.0  ->  (0.214286, 5.952381, 0.000000)   one takes it all
    l1_share = 0.5  ->  (0.732558, 2.918605, 2.918605)   split evenly
    l1_share = 0.0  ->  (1.227273, 2.863636, 2.863636)   split evenly

The first row is not a bug and not a preference. With identical columns the L1
objective is genuinely flat between the two corners, so the sweep's answer is
whichever it happened to reach first, and reversing the column order reverses
the answer. Any share of L2 at all makes the split unique, because a sum of
squares is smallest when the two are equal.

The same effect on merely correlated columns, two readings of one quantity
differing by noise at a correlation of 0.99996, target ``3 x1 + 3 x2 + x3``::

    l1_share = 1.0  ->  4.1334, 1.8350   the twins split more than two to one
    l1_share = 0.9  ->  2.9458, 2.9271   both near the 3.0 they were built from

Scale sensitivity
-----------------
Inherited from both parents, and worse than either. ``l1_share`` mixes a
threshold compared against ``x_j . r_j`` with a term added to ``x_j . x_j``, and
those two carry different powers of a column's units, so the *mixture* shifts
when a column is rescaled and not only the strength. Standardize first; see
:class:`~oop_ml.numpy.preprocessing.standardization.standardizer.Standardizer`.
"""

from __future__ import annotations

from pydantic import Field

from oop_ml.numpy.regression.penalised.coordinate_descent import (
    CoordinateDescentRegressor,
)


class ElasticNetRegression(CoordinateDescentRegressor):
    """Least squares under both penalties, fit by coordinate descent.

    Parameters
    ----------
    penalty:
        Overall strength of the penalty, split between the two shapes by
        ``l1_share``. Zero reproduces ordinary least squares at any share.
    l1_share:
        How much of that strength is spent on the absolute-value penalty, from
        0 for pure ridge to 1 for pure lasso. The default halves it, which is
        the neutral reading of a mixture and the engine's default too.
    max_iterations:
        Inherited. Cap on the number of full sweeps through the coefficients.
    tolerance:
        Inherited. Stop once no coefficient moved more than this during a
        whole sweep.
    fit_intercept:
        Inherited. The intercept, when fitted, carries neither penalty.
    """

    l1_share: float = Field(default=0.5, ge=0.0, le=1.0)

    def _penalised_optimum(self, correlation: float, column_norm: float) -> float:
        """Shave the numerator, and enlarge the denominator.

        One line holding both penalties, and each does its own job. The
        threshold is what can send a coefficient to exactly zero; the addition
        to the norm is what keeps columns describing one effect from collapsing
        onto whichever of them the sweep reached first.

        At ``l1_share = 1`` the addition is zero and this is lasso's update
        character for character. At ``l1_share = 0`` the threshold is zero, so
        the soft threshold passes its argument through untouched and what is
        left is the coordinate form of ridge's closed solution.
        """
        threshold = self.penalty * self.l1_share / 2
        ridge_addition = self.penalty * (1.0 - self.l1_share)

        return self._soft_threshold(correlation, threshold) / (
            column_norm + ridge_addition
        )
