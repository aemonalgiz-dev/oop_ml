"""Logistic regression: a linear boundary, fitted by maximum likelihood.

Theory
------
Point ``MultipleLinearRegression`` at a 0/1 target and it will fit happily, and
it will answer the wrong question. A hyperplane is unbounded, so it predicts
-0.3 and 1.4 without embarrassment, and neither of those is a probability.
Squared error compounds the problem by charging the same price for being wrong
by 0.4 on a near-certainty as for being wrong by 0.4 on a coin flip, when those
are not remotely the same mistake.

The way out is to stop modelling the probability directly and model something
unbounded instead. A probability lives in ``[0, 1]``. The odds, ``p / (1 - p)``,
live in ``[0, inf)``, so one bound is gone. The log-odds live in
``(-inf, inf)``, which is a space a hyperplane can reach every corner of::

    log(p / (1 - p))  =  b0 + b1 x1 + ... + bp xp

Invert that and the probability comes back through the sigmoid::

    p  =  1 / (1 + exp(-(X b)))

Notice what has not changed. The model is still linear in the coefficients, so
the design matrix, the ones column standing in for the bias, and pairing weights
with feature names all carry across untouched. That is why this shares
:class:`~oop_ml.core.base.linear_model.LinearModel` with the regressors and why
"linear" was never a statement about the shape of the predictions.

Why not least squares
---------------------
Beyond the wrong penalty, there is a harder reason. Squared error against a
sigmoid is **not convex** in ``b``. Scanning the Hessian over a grid on the
worked example below turns up a minimum eigenvalue of -6.125 at
``b = (-2.8, 0.2)``, and a negative eigenvalue means a saddle, which means local
minima, which means where you start decides where you finish.

Maximum likelihood has no such problem. Each observation contributes ``p`` when
its label is 1 and ``1 - p`` when it is 0, which folds into one expression::

    LL(b)  =  sum( y log(p) + (1 - y) log(1 - p) )

That is cross-entropy with the sign flipped. Its Hessian is::

    -X.T diag(p (1 - p)) X

a Gram matrix with non-negative weights, so it is negative semi-definite for
*any* ``b``, by construction rather than by luck. Checked analytically over the
same grid, the most positive eigenvalue was 1.85e-14, which is zero to floating
point. One bowl, one bottom, and the starting point stops mattering.

Mathematics
-----------
Differentiating the log-likelihood, the sigmoid's derivative ``p(1 - p)``
cancels against the logarithm's, and what survives is::

    dLL/db  =  X.T (y - p)

Set that beside ordinary least squares, whose gradient is ``X.T (y - X b)``.
The shape is identical: a residual, projected back through the design matrix.
The only difference is that ``p`` has been through a sigmoid on the way. This is
the reason ``GradientDescentRegression`` was worth building first, since the
walk, the learning rate, the convergence test and the reporting of
``epochs_run`` all transfer, and the one line that changes is where ``p`` comes
from.

There is a sanity check hiding in it. At ``b = 0`` every ``p`` is exactly 0.5,
so the gradient reduces to ``X.T (y - 0.5)``.

Why there is no closed form
---------------------------
Setting ``X.T (y - p) = 0`` leaves ``b`` inside a sigmoid inside the equation,
and no rearrangement gets it out; the relation is transcendental rather than
merely awkward. Ridge had a closed form. Lasso had none because ``abs(b)`` has
no derivative at zero. Logistic regression has none because the algebra cannot
be inverted at all, so gradient ascent stops being the slow route to what
``solve`` does instantly and becomes the only route there is.

Worked example
--------------
Eight students, hours studied against pass or fail, ascending at a learning rate
of 0.5::

      epoch   intercept     slope   log-likelihood
          0      0.0000    0.0000        -5.545177
          1      0.0000    0.2031        -5.294645
         10     -0.4341    0.2770        -4.933636
        100     -2.0290    0.7431        -4.308557
       1000     -2.4383    0.8637        -4.286217
      10000     -2.4383    0.8637        -4.286217

scikit-learn, unpenalised, agrees to 1.6e-09.

Two coefficients, two readings. The decision boundary sits where ``p = 0.5``,
which is where the log-odds are zero, so it lands at
``hours = -intercept / slope = 2.82``. And ``exp(slope) = 2.37`` says each extra
hour multiplies the *odds* of passing by 2.37. That is what a logistic
coefficient means. Not a change in probability, which is not constant along the
curve, but a constant multiplier on the odds.

Separation, and why it is not a bug
-----------------------------------
Run the same fit on cleanly separated data, where every failure sits below 2.5
hours and every pass above 3, and it does not converge. The coefficients climb
without limit; scikit-learn reaches an intercept of -114.57 on the same eight
rows.

Nothing is broken. When the classes are perfectly separable the maximum
likelihood estimate **does not exist**. Any boundary drawn in the gap classifies
every row correctly, and pushing ``abs(b)`` higher makes the predicted
probabilities more confident, so the likelihood keeps rising toward 1 without
ever attaining a maximum. This is not an exotic case either; it turns up
routinely with few rows or many features, and it is the reason scikit-learn
regularises by default rather than offering an unpenalised fit as the norm.

``converged`` is therefore load-bearing here in a way it is not for ridge. A
run that quietly exhausted ``max_epochs`` with coefficients still climbing has
not found an answer, and reporting its numbers as though it had is how a
separated dataset turns into a confidently meaningless model.
"""

from __future__ import annotations

from pydantic import Field

from oop_ml.core.base.iterative_solver import IterativeSolver
from oop_ml.core.data.column import Column
from oop_ml.core.data.design_matrix import DesignMatrix
from oop_ml.core.data.probabilities import Probabilities
from oop_ml.core.types import FloatArray
from oop_ml.numpy.classification.linear_classifier import LinearClassifier
from oop_ml.numpy.classification.logistic import sigmoid


class LogisticRegression(IterativeSolver, LinearClassifier):
    """A linear decision boundary, fitted by gradient ascent on the likelihood.

    Parameters
    ----------
    learning_rate:
        How far to step along the gradient each epoch. Too small and the walk
        runs out of patience before arriving; too large and it overshoots. The
        usable band is a property of the data, which is why there is no default
        that is right everywhere.
    max_epochs:
        How many steps to take before giving up. A limit on patience rather than
        a safety rail, although on separable data it is the only thing that ends
        the run at all.
    tolerance:
        Stop once no coefficient moved further than this in a whole epoch.
    threshold:
        The probability above which a row is called positive. Defaults to 0.5,
        which is the point where the log-odds are zero. Lower it to catch more
        of the real positives at the cost of more false alarms, which is the
        precision against recall trade made explicit as a number.
    fit_intercept:
        Inherited from :class:`~oop_ml.core.base.linear_model.LinearModel`.
    """

    learning_rate: float = Field(default=0.1, gt=0.0)
    max_epochs: int = Field(default=10_000, gt=0)
    tolerance: float = Field(default=1e-8, gt=0.0)
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)

    @property
    def _pass_limit(self) -> int:
        return self.max_epochs

    @property
    def epochs_run(self) -> int:
        """How many epochs the ascent actually took."""
        return self._completed_passes

    @staticmethod
    def _sigmoid(linear_predictor: FloatArray) -> Probabilities:
        """Map log-odds onto a probability: ``1 / (1 + exp(-z))``.

        Worth writing carefully rather than literally. In ``1 / (1 + exp(-z))``
        the exponential overflows once ``z`` drops below about -709, which a
        separated dataset will happily arrange. The value that comes back is
        still correct, since ``1 / inf`` is 0, so this costs an accumulating
        stream of RuntimeWarnings rather than a wrong answer; it is noise from
        the very rows the model is most certain about.

        Branching on the sign removes it. Use ``exp(z) / (1 + exp(z))`` where
        ``z`` is negative and the original form where it is not, so the
        exponential is only ever handed a non-positive number.

        Parameters
        ----------
        linear_predictor:
            ``z = X b``, one entry per observation.

        Returns
        -------
        FloatArray
            Probabilities in ``[0, 1]``, one per observation.
        """
        # Shared with NewtonLogisticRegression rather than spelled twice: the
        # branch-on-sign trick is easy to get subtly wrong, and two copies is
        # two chances to do so.
        return sigmoid(linear_predictor)

    def _gradient(
        self,
        design_matrix: DesignMatrix,
        target_values: Column,
        weights: FloatArray,
    ) -> FloatArray:
        """The gradient of the log-likelihood: ``X.T (y - p)``.

        Note that this is an *ascent* direction. The log-likelihood is being
        maximised, so the step is added rather than subtracted, which is the one
        sign difference from ``GradientDescentRegression``.

        Dividing by the sample count keeps the step size independent of how many
        rows there are, exactly as the regressor does.

        Parameters
        ----------
        design_matrix:
            ``X``, already carrying the ones column when ``fit_intercept`` is
            set.
        target_values:
            ``y``, the 0/1 labels.
        weights:
            ``b`` as it currently stands.

        Returns
        -------
        FloatArray
            One partial derivative per parameter.
        """
        probabilities = self._sigmoid(design_matrix.values @ weights)
        differences = target_values.values - probabilities.values

        return design_matrix.values.T @ differences / target_values.n_samples

    def _step(
        self,
        design_matrix: DesignMatrix,
        target_column: Column,
        weights: FloatArray,
    ) -> FloatArray:
        """One epoch of gradient ascent: the gradient scaled by the rate.

        Positive, because the log-likelihood is being maximised. The learning
        rate is the whole of what stands in for curvature here, which is why it
        has to be supplied and why a badly chosen one diverges.
        """
        return self.learning_rate * self._gradient(
            design_matrix, target_column, weights
        )
