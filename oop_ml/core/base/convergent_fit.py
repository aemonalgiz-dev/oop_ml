"""The half of an iterative fit that is not the step, for models with no design matrix.

Why this exists now and not earlier
------------------------------------
:class:`~oop_ml.core.base.iterative_solver.IterativeSolver` already holds this
machinery, and its own module docstring named the condition under which it
should be split in two:

    Closing it would mean splitting this class in two, one half holding the
    state and the reporting and the other adding the step loop on top, which is
    worth doing if a fifth iterative model turns up and not obviously worth it
    for one.

That condition has been met several times over. ``IterativeSolver`` serves
three linear models; ``LassoRegression`` keeps its own copy because a
coordinate sweep has no step vector to hand back; ``KMeans`` keeps a third
because it is not a linear model at all and ``IterativeSolver`` inherits from
:class:`~oop_ml.core.base.linear_model.LinearModel`. Adding a self-organising
map, a restricted Boltzmann machine and a Hebbian projection would have made
six copies of ``tolerance``, ``_converged`` and the pass counter.

So this is the lower half, and it inherits from
:class:`~oop_ml.core.base.estimator.Fittable` rather than from anything that
knows what a design matrix is. It is deliberately *not* a loop. The three
models that will use it iterate over quite different things, one over epochs
presenting rows, one over Gibbs sampling steps, one over sweeps of a
coefficient, and forcing them through a shared loop would mean a step method
whose arguments differ per subclass, which is the abstraction failing rather
than paying.

What it is instead is the bookkeeping, and the argument for centralising that
is written in the record. Two of the three original copies incremented the pass
counter *after* the convergence break, so a fit that settled immediately
reported having run zero passes while also reporting that it had converged.
That is the class of mistake which is invisible in every test that only checks
the answer.

What it deliberately does not decide
-------------------------------------
The name of the cap. A pass over the data is an *epoch* to anything presenting
rows one at a time, an *iteration* to Newton's method, and a *sweep* to
coordinate descent, and the word carries meaning. Each model keeps its own
public field and maps it onto :meth:`_pass_limit`, which is the same choice
``IterativeSolver`` made and for the same reason.

Whether convergence is even the right question
-----------------------------------------------
For a walk that descends an objective, ``converged`` means the answer stopped
moving and is therefore trustworthy. For a walk that only *approximates* a
gradient, it means something weaker, and the models that will inherit this are
mostly of the second kind. A restricted Boltzmann machine trained by
contrastive divergence is not descending the likelihood at all, it is
descending an approximation to it whose error does not vanish, so a settled
walk there means the weights stopped moving and not that the model reached a
maximum of anything. Each model's own docstring is obliged to say which of the
two its ``converged`` means, and this base class says only that the movement
fell below the tolerance.

Not persisted here
-------------------
``LEARNED_STATE`` is left empty deliberately. A subclass lists
``_passes_run`` and ``_converged`` alongside its own learned attributes, the
way ``IterativeSolver`` does, so that a class which happens to inherit this but
is not persistable does not acquire a format contract by accident.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from pydantic import Field, PrivateAttr

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.types import FloatArray


class ConvergentFit(Fittable):
    """A fit that walks to its answer and reports whether it arrived.

    Parameters
    ----------
    tolerance:
        Stop once nothing moved further than this in a whole pass. Subclasses
        may raise or lower the default where their convergence affords it.

    Notes
    -----
    A subclass supplies :meth:`_pass_limit`, runs whatever loop suits it, and
    calls :meth:`_record_walk` once at the end. Reading :attr:`converged` or
    the pass count before that raises ``NotFittedError`` like every other
    learned attribute in this library.
    """

    tolerance: float = Field(default=1e-8, gt=0.0)

    _passes_run: int | None = PrivateAttr(default=None)
    _converged: bool | None = PrivateAttr(default=None)

    @property
    @abstractmethod
    def _pass_limit(self) -> int:
        """The cap on passes, under whatever name this model gives it."""

    @property
    def _completed_passes(self) -> int:
        """How many passes the walk took, for a model to expose under its name.

        Private, because the public spelling differs per model. A model with an
        ``epochs`` cap exposes ``epochs_run`` and a model with an
        ``iterations`` cap exposes ``iterations_run``, and both read this.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._passes_run is not None
        return self._passes_run

    @property
    def converged(self) -> bool:
        """Whether the walk settled, rather than running out of passes.

        ``False`` means something was still moving when the cap was reached,
        and it is the attribute to check before trusting a fit.

        What ``True`` licences depends on the model, and the module docstring
        says why. For a walk descending an objective it means the answer
        stopped moving because it had arrived. For a walk following an
        approximate gradient it means only that the weights stopped moving,
        which is a weaker claim, and each such model says so in its own
        docstring rather than letting this one imply otherwise.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._converged is not None
        return self._converged

    def _has_converged(self, movement: FloatArray | float) -> bool:
        """Whether this pass moved everything less than ``tolerance``.

        Measured on the movement rather than on the change in the objective,
        which is the choice ``IterativeSolver`` made and defends: near an
        optimum the objective is flat, so the improvement it reports goes as the
        *square* of the parameter error and reaches zero in floating point while
        the parameters are still visibly moving. The movement is in the units
        the caller reads back and needs no reference value.

        Accepts a scalar as well as a block, because not every walk here moves
        an array. A Hopfield settling moves a count of flipped units and a
        contrastive divergence step moves a whole weight matrix, and both are
        the same question asked of a different shape.

        Parameters
        ----------
        movement:
            How far things moved this pass, in the parameters' own units.

        Returns
        -------
        bool
            True when the largest single movement is below ``tolerance``.
        """
        return bool(
            np.max(np.abs(np.asarray(movement, dtype=np.float64))) < self.tolerance
        )

    def _record_walk(self, passes_run: int, converged: bool) -> None:
        """Record how the walk ended. Call once, at the end of ``fit``.

        Deliberately separate from :meth:`_mark_fitted`, so that a fit which
        raises partway through leaves neither set. That is the same
        commit-nothing-until-everything-succeeded pattern the serving audit
        established for every other fit here.

        Parameters
        ----------
        passes_run:
            How many passes actually ran. Counted so that a walk settling on
            its first pass reports 1, not 0. Two of the three copies this class
            replaces got that wrong in the other direction, reporting 0 passes
            alongside ``converged=True``.
        converged:
            Whether the walk settled rather than exhausting :meth:`_pass_limit`.
        """
        self._passes_run = passes_run
        self._converged = converged
