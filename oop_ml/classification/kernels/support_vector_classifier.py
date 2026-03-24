"""The widest gap between two classes, drawn in a space nobody builds.

Theory
------
Logistic regression finds *a* separating boundary. When the classes genuinely
separate there are infinitely many, and it settles on one by minimising log
loss, which keeps pushing every point further from the boundary forever. A
support vector machine asks a sharper question: of all the boundaries that
separate the classes, **which one leaves the widest empty corridor around
itself?**

That corridor is the margin, and maximising it is a claim about generalisation.
A boundary squeezed against the training points has no room to be wrong; one
sitting in the middle of a wide gap can be moved a long way before it starts
making mistakes.

Getting to the dual, which is where the kernels get in
------------------------------------------------------
Write the boundary as ``w . x + b``, and require every point to sit on the
correct side by at least 1::

    y_i (w . x_i + b)  >=  1

The corridor's width turns out to be ``2 / ||w||``, so maximising the margin is
minimising ``||w||^2 / 2`` subject to those constraints. Attach a Lagrange
multiplier ``a_i`` to each constraint, and the stationarity condition gives::

    w  =  sum_i  a_i y_i x_i

Substituting that back eliminates ``w`` entirely and leaves a problem in the
multipliers alone::

    maximise   sum_i a_i  -  (1/2) sum_i sum_j  a_i a_j y_i y_j (x_i . x_j)
    subject to 0 <= a_i <= C   and   sum_i a_i y_i = 0

That is the dual, and look at what survived: the rows appear only as
``x_i . x_j``. Replace that with ``K(x_i, x_j)`` and the whole thing runs in the
implied space, unchanged. The kernel trick is not bolted on here -- the dual was
already in exactly the form that admits it.

Support vectors, and why the model is small
--------------------------------------------
The multipliers have a striking property. At the optimum, ``a_i`` is non-zero
**only** for points on or inside the margin. Every point comfortably on the
correct side gets ``a_i = 0`` and drops out of ``w`` entirely.

Those surviving points are the support vectors, and they are the model. Delete
every other training row and refit, and you get an identical boundary. That is
why this scales where kernel ridge does not: kernel ridge keeps every row,
because its dual weights are all non-zero, while a support vector machine
typically keeps a small fraction.

``C``, and what soft margin means
----------------------------------
Real classes overlap, and then no boundary satisfies every constraint. The
softened version allows violations and charges for them, with ``C`` setting the
price: the multipliers are capped at ``C``, so no single point can pull on the
boundary harder than that.

Small ``C`` is a wide margin bought by tolerating misclassifications -- more
bias, less variance. Large ``C`` insists on separating the training data and
will contort the boundary to do it. At ``C`` large with an RBF kernel of large
gamma, a support vector machine will fit any labelling at all, including a
random one, which is exactly the regime where it learns nothing.

How this one is solved
-----------------------
By projected gradient ascent on the dual, which is the plainest correct method
rather than the fastest. The gradient of the dual objective is
``1 - Q a`` where ``Q_ij = y_i y_j K(x_i, x_j)``; step along it, then project
back into the box ``0 <= a_i <= C``.

The equality constraint ``sum a_i y_i = 0`` comes from the intercept, and this
implementation drops it by absorbing the intercept into the kernel -- adding a
constant to every kernel value is equivalent to appending a constant feature,
which gives the boundary its offset without a separate term. That is a real
simplification and it is the reason Sequential Minimal Optimisation, the
standard solver, is not needed here: SMO exists precisely to respect that
equality constraint while updating multipliers in pairs.

The consequence worth stating: this is correct and it is slow. SMO or a
quadratic programming solver would be the fast route, and that is an
optimisation rather than a different model.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

from oop_ml.core.base.estimator import Classifier
from oop_ml.core.data.column import Column
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.probabilities import Probabilities
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.kernel.functions import Kernel, LinearKernel
from oop_ml.core.kernel.matrix import KernelMatrix
from oop_ml.core.types import FloatArray
from oop_ml.core.validation import ValueRole

SUPPORT_VECTOR_THRESHOLD = 1e-08
"""Above this multiplier, a training row counts as a support vector.

Exactly zero is what the mathematics says and not what a floating-point solver
returns; a point comfortably outside the margin lands at 1e-15 rather than 0.
Counting those as support vectors would report every training row as one and
lose the property that makes the model small.
"""


class SupportVector:
    """One training row that touches the margin, and how hard it pulls.

    Parameters
    ----------
    position:
        Which training row this was, so a caller can go back to it.
    multiplier:
        The dual variable. Strictly positive by construction -- a row with a
        zero multiplier is not a support vector, it is a row the boundary does
        not depend on.
    label:
        The row's class as ``-1`` or ``+1``, which is the encoding the dual is
        written in.

    Raises
    ------
    InvalidValuesError
        If the multiplier is not positive, or the label is not -1 or +1.
    """

    __slots__ = ("_label", "_multiplier", "_position")

    def __init__(self, position: int, multiplier: float, label: float) -> None:
        if multiplier <= 0.0:
            raise InvalidValuesError(
                f"a support vector has a positive multiplier; got {multiplier}. "
                f"A row with a zero multiplier is not one."
            )

        if label not in (-1.0, 1.0):
            raise InvalidValuesError(
                f"the dual is written in -1/+1 labels; got {label}"
            )

        self._position = int(position)
        self._multiplier = float(multiplier)
        self._label = float(label)

    @property
    def position(self) -> int:
        """Which training row this was."""
        return self._position

    @property
    def multiplier(self) -> float:
        """How hard this row pulls on the boundary."""
        return self._multiplier

    @property
    def label(self) -> float:
        """This row's class, as -1 or +1."""
        return self._label

    def is_at_the_cap(self, capacity: float) -> bool:
        """Whether this row's multiplier hit ``C``.

        A row at the cap is one the soft margin gave up on -- it sits inside
        the corridor or on the wrong side of the boundary, and the only thing
        stopping it pulling harder is the cap itself. Counting them is the
        quickest read on whether ``C`` is too small for the data.
        """
        return self._multiplier >= capacity * (1.0 - 1e-09)

    def __repr__(self) -> str:
        return (
            f"SupportVector(row={self._position}, "
            f"multiplier={self._multiplier:.4f}, label={self._label:+.0f})"
        )


class SupportVectors:
    """The rows the boundary actually depends on.

    Deleting every training row that is not in here and refitting gives the
    same boundary, which is the sense in which these *are* the model.

    Raises
    ------
    InvalidValuesError
        If two entries name the same training row.
    """

    __slots__ = ("_n_training_rows", "_vectors")

    def __init__(self, vectors: Sequence[SupportVector], n_training_rows: int) -> None:
        positions = [vector.position for vector in vectors]

        if len(set(positions)) != len(positions):
            raise InvalidValuesError(
                f"a training row can appear once; got positions {positions}"
            )

        self._vectors = tuple(vectors)
        self._n_training_rows = int(n_training_rows)

    @property
    def n_vectors(self) -> int:
        """How many rows the boundary depends on."""
        return len(self._vectors)

    @property
    def n_training_rows(self) -> int:
        """How many rows the fit saw."""
        return self._n_training_rows

    @property
    def share_of_training_rows(self) -> float:
        """What fraction of the training set survived into the model.

        The number that separates this from kernel ridge, which keeps all of
        them. A share near 1 means the fit found no structure and every point
        is touching the margin, which usually means ``C`` is far too large or
        the kernel far too flexible.
        """
        if self._n_training_rows == 0:
            return 0.0

        return self.n_vectors / self._n_training_rows

    def positions(self) -> list[int]:
        """Which training rows these were."""
        return [vector.position for vector in self._vectors]

    def n_at_the_cap(self, capacity: float) -> int:
        """How many multipliers hit ``C``, so how many the margin gave up on."""
        return sum(1 for vector in self._vectors if vector.is_at_the_cap(capacity))

    def __iter__(self) -> Iterator[SupportVector]:
        return iter(self._vectors)

    def __len__(self) -> int:
        return self.n_vectors

    def __repr__(self) -> str:
        return (
            f"SupportVectors({self.n_vectors} of {self._n_training_rows}, "
            f"{self.share_of_training_rows:.1%})"
        )


class SupportVectorClassifier(Classifier[Sequence[Feature], Feature]):
    """Separate two classes by the widest corridor the kernel allows.

    Parameters
    ----------
    kernel:
        Which space to draw the boundary in. With a linear kernel this is the
        ordinary maximum-margin classifier.
    capacity:
        ``C``: the price of a margin violation, and the cap on any single
        multiplier. Small values buy a wide margin by tolerating mistakes;
        large values insist on separating the training data.
    learning_rate:
        The step size for the projected gradient ascent on the dual.
    max_epochs:
        The ceiling on ascent steps.
    tolerance:
        Stop when no multiplier moves further than this in a step.

    Raises
    ------
    InvalidValuesError
        If any parameter is outside its range.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    kernel: Kernel = LinearKernel()
    capacity: float = Field(default=1.0, gt=0.0)
    learning_rate: float = Field(default=0.001, gt=0.0)
    max_epochs: int = Field(default=1000, ge=1)
    tolerance: float = Field(default=1e-06, gt=0.0)

    _multipliers: FloatArray | None = PrivateAttr(default=None)
    _signed_labels: FloatArray | None = PrivateAttr(default=None)
    _training_rows: RowBlock | None = PrivateAttr(default=None)
    _epochs_run: int | None = PrivateAttr(default=None)

    @property
    def support_vectors(self) -> SupportVectors:
        """The rows the boundary depends on.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._multipliers is not None
        assert self._signed_labels is not None
        assert self._training_rows is not None

        return SupportVectors(
            [
                SupportVector(position, float(multiplier), float(label))
                for position, (multiplier, label) in enumerate(
                    zip(self._multipliers, self._signed_labels, strict=True)
                )
                if multiplier > SUPPORT_VECTOR_THRESHOLD
            ],
            self._training_rows.n_rows,
        )

    @property
    def multipliers(self) -> FloatArray:
        """Every dual variable, including the zeros.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._multipliers is not None
        return self._multipliers.copy()

    @property
    def epochs_run(self) -> int:
        """Ascent steps taken.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._epochs_run is not None
        return self._epochs_run

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Maximise the dual objective over the multipliers.

        Converts the 0/1 target to the -1/+1 encoding the dual is written in,
        builds the Gram matrix, and hands both to :meth:`_ascend`.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonEqualArrayLengthError
            If the features and target are different lengths.
        SingleClassError
            If the target does not hold both classes, since there is then no
            boundary to place.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_aligned_with(target_values)

        target_column = Column.of(target_values, ValueRole.TARGET_VALUES)
        target_column.check_has_both_classes()

        names = tuple(feature.name for feature in feature_set)
        rows = self._as_rows(feature_set, names)

        self._training_rows = rows
        self._signed_labels = np.where(target_column.values > 0.5, 1.0, -1.0)
        self._multipliers = self._ascend(
            self.kernel.between(rows, rows), self._signed_labels
        )
        self._mark_fitted()

        return self

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """Which side of the boundary each row falls on, as 0 or 1.

        The sign of :meth:`decision_values`, mapped back from the -1/+1 the
        dual works in to the 0/1 the rest of this library speaks.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        return Predictions.already_checked(
            np.where(self.decision_values(input_values) >= 0.0, 1.0, 0.0)
        )

    def predict_probability(self, input_values: Sequence[Feature]) -> Probabilities:
        """A bounded score per row, and **not** a calibrated probability.

        A support vector machine has no probability model. It maximises a
        margin, and nothing in that objective produces a likelihood. What this
        returns is the decision value squashed through a logistic so it lands in
        ``[0, 1]``, which the ``Classifier`` frame requires and which is
        monotonic in the distance from the boundary -- so thresholding it ranks
        rows correctly.

        It is not a probability. 0.9 here does not mean nine times in ten. The
        honest way to get one is Platt scaling: fit a one-dimensional logistic
        regression to the decision values on held-out rows. That is a separate
        model and is deliberately not hidden inside this one.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        values = self.decision_values(input_values)

        return Probabilities(1.0 / (1.0 + np.exp(-np.clip(values, -500.0, 500.0))))

    def decision_values(self, input_values: Sequence[Feature]) -> FloatArray:
        """Signed distance from the boundary, in the implied space's units.

        ``sum_i a_i y_i K(x, x_i)`` over the training rows -- and since ``a_i``
        is zero for everything but the support vectors, only they contribute.
        Positive means the ``+1`` side.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        against_training = self.query_matrix(input_values)

        assert self._multipliers is not None
        assert self._signed_labels is not None

        return against_training.values @ (self._multipliers * self._signed_labels)

    def _ascend(
        self, kernel_matrix: KernelMatrix, signed_labels: FloatArray
    ) -> FloatArray:
        """Maximise the dual objective by projected gradient ascent.

        The concept, and the only part of this class that is the algorithm.

        The objective is::

            sum_i a_i  -  (1/2) a' Q a        where  Q_ij = y_i y_j K(x_i, x_j)

        and its gradient with respect to ``a`` is ``1 - Q a`` -- a vector of
        ones minus the matrix product. Then:

        1. Build ``Q`` from the kernel matrix and the labels. The outer product
           ``y y'`` times the kernel values, element-wise.
        2. Start every multiplier at zero.
        3. Repeat up to ``max_epochs`` times:

           a. Step along the gradient: ``a + learning_rate * (1 - Q a)``.
              **Ascent, so the step is added.** This maximises rather than
              minimises, which is the opposite sign from every other iterative
              solver in this library, and getting it backwards produces
              multipliers that all collapse to zero -- a model that says the
              same thing about every row and looks merely useless rather than
              wrong.
           b. Project back into the box: clip below at 0 and above at
              ``capacity``. This is what makes it *projected* ascent, and it is
              how the constraints are enforced -- step freely, then put the
              answer back where it is allowed to be.
           c. Stop when the largest change in any multiplier is at most
              ``tolerance``.

        4. Record how many steps ran in ``self._epochs_run`` before returning.

        The projection is the whole reason this works without a quadratic
        programming solver. Clipping is the exact projection onto a box, so
        each step lands on the true nearest allowed point rather than an
        approximation of it.

        Parameters
        ----------
        kernel_matrix:
            The training Gram matrix, square.
        signed_labels:
            The target as -1 and +1, one per training row.

        Returns
        -------
        FloatArray
            The multipliers, shape ``(n_training_rows,)``, every one in
            ``[0, capacity]``. Do not set ``_fitted`` here.
        """
        kernel_matrix.check_square()
        quadratic = np.outer(signed_labels, signed_labels) * kernel_matrix.values
        multipliers = np.zeros(kernel_matrix.n_left)
        self._epochs_run = 0

        for _ in range(self.max_epochs):
            self._epochs_run += 1
            stepped = multipliers + self.learning_rate * (1.0 - quadratic @ multipliers)
            moved = np.clip(stepped, 0.0, self.capacity)
            shift = float(np.max(np.abs(moved - multipliers)))
            multipliers = moved

            if shift <= self.tolerance:
                break

        return multipliers

    def _as_rows(self, feature_set: FeatureSet, names: tuple[str, ...]) -> RowBlock:
        """The features as a row block, in the given order."""
        ordered = FeatureSet.matching(names, list(feature_set))

        return rows_of(
            np.column_stack([ordered.column(name).values for name in names]), names
        )

    def query_matrix(self, input_values: Sequence[Feature]) -> KernelMatrix:
        """The kernel matrix pairing query rows against the training rows.

        Shape ``(n_queries, n_training)``, so not square and not symmetric.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        self._check_fitted()
        assert self._training_rows is not None

        fitted = self._training_rows.feature_names
        supplied = {feature.name for feature in input_values}

        if supplied != set(fitted):
            raise InvalidValuesError(
                f"expected exactly the fitted features {sorted(fitted)}; "
                f"got {sorted(supplied)}"
            )

        return self.kernel.between(
            self._as_rows(FeatureSet(input_values), fitted), self._training_rows
        )

    @property
    def signed_labels(self) -> FloatArray:
        """The training target as -1 and +1, which is what the dual uses.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._signed_labels is not None
        return self._signed_labels.copy()

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"SupportVectorClassifier({self.kernel!r}, unfitted)"

        return (
            f"SupportVectorClassifier({self.kernel!r}, capacity={self.capacity}, "
            f"{self.support_vectors!r})"
        )
