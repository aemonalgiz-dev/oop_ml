"""Inner products in a space nobody builds.

The trick, stated plainly
-------------------------
Every model here that draws a straight boundary is stuck with straight
boundaries. The usual escape is to add features -- squares, products, the
``PolynomialFeatures`` transformer already in this library -- and then fit the
same linear model in the wider space, where a straight surface in the new
coordinates is a curved one in the old.

That works and it does not scale. Degree ``d`` over ``p`` features costs
``C(p + d, d)`` columns: 20 features at degree 5 is 53,130 columns, and at
degree 10 it is 30 million. The transformed matrix stops fitting in memory long
before the mathematics runs out.

Now notice something about the models. Look at what a linear model actually
does with its inputs, and in a great many of them the raw rows appear only
inside **inner products** -- ``x_i . x_j`` between two training rows, or
``x . x_i`` between a query and a training row. Never individually. If that is
true, then a function ``K(a, b)`` returning *what the inner product would have
been in the expanded space* is all the model needs, and the expanded space
itself never has to exist.

That is the kernel trick, and it is a statement about what a model reads rather
than a clever approximation. The answer is exact.

Worked, on the smallest interesting case
-----------------------------------------
Take two features and the degree-2 expansion
``phi(x) = (x1^2, x2^2, sqrt(2) x1 x2)``. For ``a = (1, 2)`` and
``b = (3, 4)``::

    phi(a) = (1, 4, 2.8284)
    phi(b) = (9, 16, 16.9706)

    phi(a) . phi(b)  =  9 + 64 + 48  =  121

    (a . b)^2        =  (3 + 8)^2    =  121

Same number. The right-hand side never built a three-dimensional vector, and it
would not have built a 53,130-dimensional one either. At degree 2 in 2
dimensions the saving is nothing; the point is that the left side grows and the
right side does not.

What makes a function a kernel
-------------------------------
Not every two-argument function is one. ``K`` is a valid kernel when it is
symmetric and its Gram matrix is positive semi-definite for every finite set of
inputs -- Mercer's condition. That is exactly the condition under which some
feature map ``phi`` exists with ``K(a, b) = phi(a) . phi(b)``, which is what
lets a model reason as though the expanded space were real.

The practical consequence is that you cannot invent kernels freely. A function
that fails the condition produces a Gram matrix with negative eigenvalues, and
then the optimisation problems built on it are no longer convex -- the solve
may fail, or succeed and return something that is not a minimum of anything.

The four here
--------------
``LinearKernel`` is ``a . b``, the trivial case where the expanded space is the
original one. It exists because it makes every kernel model a strict
generalisation of its linear counterpart, and because it is the control: if a
kernel model with a linear kernel does not match the linear model, one of them
is wrong.

``PolynomialKernel`` is ``(gamma * a . b + constant)^degree``, whose feature map
is every product of up to ``degree`` original features. Finite-dimensional, and
the one where the size argument above is easiest to see.

``RadialBasisKernel`` is ``exp(-gamma * ||a - b||^2)``. Its feature map is
**infinite-dimensional** -- expand the exponential as a power series and every
degree appears -- which is the sharpest form of the argument, since no amount of
memory would let you build the columns. It is also the most useful default,
because it depends only on the distance between two points and so makes no
assumption about direction.

``SigmoidKernel`` is ``tanh(gamma * a . b + constant)``, and it is included with
a warning: it is **not** positive semi-definite for all parameter choices, so it
is not always a kernel at all. It is here because it appears everywhere in
practice and because a closed set of kernels that quietly omitted it would be
hiding the fact that the condition can fail.

Why these are classes and not an enum
--------------------------------------
:class:`~oop_ml.core.distance.metric.DistanceMetric` is a closed enum, and this
deliberately is not. The difference is parameters: five of the six distances
take none, so an enum member names the whole choice. A kernel is not fully
specified until its ``gamma``, ``degree`` and ``constant`` are fixed, and an
enum member cannot carry those -- it would have to be paired with a bag of
keyword arguments whose meaning changes depending on which member it sits
beside, which is the magic-string problem wearing a different hat. The class
*is* the closed set here, and it is closed the same way: a subclass that does
not exist cannot be passed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from oop_ml.core.data.row_block import RowBlock
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.kernel.matrix import KernelMatrix


class Kernel(BaseModel, ABC):
    """What the inner product would have been, in a space never constructed.

    A pydantic model rather than a plain class, so a kernel's parameters are
    validated where every other hyperparameter in this library is -- at
    construction, not at the first fit that happens to use them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def between(self, left: RowBlock, right: RowBlock) -> KernelMatrix:
        """The Gram matrix pairing every row of ``left`` with every row of ``right``.

        Shape ``(left.n_rows, right.n_rows)``. Entry ``[i, j]`` is
        ``K(left_i, right_j)``.

        Both blocks must be over the same features in the same order, since an
        inner product across mismatched columns is arithmetic rather than a
        measurement. This is the template; subclasses supply
        :meth:`_between_blocks`.

        Raises
        ------
        InvalidValuesError
            If the two blocks are not over the same features.
        """
        if left.feature_names != right.feature_names:
            raise InvalidValuesError(
                f"a kernel pairs rows over the same features; got "
                f"{list(left.feature_names)} against {list(right.feature_names)}"
            )

        return KernelMatrix(self._between_blocks(left, right))

    @abstractmethod
    def _between_blocks(self, left: RowBlock, right: RowBlock) -> np.ndarray:
        """The kernel value for every pair, as a raw array.

        Called with the feature check already passed.
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """The formula this kernel computes, for reports and reprs."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.description})"


class LinearKernel(Kernel):
    """``K(a, b) = a . b``: the plain inner product, no expansion at all.

    The identity case, and the control. A kernel model using this must agree
    with its non-kernel counterpart -- kernel ridge with a linear kernel is
    ridge regression -- and where the two disagree, one of them has a bug.
    """

    @property
    def description(self) -> str:
        return "a . b"

    def _between_blocks(self, left: RowBlock, right: RowBlock) -> np.ndarray:
        return left.values @ right.values.T


class PolynomialKernel(Kernel):
    """``K(a, b) = (gamma * a . b + constant) ** degree``.

    The feature map is every product of up to ``degree`` original features, so
    the space it stands in for is finite but large -- ``C(p + d, d)`` columns,
    which is 53,130 at 20 features and degree 5.

    Parameters
    ----------
    degree:
        How high the products go. Degree 1 with the default gamma and a
        constant of 0 is the linear kernel.
    gamma:
        Scales the inner product before the power. Large values make the kernel
        sensitive to the raw magnitude of the features, which is why scaling
        the inputs matters here as much as it does for a distance.
    constant:
        Added before the power. Non-zero is what admits the *lower*-degree
        terms into the feature map; at ``constant=0`` only the terms of exactly
        ``degree`` appear.

    Raises
    ------
    InvalidValuesError
        If ``degree`` is below 1, if ``gamma`` is not positive, or if
        ``constant`` is negative -- a negative constant can make the Gram
        matrix indefinite, so the function stops being a kernel.
    """

    degree: int = Field(default=3, ge=1)
    gamma: float = Field(default=1.0, gt=0.0)
    constant: float = Field(default=1.0, ge=0.0)

    @property
    def description(self) -> str:
        return f"({self.gamma} a . b + {self.constant}) ** {self.degree}"

    def _between_blocks(self, left: RowBlock, right: RowBlock) -> np.ndarray:
        products = self.gamma * (left.values @ right.values.T) + self.constant

        return np.power(products, self.degree)


class RadialBasisKernel(Kernel):
    """``K(a, b) = exp(-gamma * ||a - b|| ** 2)``: the Gaussian kernel.

    The one whose implied space is genuinely infinite-dimensional. Expanding the
    exponential as a power series produces terms of every degree, so no finite
    expansion of the features could reproduce it -- which makes this the case
    where the trick is not a saving but the only way.

    It depends only on the distance between two points, so it assumes nothing
    about direction, and its value falls smoothly from 1 at zero distance
    towards 0 as points separate. That gives it a natural reading: it is a
    similarity, and ``gamma`` sets how quickly similarity decays.

    Parameters
    ----------
    gamma:
        How fast similarity falls with distance. Large gamma makes every point
        similar only to its immediate neighbours, which lets a model fit
        anything and generalise to nothing; small gamma makes everything look
        alike and the model underfits. It is the hyperparameter of this kernel
        and it genuinely has to be tuned.

    Raises
    ------
    InvalidValuesError
        If ``gamma`` is not positive. At zero every pair has kernel value 1 and
        the Gram matrix is rank one, so there is nothing to fit.
    """

    gamma: float = Field(default=1.0, gt=0.0)

    @property
    def description(self) -> str:
        return f"exp(-{self.gamma} ||a - b|| ** 2)"

    def _between_blocks(self, left: RowBlock, right: RowBlock) -> np.ndarray:
        left_squares = np.sum(left.values * left.values, axis=1)[:, None]
        right_squares = np.sum(right.values * right.values, axis=1)[None, :]
        squared_gaps = (
            left_squares - 2.0 * (left.values @ right.values.T) + (right_squares)
        )

        return np.exp(-self.gamma * np.maximum(squared_gaps, 0.0))


class SigmoidKernel(Kernel):
    """``K(a, b) = tanh(gamma * a . b + constant)``.

    Included with a caveat rather than silently: this is **not** positive
    semi-definite for every choice of ``gamma`` and ``constant``, so for some
    settings it is not a kernel at all and the Gram matrix has negative
    eigenvalues. Models built on it can then fail to converge, or converge to
    something that is not the minimum of anything.

    It is here because it is used widely enough that omitting it would look
    like an oversight, and because a closed set of kernels that quietly
    excluded the awkward one would be hiding the fact that Mercer's condition
    is a condition rather than a formality.

    Parameters
    ----------
    gamma:
        Scales the inner product inside the ``tanh``.
    constant:
        Shifts it. Negative values are the usual choice in practice and are
        also where the positive semi-definiteness is most often lost.
    """

    gamma: float = Field(default=1.0, gt=0.0)
    constant: float = 0.0

    @property
    def description(self) -> str:
        return f"tanh({self.gamma} a . b + {self.constant})"

    def _between_blocks(self, left: RowBlock, right: RowBlock) -> np.ndarray:
        return np.tanh(self.gamma * (left.values @ right.values.T) + self.constant)
