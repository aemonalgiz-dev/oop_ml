"""A layer that adds back what it read, which is what makes depth possible.

The problem it addresses
------------------------
Stacking layers is supposed to buy capacity, and past a certain depth it stops
doing so. The observation that named this is that a deeper network can be
*worse on its training data* than a shallower one -- not overfitting, which is a
gap between training and held-out, but a failure to fit at all. Something about
depth was making the fit harder rather than the model weaker.

The mechanism is in the backward walk. Blame arriving at the top of a chain is
multiplied, at every layer it passes through, by that layer's weights and by the
slope of its bend. Thirty layers means thirty such factors multiplied together,
and a product of thirty numbers is either very small or very large unless each
one is remarkably close to one. Small is the usual outcome, and it is silent:
the fit converges, the loss falls, and the bottom of the network has learned
almost nothing because almost no blame reached it. That is the same product
:mod:`~oop_ml.core.network.recurrent` records for a walk through time, and it is
the same failure written along depth instead.

What the addition does about it
--------------------------------
This layer answers with ``inner(x) + x``. Differentiate that and the two terms
separate::

    d (inner(x) + x) / dx  =  d inner(x) / dx  +  1

so the blame passed down is what the inner layer passed down *plus the arriving
block itself*. The second term is multiplied by one, and it is multiplied by one
however many of these are stacked: a chain of thirty residual blocks has a route
from the top to the bottom along which nothing is multiplied by any weight at
all. Whatever the inner layers do to the first term, the second arrives intact.

Read forward it says the same thing differently. The block starts as an identity
and the inner layer is asked only for the *difference* from it, so a layer that
has learned nothing yet is harmless rather than destructive, and a network can
be made deeper without first being made worse.

Why the shape is refused at construction
-----------------------------------------
The addition only means anything when the inner layer answers with what it
reads. Otherwise there is nothing to add ``x`` to -- or, worse, there is
something numpy will happily broadcast, which produces a layer that runs and
computes a quantity nobody asked for.

So the check is an integer comparison made before any data exists, which is this
package's whole promise. It compares extents rather than counts, for the reason
:mod:`~oop_ml.core.network.flatten` sets out at length: ``Flatten((2, 2))`` reads
``(2, 2)`` and answers ``(4,)``, which hold the same four numbers and are not
the same arrangement, and a count comparison would have accepted it.

Why it holds a layer rather than being one
-------------------------------------------
Every other layer in this package is a leaf. This one wraps, and that is the
first structural thing here that is not a chain: a :class:`LayerStack` is a list,
and a skip is an edge that bypasses part of it. Modelling that as a *layer
holding a layer* rather than as a change to the stack's topology keeps every
join in the stack exactly as it was -- a residual has one shape, reads what it
reads, answers what it answers, and is checked by ``follows`` like anything else.
The stack learns nothing new, and the bypass is a fact about one member.

Everything else follows from delegation. The shape is the inner layer's shape.
The gradient is the inner layer's gradient, unchanged, because the addition has
no parameters of its own. A step is a new residual around the stepped inner
layer, which keeps the rule that training does not mutate. And
:meth:`Residual._read_block` is the inner layer's, so the wrapper never invents a
reading rule -- whatever the layer inside states about what it can be handed is
what the wrapper states.

That last delegation is currently unreachable, and it is recorded rather than
quietly relied upon. The only layer in the library that overrides ``_read_block``
is :class:`~oop_ml.core.network.embedding.Embedding`, and an embedding answers
with an extent its input does not have, so the shape refusal above turns away any
residual around one. The delegation is therefore a statement about which object
owns the rule, checked by a test double, rather than a configuration that exists
today.

Why the response carries the inner one
---------------------------------------
The backward step has to ask the inner layer for its correction, and the inner
layer needs the response *it* produced -- its own scores, its own record of what
it read. Recomputing that means running the inner layer forward again, and for a
layer with a draw in it, dropout being the case that matters, the second run is
not the first run. So :class:`ResidualResponse` keeps it, which is the same
argument :class:`~oop_ml.core.network.dropout.DropoutResponse` makes about its
mask and :class:`~oop_ml.core.network.recurrent.RecurrentResponse` about its
trace.
"""

from __future__ import annotations

from oop_ml.core.exceptions import ShapeMismatchError
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray


class ResidualResponse(LayerResponse):
    """A layer response that also remembers what the inner layer answered.

    The backward step delegates to the inner layer, and the inner layer's
    ``correction_for`` wants the response it produced rather than one describing
    the sum. Rebuilding it would mean a second forward pass through the inner
    layer, which is wrong for any layer holding a draw -- a dropout inside a
    residual would be asked to justify a mask other than the one it used.

    So the inner response travels on this one, which is the pattern
    :class:`~oop_ml.core.network.dropout.DropoutResponse` and
    :class:`~oop_ml.core.network.recurrent.RecurrentResponse` already establish:
    a layer that carries one more thing gets a response type that says so,
    instead of a field on the type every layer shares.
    """

    __slots__ = ("_inner_response",)

    @classmethod
    def recording(
        cls,
        inputs: FloatArray,
        outputs: FloatArray,
        inner_response: LayerResponse,
    ) -> ResidualResponse:
        """Wrap a forward pass this layer just built, along with the inner one.

        The :meth:`~oop_ml.core.network.layer.LayerResponse.already_checked`
        pattern, sound for the same reason: the caller allocated the sum, filled
        it, and has shared it with nobody. The inner response's own blocks were
        frozen by whichever layer produced them.

        Parameters
        ----------
        inputs:
            The block this layer read, which is also the block that was added
            back.
        outputs:
            The inner layer's answer plus that block. Used as the scores as
            well, since a residual applies no bend and so has no pre-activation
            value distinct from its answer.
        inner_response:
            What the inner layer did, kept for the backward step.
        """
        response = cls.__new__(cls)
        inputs.setflags(write=False)
        outputs.setflags(write=False)
        response._inputs = inputs
        response._scores = outputs
        response._outputs = outputs
        response._inner_response = inner_response
        return response

    @property
    def inner_response(self) -> LayerResponse:
        """What the inner layer answered, and everything it kept about doing so."""
        return self._inner_response

    def __eq__(self, other: object) -> bool:
        # The inner response is why this class exists, so two residuals whose
        # inner layers did different things are not the same response however
        # well their sums line up. The base settles the type check and the
        # blocks; this adds the one field it does not know about.
        #
        # The base answers NotImplemented when the two types differ, and
        # ``bool(NotImplemented)`` is True, so coercing its answer would report
        # two different types equal -- the exact failure this override exists to
        # prevent. It is guarded explicitly.
        if not isinstance(other, ResidualResponse):
            return NotImplemented
        settled = super().__eq__(other)
        if settled is NotImplemented or settled is False:
            return settled
        return bool(self._inner_response == other._inner_response)

    def __repr__(self) -> str:
        return f"ResidualResponse(n_rows={self.n_rows!r})"


class Residual(Layer):
    """One layer, plus a route around it that multiplies the slope by one.

    Parameters
    ----------
    inner:
        The layer to wrap. It must answer with exactly what it reads, since the
        block it read is added to its answer.

    Raises
    ------
    ShapeMismatchError
        If the inner layer's shape does not read and answer the same
        arrangement. Extents rather than counts, so a layer reading ``(2, 2)``
        and answering ``(4,)`` is refused even though the addition would have a
        number of values to work with.

    Notes
    -----
    The residual itself holds no parameters. Its gradient is whatever the inner
    layer's is, including ``None`` when the inner layer has nothing to learn, and
    a step rebuilds the wrapper around the stepped inner layer.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Layer) -> None:
        if inner.shape.reads != inner.shape.answers:
            raise ShapeMismatchError(
                "a residual adds what a layer read to what it answered, so the "
                f"layer inside must answer with what it reads; this one reads "
                f"{inner.shape.reads} and answers {inner.shape.answers}"
            )

        self._inner = inner

    @property
    def shape(self) -> LayerShape:
        """The inner layer's shape, which reads and answers the same thing.

        A residual moves no join, so it can be inserted into a sound stack
        without making it unsound, and a stack containing one validates by the
        same exact extent equality as any other.
        """
        return self._inner.shape

    @property
    def inner(self) -> Layer:
        """The layer this one wraps."""
        return self._inner

    def _read_block(self, inputs: object) -> FloatArray:
        """Whatever the inner layer's reading rule is.

        The wrapper states no rule of its own. A residual around a layer that
        reads measurements reads measurements; a residual around a layer that
        reads whole numbers would read whole numbers, because the object that
        knows what it can be handed is the one inside.

        See the module docstring for why that second case cannot arise today:
        the only layer overriding this method answers with an extent its input
        does not have, and is therefore refused by the shape check at
        construction. The delegation is a statement about ownership rather than
        a live configuration, and a test double is what holds the wrapper to it.

        Raises
        ------
        InvalidValuesError
            Whatever the inner layer refuses, refused in its words.
        """
        return self._inner._read_block(inputs)

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        """The inner layer's answer, plus the block that was handed to it.

        Parameters
        ----------
        inputs:
            ``(n_rows, *shape.reads)``, already validated by
            :meth:`~oop_ml.core.network.layer.Layer.respond_to`.
        purpose:
            Forwarded to the inner layer unchanged. A dropout or a batch
            normalisation inside a residual must be told which kind of pass this
            is, and a wrapper that dropped the keyword would silently make it
            predict while the network trained.

        Returns
        -------
        ResidualResponse
            Carrying the sum, the block that was read, and the inner layer's own
            response for the backward step.

        Notes
        -----
        The inner layer is asked through its public ``respond_to``, so it
        applies its own reading rule and its own shape checks. That is one
        redundant coercion per pass, since this layer has already run the same
        rule, and it is accepted rather than worked around: reaching past a
        layer's entry point to save a copy would mean this wrapper deciding
        which of another layer's checks it can skip.
        """
        inner_response = self._inner.respond_to(inputs, purpose)
        outputs = inner_response.outputs + inputs

        return ResidualResponse.recording(
            inputs=inputs, outputs=outputs, inner_response=inner_response
        )

    def _checked_inner(self, response: LayerResponse) -> LayerResponse:
        """The inner response this one carries, refusing one that has none.

        The companion to
        :meth:`~oop_ml.core.network.layer.Layer._checked_arriving`, and the same
        kind of check: not about the data, about whether these two objects belong
        together. A response from the inner layer alone has the right
        arrangement and the right row count and satisfies that method perfectly,
        while carrying no record of the addition at all.

        Raises
        ------
        ShapeMismatchError
            If the response is not a :class:`ResidualResponse`.
        """
        if not isinstance(response, ResidualResponse):
            raise ShapeMismatchError(
                "a residual's backward step needs the inner response its own "
                "forward pass kept, and this response carries none -- it came "
                "from the inner layer alone, or from another layer of this shape"
            )
        return response.inner_response

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """The inner layer's correction, with the arriving block added back.

        Parameters
        ----------
        response:
            What this layer did on the way up. Must be a
            :class:`ResidualResponse`, since the inner layer's own response
            cannot be recovered from the sum.
        arriving:
            ``(n_rows, *shape.answers)``, the slope of the loss at this layer's
            outputs.

        Returns
        -------
        LayerCorrection
            ``gradient`` is the inner layer's, unchanged, because the addition
            has no parameters. ``passed_down`` is what the inner layer passed
            down plus the arriving block, which is the route multiplied by one.

        Raises
        ------
        ShapeMismatchError
            If the response did not come from a layer of this shape, if the
            arriving block does not describe this layer's outputs for the rows
            the response holds, or if the response carries no inner response.
        InvalidValuesError
            If the arriving block cannot be read as a float array.

        Notes
        -----
        The whole of it is ``+ block``, and dropping it is the mistake worth
        naming because it does not announce itself. The layer still runs, the
        loss still falls, and the network is exactly the deep chain the addition
        was introduced to rescue -- so the symptom is a fit that is merely
        disappointing rather than an error anybody can point at. The spec's
        finite difference is what settles it, and it settles it on the *inputs*
        rather than on the parameters, since the parameter slopes are identical
        with the skip and without it.

        The arriving block reaches the inner layer whole rather than reduced.
        Both terms of ``d (inner(x) + x) / dx`` are taken at the same arriving
        slope, so there is nothing to divide between them.
        """
        block = self._checked_arriving(response, arriving)
        inner_response = self._checked_inner(response)

        inner_correction = self._inner.correction_for(inner_response, block)

        return LayerCorrection(
            passed_down=inner_correction.passed_down + block,
            gradient=inner_correction.gradient,
        )

    def stepped_by(
        self, gradient: LayerGradient | None, learning_rate: float
    ) -> Residual:
        """A new residual around the inner layer, stepped.

        The wrapper owns no parameters, so it moves nothing itself and hands the
        gradient straight through. It is rebuilt rather than answering with
        ``self`` because the layer it holds may well have changed, and a wrapper
        pointing at the old one would quietly train nothing.

        Parameters
        ----------
        gradient:
            What the inner layer's parameters want to change by, or ``None``
            when it has none. Passed on without being read here.
        learning_rate:
            How far to move. Likewise.

        Returns
        -------
        Residual
            A residual of the same shape around the stepped inner layer. When
            the inner layer answers with itself, as a layer with nothing to
            learn does, the new wrapper holds that same object.

        Raises
        ------
        ShapeMismatchError
            Whatever the inner layer refuses, refused in its words -- a layer
            with parameters asked to step by nothing, say.
        """
        return Residual(self._inner.stepped_by(gradient, learning_rate))

    def __repr__(self) -> str:
        return f"Residual(inner={self._inner!r})"
