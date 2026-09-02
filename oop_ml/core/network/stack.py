"""Layers joined into a chain, which is where the shape goal is actually kept.

Why a predicate was not enough
------------------------------
:meth:`~oop_ml.core.network.shape.LayerShape.follows` answers whether two
layers fit together, and an answer nobody is obliged to read is not a
guarantee. A caller can build a network out of layers that do not chain,
discard every ``follows`` result, and discover the disagreement hours later
inside a matrix multiply. That is precisely the free ``check_*`` function this
library's design notes rule out: a rule spanning several values belongs to an
object that enforces it in its constructor, not to a guard every caller has to
remember.

:class:`LayerStack` is that object. It refuses to exist unless every join
holds, so a stack that has been constructed is a stack whose shape is already
known to be sound, and no forward pass anywhere has to re-establish it.

What the refusal actually costs to compute
-------------------------------------------
Nothing, and that is the point worth keeping. Every number involved is settled
by the layers themselves, so validating a whole network is a walk down a list
of integers::

    layer 1   (4 -> 8)
    layer 2   (8 -> 8)
    layer 3   (8 -> 3)

Three layers, two joins, two comparisons. No data is read, no memory is
allocated for a batch, and no epoch is begun. A network that cannot work is
rejected in microseconds rather than after however long it takes to reach the
first bad multiply, and the error names the join rather than the array.

The two facts the data supplies
-------------------------------
Only the ends. The first layer's input width has to match the number of
features, and the last layer's output width has to match what the task wants:
one column for a regression or a binary decision, one per class otherwise.
Everything between those two is internal and settled here.

That is the whole of the claim made when this package began. A network's shape
is decidable before a single row arrives, and the only reason it might not be
is that nobody built an object to decide it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from oop_ml.core.exceptions import EmptyValuesError, ShapeMismatchError
from oop_ml.core.network.gradient import BackwardPass, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.loss import Loss
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray


class StackResponse:
    """What every layer did, in order, on one block of rows.

    A backward pass needs all of it, since the correction owed to a layer
    depends on the scores it formed and the outputs it handed upward, so the
    intermediate results are kept rather than discarded on the way through.

    The final layer's outputs are the network's answer, and
    :attr:`outputs` is that block, so an ordinary caller never indexes into
    the sequence at all.

    Parameters
    ----------
    responses:
        One :class:`~oop_ml.core.network.layer.LayerResponse` per layer, in the
        order the layers were visited.

    Raises
    ------
    EmptyValuesError
        If no responses are supplied. A stack holds at least one layer, so a
        pass through it produces at least one response.
    """

    __slots__ = ("_responses",)

    def __init__(self, responses: Sequence[LayerResponse]) -> None:
        if not responses:
            raise EmptyValuesError(
                "a pass through a stack produces one response per layer"
            )

        self._responses = tuple(responses)

    @property
    def outputs(self) -> FloatArray:
        """``(n_rows, n_outputs)``, the last layer's answer."""
        return self._responses[-1].outputs

    def __len__(self) -> int:
        """How many layers answered."""
        return len(self._responses)

    def __getitem__(self, position: int) -> LayerResponse:
        """What the layer at ``position`` did.

        A backward pass needs exactly this: layer ``k``'s own scores, and
        layer ``k - 1``'s outputs, which were the block layer ``k`` read.
        """
        return self._responses[position]

    def __iter__(self) -> Iterator[LayerResponse]:
        """Each layer's response, bottom to top."""
        return iter(self._responses)

    def __repr__(self) -> str:
        return f"StackResponse(n_layers={len(self._responses)!r})"


class LayerStack:
    """Layers in order, refusing to exist unless every join holds.

    Parameters
    ----------
    layers:
        At least one, each reading exactly what the one beneath it answers.

    Raises
    ------
    EmptyValuesError
        If no layers are supplied. There is no network to describe.
    ShapeMismatchError
        If any layer does not read what the layer beneath it answers. The
        message names the join and both widths, since the mistake is at a
        specific seam and saying which one is the entire value of checking
        early.
    """

    __slots__ = ("_layers", "_shape")

    def __init__(self, layers: Sequence[Layer]) -> None:
        if not layers:
            raise EmptyValuesError("a stack needs at least one layer")

        ordered = tuple(layers)
        for position, (beneath, above) in enumerate(
            zip(ordered, ordered[1:], strict=False)
        ):
            if not above.shape.follows(beneath.shape):
                # Named by arrangement rather than by count, because the join
                # is checked by arrangement. An earlier version reported the
                # two element counts, and on the case this package exists to
                # catch it printed "layer 0 answers with 5408 numbers and layer
                # 1 reads 5408" -- a refusal whose own message says the two
                # agree. A reader meeting that goes looking for a bug in the
                # library rather than for the missing Flatten.
                raise ShapeMismatchError(
                    f"layer {position} answers with {beneath.shape.answers} and "
                    f"layer {position + 1} reads {above.shape.reads}"
                    + (
                        f"; both hold {beneath.shape.n_outputs} numbers, so it "
                        "is the arrangement that disagrees and not the width"
                        if beneath.shape.n_outputs == above.shape.n_inputs
                        else ""
                    )
                )

        self._layers = ordered
        # The *arrangements*, not the counts. Reading `n_inputs` here collapsed
        # a picture to its element count, so a stack beginning with a
        # convolution over (1, 28, 28) reported that it read (784,). Every join
        # inside it was still checked correctly, since `follows` compares
        # extents, but the shape the stack reported about itself was one no
        # layer in it would accept: a caller building a block from
        # `stack.shape.reads` was refused by the stack's own first layer.
        self._shape = LayerShape(
            n_inputs=ordered[0].shape.reads,
            n_outputs=ordered[-1].shape.answers,
        )

    @property
    def shape(self) -> LayerShape:
        """What the network reads and what it answers with, end to end.

        The first layer's arrangement paired with the last layer's. Everything
        between them is internal, already agreed, and no longer anybody's
        business.

        Arrangements rather than counts, so that a block built from
        ``shape.reads`` is a block this stack accepts. That is the whole point
        of the property and it is what makes it worth reading back: a
        convolutional stack says it reads ``(1, 28, 28)``, which is true, where
        ``(784,)`` would be a shape none of its layers would take.
        """
        return self._shape

    def __len__(self) -> int:
        """How many layers the stack holds."""
        return len(self._layers)

    def __getitem__(self, position: int) -> Layer:
        """The layer at ``position``, counting from the input end."""
        return self._layers[position]

    def __iter__(self) -> Iterator[Layer]:
        """The layers, bottom to top."""
        return iter(self._layers)

    def respond_to(
        self, inputs: FloatArray, purpose: PassPurpose = PassPurpose.PREDICTING
    ) -> StackResponse:
        """Push a block of rows all the way up.

        Only the first layer's width is worth checking, and even that is
        checked by the layer itself. Every interior join was settled at
        construction, so nothing here re-establishes anything.

        Parameters
        ----------
        inputs:
            ``(n_rows, n_inputs)``, its width matching :attr:`shape`.
        purpose:
            Why the pass is happening, handed to every layer unchanged. The
            default answers deterministically, so a caller who never thinks
            about it gets predictions rather than perturbed ones.
            :meth:`backward_pass` states :attr:`PassPurpose.TRAINING` for
            itself, which is why an ordinary training loop never has to.

        Returns
        -------
        StackResponse
            One response per layer, in order, whose :attr:`StackResponse.outputs`
            is the network's answer.

        Raises
        ------
        ShapeMismatchError
            If the block is not two-dimensional, or its width is not what the
            first layer reads.
        EmptyValuesError
            If the block holds no rows.
        InvalidValuesError
            If the block cannot be read as a float array, or any entry in it is
            not finite.
        """
        return self._response_for(inputs, purpose)

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> StackResponse:
        """The forward pass, layer by layer.

        Parameters
        ----------
        inputs:
            ``(n_rows, n_inputs)`` for the first layer. Its width is validated
            by that layer, not here.
        purpose:
            Passed to each layer in turn. The stack does not read it, and
            deliberately does not decide it either -- one layer perturbing
            while another does not would be a network in two states at once.

        Returns
        -------
        StackResponse
            Every layer's response, in visiting order.

        Notes
        -----
        Each layer reads what the one beneath it answered, so the block handed
        onward is the previous response's ``outputs`` and the first block is
        the caller's own. Every response is kept, not only the last, because a
        backward pass needs the scores each layer formed.

        Nothing here checks a width. The interior joins were settled when the
        stack was built, and the first layer checks its own, which is the
        entire payoff of having refused a bad chain at construction.
        """
        responses: list[LayerResponse] = []
        block = inputs

        for layer in self._layers:
            response = layer.respond_to(block, purpose)
            responses.append(response)
            block = response.outputs

        return StackResponse(responses)

    def backward_pass(
        self, inputs: FloatArray, targets: FloatArray, loss: Loss
    ) -> BackwardPass:
        """Run the network forward, score it, and walk the blame back down.

        Parameters
        ----------
        inputs:
            ``(n_rows, n_inputs)``, matching :attr:`shape`.
        targets:
            ``(n_rows, n_outputs)``, matching what the last layer answers with.
        loss:
            What to score the answers by, and what starts the walk.

        Returns
        -------
        BackwardPass
            The loss, and one
            :class:`~oop_ml.core.network.gradient.LayerGradient` per layer,
            bottom to top, so the caller never has to reverse anything.

        Raises
        ------
        ShapeMismatchError
            If the blocks do not match this network's two ends.
        EmptyValuesError
            If there are no rows.
        InvalidValuesError
            If either block carries a non-finite entry.

        Notes
        -----
        The forward pass is :meth:`respond_to`, and its
        :class:`StackResponse` is why every layer's *scores* were kept rather
        than only its outputs: a bend's slope is taken at the score that
        produced it.

        Then the walk, top to bottom, over ``zip(reversed(self),
        reversed(forward))``. Both of those support ``reversed`` directly, and
        each response carries the block its own layer read, so nothing has to
        be paired up by hand::

            correction = layer.correction_for(response, arriving)
            gradients.append(correction.gradient)
            arriving = correction.passed_down

        ``arriving`` is the only thing carried between iterations. It starts as
        the loss's slope at the final outputs and leaves each layer as the
        slope at the outputs of the layer beneath. That single variable is the
        backward pass.

        An earlier version of this scaffold made ``LayerResponse`` hold only
        the scores and outputs, which forced the walk to rebuild the list of
        what each layer read by shifting the outputs down one and pushing the
        caller's block on the front. That shift was reconstructing something
        the forward pass had known and discarded, and moving ``inputs`` onto
        the response removed it.

        Every shape in those four lines is forced, which is the useful thing
        to lean on while writing it. If a product does not conform, the
        transpose is on the wrong side.

        The gradients come out top to bottom and :class:`BackwardPass` wants
        layer order, so they get reversed once at the end.
        """
        # Stated, not defaulted. A backward pass is training by definition, and
        # a dropout layer that answered deterministically here would have the
        # gradient describe a network the step is not about to build.
        forward = self.respond_to(inputs, PassPurpose.TRAINING)

        # Where the walk starts. The loss reports its slope at the final
        # outputs, and because the last layer is linear those outputs are the
        # scores, so this is already the top layer's arriving block.
        measurement = loss.measure(forward.outputs, targets)
        arriving = measurement.gradient

        gradients: list[LayerGradient | None] = []
        for layer, response in zip(reversed(self), reversed(forward), strict=True):
            correction = layer.correction_for(response, arriving)
            gradients.append(correction.gradient)
            arriving = correction.passed_down

        # Appended top to bottom; BackwardPass wants layer order.
        return BackwardPass(loss=measurement.value, gradients=list(reversed(gradients)))

    def stepped_by(self, backward: BackwardPass, learning_rate: float) -> LayerStack:
        """A new stack with every parameter nudged against its slope.

        Parameters
        ----------
        backward:
            The gradients to move against, one per layer in layer order.
        learning_rate:
            How far to move. The calculus primer's eta, with the same two
            failure modes at either end.

        Returns
        -------
        LayerStack
            A new stack, same shapes and same bends, moved parameters.

        Raises
        ------
        ShapeMismatchError
            If the gradients do not describe this stack's layers.

        Notes
        -----
        Downhill is *against* the slope, so the step subtracts. Each layer is
        rebuilt through
        :meth:`~oop_ml.core.network.layer.DenseLayer.with_parameters` rather than
        mutated, which is what keeps every neuron's weights frozen and every
        layer's cached matrix honest.
        """
        return LayerStack(
            [
                layer.stepped_by(gradient, learning_rate)
                for layer, gradient in zip(self, backward, strict=True)
            ]
        )

    def __repr__(self) -> str:
        # The arrangements, matching every layer's own repr. A stack that reads
        # pictures should not describe itself in a vocabulary none of its
        # layers uses.
        return (
            f"LayerStack(n_layers={len(self._layers)!r}, "
            f"reads={self._shape.reads!r}, "
            f"answers={self._shape.answers!r})"
        )
