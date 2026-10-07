"""A stack of this library's layers with torch doing the arithmetic.

The one thing torch does not do, and this does
-----------------------------------------------
``torch.nn.Sequential(nn.Linear(64, 32), nn.Linear(16, 8))`` constructs without
complaint. The second layer cannot read what the first answers, and torch will
say so at the first forward pass, from inside a matrix multiply, after the
data has been loaded and the training loop has begun. A caller who wrote that
line finds out later, and finds out in a traceback naming shapes rather than
layers. torch's own answer to the problem is the lazy modules, which defer the
question until data arrives, which is the opposite of answering it.

This library's :class:`~oop_ml.core.network.stack.LayerStack` refuses the same
chain at construction, in integer comparisons, naming the seam, before a row
of data exists. The whole of this module is that refusal placed in front of
torch. A stack here checks every join exactly as the numpy stack does, and
only once every join holds does it build a module. So a network that cannot
work is refused before any tensor has been allocated, and the refusal says
``layer 0 answers with (8, 26, 26) and layer 1 reads (5408,)`` rather than
``mat1 and mat2 shapes cannot be multiplied``.

Which backend owns what
------------------------
The numpy layers are the description and the source of truth. The torch
modules are a translation of them, rebuilt whenever the description changes.
That division is what lets this class keep the numpy stack's interface
exactly, and it decides three things.

``respond_to`` and ``backward_pass`` are torch's. The forward pass runs
through the translated modules and the gradient comes from autograd, which is
what a torch backend is for. The gradient is handed back arranged the way the
numpy layer's own ``correction_for`` would arrange it, so the two backends'
backward passes compare block for block, and ``test/pytorch/test_stack.py``
makes that comparison on every translated layer. It is the strongest check
either backend has: a hand-written backward pass agreeing with automatic
differentiation to 1e-10 is not a thing that happens by accident.

``stepped_by`` is the numpy layers' own. A torch gradient arranged as a numpy
gradient is one the numpy layer's ``stepped_by`` already knows how to apply,
so the step is the same subtraction the numpy stack performs, and the new
stack is built from the stepped descriptions, which translates them again.
That is a re-translation per step, measured below, and it buys the thing that
makes the two backends interchangeable rather than merely alike: a stack
trained here can be handed to the numpy ``LayerStack`` unchanged, because its
layers are numpy layers that were never anything else.

``module`` is torch's, for a caller who wants torch's own training loop. It is
the same modules the stack runs, so what that caller trains is what the stack
answers with, until the stack is stepped and rebuilds them.

What is pure and what is not
-----------------------------
The numpy stack's ``backward_pass`` changes nothing. torch's batch
normalisation advances its running statistics on every training forward
pass, so the translation of that one layer rewrites the module's buffers from
the description after each pass, and a test asserts that two consecutive
backward passes read the same running figures. The dropout module draws from
torch's global generator rather than from a generator the layer holds, which
is torch's design and is recorded in the dropout translation.

What the re-translation costs
------------------------------
Measured on a stack of one convolution, one pooling, one flattening and one
dense layer over sixteen ``(1, 8, 8)`` pictures, median of seven samples of
thirty calls after a warm-up, each backend in its own process::

                   numpy      torch
    forward      0.132 ms   0.333 ms
    backward     0.460 ms   0.809 ms
    stepped_by   0.036 ms   0.464 ms   of which 0.358 ms is the re-translation
    whole round  0.596 ms   1.546 ms

So a training step here costs 2.6x the numpy step at this size, and most of
the difference is building modules rather than running them. The honest
statement is that this backend is not where to go for speed on a network that
small; it is where to go to let autograd check a backward pass, and to hand a
``torch.nn.Module`` to anything that wants one.

The two processes are not a nicety. Timed in one process, with torch calls
interleaved, the numpy backward pass read 1.21 ms, which is slower than the
numpy backward-and-step round of 0.72 ms measured moments later, and a part
cannot cost more than the whole it belongs to. torch's thread pool spins for
a while after a torch call and takes the cores from whatever numpy call comes
next, so a numpy timing taken after torch work measures torch's idling.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import nn

from oop_ml.core.exceptions import EmptyValuesError, ShapeMismatchError
from oop_ml.core.network.blocks import as_block, check_finite
from oop_ml.core.network.gradient import BackwardPass, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.loss import Loss
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.network.stack import StackResponse
from oop_ml.core.types import FloatArray
from oop_ml.pytorch.layers import Translation, translation_of
from oop_ml.pytorch.losses import torch_loss_for
from oop_ml.pytorch.plumbing import as_array, as_tensor


class LayerStack:
    """Layers in order, refused at construction unless every join holds, run by torch.

    Parameters
    ----------
    layers:
        At least one of this library's layers, each reading exactly what the
        one beneath it answers.

    Raises
    ------
    EmptyValuesError
        If no layers are supplied.
    ShapeMismatchError
        If any layer does not read what the layer beneath it answers. Raised
        before any module is built, naming the join and both arrangements, and
        saying when the two hold the same number of values that it is the
        arrangement disagreeing and not the width.
    InvalidValuesError
        If a layer has no translation, by way of
        :func:`~oop_ml.pytorch.layers.translation_of`, which names the layer
        and the reason. Also raised before any module is built, since the
        joins are checked first and a stack that cannot be built is refused at
        the cheapest point that can tell.
    """

    __slots__ = ("_layers", "_module", "_shape", "_translations")

    def __init__(self, layers: Sequence[Layer]) -> None:
        if not layers:
            raise EmptyValuesError("a stack needs at least one layer")

        ordered = tuple(layers)
        for position, (beneath, above) in enumerate(
            zip(ordered, ordered[1:], strict=False)
        ):
            if not above.shape.follows(beneath.shape):
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

        # Only now, with every join known to hold, is a torch module built.
        # The order is the point: a network that cannot work is refused by
        # integer comparisons before a single tensor exists.
        translations = tuple(translation_of(layer) for layer in ordered)

        self._layers = ordered
        self._translations = translations
        self._module = nn.Sequential(*(one.module for one in translations))
        self._shape = LayerShape(
            n_inputs=ordered[0].shape.reads,
            n_outputs=ordered[-1].shape.answers,
        )

    @property
    def shape(self) -> LayerShape:
        """What the network reads and what it answers with, end to end."""
        return self._shape

    @property
    def module(self) -> nn.Sequential:
        """The translated layers as one torch module, for torch's own loops.

        The same modules this stack runs, so training them with an optimiser
        changes what the stack answers. A caller doing that has left the
        library's immutable step behind and taken torch's in-place one, which
        is a legitimate thing to do and is the reason this property exists.
        """
        return self._module

    def __len__(self) -> int:
        """How many layers the stack holds."""
        return len(self._layers)

    def __getitem__(self, position: int) -> Layer:
        """The layer at ``position``, counting from the input end.

        The numpy description, exactly as a numpy stack would hand it out,
        and not the torch module. A caller who wants the module has
        :attr:`module`.
        """
        return self._layers[position]

    def __iter__(self) -> Iterator[Layer]:
        """The layers, bottom to top."""
        return iter(self._layers)

    def _checked_block(self, inputs: object) -> FloatArray:
        """The block as a float array, refused unless this stack can read it.

        The same four refusals :meth:`~oop_ml.core.network.layer.Layer.respond_to`
        makes, in the same words, so that handing a bad block to either backend
        reads the same.
        """
        block = as_block(inputs, "inputs")
        check_finite(block, "inputs")

        if block.ndim < 2:
            raise ShapeMismatchError(
                "a layer reads a block whose first axis is the rows, got "
                f"{block.ndim} dimensions"
            )
        if block.shape[0] == 0:
            raise EmptyValuesError("a layer needs at least one row to respond to")
        if block.shape[1:] != self._shape.reads:
            raise ShapeMismatchError(
                f"this layer reads {self._shape.reads}, got a block arranged "
                f"{block.shape[1:]}"
            )

        return block

    def respond_to(
        self, inputs: FloatArray, purpose: PassPurpose = PassPurpose.PREDICTING
    ) -> StackResponse:
        """Push a block of rows all the way up, through torch, without a gradient.

        Parameters
        ----------
        inputs:
            ``(n_rows, *shape.reads)``.
        purpose:
            Why the pass is happening, handed to every module as its train or
            eval mode. The default answers deterministically, as it does in
            numpy and for the same reason.

        Returns
        -------
        StackResponse
            One :class:`~oop_ml.core.network.layer.LayerResponse` per layer,
            as arrays, whose ``outputs`` is the network's answer.

        Raises
        ------
        ShapeMismatchError
            If the block does not have rows on its first axis, or is not
            arranged the way the first layer reads.
        EmptyValuesError
            If the block holds no rows.
        InvalidValuesError
            If the block cannot be read as a float array, holds a non-finite
            entry, or, for a stack beginning with an embedding, holds a
            position that is not a whole number in range.
        """
        block = self._checked_block(inputs)
        self._module.train(purpose == PassPurpose.TRAINING)

        responses: list[LayerResponse] = []
        with torch.no_grad():
            current = self._translations[0].read(block)
            for translation in self._translations:
                read = current
                response = translation.respond(current)
                responses.append(
                    LayerResponse.already_checked(
                        inputs=as_array(read),
                        scores=as_array(response.scores),
                        outputs=as_array(response.outputs),
                    )
                )
                current = response.outputs

        if purpose == PassPurpose.TRAINING:
            self._restore_state()

        return StackResponse(responses)

    def backward_pass(
        self, inputs: FloatArray, targets: FloatArray, loss: Loss
    ) -> BackwardPass:
        """Run the network forward, score it, and let autograd walk the blame down.

        Parameters
        ----------
        inputs:
            ``(n_rows, *shape.reads)``.
        targets:
            ``(n_rows, *shape.answers)``, the truth in the shape the last
            layer answers with.
        loss:
            What to score by. Translated to the criterion that measures the
            same number, so the loss reported here is the loss the numpy stack
            would report.

        Returns
        -------
        BackwardPass
            The loss, and one gradient per layer bottom to top, each arranged
            as the numpy layer arranges its own, so that this stack's
            :meth:`stepped_by` and the numpy stack's agree on what to do with
            it.

        Raises
        ------
        ShapeMismatchError
            If the blocks do not match this network's two ends.
        EmptyValuesError
            If there are no rows.
        InvalidValuesError
            If either block carries a non-finite entry, or the loss has no
            criterion.

        Notes
        -----
        Stated as a training pass, as the numpy stack states it, so a dropout
        draws and a batch normalisation standardises by the batch. What a
        training pass does to torch's own buffers is undone afterwards, which
        is what keeps this method as free of side effects as the one it
        mirrors; see the module docstring.
        """
        block = self._checked_block(inputs)
        target_block = as_block(targets, "targets")
        check_finite(target_block, "targets")

        expected = (block.shape[0], *self._shape.answers)
        if target_block.shape != expected:
            raise ShapeMismatchError(
                f"outputs {expected} and targets {target_block.shape} describe "
                "the same rows and must be the same shape"
            )

        criterion = torch_loss_for(loss)

        self._module.train(True)
        self._module.zero_grad(set_to_none=True)

        current = self._translations[0].read(block)
        for translation in self._translations:
            current = translation.respond(current).outputs
        value = criterion(current, as_tensor(target_block))
        value.backward()

        gradients: list[LayerGradient | None] = [
            translation.read_gradient() for translation in self._translations
        ]
        self._restore_state()

        return BackwardPass(loss=float(value.item()), gradients=gradients)

    def stepped_by(self, backward: BackwardPass, learning_rate: float) -> LayerStack:
        """A new stack with every parameter nudged against its slope.

        The numpy layers' own step, applied to a torch gradient, which is
        possible because :meth:`backward_pass` arranges each gradient the way
        the layer that owns it expects. The new stack is built from the
        stepped descriptions and translates them afresh, so its modules carry
        the new numbers and its layers remain numpy layers a numpy stack would
        accept unchanged.

        Parameters
        ----------
        backward:
            The gradients to move against, one per layer in layer order.
        learning_rate:
            How far to move.

        Returns
        -------
        LayerStack
            A new stack, same shapes and same bends, moved parameters.

        Raises
        ------
        ShapeMismatchError
            If the gradients do not describe this stack's layers.
        """
        return LayerStack(
            [
                layer.stepped_by(gradient, learning_rate)
                for layer, gradient in zip(self._layers, backward, strict=True)
            ]
        )

    def _restore_state(self) -> None:
        for translation in self._translations:
            translation.restore_state()

    @property
    def translations(self) -> tuple[Translation, ...]:
        """Each layer's translation, bottom to top, for a reader of the seam."""
        return self._translations

    def __repr__(self) -> str:
        return (
            f"LayerStack(n_layers={len(self._layers)!r}, "
            f"reads={self._shape.reads!r}, "
            f"answers={self._shape.answers!r})"
        )
