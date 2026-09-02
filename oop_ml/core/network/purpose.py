"""Why a forward pass is happening, which some layers need to know.

Most layers do not care
-----------------------
A dense layer, a convolution, a pooling window and a flattening all do exactly
the same thing whether they are learning or answering. They read a block and
transform it, and nothing about the reason changes the arithmetic.

Two do care, and they are unavoidable
--------------------------------------
:class:`~oop_ml.core.network.dropout.Dropout` deliberately damages its own
answers while learning, so that the layers above cannot come to depend on any
one unit. Doing that while *answering* would make the model's prediction a
matter of luck, so it must stop.

Batch normalisation has the same shape of problem from the other direction. It
centres and scales each column by the statistics of the batch in front of it,
which is fine while learning, and unusable while answering: a prediction would
depend on which other rows happened to be in the same request, and a single row
on its own has no spread to divide by. So it remembers the statistics it saw
during training and uses those instead.

Why an enum rather than a flag
-------------------------------
``respond_to(block, True)`` at a call site says nothing, and the reader has to
go and look up what the second parameter means. This library's convention is
that a function needing to know *which* of a closed set of things it is looking
at takes an enum, so that a wrong value is a name error in the caller's source
rather than a plausible-looking boolean.

The default is :attr:`PassPurpose.PREDICTING`, and that direction is chosen
deliberately. Forgetting to say "training" costs a slightly slower descent,
since dropout stops perturbing and batch norm uses stale statistics; forgetting
to say "predicting" would make every answer the model gives random. Of the two
mistakes, the default should protect against the worse one.

A caller who trains through
:meth:`~oop_ml.core.network.stack.LayerStack.backward_pass` never has to think
about it, since that method knows why it is running and says so.
"""

from __future__ import annotations

from enum import StrEnum


class PassPurpose(StrEnum):
    """Why a block is being pushed through a network.

    Attributes
    ----------
    PREDICTING:
        The answer is wanted. Every layer behaves deterministically, and
        nothing about one row's answer may depend on which other rows it
        travelled with.
    TRAINING:
        The answer is wanted only so that a backward pass can follow. Layers
        may perturb, and may read statistics from the batch.
    """

    PREDICTING = "predicting"
    TRAINING = "training"
