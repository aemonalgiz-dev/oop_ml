"""The chores every translation shares: arrays in, tensors through, arrays out.

One dtype, and it is the numpy backend's
-----------------------------------------
Everything this backend builds is ``float64``. torch's own default is
``float32``, and the reason to override it is the whole point of having a
second backend behind one API: the agreement tests between the two are to
the last bits, and a ``float32`` forward pass parts company with a
``float64`` one around the seventh significant figure, which is enough to
turn every one of those tests into a tolerance argument. A caller who wants
the speed of single precision has the torch module itself and ``.float()``.

Why ``torch.tensor`` and never ``torch.from_numpy``
----------------------------------------------------
Every learned array in this library is frozen, which is the invariant
``test/core/test_encapsulation.py`` holds. ``torch.from_numpy`` builds a
tensor that shares the array's memory and warns, every time, that the array
is not writable. ``torch.tensor`` copies, so the tensor is the backend's own
and the array stays what its owner made it. The copy is the cost of not
sharing a buffer across the boundary, and it is the same choice
:class:`~oop_ml.core.data.column.Column` makes on the way in.
"""

from __future__ import annotations

import numpy as np
import torch

from oop_ml.core.types import FloatArray

#: What every parameter, buffer and activation here is built in.
TENSOR_DTYPE = torch.float64

#: What a block of positions is handed to an embedding as.
POSITION_DTYPE = torch.int64


def as_tensor(block: FloatArray) -> torch.Tensor:
    """A private ``float64`` tensor holding a copy of ``block``.

    Parameters
    ----------
    block:
        Any float array, frozen or not. It is read and never aliased.

    Returns
    -------
    torch.Tensor
        The same values, on the CPU, carrying no gradient history.
    """
    return torch.tensor(np.asarray(block, dtype=np.float64), dtype=TENSOR_DTYPE)


def as_positions(block: FloatArray) -> torch.Tensor:
    """A private ``int64`` tensor of positions, for an embedding to index with.

    The caller has already established that every entry is a whole number in
    range, which is the embedding translation's job and not this function's.
    Casting first and checking afterwards would destroy the evidence, which is
    the failure the numpy embedding's own ``_read_block`` names.
    """
    return torch.tensor(np.asarray(block, dtype=np.int64), dtype=POSITION_DTYPE)


def as_array(tensor: torch.Tensor) -> FloatArray:
    """A private ``float64`` array holding a copy of ``tensor``.

    Detached from any gradient history and copied off the tensor's storage,
    so the array is the library's own and can be frozen by whatever value
    object receives it without reaching back into torch.
    """
    return np.array(tensor.detach().cpu().numpy(), dtype=np.float64, copy=True)


def copy_into(parameter: torch.Tensor, block: FloatArray) -> None:
    """Write ``block`` into ``parameter`` in place, outside the gradient tape.

    How every translation hands a layer's learned numbers to the module that
    will carry them. The shapes must already agree, which the translation
    guarantees by having built the module from the same layer.
    """
    with torch.no_grad():
        parameter.copy_(as_tensor(block))


def required(parameter: torch.Tensor | None, role: str) -> torch.Tensor:
    """A parameter or buffer torch types as optional and this backend never omits.

    ``nn.Linear.bias`` is ``None`` when built with ``bias=False``, and a batch
    normalisation's running buffers are ``None`` when built without tracking,
    so torch annotates all of them as optional. Every module here is built
    with both, so an absence is a defect in the backend's own construction
    rather than anything a caller handed in, and it is reported as one.

    Raises
    ------
    RuntimeError
        If the parameter is absent.
    """
    if parameter is None:
        raise RuntimeError(f"{role} is absent on a module this backend built with it")
    return parameter


def slope_of(parameter: torch.Tensor, role: str) -> FloatArray:
    """The gradient a backward pass left on ``parameter``, as an array.

    Raises
    ------
    RuntimeError
        If no backward pass has run since the gradients were last cleared.
        A bare ``RuntimeError`` rather than one of this library's own, because
        reaching this is a defect in the backend's own sequencing and not
        anything a caller handed in; the stack only reads slopes after it has
        called ``backward``.
    """
    if parameter.grad is None:
        raise RuntimeError(
            f"no gradient on {role}; the backward pass has not run, or the "
            "gradients were cleared before they were read"
        )
    return as_array(parameter.grad)
