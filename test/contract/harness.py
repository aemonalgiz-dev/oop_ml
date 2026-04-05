"""The helpers that hold every backend to one contract.

"Same encapsulations" is the promise the three backends make, and a promise is
only worth what checks it. This package runs one spec against every backend,
parametrized, so a model that exists in two backends is tested identically in
both and a divergence is a red test rather than a surprise in production.

A backend may decline a model, but it may not forget one. Each backend package
declares ``NOT_PROVIDED``, a mapping from model name to the reason it is absent,
and :func:`provided` skips with that reason. A model that is neither exported
nor declined fails :mod:`test_every_model_is_accounted_for`, which is what
turns "the backends are the same" from an intention into a thing CI enforces.

The numpy backend is the reference. It exports everything and declines nothing,
so its ``NOT_PROVIDED`` is empty and every contract test runs against it.
"""

from __future__ import annotations

from types import ModuleType

import pytest

#: Every backend, by the name of its package under ``oop_ml``. pytorch joins
#: this list when it exists; until then listing it would be a lie the harness
#: cannot tell from a missing install.
BACKENDS: tuple[str, ...] = ("numpy", "scikit")


def declined(backend: ModuleType) -> dict[str, str]:
    """What this backend says it does not provide, and why. Empty for numpy."""
    return getattr(backend, "NOT_PROVIDED", {})


def provided(backend: ModuleType, model_name: str) -> type:
    """The named model from this backend, or a skip carrying the declared reason.

    Skipping is the honest outcome for a declined model, and it is only honest
    because :mod:`test_every_model_is_accounted_for` separately refuses any
    model that is missing without a declaration. Together the two mean a skip
    here always has a reason a person wrote down.
    """
    model_type = getattr(backend, model_name, None)
    if model_type is not None:
        return model_type
    reason = declined(backend).get(model_name)
    if reason is None:
        pytest.fail(
            f"{backend.__name__} neither provides nor declines {model_name}; "
            "add it to the backend or to its NOT_PROVIDED with a reason"
        )
    pytest.skip(f"{backend.__name__} declines {model_name}: {reason}")
