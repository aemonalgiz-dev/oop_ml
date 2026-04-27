"""Which classes a document may name, per backend.

The format is shared and the models are not, so the machinery lives in
``core`` and each backend hands it a mapping of the classes it can restore.
Core never imports a backend, and a backend never learns about another one.

Why a document names its backend
---------------------------------
A bare class name was unambiguous while one backend existed. It is not now:
``RidgeRegression`` names two classes, and the contract is that loading a
document written by one gives back that one. So the backend travels with the
document and the name is resolved inside it.

Why a backend declines some of its own models
----------------------------------------------
A document holds what a model learned, and eleven of the scikit backend's
wrappers hold exactly that and nothing else. The other fifteen keep the fitted
engine and answer through it, so restoring their learned state would produce a
model that cannot predict. Rather than hand back something that looks fitted
and raises on use, those decline by name with a reason, the same way an absent
model does. A backend can decline; it cannot forget.
"""

from __future__ import annotations

from pydantic import BaseModel

from oop_ml.core.exceptions import InvalidDocumentError

#: Backend name to the classes it can restore, filled by each backend at import.
_REGISTRIES: dict[str, dict[str, type[BaseModel]]] = {}

#: Backend name to the models it deliberately cannot save, and why.
_DECLINED: dict[str, dict[str, str]] = {}

NUMPY_BACKEND = "numpy"
"""The backend a document without a stated one came from.

Documents written before the format named a backend are all from the
from-scratch models, because it was the only backend that existed when they
were written. Reading them as such is a fact about those files rather than a
guess.
"""


def register_backend(
    name: str,
    persistable: dict[str, type[BaseModel]],
    declined: dict[str, str] | None = None,
) -> None:
    """Declare which of a backend's models can be saved, and which cannot.

    Parameters
    ----------
    name:
        What documents written by this backend will record.
    persistable:
        Class name to class, for every model this backend can restore.
    declined:
        Class name to the reason it cannot be, for models this backend has but
        cannot bring back to working order.
    """
    _REGISTRIES[name] = dict(persistable)
    _DECLINED[name] = dict(declined or {})


def registered_backends() -> tuple[str, ...]:
    """Every backend that has been imported and has registered."""
    return tuple(sorted(_REGISTRIES))


def persistable_types(backend: str) -> dict[str, type[BaseModel]]:
    """What that backend can restore.

    Raises
    ------
    InvalidDocumentError
        If the backend has not registered, which for an optional backend means
        it was never imported rather than that it does not exist.
    """
    if backend not in _REGISTRIES:
        known = ", ".join(registered_backends()) or "none"
        raise InvalidDocumentError(
            f"this document was written by the {backend!r} backend, which is "
            f"not loaded; import oop_ml.{backend} before reading it. Loaded "
            f"backends: {known}"
        )
    return _REGISTRIES[backend]


def declined_reason(backend: str, name: str) -> str | None:
    """Why that backend cannot save that model, if it has said so."""
    return _DECLINED.get(backend, {}).get(name)


def backend_of(model_type: type) -> str:
    """Which backend a class belongs to, read from where it is defined.

    Read from the module rather than asked of the class, so a backend cannot
    forget to answer and no model carries a field naming its own package.
    """
    parts = model_type.__module__.split(".")
    if len(parts) >= 2 and parts[0] == "oop_ml" and parts[1] in _REGISTRIES:
        return parts[1]
    return NUMPY_BACKEND
