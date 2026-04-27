"""Saving and loading the from-scratch models.

The format lives in :mod:`oop_ml.core.persistence`; what lives here is the
closed registry of which models this backend can restore, registered on import
so that reading a document written by this backend resolves its class names
inside it.
"""

from __future__ import annotations

from oop_ml.core.persistence.document import ModelDocument
from oop_ml.core.persistence.store import (
    build_model,
    load_model,
    model_document,
    save_model,
)
from oop_ml.numpy.persistence.registry import PERSISTABLE_TYPES

__all__ = [
    "PERSISTABLE_TYPES",
    "ModelDocument",
    "build_model",
    "load_model",
    "model_document",
    "save_model",
]
