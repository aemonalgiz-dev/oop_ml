"""Every model the reference backend exports is provided or declined by every other.

This is the test that stops a backend forgetting a model. The numpy backend is
the reference; its exports define what "every model" means. For each other
backend, each of those names must be either exported or listed in that
backend's ``NOT_PROVIDED`` with a reason, and never both, since a model that is
present and declined at once is a contradiction somebody should resolve.
"""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

import pytest

from oop_ml import numpy as reference
from oop_ml.core.base.estimator import Fittable

from .harness import BACKENDS, declined


def _is_model(name: str) -> bool:
    """Whether a reference export is a concrete model rather than vocabulary.

    Two kinds of export are not models and are not held to account. Value
    objects such as responses and gradients belong to the numpy implementation.
    Abstract bases are the numpy backend's own frame, and a backend that wraps
    an engine has no reason to reproduce another backend's frame. The first run
    of this test flagged exactly three of them as "forgotten", which is the
    harness doing its job and the reason this predicate says *concrete*.

    The three are ``LinearFeatureRegressor``, ``FeatureScaler`` and
    ``LinearClassifier``, and they are named because the scikit backend does
    not treat them alike. It copies the first two and inherits the third
    across the backend boundary, so ``LogisticRegression`` there runs a numpy
    module's ``fit`` and ``predict``. Settling that is a decision about where
    a shared frame lives rather than about this predicate, which excludes all
    three either way.
    """
    candidate = getattr(reference, name)
    return (
        isinstance(candidate, type)
        and issubclass(candidate, Fittable)
        and not inspect.isabstract(candidate)
    )


REFERENCE_MODELS: tuple[str, ...] = tuple(
    name for name in reference.__all__ if _is_model(name)
)


def test_the_reference_declines_nothing() -> None:
    assert declined(reference) == {}


def test_the_reference_exports_models() -> None:
    """A guard on the guard. If this list were empty the tests below would
    pass vacuously and prove nothing."""
    assert len(REFERENCE_MODELS) >= 30


@pytest.mark.parametrize("backend_name", [one for one in BACKENDS if one != "numpy"])
def test_every_reference_model_is_provided_or_declined(backend_name: str) -> None:
    backend: ModuleType = importlib.import_module(f"oop_ml.{backend_name}")
    absent = declined(backend)

    forgotten = [
        name
        for name in REFERENCE_MODELS
        if getattr(backend, name, None) is None and name not in absent
    ]
    contradicted = [
        name
        for name in REFERENCE_MODELS
        if getattr(backend, name, None) is not None and name in absent
    ]

    assert not forgotten, f"{backend_name} neither provides nor declines {forgotten}"
    assert not contradicted, f"{backend_name} both provides and declines {contradicted}"


@pytest.mark.parametrize("backend_name", [one for one in BACKENDS if one != "numpy"])
def test_a_declined_model_names_a_reason(backend_name: str) -> None:
    backend: ModuleType = importlib.import_module(f"oop_ml.{backend_name}")
    for name, reason in declined(backend).items():
        assert name in REFERENCE_MODELS, (
            f"{backend_name} declines {name}, which is not a reference model"
        )
        assert reason.strip(), f"{backend_name} declines {name} with no reason"
