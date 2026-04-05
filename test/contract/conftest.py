"""Fixtures for the contract suite. The helpers live in :mod:`harness`."""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

from .harness import BACKENDS


@pytest.fixture(params=BACKENDS, ids=BACKENDS)
def backend(request: pytest.FixtureRequest) -> ModuleType:
    """One backend package per parametrized run."""
    return importlib.import_module(f"oop_ml.{request.param}")
