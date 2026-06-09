"""Which package may import which, checked rather than remembered.

The rule is one arrow. ``core`` is shared vocabulary and imports nothing from
a backend; every backend imports ``core``; and no backend imports another.
That last clause is the one worth a test, because it held only by accident
until the shared value objects were moved into ``core``, and nothing about the
code makes it obvious when it stops holding.

``core/natural_language_processing`` is inside ``core`` and so is held to the
same rule by the same scan: a token id has to mean one thing whichever backend
reads it, so nothing there imports a backend. One more arrow runs inside
``core``: the frame does not reach up for a tokenizer, because a tokenizer is
built on the frame rather than being part of it, and a test says so.

Two things break if a backend reaches into another. The dependency becomes a
lie, since ``oop_ml.scikit`` would need the from-scratch implementations
installed to answer a question the engine answers. And a caller who fits a
``Standardizer`` from one backend and one from the other would be handed types
from different packages that happen to have the same name, which is exactly
the confusion the shared vocabulary exists to prevent.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "oop_ml"

BACKENDS = ("numpy", "scikit")

#: The one domain package inside core that is built on the frame rather than
#: being part of it.
TOKENIZATION = "core/natural_language_processing"


def _modules_under(package: str) -> list[pathlib.Path]:
    return sorted((PACKAGE_ROOT / package).rglob("*.py"))


def _imported_packages(path: pathlib.Path) -> set[str]:
    """Every ``oop_ml.*`` module this file imports, by dotted name."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("oop_ml"):
                found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith("oop_ml"))
    return found


def _relative(path: pathlib.Path) -> str:
    return path.relative_to(PACKAGE_ROOT.parent).as_posix()


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_backend_imports_no_other_backend(backend: str) -> None:
    others = [name for name in BACKENDS if name != backend]

    offences: list[str] = []
    for path in _modules_under(backend):
        for imported in _imported_packages(path):
            for other in others:
                if imported.startswith(f"oop_ml.{other}"):
                    offences.append(f"{_relative(path)} imports {imported}")

    assert not offences, (
        f"oop_ml.{backend} reaches into another backend:\n  "
        + "\n  ".join(sorted(offences))
        + "\nWhatever it wants is shared vocabulary and belongs in oop_ml.core."
    )


def test_core_imports_no_backend() -> None:
    """The arrow points one way. Core is what backends are written against,
    and the scan covers the tokenizers under it too."""
    offences: list[str] = []
    for path in _modules_under("core"):
        for imported in _imported_packages(path):
            if any(imported.startswith(f"oop_ml.{name}") for name in BACKENDS):
                offences.append(f"{_relative(path)} imports {imported}")

    assert not offences, (
        "oop_ml.core imports a backend, which inverts the layering:\n  "
        + "\n  ".join(sorted(offences))
    )


def test_the_frame_does_not_reach_up_for_a_tokenizer() -> None:
    """Nothing in core outside the tokenizers imports the tokenizers. A
    tokenizer is built on the frame; the frame does not depend on one."""
    tokenization_modules = set(_modules_under(TOKENIZATION))
    offences: list[str] = []
    for path in _modules_under("core"):
        if path in tokenization_modules:
            continue
        for imported in _imported_packages(path):
            if imported.startswith("oop_ml.core.natural_language_processing"):
                offences.append(f"{_relative(path)} imports {imported}")

    assert not offences, (
        "the frame imports oop_ml.core.natural_language_processing:\n  "
        + "\n  ".join(sorted(offences))
    )


def test_the_tokenizers_do_import_the_frame() -> None:
    """The guard on the guard above, as for the backends: an empty package or a
    broken scan would pass vacuously."""
    reaches_frame = any(
        any(
            imported.startswith("oop_ml.core.")
            and not imported.startswith("oop_ml.core.natural_language_processing")
            for imported in _imported_packages(path)
        )
        for path in _modules_under(TOKENIZATION)
    )

    assert reaches_frame


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_backend_does_import_core(backend: str) -> None:
    """A guard on the guards above, which would pass vacuously on an empty
    package or if the import scan silently stopped finding anything."""
    reaches_core = any(
        any(imported.startswith("oop_ml.core") for imported in _imported_packages(path))
        for path in _modules_under(backend)
    )

    assert reaches_core


def test_the_scan_sees_the_modules_it_thinks_it_does() -> None:
    """The other guard on the guards. If these globs went empty the layering
    tests would all pass while checking nothing at all."""
    assert len(_modules_under("core")) > 40
    assert len(_modules_under("numpy")) > 30
    assert len(_modules_under("scikit")) >= 6
    assert len(_modules_under(TOKENIZATION)) > 10
