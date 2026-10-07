"""Whether the engine this backend wraps is present, and new enough to trust.

The shape of this module is :mod:`oop_ml.scikit.engine_version`'s, and the
reason it exists is the first of that module's two: ``oop_ml`` depends on
numpy, scipy and pydantic and nothing else, so ``import oop_ml`` and
``import oop_ml.numpy`` work with torch absent. This backend is the one part
of the package that does not, and the bare
``ModuleNotFoundError: No module named 'torch'`` a reader would otherwise
meet says nothing about the extra that fixes it.

The floor is a different kind of floor
---------------------------------------
The scikit floor guards a *quiet* failure. Below 1.9 that engine accepts a
threshold and ignores it, and a wrapper keeps working while answering
differently, which is why that check lives at import rather than in packaging
metadata nothing enforces.

Nothing here is quiet. Two of the five loss translations do not exist below
1.10: cross entropy only accepts probability targets, which is what a one-hot
block is, from that release, and the Huber loss arrived in 1.9. Every module
this backend builds is also constructed with the ``dtype`` factory keyword,
which the module constructors gained in 1.9. Below the floor each of those is
a ``TypeError`` from the engine naming a keyword, which is loud and is also
unhelpful, since the keyword it names is not the thing a reader has to change.
So the floor is here for the sentence rather than for the safety, and it is
the version the translations are known to need rather than the one they were
measured on, which is 2.14 on a CPU.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

MINIMUM_ENGINE_VERSION = (1, 10)
"""The oldest torch whose behaviour this backend is written against.

Set by ``torch.nn.functional.cross_entropy`` accepting probability targets,
which it does from 1.10. Raise this floor when a translation starts depending
on something newer, and say which one in the module docstring above.
"""

ENGINE_PACKAGE = "torch"
INSTALL_HINT = "pip install 'oop_ml[pytorch]'"


def _readable(numbers: tuple[int, ...]) -> str:
    return ".".join(str(number) for number in numbers)


def _release_of(raw: str) -> tuple[int, ...]:
    """The leading numeric parts of a version string.

    Stops at the first part that is not a whole number, so ``2.14.1+cpu``
    compares as ``(2, 14, 1)`` and a release candidate compares by its release
    alone rather than raising. The build tag torch appends after a plus is the
    usual thing to meet here, since the CPU wheels all carry one.
    """
    parts: list[int] = []
    for part in raw.split("."):
        digits = ""
        for character in part:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_engine_available() -> None:
    """Refuse to import this backend without a torch it can trust.

    Raises
    ------
    ImportError
        If torch is absent, naming the extra that installs it; or if it is
        present but older than :data:`MINIMUM_ENGINE_VERSION`, naming the
        version found and which translations stop existing below the floor.
        An ImportError rather than an :class:`~oop_ml.core.exceptions.MLLibError`,
        because this is a failure to import a package rather than a failure of
        anything this library models, and a caller guarding an optional backend
        writes ``except ImportError``.
    """
    try:
        found = version(ENGINE_PACKAGE)
    except PackageNotFoundError:
        raise ImportError(
            f"oop_ml.pytorch needs {ENGINE_PACKAGE}, which is not installed. "
            f"It is an optional extra, since oop_ml.numpy needs nothing "
            f"beyond numpy, scipy and pydantic. Install it with "
            f"{INSTALL_HINT}"
        ) from None

    if _release_of(found) < MINIMUM_ENGINE_VERSION:
        raise ImportError(
            f"oop_ml.pytorch needs {ENGINE_PACKAGE} "
            f"{_readable(MINIMUM_ENGINE_VERSION)} or newer and found {found}. "
            f"Below that, cross entropy does not accept the one-hot targets "
            f"this library's SoftmaxCrossEntropy is defined over, and the Huber "
            f"loss does not exist, so two of the five loss translations cannot "
            f"be built. Upgrade with {INSTALL_HINT}"
        )
