"""Whether the engine this backend wraps is present, and new enough to trust.

Two failures this module turns into sentences a reader can act on.

The engine is optional
----------------------
``oop_ml`` itself depends on numpy, scipy and pydantic and nothing else, so
``import oop_ml`` and ``import oop_ml.numpy`` work with scikit-learn absent.
This backend is the one part that does not, and the bare
``ModuleNotFoundError: No module named 'sklearn'`` a reader would otherwise
meet says nothing about the extra that fixes it.

The engine is version sensitive, and quietly so
------------------------------------------------
:class:`~oop_ml.scikit.regression.MultipleLinearRegression` pins
``LinearRegression.tol`` at machine epsilon so that a singular value below it
is dropped from the rank, which is how that wrapper detects collinearity. The
field was added to the engine in 1.7 and, on dense data, only became the
``cond`` argument of ``scipy.linalg.lstsq`` in 1.9. So on 1.7 and 1.8 the
wrapper constructs happily, the threshold is accepted and ignored, and the
engine falls back to its own default of 1e-6, which is ten orders looser than
exact singularity. Two independent columns on different scales are then
refused as collinear where the numpy backend fits them, and nothing anywhere
says why.

That is the worst shape a version floor can have: not a crash, but a wrapper
that keeps working and quietly answers differently. Hence a check at import
rather than a line in the packaging metadata that nothing enforces.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

MINIMUM_ENGINE_VERSION = (1, 9)
"""The oldest scikit-learn whose behaviour this backend is written against.

Set by ``LinearRegression.tol``, which is interpreted as ``cond`` on dense
data only from 1.9. Raise this floor when a wrapper starts depending on
something newer; do not lower it without checking what the collinearity
refusal does at that version.
"""

ENGINE_PACKAGE = "scikit-learn"
INSTALL_HINT = "pip install 'oop_ml[scikit]'"


def _readable(numbers: tuple[int, ...]) -> str:
    return ".".join(str(number) for number in numbers)


def _release_of(raw: str) -> tuple[int, ...]:
    """The leading numeric parts of a version string.

    Stops at the first part that is not a whole number, so a release candidate
    or a development build compares by its release alone rather than raising.
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
    """Refuse to import this backend without a scikit-learn it can trust.

    Raises
    ------
    ImportError
        If scikit-learn is absent, naming the extra that installs it; or if it
        is present but older than :data:`MINIMUM_ENGINE_VERSION`, naming the
        version found and what goes wrong at it. An ImportError rather than an
        :class:`~oop_ml.core.exceptions.MLLibError`, because this is a failure
        to import a package rather than a failure of anything this library
        models, and a caller guarding an optional backend writes
        ``except ImportError``.
    """
    try:
        found = version(ENGINE_PACKAGE)
    except PackageNotFoundError:
        raise ImportError(
            f"oop_ml.scikit needs {ENGINE_PACKAGE}, which is not installed. "
            f"It is an optional extra, since oop_ml.numpy needs nothing "
            f"beyond numpy, scipy and pydantic. Install it with "
            f"{INSTALL_HINT}"
        ) from None

    if _release_of(found) < MINIMUM_ENGINE_VERSION:
        raise ImportError(
            f"oop_ml.scikit needs {ENGINE_PACKAGE} "
            f"{_readable(MINIMUM_ENGINE_VERSION)} or newer and found {found}. "
            f"Below that, LinearRegression.tol is accepted and ignored on "
            f"dense data, so MultipleLinearRegression's collinearity "
            f"threshold silently reverts to the engine's default of 1e-6 and "
            f"independent columns on different scales are refused as "
            f"collinear. Upgrade with {INSTALL_HINT}"
        )
