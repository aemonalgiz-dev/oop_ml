"""The guard that refuses an engine this backend cannot trust.

The absent case is ordinary. The too-old case is the one worth having, because
below scikit-learn 1.9 nothing fails: ``LinearRegression.tol`` is accepted and
ignored on dense data, so the collinearity threshold reverts to the engine's
default of 1e-6 and independent columns on different scales start being
refused as collinear. A version floor that only lives in packaging metadata
would not stop that, since metadata is honoured at install time and says
nothing to somebody who upgraded around it.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError

import pytest

from oop_ml.scikit import engine_version
from oop_ml.scikit.engine_version import (
    MINIMUM_ENGINE_VERSION,
    _release_of,
    check_engine_available,
)


class TestTheReleaseReading:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1.9.0", (1, 9, 0)),
            ("1.10.0", (1, 10, 0)),
            ("2.0", (2, 0)),
            ("1.9", (1, 9)),
        ],
    )
    def test_a_plain_release_reads_as_its_numbers(
        self, raw: str, expected: tuple[int, ...]
    ) -> None:
        assert _release_of(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1.10.0rc1", (1, 10, 0)),
            ("1.9.0.dev0", (1, 9, 0)),
            ("2.0b1", (2, 0)),
        ],
    )
    def test_a_prerelease_compares_by_its_release(
        self, raw: str, expected: tuple[int, ...]
    ) -> None:
        # A release candidate for a version above the floor must pass, so the
        # trailing letters are dropped rather than raising.
        assert _release_of(raw) == expected

    def test_ten_sorts_above_nine_rather_than_below_it(self) -> None:
        # The reason this is a tuple of numbers and not a string comparison.
        assert _release_of("1.10.0") > _release_of("1.9.0")

    def test_something_with_no_numbers_at_all_reads_as_nothing(self) -> None:
        # An empty tuple sorts below the floor, so an unreadable version is
        # refused rather than waved through.
        assert _release_of("unknown") == ()
        assert _release_of("unknown") < MINIMUM_ENGINE_VERSION


class TestTheGuard:
    def test_the_installed_engine_passes(self) -> None:
        check_engine_available()

    def test_an_absent_engine_names_the_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing(name: str) -> str:
            raise PackageNotFoundError(name)

        monkeypatch.setattr(engine_version, "version", missing)

        with pytest.raises(ImportError) as raised:
            check_engine_available()

        message = str(raised.value)
        assert "not installed" in message
        assert "oop_ml[scikit]" in message

    @pytest.mark.parametrize("found", ["1.8.2", "1.7.0", "1.4.0", "0.24.2"])
    def test_an_old_engine_is_refused_by_name(
        self, monkeypatch: pytest.MonkeyPatch, found: str
    ) -> None:
        monkeypatch.setattr(engine_version, "version", lambda name: found)

        with pytest.raises(ImportError) as raised:
            check_engine_available()

        message = str(raised.value)
        assert found in message
        assert "collinear" in message

    @pytest.mark.parametrize("found", ["1.9.0", "1.9.3", "1.10.0", "2.0.0"])
    def test_the_floor_and_anything_above_it_passes(
        self, monkeypatch: pytest.MonkeyPatch, found: str
    ) -> None:
        monkeypatch.setattr(engine_version, "version", lambda name: found)

        check_engine_available()

    def test_the_floor_is_the_version_the_pin_needs(self) -> None:
        # MultipleLinearRegression pins LinearRegression.tol, which is only
        # interpreted as lstsq's cond on dense data from 1.9. Lowering this
        # without changing that wrapper reintroduces the bug it exists to fix.
        assert MINIMUM_ENGINE_VERSION == (1, 9)
