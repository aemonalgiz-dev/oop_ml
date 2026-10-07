"""The guard that refuses a torch this backend cannot build its translations on.

The absent case is the ordinary one, and the sentence it produces has to name
the extra rather than the engine, since ``No module named 'torch'`` tells a
reader nothing about ``oop_ml[pytorch]``. The too-old case is loud rather than
quiet, unlike the scikit backend's, so the test for it is about the wording
and not about a wrong answer being prevented.

The version string torch reports carries a build tag after a plus, as in
``2.14.1+cpu``, and every CPU wheel has one. The reading has to stop at the
plus and still compare the release numerically, which is the one case here the
scikit guard never had to meet.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError

import pytest

from oop_ml.pytorch import engine_version
from oop_ml.pytorch.engine_version import (
    INSTALL_HINT,
    MINIMUM_ENGINE_VERSION,
    _release_of,
    check_engine_available,
)


class TestTheReleaseReading:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2.14.1", (2, 14, 1)),
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
            ("2.14.1+cpu", (2, 14, 1)),
            ("2.3.0+cu121", (2, 3, 0)),
            ("2.1.0.dev20230801+cpu", (2, 1, 0)),
        ],
    )
    def test_a_build_tag_after_the_plus_is_dropped(
        self, raw: str, expected: tuple[int, ...]
    ) -> None:
        """Every CPU wheel reports one, so this is the common case and not
        the odd one."""
        assert _release_of(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2.0.0rc1", (2, 0, 0)),
            ("1.10.0.dev0", (1, 10, 0)),
        ],
    )
    def test_a_prerelease_compares_by_its_release(
        self, raw: str, expected: tuple[int, ...]
    ) -> None:
        assert _release_of(raw) == expected

    def test_ten_sorts_above_nine_rather_than_below_it(self) -> None:
        assert _release_of("1.10.0") > _release_of("1.9.0")

    def test_something_with_no_numbers_at_all_reads_as_nothing(self) -> None:
        assert _release_of("nightly") == ()


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

        assert INSTALL_HINT in str(raised.value)
        assert "optional" in str(raised.value)

    def test_a_too_old_engine_names_the_floor_and_what_stops_working(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_version, "version", lambda name: "1.9.0")

        with pytest.raises(ImportError) as raised:
            check_engine_available()

        message = str(raised.value)
        assert ".".join(str(number) for number in MINIMUM_ENGINE_VERSION) in message
        assert "1.9.0" in message
        assert "cross entropy" in message
        assert INSTALL_HINT in message

    def test_the_floor_itself_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A floor is inclusive, or the version it names would be refused."""
        floor = ".".join(str(number) for number in MINIMUM_ENGINE_VERSION) + ".0"
        monkeypatch.setattr(engine_version, "version", lambda name: floor)

        check_engine_available()
