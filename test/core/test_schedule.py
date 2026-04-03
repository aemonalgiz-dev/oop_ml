"""Spec for a quantity that shrinks as a walk proceeds.

Three claims carry this file and the rest is bounds checking.

The first is that the endpoints land *exactly*. A schedule is asked for a
fraction of the way through a walk, and the arithmetic that computes that
fraction is the one place an off-by-one hides. Pass one must report the
starting value to the last bit and the final pass must report the ending value
to the last bit, because anything else means the fraction was computed from the
wrong denominator and every value in between is quietly wrong by a little.

The second is that a decay is a fraction of the *run* rather than a rate per
pass. That is the whole reason ``value_at`` takes the total rather than
remembering one, and the test that pins it runs the same schedule over 100
passes and over 1000 and asserts the curves agree at matching fractions. A
schedule that decayed per pass would silently change meaning when a caller
raised a cap, which is exactly the class of quiet mistake this library keeps a
list of.

The third is the contrast between the two decays, which is the only reason to
have both. From 0.5 to 0.005, halfway through the walk a linear decay is at
0.2525 and an exponential one at 0.0500, five times smaller. Those numbers are
in the module docstring and they are asserted here, because a docstring carrying
a measurement nobody checks is a claim rather than a fact -- and writing this
file caught exactly that. It said "at pass 50 of 100", which is 49/99 of the way
through rather than half, where the values are 0.2550 and 0.0512. The prose was
out by one pass and nothing would have noticed.

The single-pass walk gets its own tests. There is no interval to be a fraction
of, so the honest answer is the starting value, and computing it without that
check divides by zero on a configuration nothing else refuses.
"""

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.schedule import (
    ConstantSchedule,
    ExponentialDecaySchedule,
    LinearDecaySchedule,
    Schedule,
)

#: One of each, for the claims that hold across the whole family.
EVERY_SCHEDULE = [
    ConstantSchedule(value=0.3),
    LinearDecaySchedule(start=5.0, end=1.0),
    ExponentialDecaySchedule(start=0.5, end=0.005),
]


class TestTheEndpointsLandExactly:
    """Where an off-by-one in the elapsed fraction would show itself."""

    @pytest.mark.parametrize("schedule", EVERY_SCHEDULE)
    def test_the_first_pass_reports_the_starting_value(
        self, schedule: Schedule
    ) -> None:
        assert schedule.value_at(1, 100) == schedule.value_at(1, 1000)

    def test_a_linear_decay_starts_and_ends_on_its_bounds(self) -> None:
        schedule = LinearDecaySchedule(start=5.0, end=1.0)

        assert schedule.value_at(1, 200) == 5.0
        assert schedule.value_at(200, 200) == 1.0

    def test_an_exponential_decay_starts_and_ends_on_its_bounds(self) -> None:
        """Written as start times a ratio raised to the elapsed fraction, so the
        last pass lands on the ending value rather than near it."""
        schedule = ExponentialDecaySchedule(start=0.5, end=0.005)

        assert schedule.value_at(1, 100) == 0.5
        assert schedule.value_at(100, 100) == pytest.approx(0.005, abs=1e-15)


class TestTheDecayIsAFractionOfTheRun:
    """Not a rate per pass, which is why the total is a parameter.

    A schedule that decayed per pass would mean something different the moment
    a caller raised the cap, and the value it reported would still look
    entirely reasonable.
    """

    @pytest.mark.parametrize("schedule", EVERY_SCHEDULE)
    @pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_the_same_fraction_gives_the_same_value_at_any_length(
        self, schedule: Schedule, fraction: float
    ) -> None:
        # 101 and 1001 rather than 100 and 1000, so that a quarter of the
        # way through is exactly 25/100 and exactly 250/1000. At 100 and 1000
        # the two fractions are 25/99 and 250/999, which differ, and the test
        # would be measuring its own arithmetic rather than the schedule's.
        short = schedule.value_at(1 + round(fraction * 100), 101)
        long = schedule.value_at(1 + round(fraction * 1000), 1001)

        assert short == pytest.approx(long)


class TestTheTwoDecaysDifferAndTheNumbersAreTheReason:
    """The only argument for having both, asserted rather than described."""

    def test_a_linear_decay_is_at_its_midpoint_halfway_through(self) -> None:
        schedule = LinearDecaySchedule(start=0.5, end=0.005)

        assert schedule.value_at(50, 99) == pytest.approx(0.2525, abs=5e-4)

    def test_an_exponential_decay_is_far_below_its_midpoint(self) -> None:
        """It spends most of the walk small, which is what a rate wants."""
        schedule = ExponentialDecaySchedule(start=0.5, end=0.005)

        assert schedule.value_at(50, 99) == pytest.approx(0.0500, abs=5e-4)

    def test_the_exponential_one_is_five_times_smaller_at_the_midpoint(
        self,
    ) -> None:
        straight = LinearDecaySchedule(start=0.5, end=0.005).value_at(50, 99)
        geometric = ExponentialDecaySchedule(start=0.5, end=0.005).value_at(50, 99)

        assert straight / geometric == pytest.approx(5.0, abs=0.1)


class TestEachCurveIsWhatItSaysItIs:
    def test_a_constant_schedule_never_moves(self) -> None:
        schedule = ConstantSchedule(value=0.3)

        assert [schedule.value_at(one, 10) for one in range(1, 11)] == [0.3] * 10

    @pytest.mark.parametrize(
        "schedule",
        [
            LinearDecaySchedule(start=5.0, end=1.0),
            ExponentialDecaySchedule(start=0.5, end=0.005),
        ],
    )
    def test_a_decay_never_rises(self, schedule: Schedule) -> None:
        walked = [schedule.value_at(one, 40) for one in range(1, 41)]

        assert all(
            later <= earlier
            for earlier, later in zip(walked[:-1], walked[1:], strict=True)
        )

    def test_a_linear_schedule_may_grow(self) -> None:
        """Nothing in the mathematics forbids it, so nothing here does, and a
        growing neighbourhood is worth being able to demonstrate failing."""
        schedule = LinearDecaySchedule(start=1.0, end=4.0)

        assert schedule.value_at(1, 10) == 1.0
        assert schedule.value_at(10, 10) == 4.0

    def test_a_linear_decay_defaults_to_reaching_zero(self) -> None:
        assert LinearDecaySchedule(start=2.0).value_at(10, 10) == 0.0


class TestASinglePassWalk:
    """No interval to be a fraction of, and a division by zero without a guard."""

    @pytest.mark.parametrize("schedule", EVERY_SCHEDULE)
    def test_it_answers_rather_than_dividing_by_zero(self, schedule: Schedule) -> None:
        assert schedule.value_at(1, 1) == schedule.value_at(1, 500)

    def test_it_reports_the_starting_value(self) -> None:
        assert LinearDecaySchedule(start=5.0, end=1.0).value_at(1, 1) == 5.0


class TestBoundsAreRefused:
    """A schedule asked about pass zero would extrapolate past its own start."""

    @pytest.mark.parametrize("schedule", EVERY_SCHEDULE)
    @pytest.mark.parametrize(
        ("pass_number", "total_passes"),
        [(0, 10), (-1, 10), (11, 10), (1, 0), (1, -3)],
        ids=[
            "pass zero",
            "negative pass",
            "past the end",
            "no passes",
            "negative walk",
        ],
    )
    def test_it_is_refused_in_the_library_s_own_words(
        self, schedule: Schedule, pass_number: int, total_passes: int
    ) -> None:
        with pytest.raises(InvalidValuesError):
            schedule.value_at(pass_number, total_passes)

    @pytest.mark.parametrize("schedule", EVERY_SCHEDULE)
    def test_the_first_and_last_passes_are_inside_the_walk(
        self, schedule: Schedule
    ) -> None:
        schedule.value_at(1, 10)
        schedule.value_at(10, 10)


class TestConfigurationIsRefusedAtConstruction:
    """Where every other hyperparameter in this library is refused."""

    def test_a_negative_constant_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            ConstantSchedule(value=-1.0)

    @pytest.mark.parametrize("bound", ["start", "end"])
    def test_a_negative_linear_bound_is_refused(self, bound: str) -> None:
        with pytest.raises(ValidationError):
            LinearDecaySchedule(**{"start": 1.0, "end": 1.0, bound: -0.5})

    @pytest.mark.parametrize("bound", ["start", "end"])
    @pytest.mark.parametrize("value", [0.0, -1.0], ids=["zero", "negative"])
    def test_an_exponential_bound_that_is_not_positive_is_refused(
        self, bound: str, value: float
    ) -> None:
        """A geometric fall multiplies by a fixed ratio, so it cannot reach zero
        and a negative bound would make the value alternate in sign. This is the
        one place a schedule refuses what a linear decay accepts, and the
        refusal is the mathematics rather than a policy."""
        with pytest.raises(ValidationError):
            ExponentialDecaySchedule(**{"start": 1.0, "end": 1.0, bound: value})

    def test_a_linear_decay_accepts_the_zero_an_exponential_refuses(self) -> None:
        """The contrast, stated directly so the asymmetry reads as deliberate."""
        assert LinearDecaySchedule(start=1.0, end=0.0).value_at(10, 10) == 0.0

    @pytest.mark.parametrize(
        "schedule_type",
        [ConstantSchedule, LinearDecaySchedule, ExponentialDecaySchedule],
    )
    def test_an_unrecognised_keyword_is_refused(
        self, schedule_type: type[Schedule]
    ) -> None:
        """extra="forbid", for the reason the whole library sets it: a
        misspelled hyperparameter otherwise leaves a plausible default in
        place and says nothing."""
        with pytest.raises(ValidationError):
            schedule_type(**{"start": 1.0, "end": 0.5, "value": 1.0, "decay_rate": 0.9})


class TestTheBaseIsAbstract:
    def test_a_schedule_cannot_be_built_directly(self) -> None:
        with pytest.raises(TypeError):
            Schedule()  # type: ignore[abstract]
