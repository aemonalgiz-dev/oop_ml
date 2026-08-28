"""Spec for the FeatureImportances value object.

Green already, since the object is plumbing rather than a concept. It is here
because the invariant it enforces is the whole reason it exists: shares that do
not sum to 1 are not shares, and a producer that hands over raw totals without
normalising them should be caught at the boundary rather than three call sites
later when a report says a feature accounts for 340% of the answer.
"""

import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
)
from oop_ml.core.importance.importances import (
    FeatureContribution,
    FeatureImportance,
    FeatureImportances,
)

NAMES = ["studied", "slept", "noise"]
SCORES = [6.0, 3.0, 1.0]
SHARES = [0.6, 0.3, 0.1]


@pytest.fixture
def importances() -> FeatureImportances:
    return FeatureImportances.from_scores(NAMES, SCORES)


class TestFromScores:
    """Normalising raw totals, which is how every producer builds one."""

    def test_divides_by_the_total(self, importances: FeatureImportances) -> None:
        assert [one.value for one in importances] == pytest.approx(SHARES)

    def test_keeps_the_order_the_fit_saw(self, importances: FeatureImportances) -> None:
        assert [one.name for one in importances] == NAMES

    def test_shares_sum_to_one(self, importances: FeatureImportances) -> None:
        assert sum(one.value for one in importances) == pytest.approx(1.0)

    def test_a_feature_that_earned_nothing_still_gets_an_entry(self) -> None:
        """A zero is a finding, and dropping it would hide the finding."""
        importances = FeatureImportances.from_scores(NAMES, [6.0, 4.0, 0.0])

        assert len(importances) == 3
        assert importances["noise"] == 0.0

    def test_rejects_scores_that_total_zero(self) -> None:
        """A tree that never split has no shares, not shares of zero."""
        with pytest.raises(InvalidValuesError):
            FeatureImportances.from_scores(NAMES, [0.0, 0.0, 0.0])

    def test_rejects_a_negative_score(self) -> None:
        with pytest.raises(InvalidValuesError):
            FeatureImportances.from_scores(NAMES, [6.0, -1.0, 1.0])

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(NonEqualArrayLengthError):
            FeatureImportances.from_scores(NAMES, [6.0, 3.0])

    def test_rejects_no_features(self) -> None:
        with pytest.raises(EmptyValuesError):
            FeatureImportances.from_scores([], [])


class TestFromContributions:
    """Totalling many credits per feature, which is what a tree walk produces.

    The exam tree is the worked case: the root credits ``slept`` with
    ``15 * 0.2133 = 3.2`` and the node below credits ``studied`` with
    ``9 * 0.2778 = 2.5``, which normalise to 0.5614 and 0.4386.
    """

    def test_totals_repeated_credits_to_one_feature(self) -> None:
        """A feature winning several splits collects several contributions."""
        importances = FeatureImportances.from_contributions(
            ["slept", "studied"],
            [
                FeatureContribution("slept", 2.0),
                FeatureContribution("studied", 2.5),
                FeatureContribution("slept", 1.2),
            ],
        )

        assert importances["slept"] == pytest.approx(3.2 / 5.7)
        assert importances["studied"] == pytest.approx(2.5 / 5.7)

    def test_a_feature_with_no_contributions_gets_a_zero(self) -> None:
        """Why the names are passed separately from the contributions.

        A feature that never won a split contributes nothing and would
        otherwise be missing from the result entirely.
        """
        importances = FeatureImportances.from_contributions(
            ["slept", "studied", "noise"],
            [FeatureContribution("slept", 3.2), FeatureContribution("studied", 2.5)],
        )

        assert len(importances) == 3
        assert importances["noise"] == 0.0

    def test_keeps_the_order_the_fit_saw(self) -> None:
        importances = FeatureImportances.from_contributions(
            ["noise", "slept"], [FeatureContribution("slept", 1.0)]
        )

        assert [one.name for one in importances] == ["noise", "slept"]

    def test_order_of_contributions_does_not_matter(self) -> None:
        forwards = FeatureImportances.from_contributions(
            ["a", "b"],
            [FeatureContribution("a", 1.0), FeatureContribution("b", 3.0)],
        )
        backwards = FeatureImportances.from_contributions(
            ["a", "b"],
            [FeatureContribution("b", 3.0), FeatureContribution("a", 1.0)],
        )

        assert forwards["a"] == pytest.approx(backwards["a"])

    def test_rejects_a_contribution_to_an_unfitted_feature(self) -> None:
        """A typo in a feature name would otherwise vanish into a total."""
        with pytest.raises(InvalidValuesError):
            FeatureImportances.from_contributions(
                ["slept"], [FeatureContribution("slpet", 1.0)]
            )

    def test_rejects_contributions_that_total_zero(self) -> None:
        with pytest.raises(InvalidValuesError):
            FeatureImportances.from_contributions(
                ["slept"], [FeatureContribution("slept", 0.0)]
            )

    def test_rejects_no_names(self) -> None:
        with pytest.raises(EmptyValuesError):
            FeatureImportances.from_contributions([], [])

    def test_a_contribution_cannot_be_negative(self) -> None:
        """Both producers measure something removed or lost."""
        with pytest.raises(InvalidValuesError):
            FeatureContribution("slept", -1.0)

    def test_a_contribution_is_not_bounded_by_one(self) -> None:
        """The difference from a share, and the reason for the second class."""
        assert FeatureContribution("slept", 3.2).amount == 3.2


class TestReading:
    """Getting a number back out."""

    def test_addressable_by_name(self, importances: FeatureImportances) -> None:
        assert importances["studied"] == pytest.approx(0.6)
        assert importances.value_for("slept") == pytest.approx(0.3)

    def test_ranked_puts_the_largest_first(
        self, importances: FeatureImportances
    ) -> None:
        assert [one.name for one in importances.ranked()] == [
            "studied",
            "slept",
            "noise",
        ]

    def test_most_important_is_the_largest_share(
        self, importances: FeatureImportances
    ) -> None:
        assert importances.most_important.name == "studied"

    def test_knows_which_features_it_holds(
        self, importances: FeatureImportances
    ) -> None:
        assert "slept" in importances
        assert "unseen" not in importances

    def test_asking_for_an_unknown_feature_raises(
        self, importances: FeatureImportances
    ) -> None:
        with pytest.raises(InvalidValuesError):
            _ = importances["unseen"]


class TestInvariant:
    """What the constructor refuses."""

    def test_rejects_shares_that_do_not_sum_to_one(self) -> None:
        with pytest.raises(InvalidValuesError):
            FeatureImportances(
                [FeatureImportance("a", 0.5), FeatureImportance("b", 0.2)]
            )

    def test_rejects_a_duplicate_name(self) -> None:
        with pytest.raises(NonUniqueFeaturesError):
            FeatureImportances(
                [FeatureImportance("a", 0.5), FeatureImportance("a", 0.5)]
            )

    def test_rejects_no_importances(self) -> None:
        with pytest.raises(EmptyValuesError):
            FeatureImportances([])

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_rejects_a_share_outside_zero_and_one(self, value: float) -> None:
        with pytest.raises(InvalidValuesError):
            FeatureImportance("a", value)

    def test_rejects_a_blank_name(self) -> None:
        with pytest.raises(InvalidValuesError):
            FeatureImportance("   ", 1.0)
