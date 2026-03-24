"""Spec for what a decomposition learned.

Green already, since these objects are plumbing rather than a concept. They are
specified anyway because three of their invariants exist to catch failures that
are otherwise silent, and an invariant nothing exercises is a comment.

``TestOrdering`` is the one that matters most. ``numpy.linalg.eigh`` returns
eigenvalues ascending, PCA wants them descending, and a fit that forgets to
reverse them produces finite, plausible numbers in every direction a caller
might look. The constructor is the only thing standing between that mistake and
a model that quietly reports its worst direction as its best.
"""

import numpy as np
import pytest

from oop_ml.core.data.coefficients import Coefficient, Coefficients
from oop_ml.core.decomposition.components import (
    PrincipalComponent,
    PrincipalComponents,
)
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
)

DIAGONAL = 0.7071067811865476
NAMES = ("first", "second")


def loadings(*weights: float, names: tuple[str, ...] = NAMES) -> Coefficients:
    """Weights bound to feature names, in order."""
    return Coefficients(
        [Coefficient(name, weight) for name, weight in zip(names, weights, strict=True)]
    )


def component(
    name: str, variance: float, *weights: float, names: tuple[str, ...] = NAMES
) -> PrincipalComponent:
    """One direction, named, with its variance."""
    return PrincipalComponent(name, loadings(*weights, names=names), variance)


def rotated_pair(total_variance: float = 5.0) -> PrincipalComponents:
    """The two directions the ``ROTATED_ELLIPSE`` fixture is known to give."""
    return PrincipalComponents(
        [
            component("component_1", 4.0, DIAGONAL, DIAGONAL),
            component("component_2", 1.0, DIAGONAL, -DIAGONAL),
        ],
        total_variance,
    )


class TestOneComponent:
    """A direction, its variance, and its loadings."""

    def test_reports_its_variance(self) -> None:
        assert component("component_1", 4.0, 1.0, 0.0).variance == pytest.approx(4.0)

    def test_loadings_are_addressable_by_feature_name(self) -> None:
        """The reason a loading vector is Coefficients and not an array."""
        one = component("component_1", 4.0, DIAGONAL, -DIAGONAL)

        assert one.loading_for("first") == pytest.approx(DIAGONAL)
        assert one.loadings["second"] == pytest.approx(-DIAGONAL)

    def test_direction_is_the_loadings_in_feature_order(self) -> None:
        one = component("component_1", 4.0, 0.6, 0.8)

        assert np.allclose(one.direction, [0.6, 0.8])
        assert one.feature_names == NAMES

    def test_asking_for_an_unweighted_feature_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            component("component_1", 4.0, 1.0, 0.0).loading_for("unseen")

    def test_a_direction_must_have_length_one(self) -> None:
        """A component says which way, not how far.

        Scaling the vector scales every transformed coordinate with it while
        leaving the reported variance alone, so the two stop describing the
        same thing.
        """
        with pytest.raises(InvalidValuesError):
            component("component_1", 4.0, 2.0, 0.0)

    def test_a_variance_cannot_be_negative(self) -> None:
        with pytest.raises(InvalidValuesError):
            component("component_1", -1.0, 1.0, 0.0)

    def test_a_blank_name_is_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            component("   ", 4.0, 1.0, 0.0)


class TestOrdering:
    """The invariant that catches an unreversed eigendecomposition."""

    def test_increasing_variance_is_rejected(self) -> None:
        """Exactly what ``eigh`` hands back if nothing reverses it."""
        with pytest.raises(InvalidValuesError):
            PrincipalComponents(
                [
                    component("component_1", 1.0, DIAGONAL, -DIAGONAL),
                    component("component_2", 4.0, DIAGONAL, DIAGONAL),
                ],
                5.0,
            )

    def test_equal_variances_are_allowed(self) -> None:
        """A tie is arbitrary, not wrong, and comes back off in the last bits.

        Two directions through pure noise, or a perfectly symmetric cloud,
        genuinely explain the same amount. Comparing strictly would refuse a
        correct decomposition.
        """
        components = PrincipalComponents(
            [
                component("component_1", 2.0, 1.0, 0.0),
                component("component_2", 2.0 + 1e-15, 0.0, 1.0),
            ],
            4.0,
        )

        assert components.n_components == 2


class TestOrthogonality:
    """The other invariant that would otherwise fail quietly."""

    def test_non_perpendicular_directions_are_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            PrincipalComponents(
                [
                    component("component_1", 4.0, 1.0, 0.0),
                    component("component_2", 1.0, DIAGONAL, DIAGONAL),
                ],
                5.0,
            )

    def test_perpendicular_directions_are_accepted(self) -> None:
        assert rotated_pair().n_components == 2


class TestExplainedVariance:
    """Shares, and the denominator they are taken against."""

    def test_shares_are_variance_over_the_stated_total(self) -> None:
        assert rotated_pair().variance_shares == pytest.approx((0.8, 0.2))

    def test_cumulative_shares_are_the_running_total(self) -> None:
        assert rotated_pair().cumulative_shares == pytest.approx((0.8, 1.0))

    def test_a_truncated_decomposition_does_not_claim_everything(self) -> None:
        """The reason the total is supplied rather than summed.

        One component out of a total of 5.0 explains 0.8, not 1.0. Summing what
        the object happens to hold would make every truncated fit report full
        explanation, which is precisely the claim being asked about.
        """
        kept = PrincipalComponents(
            [component("component_1", 4.0, DIAGONAL, DIAGONAL)], 5.0
        )

        assert kept.variance_shares == pytest.approx((0.8,))
        assert kept.kept_variance == pytest.approx(4.0)
        assert kept.total_variance == pytest.approx(5.0)

    def test_explaining_more_than_the_total_is_rejected(self) -> None:
        """A share above 1 is a denominator from the wrong data."""
        with pytest.raises(InvalidValuesError):
            rotated_pair(total_variance=3.0)

    def test_a_total_of_zero_is_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            PrincipalComponents([component("component_1", 0.0, 1.0, 0.0)], 0.0)


class TestChoosingHowMany:
    """Reading a component count off the shares."""

    @pytest.mark.parametrize(
        ("share", "expected"), [(0.5, 1), (0.8, 1), (0.85, 2), (1.0, 2)]
    )
    def test_counts_the_components_needed_to_reach_a_share(
        self, share: float, expected: int
    ) -> None:
        assert rotated_pair().n_components_for(share) == expected

    def test_a_share_the_kept_components_never_reach_raises(self) -> None:
        kept = PrincipalComponents(
            [component("component_1", 4.0, DIAGONAL, DIAGONAL)], 5.0
        )

        with pytest.raises(InvalidValuesError):
            kept.n_components_for(0.95)

    @pytest.mark.parametrize("share", [0.0, -0.1, 1.5])
    def test_a_share_outside_zero_to_one_raises(self, share: float) -> None:
        with pytest.raises(InvalidValuesError):
            rotated_pair().n_components_for(share)


class TestReadingTheGroup:
    """Getting a component back out."""

    def test_addressable_by_name(self) -> None:
        assert rotated_pair()["component_1"].variance == pytest.approx(4.0)

    def test_knows_which_components_it_holds(self) -> None:
        components = rotated_pair()

        assert "component_2" in components
        assert "component_9" not in components

    def test_asking_for_an_unknown_component_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            _ = rotated_pair()["component_9"]

    def test_iterates_in_order(self) -> None:
        assert [one.name for one in rotated_pair()] == ["component_1", "component_2"]

    def test_directions_are_one_component_per_row(self) -> None:
        """The orientation a projection wants."""
        directions = rotated_pair().directions

        assert directions.shape == (2, 2)
        assert np.allclose(directions[0], [DIAGONAL, DIAGONAL])

    def test_reports_the_features_it_came_from(self) -> None:
        components = rotated_pair()

        assert components.feature_names == NAMES
        assert components.n_features == 2
        assert len(components) == 2


class TestWhatTheGroupRefuses:
    """The remaining constructor invariants."""

    def test_no_components_is_rejected(self) -> None:
        with pytest.raises(EmptyValuesError):
            PrincipalComponents([], 5.0)

    def test_a_duplicate_name_is_rejected(self) -> None:
        with pytest.raises(NonUniqueFeaturesError):
            PrincipalComponents(
                [
                    component("component_1", 4.0, 1.0, 0.0),
                    component("component_1", 1.0, 0.0, 1.0),
                ],
                5.0,
            )

    def test_components_over_different_features_are_rejected(self) -> None:
        """Position i of one direction has to mean what position i of the next does."""
        with pytest.raises(InvalidValuesError):
            PrincipalComponents(
                [
                    component("component_1", 4.0, 1.0, 0.0),
                    component(
                        "component_2", 1.0, 0.0, 1.0, names=("first", "different")
                    ),
                ],
                5.0,
            )
