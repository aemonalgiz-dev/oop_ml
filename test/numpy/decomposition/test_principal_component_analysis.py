"""Spec for PCA -- red until ``_solve``, ``transform`` and ``inverse_transform`` land.

``ROTATED_ELLIPSE`` is a fixture whose whole decomposition was done by hand:
centred, its covariance is ``[[2.5, 1.5], [1.5, 2.5]]``, whose eigenvalues are
``2.5 +/- 1.5``. So the answers are exactly 4.0 and 1.0, along ``(1, 1)/sqrt(2)``
and ``(1, -1)/sqrt(2)``, and every assertion below is against a number that was
known before the code was written.

Its means are 10 and 100 rather than 0, and far apart on purpose:
``test_it_centres_before_decomposing`` is what fails if the centring is skipped,
and it fails loudly rather than by a few decimal places.

**No test asserts a sign.** An eigenvector's sign is arbitrary -- ``v`` and
``-v`` are the same direction with the same variance -- and different platforms
and library versions legitimately return different ones. Everything here
compares absolute values, or compares a quantity that is sign-free.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    AllSameValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
    NotFittedError,
    TooFewValuesError,
)
from oop_ml.numpy.decomposition.principal_component_analysis import (
    PrincipalComponentAnalysis,
)
from test.fixtures import (
    MISMATCHED_UNITS,
    MISMATCHED_UNITS_RAW_SMALL_LOADING_CEILING,
    ROTATED_ELLIPSE,
    ROTATED_ELLIPSE_CUMULATIVE_SHARES,
    ROTATED_ELLIPSE_FIRST_COORDINATES,
    ROTATED_ELLIPSE_LOADING,
    ROTATED_ELLIPSE_SHARES,
    ROTATED_ELLIPSE_TOTAL_VARIANCE,
    ROTATED_ELLIPSE_VARIANCES,
)


def fitted(n_components: int | None = None) -> PrincipalComponentAnalysis:
    """A model fitted to the hand-computed fixture."""
    return PrincipalComponentAnalysis(n_components=n_components).fit(
        ROTATED_ELLIPSE.input_features
    )


def column_of(features: list[Feature], name: str) -> np.ndarray:
    """One named column out of a transform's output."""
    return next(feature.values for feature in features if feature.name == name)


class TestWhatItLearns:
    """The decomposition itself, against numbers computed by hand."""

    def test_finds_a_direction_per_feature_by_default(self) -> None:
        """Keeping everything is a rotation, and a rotation loses nothing."""
        components = fitted().components

        assert components.n_components == 2
        assert components.n_features == 2

    def test_variances_are_the_eigenvalues(self) -> None:
        assert [one.variance for one in fitted().components] == pytest.approx(
            list(ROTATED_ELLIPSE_VARIANCES)
        )

    def test_the_first_component_lies_along_the_spread(self) -> None:
        """The cloud is stretched along (1, 1), so the loadings are equal.

        Compared by absolute value, since the sign of an eigenvector carries no
        meaning.
        """
        first = fitted().components["component_1"]

        assert abs(first.loading_for("first")) == pytest.approx(ROTATED_ELLIPSE_LOADING)
        assert abs(first.loading_for("second")) == pytest.approx(
            ROTATED_ELLIPSE_LOADING
        )

    def test_the_two_components_are_perpendicular(self) -> None:
        directions = fitted().components.directions

        assert float(directions[0] @ directions[1]) == pytest.approx(0.0, abs=1e-12)

    def test_components_come_out_ranked(self) -> None:
        """Descending, which is the reverse of what ``eigh`` hands back."""
        variances = [one.variance for one in fitted().components]

        assert variances == sorted(variances, reverse=True)

    def test_reports_the_share_of_variance_explained(self) -> None:
        components = fitted().components

        assert components.variance_shares == pytest.approx(ROTATED_ELLIPSE_SHARES)
        assert components.cumulative_shares == pytest.approx(
            ROTATED_ELLIPSE_CUMULATIVE_SHARES
        )

    def test_it_centres_before_decomposing(self) -> None:
        """The fixture's means are 10 and 100, so skipping this is unmissable.

        Uncentred, the first component points almost exactly at (0, 1) -- it
        finds where the cloud sits rather than how it is shaped -- and the
        reported total variance runs to five figures instead of 5.0.
        """
        components = fitted().components

        assert components.total_variance == pytest.approx(
            ROTATED_ELLIPSE_TOTAL_VARIANCE
        )
        assert abs(components["component_1"].loading_for("first")) == pytest.approx(
            ROTATED_ELLIPSE_LOADING
        )


class TestKeepingFewer:
    """Truncation, and what it does and does not change."""

    def test_keeps_only_the_components_asked_for(self) -> None:
        assert fitted(n_components=1).components.n_components == 1

    def test_a_kept_component_is_unchanged_by_the_truncation(self) -> None:
        """Dropping the second does not alter the first: they are independent."""
        assert fitted(n_components=1).components["component_1"].variance == (
            pytest.approx(fitted().components["component_1"].variance)
        )

    def test_the_share_is_still_taken_against_the_whole(self) -> None:
        """One direction of 4.0 out of a total of 5.0 explains 0.8, not 1.0.

        Summing only the kept components would report full explanation from a
        fit that discarded a fifth of the variance.
        """
        components = fitted(n_components=1).components

        assert components.total_variance == pytest.approx(
            ROTATED_ELLIPSE_TOTAL_VARIANCE
        )
        assert components.variance_shares == pytest.approx((0.8,))

    def test_choosing_the_count_from_the_shares(self) -> None:
        """Fit everything once, ask what reaches a share, refit at that count."""
        assert fitted().components.n_components_for(0.75) == 1


class TestTransforming:
    """Projecting rows onto the learned directions."""

    def test_produces_one_feature_per_component(self) -> None:
        transformed = fitted().transform(ROTATED_ELLIPSE.input_features)

        assert [feature.name for feature in transformed] == [
            "component_1",
            "component_2",
        ]

    def test_truncating_narrows_the_table(self) -> None:
        """The whole reason to do this."""
        transformed = fitted(n_components=1).transform(ROTATED_ELLIPSE.input_features)

        assert len(transformed) == 1
        assert len(transformed[0].values) == ROTATED_ELLIPSE.n_samples

    def test_the_coordinates_are_the_hand_computed_ones(self) -> None:
        """Centred row 0 is (2, 2), and (2 + 2)/sqrt(2) is 2.8284.

        Absolute values, since flipping the component flips every coordinate
        along it and describes the same rotation.
        """
        transformed = fitted().transform(ROTATED_ELLIPSE.input_features)

        assert np.abs(column_of(transformed, "component_1")) == pytest.approx(
            np.abs(ROTATED_ELLIPSE_FIRST_COORDINATES)
        )

    def test_the_transformed_columns_carry_the_reported_variances(self) -> None:
        """What "variance along this direction" means, checked directly."""
        transformed = fitted().transform(ROTATED_ELLIPSE.input_features)

        assert [
            float(np.var(feature.values, ddof=1)) for feature in transformed
        ] == pytest.approx(list(ROTATED_ELLIPSE_VARIANCES))

    def test_the_transformed_columns_are_uncorrelated(self) -> None:
        """What orthogonality buys, and the property PCA is usually wanted for."""
        transformed = fitted().transform(ROTATED_ELLIPSE.input_features)
        covariance = np.cov(
            np.column_stack([feature.values for feature in transformed]), rowvar=False
        )

        assert float(covariance[0, 1]) == pytest.approx(0.0, abs=1e-12)

    def test_the_transformed_columns_are_centred(self) -> None:
        """The fit's means were subtracted, so every coordinate averages zero."""
        transformed = fitted().transform(ROTATED_ELLIPSE.input_features)

        for feature in transformed:
            assert float(np.mean(feature.values)) == pytest.approx(0.0, abs=1e-12)

    def test_column_order_does_not_matter(self) -> None:
        """Matched by name, like every other model here."""
        first, second = ROTATED_ELLIPSE.input_features
        forwards = fitted().transform([first, second])
        backwards = fitted().transform([second, first])

        assert np.allclose(
            column_of(forwards, "component_1"), column_of(backwards, "component_1")
        )

    def test_new_rows_are_centred_on_the_fit_s_means(self) -> None:
        """Not on their own, which would be the leak the split exists to stop.

        These two rows average (11, 101). Re-centred on themselves the first
        would land at the origin; centred on the fit's (10, 100) it does not.
        """
        model = fitted()
        transformed = model.transform(
            [Feature("first", [12.0, 10.0]), Feature("second", [102.0, 100.0])]
        )

        assert abs(float(column_of(transformed, "component_1")[0])) == pytest.approx(
            abs(ROTATED_ELLIPSE_FIRST_COORDINATES[0])
        )


class TestReconstructing:
    """The trip back, and what truncation costs."""

    def test_keeping_everything_is_lossless(self) -> None:
        """A full decomposition is a rotation, and a rotation is invertible."""
        model = fitted()
        rebuilt = model.inverse_transform(
            model.transform(ROTATED_ELLIPSE.input_features)
        )

        for original, restored in zip(
            ROTATED_ELLIPSE.input_features, rebuilt, strict=True
        ):
            assert np.allclose(restored.values, original.values)

    def test_it_restores_the_original_feature_names(self) -> None:
        model = fitted()
        rebuilt = model.inverse_transform(
            model.transform(ROTATED_ELLIPSE.input_features)
        )

        assert [feature.name for feature in rebuilt] == list(
            ROTATED_ELLIPSE.feature_names
        )

    def test_truncating_loses_exactly_the_discarded_variance(self) -> None:
        """The reconstruction error is what the dropped component held.

        Squared error per row, totalled and divided by ``n - 1``, comes to the
        second component's variance of 1.0. That equality is what makes PCA a
        *best* rank-one approximation rather than merely a small one.
        """
        model = fitted(n_components=1)
        rebuilt = model.inverse_transform(
            model.transform(ROTATED_ELLIPSE.input_features)
        )

        squared_error = sum(
            float(np.sum((restored.values - original.values) ** 2))
            for original, restored in zip(
                ROTATED_ELLIPSE.input_features, rebuilt, strict=True
            )
        )

        assert squared_error / (ROTATED_ELLIPSE.n_samples - 1) == pytest.approx(
            ROTATED_ELLIPSE_VARIANCES[1]
        )


class TestUnits:
    """Why ``standardize`` is a decision and not a default."""

    def test_unstandardized_the_larger_unit_takes_the_first_component(self) -> None:
        """One signal in two units, and the millimetres column wins on nothing else.

        Both columns hold the same measurement; the second is the first times a
        thousand. Variance is not unit-free, so the second's is a million times
        larger and it takes essentially the whole first component.
        """
        model = PrincipalComponentAnalysis().fit(MISMATCHED_UNITS.input_features)

        assert (
            abs(model.components["component_1"].loading_for("width_metres"))
            < MISMATCHED_UNITS_RAW_SMALL_LOADING_CEILING
        )

    def test_standardizing_gives_the_two_columns_equal_say(self) -> None:
        """Divided by their own spreads the columns become identical.

        The first component then splits evenly, which is the answer that
        reflects the data rather than the choice of unit.
        """
        model = PrincipalComponentAnalysis(standardize=True).fit(
            MISMATCHED_UNITS.input_features
        )

        assert abs(
            model.components["component_1"].loading_for("width_metres")
        ) == pytest.approx(ROTATED_ELLIPSE_LOADING, abs=1e-06)

    def test_standardizing_survives_the_round_trip(self) -> None:
        """``inverse_transform`` has to undo the scaling as well as the centring."""
        model = PrincipalComponentAnalysis(standardize=True).fit(
            MISMATCHED_UNITS.input_features
        )
        rebuilt = model.inverse_transform(
            model.transform(MISMATCHED_UNITS.input_features)
        )

        for original, restored in zip(
            MISMATCHED_UNITS.input_features, rebuilt, strict=True
        ):
            assert np.allclose(restored.values, original.values)

    def test_standardizing_a_constant_column_raises(self) -> None:
        """There is no spread to divide by, and the standardizer says so."""
        with pytest.raises(AllSameValuesError):
            PrincipalComponentAnalysis(standardize=True).fit(
                [Feature("varies", [1.0, 2.0, 3.0]), Feature("flat", [7.0, 7.0, 7.0])]
            )


class TestWhatItRefuses:
    """Guards, each raising something from the MLLibError hierarchy."""

    def test_reading_components_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = PrincipalComponentAnalysis().components

    def test_transforming_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            PrincipalComponentAnalysis().transform(ROTATED_ELLIPSE.input_features)

    def test_inverse_transforming_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            PrincipalComponentAnalysis().inverse_transform(
                [Feature("component_1", [1.0, 2.0])]
            )

    def test_a_single_row_has_no_spread_to_decompose(self) -> None:
        with pytest.raises(TooFewValuesError):
            PrincipalComponentAnalysis().fit([Feature("first", [1.0])])

    def test_asking_for_more_components_than_features_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            PrincipalComponentAnalysis(n_components=3).fit(
                ROTATED_ELLIPSE.input_features
            )

    def test_duplicate_feature_names_are_rejected(self) -> None:
        with pytest.raises(NonUniqueFeaturesError):
            PrincipalComponentAnalysis().fit(
                [Feature("same", [1.0, 2.0, 3.0]), Feature("same", [4.0, 5.0, 6.0])]
            )

    def test_transforming_without_every_fitted_feature_raises(self) -> None:
        """A component is a blend of all of them, so a missing column is fatal."""
        with pytest.raises(InvalidValuesError):
            fitted().transform([ROTATED_ELLIPSE.input_features[0]])

    def test_transforming_with_an_unknown_feature_raises(self) -> None:
        """An extra column has no loading to be weighted by."""
        with pytest.raises(InvalidValuesError):
            fitted().transform(
                [*ROTATED_ELLIPSE.input_features, Feature("extra", [1.0] * 5)]
            )

    def test_inverse_transforming_the_wrong_components_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().inverse_transform([Feature("component_1", [1.0] * 5)])


class TestFitTransform:
    """The convenience the base class provides, checked to line up."""

    def test_matches_fitting_and_transforming_separately(self) -> None:
        together = PrincipalComponentAnalysis(n_components=1).fit_transform(
            ROTATED_ELLIPSE.input_features
        )
        apart = fitted(n_components=1).transform(ROTATED_ELLIPSE.input_features)

        assert np.allclose(together[0].values, apart[0].values)
