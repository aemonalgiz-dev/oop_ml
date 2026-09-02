"""Spec for kernel PCA -- red until ``_solve`` and ``transform`` land.

``test_a_linear_kernel_matches_ordinary_pca`` is the control, and it is the only
test here that can tell you which half broke. With a linear kernel the implied
space *is* the original one, so the coordinates must match
``PrincipalComponentAnalysis``'s -- up to a sign, since an eigenvector's sign
carries no meaning in either.

The payoff test is ``test_it_unrolls_what_ordinary_pca_cannot``. Concentric
rings are linearly inseparable, and no rotation of the plane makes them
separable, so ordinary PCA cannot help however many components it keeps. An RBF
kernel pulls them apart along a single coordinate. That is a property of the
implied space rather than of any clever fitting.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    InvalidValuesError,
    NotFittedError,
    TooFewValuesError,
)
from oop_ml.core.kernel.functions import LinearKernel, RadialBasisKernel
from oop_ml.numpy.decomposition.kernel_principal_component_analysis import (
    KernelComponent,
    KernelComponents,
    KernelPrincipalComponentAnalysis,
)
from oop_ml.numpy.decomposition.principal_component_analysis import (
    PrincipalComponentAnalysis,
)
from test.fixtures import CONCENTRIC_RINGS, ROTATED_ELLIPSE


def fitted(kernel=None, n_components: int | None = None):
    """A model fitted to the hand-computed PCA fixture."""
    return KernelPrincipalComponentAnalysis(
        kernel=kernel or LinearKernel(), n_components=n_components
    ).fit(ROTATED_ELLIPSE.input_features)


def column_of(features: list[Feature], name: str) -> np.ndarray:
    """One named column out of a transform's output."""
    return next(feature.values for feature in features if feature.name == name)


class TestAgainstOrdinaryPca:
    """The control: with a linear kernel these decompose the same space."""

    def test_a_linear_kernel_matches_ordinary_pca(self) -> None:
        """Same coordinates, up to a sign that means nothing in either.

        The fixture's known eigenvalues are 4.0 and 1.0, so the first
        coordinate carries a variance of 4.0 whichever route produced it.
        """
        kernelled = fitted(LinearKernel(), n_components=2)
        ordinary = PrincipalComponentAnalysis(n_components=2).fit(
            ROTATED_ELLIPSE.input_features
        )

        through_kernel = column_of(
            kernelled.transform(ROTATED_ELLIPSE.input_features), "kernel_component_1"
        )
        through_pca = column_of(
            ordinary.transform(ROTATED_ELLIPSE.input_features), "component_1"
        )

        assert np.abs(through_kernel) == pytest.approx(np.abs(through_pca), abs=1e-06)

    def test_the_variances_match_ordinary_pca_s(self) -> None:
        """4.0 and 1.0, computed through the Gram matrix instead.

        The Gram matrix and the covariance share their non-zero eigenvalues,
        which is the identity the whole method rests on.
        """
        variances = [one.variance for one in fitted(LinearKernel()).components]

        assert variances[0] == pytest.approx(4.0, abs=1e-06)
        assert variances[1] == pytest.approx(1.0, abs=1e-06)


class TestWhatTheKernelBuys:
    """Structure no rotation of the original space can expose."""

    def test_it_unrolls_what_ordinary_pca_cannot(self) -> None:
        """Concentric rings separate along one kernel coordinate.

        Ordinary PCA can only rotate, and no rotation separates a ring from
        the ring around it -- so the inner and outer groups overlap on every
        ordinary component. Under an RBF kernel the two groups' first
        coordinates pull apart.
        """
        model = KernelPrincipalComponentAnalysis(
            kernel=RadialBasisKernel(gamma=0.05), n_components=2
        ).fit(CONCENTRIC_RINGS.input_features)

        first = column_of(
            model.transform(CONCENTRIC_RINGS.input_features), "kernel_component_1"
        )
        groups = np.asarray(CONCENTRIC_RINGS.true_groups)

        inner = first[groups == 0]
        outer = first[groups == 1]

        assert inner.max() < outer.min() or outer.max() < inner.min()

    def test_ordinary_pca_cannot_do_the_same(self) -> None:
        """The other half of the contrast, asserted rather than assumed."""
        ordinary = PrincipalComponentAnalysis(n_components=2).fit(
            CONCENTRIC_RINGS.input_features
        )
        first = column_of(
            ordinary.transform(CONCENTRIC_RINGS.input_features), "component_1"
        )
        groups = np.asarray(CONCENTRIC_RINGS.true_groups)

        inner = first[groups == 0]
        outer = first[groups == 1]

        assert not (inner.max() < outer.min() or outer.max() < inner.min())


class TestWhatItLearns:
    """Components built from rows rather than from features."""

    def test_a_component_is_built_from_training_rows(self) -> None:
        """One coefficient per row, not one per feature. That is the difference."""
        component = fitted(n_components=1).components["kernel_component_1"]

        assert component.n_training_rows == ROTATED_ELLIPSE.n_samples

    def test_a_component_has_no_loadings(self) -> None:
        """In the implied space the coordinates have no names to bind to."""
        component = fitted(n_components=1).components["kernel_component_1"]

        assert not hasattr(component, "loadings")
        assert not hasattr(component, "loading_for")

    def test_components_come_out_ranked(self) -> None:
        variances = [
            one.variance for one in fitted(RadialBasisKernel(gamma=0.5)).components
        ]

        assert variances == sorted(variances, reverse=True)

    def test_shares_are_taken_against_the_full_total(self) -> None:
        """The same denominator rule as ordinary PCA: 4.0 out of 5.0 is 0.8."""
        components = fitted(LinearKernel(), n_components=1).components

        assert components.total_variance == pytest.approx(5.0, abs=1e-06)
        assert components.variance_shares[0] == pytest.approx(0.8, abs=1e-06)

    def test_the_normalisation_makes_the_coordinates_carry_the_variance(self) -> None:
        """The step with no counterpart in ordinary PCA.

        ``eigh`` returns unit-length eigenvectors, but the direction each names
        in the implied space has length ``sqrt(lambda)``. Dividing by that is
        what makes the transformed column's variance come out as ``lambda``
        rather than ``lambda ** 2``. Skip it and everything still separates,
        scaled wrong by a different factor per component.
        """
        model = fitted(LinearKernel(), n_components=2)
        transformed = model.transform(ROTATED_ELLIPSE.input_features)

        variances = [float(np.var(feature.values, ddof=1)) for feature in transformed]

        assert variances[0] == pytest.approx(4.0, abs=1e-06)
        assert variances[1] == pytest.approx(1.0, abs=1e-06)


class TestTransforming:
    """The query side, and the centring it must not skip."""

    def test_it_produces_one_feature_per_component(self) -> None:
        transformed = fitted(n_components=2).transform(ROTATED_ELLIPSE.input_features)

        assert [feature.name for feature in transformed] == [
            "kernel_component_1",
            "kernel_component_2",
        ]

    def test_the_names_do_not_collide_with_ordinary_pca_s(self) -> None:
        """So the output of one cannot be fed to the other by accident."""
        kernelled = fitted(n_components=1).transform(ROTATED_ELLIPSE.input_features)
        ordinary = PrincipalComponentAnalysis(n_components=1).fit_transform(
            ROTATED_ELLIPSE.input_features
        )

        assert kernelled[0].name != ordinary[0].name

    def test_the_transformed_columns_are_centred(self) -> None:
        """The centring identity, read off the result."""
        transformed = fitted(n_components=2).transform(ROTATED_ELLIPSE.input_features)

        for feature in transformed:
            assert float(np.mean(feature.values)) == pytest.approx(0.0, abs=1e-08)

    def test_new_rows_are_centred_against_the_training_matrix(self) -> None:
        """Not against themselves, which would be the usual leak.

        A single query row has no mean of its own worth speaking of, so a
        model centring against the query block would produce zero here.
        """
        model = fitted(LinearKernel(), n_components=1)
        transformed = model.transform(
            [Feature("first", [12.0]), Feature("second", [102.0])]
        )

        assert abs(float(transformed[0].values[0])) > 1.0

    def test_column_order_does_not_matter(self) -> None:
        model = fitted(n_components=2)
        first, second = ROTATED_ELLIPSE.input_features

        assert np.allclose(
            column_of(model.transform([first, second]), "kernel_component_1"),
            column_of(model.transform([second, first]), "kernel_component_1"),
        )


class TestWhatIsAbsent:
    """A gap that is real rather than unimplemented."""

    def test_there_is_no_inverse_transform(self) -> None:
        """The pre-image problem: the feature map often has no inverse.

        Ordinary PCA reconstructs into the original features. Here the
        reconstruction lands in the implied space, and for an RBF kernel most
        points there are not the image of any real row.
        """
        assert not hasattr(fitted(), "inverse_transform")


class TestTheValueObjects:
    """Invariants on the components themselves."""

    def component(self, name: str, variance: float, *coefficients: float):
        return KernelComponent(name, np.array(coefficients, dtype=np.float64), variance)

    def test_increasing_variance_is_rejected(self) -> None:
        """The unreversed-``eigh`` guard, same as ordinary PCA's."""
        with pytest.raises(InvalidValuesError):
            KernelComponents(
                [
                    self.component("kernel_component_1", 1.0, 0.5, 0.5),
                    self.component("kernel_component_2", 4.0, 0.5, -0.5),
                ],
                5.0,
            )

    def test_components_from_different_row_counts_are_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            KernelComponents(
                [
                    self.component("kernel_component_1", 4.0, 0.5, 0.5),
                    self.component("kernel_component_2", 1.0, 0.5),
                ],
                5.0,
            )

    def test_a_negative_variance_is_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            self.component("kernel_component_1", -1.0, 0.5, 0.5)

    def test_explaining_more_than_the_total_is_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            KernelComponents([self.component("kernel_component_1", 4.0, 0.5, 0.5)], 3.0)


class TestWhatItRefuses:
    """Guards, each from the MLLibError hierarchy."""

    def test_reading_the_components_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = KernelPrincipalComponentAnalysis().components

    def test_transforming_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            KernelPrincipalComponentAnalysis().transform(ROTATED_ELLIPSE.input_features)

    def test_a_single_row_is_rejected(self) -> None:
        with pytest.raises(TooFewValuesError):
            KernelPrincipalComponentAnalysis().fit([Feature("first", [1.0])])

    def test_more_components_than_rows_is_rejected(self) -> None:
        """The ceiling is the row count here, not the feature count."""
        with pytest.raises(InvalidValuesError):
            KernelPrincipalComponentAnalysis(n_components=99).fit(
                ROTATED_ELLIPSE.input_features
            )

    def test_transforming_without_every_fitted_feature_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().transform([ROTATED_ELLIPSE.input_features[0]])
