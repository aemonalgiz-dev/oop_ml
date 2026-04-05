"""The contract every backend's KernelPrincipalComponentAnalysis keeps.

The control is the linear kernel, under which the implied space is the
original one and the coordinates must match ordinary PCA's hand-computed
answer: variances 4.0 and 1.0, and a first coordinate of ``4 / sqrt(2)`` for
the row centred at ``(2, 2)``, up to a sign that means nothing.

The payoff is two concentric rings. No rotation of the plane separates a ring
from the ring around it, and a radial basis kernel pulls them apart along one
coordinate. That is a property of the implied space, and it is the known
answer the fixture is built to have.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.exceptions import (
    InvalidValuesError,
    NotFittedError,
    TooFewValuesError,
)
from oop_ml.core.kernel.functions import (
    Kernel,
    LinearKernel,
    PolynomialKernel,
    RadialBasisKernel,
    SigmoidKernel,
)

from .harness import provided

_FIRST = [12.0, 8.0, 11.0, 9.0, 10.0]
_SECOND = [102.0, 98.0, 99.0, 101.0, 100.0]
ELLIPSE = [Feature("first", _FIRST), Feature("second", _SECOND)]
VARIANCES = (4.0, 1.0)
FIRST_COORDINATES = np.array([4.0, -4.0, 0.0, 0.0, 0.0]) / np.sqrt(2.0)

_ANGLES = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
_INNER = 1.0 * np.column_stack([np.cos(_ANGLES), np.sin(_ANGLES)])
_OUTER = 5.0 * np.column_stack([np.cos(_ANGLES), np.sin(_ANGLES)])
_RINGS = np.vstack([_INNER, _OUTER])
RINGS = [Feature("first", _RINGS[:, 0]), Feature("second", _RINGS[:, 1])]
RING_GROUPS = np.array([0] * 12 + [1] * 12)

#: The sigmoid kernel is not a Mercer kernel, and the two backends part on an
#: indefinite Gram matrix: numpy clamps its negative eigenvalues, the engine
#: refuses ones that are significant. At ``gamma=0.01`` on these rings the
#: worst eigenvalue is half a percent of the largest and the engine refuses;
#: at ``gamma=1e-4`` the tanh is linear to within rounding and both accept.
#: The contract covers what both backends provide, so the kernel is exercised
#: where it is positive semi-definite.
EVERY_KERNEL: list[Kernel] = [
    LinearKernel(),
    PolynomialKernel(degree=2),
    RadialBasisKernel(gamma=0.05),
    SigmoidKernel(gamma=1e-4, constant=0.0),
]


def column_of(features: list[Feature], name: str) -> np.ndarray:
    """One named column out of a transform's output."""
    return next(feature.values for feature in features if feature.name == name)


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    KernelPrincipalComponentAnalysis = provided(
        backend, "KernelPrincipalComponentAnalysis"
    )
    model = KernelPrincipalComponentAnalysis(
        kernel=RadialBasisKernel(gamma=2.0), n_components=1
    )

    assert model.kernel == RadialBasisKernel(gamma=2.0)
    assert model.n_components == 1


def test_it_fits_features_and_returns_itself(backend: ModuleType) -> None:
    KernelPrincipalComponentAnalysis = provided(
        backend, "KernelPrincipalComponentAnalysis"
    )
    model = KernelPrincipalComponentAnalysis()

    assert model.fit(ELLIPSE) is model


def test_a_linear_kernel_recovers_ordinary_pca_s_variances(
    backend: ModuleType,
) -> None:
    KernelPrincipalComponentAnalysis = provided(
        backend, "KernelPrincipalComponentAnalysis"
    )
    components = (
        KernelPrincipalComponentAnalysis(kernel=LinearKernel()).fit(ELLIPSE).components
    )

    assert components.n_components == 2
    assert [one.variance for one in components] == pytest.approx(
        list(VARIANCES), abs=1e-06
    )
    assert components.variance_shares == pytest.approx((0.8, 0.2), abs=1e-06)


def test_a_linear_kernel_recovers_ordinary_pca_s_coordinates(
    backend: ModuleType,
) -> None:
    KernelPrincipalComponentAnalysis = provided(
        backend, "KernelPrincipalComponentAnalysis"
    )
    model = KernelPrincipalComponentAnalysis(kernel=LinearKernel(), n_components=2)
    model.fit(ELLIPSE)

    transformed = model.transform(ELLIPSE)

    assert [feature.name for feature in transformed] == [
        "kernel_component_1",
        "kernel_component_2",
    ]
    assert np.allclose(
        np.abs(column_of(transformed, "kernel_component_1")),
        np.abs(FIRST_COORDINATES),
        atol=1e-06,
    )


def test_its_components_are_addressable_by_name(backend: ModuleType) -> None:
    KernelPrincipalComponentAnalysis = provided(
        backend, "KernelPrincipalComponentAnalysis"
    )
    components = (
        KernelPrincipalComponentAnalysis(n_components=1).fit(ELLIPSE).components
    )

    first = components["kernel_component_1"]

    assert first.n_training_rows == len(_FIRST)
    assert first.row_coefficients.shape == (len(_FIRST),)
    assert components.n_components == 1
    # The share is against every direction, kept and discarded, so a
    # truncated fit still reports 0.8 rather than claiming all of it.
    assert components.variance_shares == pytest.approx((0.8,), abs=1e-06)


def test_a_radial_basis_kernel_separates_the_rings(backend: ModuleType) -> None:
    KernelPrincipalComponentAnalysis = provided(
        backend, "KernelPrincipalComponentAnalysis"
    )
    model = KernelPrincipalComponentAnalysis(
        kernel=RadialBasisKernel(gamma=0.05), n_components=2
    ).fit(RINGS)

    first = column_of(model.transform(RINGS), "kernel_component_1")
    inner = first[RING_GROUPS == 0]
    outer = first[RING_GROUPS == 1]

    assert inner.max() < outer.min() or outer.max() < inner.min()


def separating_components(transformed: list[Feature]) -> list[str]:
    """The names of the components along which the two rings do not overlap."""
    return [
        feature.name
        for feature in transformed
        if feature.values[RING_GROUPS == 0].max()
        < feature.values[RING_GROUPS == 1].min()
        or feature.values[RING_GROUPS == 1].max()
        < feature.values[RING_GROUPS == 0].min()
    ]


def test_a_degree_two_polynomial_kernel_separates_the_rings_where_the_linear_cannot(
    backend: ModuleType,
) -> None:
    """The degree-2 feature map holds ``x^2 + y^2``, which is 1 on one ring and
    25 on the other, so the rings are separable along a direction the implied
    space contains. It is not the leading direction: by the rings' symmetry
    the pair ``x^2 - y^2`` and ``2xy`` share a larger variance and come
    first, so the separation appears on the third component. No direction of
    the plane itself separates a ring from the ring around it, which is what
    the linear kernel shows.

    This is the claim that catches a wrapper quietly handing every kernel to
    the engine as a linear one, which the acceptance test below cannot.
    """
    KernelPrincipalComponentAnalysis = provided(
        backend, "KernelPrincipalComponentAnalysis"
    )
    polynomial = KernelPrincipalComponentAnalysis(
        kernel=PolynomialKernel(degree=2), n_components=3
    ).fit(RINGS)
    linear = KernelPrincipalComponentAnalysis(
        kernel=LinearKernel(), n_components=2
    ).fit(RINGS)

    assert separating_components(polynomial.transform(RINGS)) == ["kernel_component_3"]
    assert separating_components(linear.transform(RINGS)) == []


@pytest.mark.parametrize(
    "kernel", EVERY_KERNEL, ids=[type(one).__name__ for one in EVERY_KERNEL]
)
def test_every_kernel_of_the_library_is_accepted(
    backend: ModuleType, kernel: Kernel
) -> None:
    KernelPrincipalComponentAnalysis = provided(
        backend, "KernelPrincipalComponentAnalysis"
    )
    model = KernelPrincipalComponentAnalysis(kernel=kernel, n_components=2).fit(RINGS)

    transformed = model.transform(RINGS)

    assert 1 <= len(transformed) <= 2
    assert all(np.all(np.isfinite(feature.values)) for feature in transformed)
    assert all(len(feature.values) == len(RING_GROUPS) for feature in transformed)


def test_writing_into_the_learned_coefficients_does_not_move_them(
    backend: ModuleType,
) -> None:
    """A fitted model's answers do not change because a caller wrote into an
    array it handed out. Both routes to the row coefficients are writeable
    float arrays and both are written into here.

    ``row_coefficients`` is the assertion that discriminates, since it is the
    one buffer a component actually owns and hands out through a copy. With
    that copy removed the write lands in the fitted direction, and this test
    goes red on both backends.
    """
    KernelPrincipalComponentAnalysis = provided(
        backend, "KernelPrincipalComponentAnalysis"
    )
    model = KernelPrincipalComponentAnalysis(kernel=LinearKernel()).fit(ELLIPSE)
    before = column_of(model.transform(ELLIPSE), "kernel_component_1")
    coefficients = np.array(model.components["kernel_component_1"].row_coefficients)

    model.components.coefficients[:] = 999.0
    model.components["kernel_component_1"].row_coefficients[:] = 999.0

    assert np.allclose(
        column_of(model.transform(ELLIPSE), "kernel_component_1"), before
    )
    assert np.allclose(
        model.components["kernel_component_1"].row_coefficients, coefficients
    )


def test_a_refused_refit_leaves_the_earlier_fit_intact(backend: ModuleType) -> None:
    """Compute into locals, assign at the end, checked rather than intended.

    A refit that raises must leave the model as the last successful fit left
    it, rather than half replaced or unfitted. This model's shared refusal is
    the row count, since a kernel is a construction field and the engine's own
    refusal of an indefinite Gram matrix has no counterpart on the numpy
    backend; that deeper one is pinned in
    ``test/scikit/test_unsupervised_translation.py``.
    """
    KernelPrincipalComponentAnalysis = provided(
        backend, "KernelPrincipalComponentAnalysis"
    )
    model = KernelPrincipalComponentAnalysis(kernel=LinearKernel()).fit(ELLIPSE)
    before = column_of(model.transform(ELLIPSE), "kernel_component_1")

    with pytest.raises(TooFewValuesError):
        model.fit([Feature("first", [1.0]), Feature("second", [2.0])])

    assert model.is_fitted
    assert model.components.n_components == 2
    assert [one.variance for one in model.components] == pytest.approx(
        list(VARIANCES), abs=1e-06
    )
    assert np.allclose(
        column_of(model.transform(ELLIPSE), "kernel_component_1"), before
    )


def test_it_refuses_more_components_than_rows(backend: ModuleType) -> None:
    KernelPrincipalComponentAnalysis = provided(
        backend, "KernelPrincipalComponentAnalysis"
    )

    with pytest.raises(InvalidValuesError):
        KernelPrincipalComponentAnalysis(n_components=6).fit(ELLIPSE)


def test_it_refuses_a_single_row(backend: ModuleType) -> None:
    KernelPrincipalComponentAnalysis = provided(
        backend, "KernelPrincipalComponentAnalysis"
    )

    with pytest.raises(TooFewValuesError):
        KernelPrincipalComponentAnalysis().fit(
            [Feature("first", [1.0]), Feature("second", [2.0])]
        )


def test_it_refuses_a_query_over_the_wrong_features(backend: ModuleType) -> None:
    KernelPrincipalComponentAnalysis = provided(
        backend, "KernelPrincipalComponentAnalysis"
    )
    model = KernelPrincipalComponentAnalysis().fit(ELLIPSE)

    with pytest.raises(InvalidValuesError):
        model.transform([ELLIPSE[0]])


def test_it_refuses_to_transform_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    KernelPrincipalComponentAnalysis = provided(
        backend, "KernelPrincipalComponentAnalysis"
    )

    with pytest.raises(NotFittedError):
        KernelPrincipalComponentAnalysis().transform(ELLIPSE)

    with pytest.raises(NotFittedError):
        _ = KernelPrincipalComponentAnalysis().components
