"""The contract every backend's GaussianMixture keeps.

The fourth clusterer, and the first that gives a row *shares* of several groups
rather than one group. That is the capability the contract is built around,
because it is the one thing no other clusterer here can do at all.

What is deliberately not asserted
----------------------------------
That it beats k-means on an elongated group. The model can describe one, having
a full covariance per component where k-means has none, but both backends start
their walk from a k-means grouping and expectation-maximisation climbs to a
local maximum, so a bad start is inherited rather than escaped. Measured on two
long flat bands stacked three apart, both backends and the engine at its own
default initialisation agree with the truth about 51% to 58% of the time, which
is chance; only a different starting point reaches the answer. The numpy spec
records that rather than the contract pretending otherwise.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml import Feature
from oop_ml.core.exceptions import NotFittedError, TooFewValuesError

from .harness import provided

_GENERATOR = np.random.default_rng(9)
_SOURCES = np.vstack(
    [
        _GENERATOR.multivariate_normal([0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]], size=80),
        _GENERATOR.multivariate_normal([6.0, 1.0], [[2.0, -1.2], [-1.2, 1.5]], size=60),
        _GENERATOR.multivariate_normal([2.0, 6.0], [[0.6, 0.0], [0.0, 0.6]], size=60),
    ]
)
#: Three Gaussians of visibly different shape and size, far enough apart that
#: the walk reaches them from a k-means start.
FEATURES = [Feature("left", _SOURCES[:, 0]), Feature("right", _SOURCES[:, 1])]
TRUE_SOURCES = np.array([0] * 80 + [1] * 60 + [2] * 60)

_CLUMP_GENERATOR = np.random.default_rng(31)
_CLUMPS = np.vstack(
    [
        _CLUMP_GENERATOR.normal(loc=[0.0, 0.0], scale=0.6, size=(60, 2)),
        _CLUMP_GENERATOR.normal(loc=[4.0, 0.0], scale=0.6, size=(60, 2)),
    ]
)
#: Two round clumps four apart, so a row halfway between them genuinely
#: belongs to neither more than the other.
CLUMP_FEATURES = [
    Feature("left", _CLUMPS[:, 0]),
    Feature("right", _CLUMPS[:, 1]),
]


def agreement(answered: object, truth: np.ndarray) -> float:
    """How often two labellings agree, allowing for the numbering.

    Cluster labels carry no meaning across fits, so the comparison takes the
    best of every way of matching one numbering onto the other. Three groups
    give six orderings, which is few enough to try them all.
    """
    from itertools import permutations

    found = np.asarray(answered).astype(int)
    n_groups = int(truth.max()) + 1

    return max(
        float((np.asarray(order)[found] == truth).mean())
        for order in permutations(range(n_groups))
    )


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    GaussianMixture = provided(backend, "GaussianMixture")

    model = GaussianMixture(n_components=4, covariance_smoothing=1e-5, random_seed=7)

    assert model.n_components == 4
    assert model.covariance_smoothing == pytest.approx(1e-5)
    assert model.random_seed == 7


def test_its_defaults_are_the_engine_s(backend: ModuleType) -> None:
    GaussianMixture = provided(backend, "GaussianMixture")

    model = GaussianMixture()

    assert model.n_components == 1
    assert model.max_iterations == 100
    assert model.tolerance == pytest.approx(1e-3)
    assert model.covariance_smoothing == pytest.approx(1e-6)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [("n_components", 0), ("tolerance", 0.0), ("covariance_smoothing", -1.0)],
)
def test_it_refuses_a_setting_that_describes_no_mixture(
    backend: ModuleType, field_name: str, invalid: float
) -> None:
    GaussianMixture = provided(backend, "GaussianMixture")

    with pytest.raises(ValidationError):
        GaussianMixture(**{field_name: invalid})


def test_it_fits_features_and_returns_itself(backend: ModuleType) -> None:
    GaussianMixture = provided(backend, "GaussianMixture")
    model = GaussianMixture(n_components=3, random_seed=0)

    assert model.fit(FEATURES) is model


def test_it_recovers_the_sources_the_fixture_was_drawn_from(
    backend: ModuleType,
) -> None:
    GaussianMixture = provided(backend, "GaussianMixture")
    model = GaussianMixture(n_components=3, random_seed=0).fit(FEATURES)

    assert agreement(model.predict(FEATURES), TRUE_SOURCES) > 0.95


def test_its_weights_are_the_shares_and_sum_to_one(backend: ModuleType) -> None:
    """80, 60 and 60 rows, so 0.4 and 0.3 and 0.3 to within what the overlap
    lets the walk resolve."""
    GaussianMixture = provided(backend, "GaussianMixture")
    model = GaussianMixture(n_components=3, random_seed=0).fit(FEATURES)

    weights = np.sort(np.asarray(model.weights))

    assert weights.sum() == pytest.approx(1.0)
    assert weights[-1] == pytest.approx(0.4, abs=0.05)


def test_every_component_has_a_shape_of_its_own(backend: ModuleType) -> None:
    """The thing k-means does not have, and the whole of why this can describe
    a tilted or elongated group. The three sources were drawn with visibly
    different covariances and the fit reports three different matrices."""
    GaussianMixture = provided(backend, "GaussianMixture")
    model = GaussianMixture(n_components=3, random_seed=0).fit(FEATURES)

    covariances = np.asarray(model.covariances)

    assert covariances.shape == (3, 2, 2)
    for component in covariances:
        assert np.allclose(component, component.T)
    assert not np.allclose(covariances[0], covariances[1])


def test_a_row_between_two_groups_is_divided_between_them(
    backend: ModuleType,
) -> None:
    """The reason to prefer this over k-means, and the answer no other
    clusterer here can give. A row halfway between two clumps comes back split
    rather than assigned to whichever is marginally nearer."""
    GaussianMixture = provided(backend, "GaussianMixture")
    model = GaussianMixture(n_components=2, random_seed=0).fit(CLUMP_FEATURES)

    between = [
        Feature("left", np.array([2.0, 0.0, 4.0])),
        Feature("right", np.array([0.0, 0.0, 0.0])),
    ]
    shares = np.asarray(model.predict_probabilities(between))

    assert np.allclose(shares.sum(axis=1), 1.0)
    assert 0.2 < shares[0].min() < 0.5
    assert shares[1].max() > 0.99
    assert shares[2].max() > 0.99


def test_the_hard_answer_is_the_largest_share(backend: ModuleType) -> None:
    GaussianMixture = provided(backend, "GaussianMixture")
    model = GaussianMixture(n_components=3, random_seed=0).fit(FEATURES)

    from_shares = np.argmax(
        np.asarray(model.predict_probabilities(FEATURES)), axis=1
    ).astype(float)

    assert np.array_equal(np.asarray(model.predict(FEATURES)), from_shares)


def test_more_components_explain_the_data_better(backend: ModuleType) -> None:
    """The likelihood is what the walk climbs, and adding a component can only
    help on the training rows, which is exactly why it cannot choose the
    number of them any more than inertia can for k-means."""
    GaussianMixture = provided(backend, "GaussianMixture")

    scores = [
        GaussianMixture(n_components=count, random_seed=0)
        .fit(CLUMP_FEATURES)
        .mean_log_likelihood
        for count in (1, 2, 3)
    ]

    assert scores == sorted(scores)


def test_it_reports_how_the_rounds_ended(backend: ModuleType) -> None:
    GaussianMixture = provided(backend, "GaussianMixture")
    model = GaussianMixture(n_components=3, random_seed=0).fit(FEATURES)

    assert model.converged is True
    assert 0 < model.iterations_run <= model.max_iterations


def test_it_refuses_more_components_than_there_are_rows(backend: ModuleType) -> None:
    GaussianMixture = provided(backend, "GaussianMixture")
    three_rows = [
        Feature("left", np.array([0.0, 1.0, 2.0])),
        Feature("right", np.array([0.0, 1.0, 4.0])),
    ]

    with pytest.raises(TooFewValuesError):
        GaussianMixture(n_components=4).fit(three_rows)


def test_it_matches_query_columns_by_name(backend: ModuleType) -> None:
    GaussianMixture = provided(backend, "GaussianMixture")
    model = GaussianMixture(n_components=3, random_seed=0).fit(FEATURES)

    reversed_order = [FEATURES[1], FEATURES[0]]

    assert np.array_equal(
        np.asarray(model.predict(reversed_order)), np.asarray(model.predict(FEATURES))
    )


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    GaussianMixture = provided(backend, "GaussianMixture")

    with pytest.raises(NotFittedError):
        GaussianMixture().predict(FEATURES)


@pytest.mark.parametrize(
    "summary",
    ["weights", "means", "covariances", "mean_log_likelihood", "converged"],
)
def test_it_refuses_to_report_its_fit_before_fit(
    backend: ModuleType, summary: str
) -> None:
    GaussianMixture = provided(backend, "GaussianMixture")

    with pytest.raises(NotFittedError):
        getattr(GaussianMixture(), summary)
