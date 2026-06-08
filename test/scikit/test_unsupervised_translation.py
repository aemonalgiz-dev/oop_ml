"""Where the scikit-learn unsupervised wrappers translate, and that it holds.

The contract suite checks each backend against a fixture's known answer and
never against the other backend. This file is the other half, as for the
regression and classification families: every translation that changes a
scale or drives the engine differently from its own ``fit`` is pinned by
fitting both sides on one fixture and asking them to agree, because a wrong
scale still fits, still looks reasonable, and still passes a loose contract.

Three things are pinned here that the contract cannot reach. The k-means
tolerance, which is relative on one side and absolute on the other, and whose
translation is only checkable from an identical start that the wrapper does
not expose, so it is checked against the engine directly. The kernel
decomposition's two scales, which the contract's linear-kernel control checks
on one kernel and this file checks on every kernel. And the Boltzmann walk,
which the wrapper drives one ``partial_fit`` at a time and which must
therefore reproduce the engine's own ``fit`` exactly, or the tolerance and
the schedule have been bought at the price of a different fit.

The one refusal this backend adds is pinned here too, since the numpy backend
accepts the same call and so the contract has no place for it. So are the
three refusals both backends make and word differently, because a contract
asserting an exception type cannot see which noun the sentence uses, and the
noun is the whole of what a reader gets.

Two atomicity claims sit here rather than in the contract for the same reason,
that they hold on one backend and not the other. Both are refits refused after
the engine has already run, which is the window the compute-into-locals rule
exists to close, and neither has a numpy counterpart. On the first the numpy
backend does not refuse the call at all; on the second it refuses and does not
stay intact.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.cluster import DBSCAN as EngineDBSCAN
from sklearn.cluster import AgglomerativeClustering as EngineAgglomerative
from sklearn.cluster import KMeans as EngineKMeans
from sklearn.neural_network import BernoulliRBM

from oop_ml import Feature, scikit
from oop_ml import numpy as reference
from oop_ml.core.clustering.linkage import Linkage
from oop_ml.core.distance.calculations import MinkowskiDistance
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.exceptions import (
    AllSameValuesError,
    InvalidValuesError,
    NotFittedError,
)
from oop_ml.core.kernel.functions import (
    Kernel,
    LinearKernel,
    PolynomialKernel,
    RadialBasisKernel,
    SigmoidKernel,
)
from oop_ml.core.schedule import ConstantSchedule
from oop_ml.scikit.unsupervised import engine_tolerance

_ANGLES = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
_UNIT = np.column_stack([np.cos(_ANGLES), np.sin(_ANGLES)])
_RINGS = np.vstack([_UNIT, 5.0 * _UNIT])
RINGS = [Feature("first", _RINGS[:, 0]), Feature("second", _RINGS[:, 1])]

#: Four wide blobs at a scale where the engine's relative tolerance and this
#: library's absolute one differ by six orders of magnitude.
_GENERATOR = np.random.default_rng(5)
_BLOBS = np.vstack(
    [
        _GENERATOR.normal(0.0, 300.0, (50, 2)) + centre
        for centre in [(0.0, 0.0), (2000.0, 0.0), (0.0, 2000.0), (1500.0, 1500.0)]
    ]
)
_START = np.array([[100.0, 100.0], [1900.0, 100.0], [100.0, 1900.0], [1400.0, 1400.0]])

_BITS = np.random.default_rng(3)
_BINARY = (_BITS.random((40, 6)) < 0.5).astype(np.float64)
BINARY = [Feature(name, _BINARY[:, position]) for position, name in enumerate("abcdef")]


def block_of(features: list[Feature]) -> np.ndarray:
    return np.column_stack([feature.values for feature in features])


_DENSITY_GENERATOR = np.random.default_rng(31)
_DENSITY_ROWS = np.vstack(
    [
        _DENSITY_GENERATOR.normal(loc=[0.0, 0.0], scale=0.6, size=(30, 2)),
        _DENSITY_GENERATOR.normal(loc=[5.0, 0.0], scale=0.6, size=(25, 2)),
        _DENSITY_GENERATOR.normal(loc=[2.5, 5.0], scale=0.6, size=(20, 2)),
        _DENSITY_GENERATOR.uniform(low=-6.0, high=11.0, size=(12, 2)),
    ]
)
#: Three blobs and a dozen scattered rows, so a fit has core points, border
#: points and noise to get right rather than only groups.
DENSITY_FEATURES = [
    Feature("left", _DENSITY_ROWS[:, 0]),
    Feature("right", _DENSITY_ROWS[:, 1]),
]


_MERGE_ROWS = np.array(
    [
        [0.0, 0.0],
        [0.2, 0.0],
        [0.0, 0.3],
        [5.0, 5.0],
        [5.3, 5.0],
        [5.0, 5.4],
        [10.0, 0.0],
    ]
)
#: Two threes and a stray, small enough that the whole merge sequence is short
#: and every height is checkable by hand.
MERGE_FEATURES = [
    Feature("left", _MERGE_ROWS[:, 0]),
    Feature("right", _MERGE_ROWS[:, 1]),
]


class TestTheKMeansToleranceScale:
    """``tol = tolerance / mean(var(X))`` lands the engine on the absolute rule."""

    def test_the_translation_undoes_the_engine_s_scaling(self) -> None:
        assert engine_tolerance(_BLOBS, 1e-8) * float(
            np.mean(np.var(_BLOBS, axis=0))
        ) == pytest.approx(1e-8)

    def test_data_with_no_variance_passes_the_tolerance_through(self) -> None:
        assert engine_tolerance(np.ones((4, 2)), 0.5) == 0.5

    @pytest.mark.parametrize("tolerance", [1e-8, 1e2, 1e4, 1e6])
    def test_the_engine_stops_when_the_absolute_rule_would(
        self, tolerance: float
    ) -> None:
        """From one fixed start, the engine under the translated tolerance
        takes the same number of passes as Lloyd's loop under this library's
        rule, at every threshold from far below the movement to far above it.
        The engine sums the squared shifts where the library takes the
        largest, so the engine may run one pass longer; on these blobs it
        does not, and the counts are asserted equal."""
        # The untyped engine reads ``init="k-means++"`` as the parameter's type.
        engine_type: Any = EngineKMeans
        engine = engine_type(
            n_clusters=4,
            init=_START,
            n_init=1,
            tol=engine_tolerance(_BLOBS, tolerance),
            max_iter=300,
        ).fit(_BLOBS)

        positions = _START.copy()
        passes = 0
        while passes < 300:
            passes += 1
            gaps = _BLOBS[:, None, :] - positions[None, :, :]
            labels = np.sum(gaps * gaps, axis=2).argmin(axis=1)
            moved = np.array(
                [_BLOBS[labels == group].mean(axis=0) for group in range(4)]
            )
            shift = float(np.max(np.sum((moved - positions) ** 2, axis=1)))
            positions = moved
            if shift <= tolerance:
                break

        assert int(engine.n_iter_) == passes


class TestTheKernelDecompositionScales:
    """Variance is ``eigenvalue / (n - 1)`` and coefficients are over ``sqrt(eigenvalue)``.

    Both scales are checked on every kernel by agreement with the numpy
    backend, up to the sign of a direction and the rotation of a degenerate
    pair. The rings are symmetric, so several eigenvalues repeat and the two
    solvers legitimately pick different bases inside the repeated subspace;
    what is invariant is the Gram matrix of the coordinates, ``T T'``, which
    is asserted instead of the columns.
    """

    @pytest.mark.parametrize(
        "kernel",
        [
            LinearKernel(),
            PolynomialKernel(degree=2),
            RadialBasisKernel(gamma=0.05),
            SigmoidKernel(gamma=1e-4, constant=0.0),
        ],
        ids=["linear", "polynomial", "radial", "sigmoid"],
    )
    def test_both_backends_report_the_same_variances_and_span(
        self, kernel: Kernel
    ) -> None:
        expected = reference.KernelPrincipalComponentAnalysis(
            kernel=kernel, n_components=3
        ).fit(RINGS)
        wrapped = scikit.KernelPrincipalComponentAnalysis(
            kernel=kernel, n_components=3
        ).fit(RINGS)

        assert [one.variance for one in wrapped.components] == pytest.approx(
            [one.variance for one in expected.components], abs=1e-9
        )
        assert wrapped.components.total_variance == pytest.approx(
            expected.components.total_variance, abs=1e-9
        )

        expected_block = block_of(expected.transform(RINGS))
        wrapped_block = block_of(wrapped.transform(RINGS))

        assert np.allclose(
            wrapped_block @ wrapped_block.T,
            expected_block @ expected_block.T,
            atol=1e-9,
        )


class TestTheBoltzmannWalk:
    """One ``partial_fit`` per epoch is the engine's ``fit``, and it is measurable."""

    def test_the_walk_reproduces_the_engine_s_own_fit_bit_for_bit(self) -> None:
        """The wrapper seeds the engine's state and then drives it an epoch at
        a time. If that is the engine's ``fit``, the weights are identical to
        the bit; a different initial draw, a different stream, or a batch of
        the wrong size would all show here."""
        wrapped = scikit.RestrictedBoltzmannMachine(
            n_hidden_units=3,
            learning_rate=ConstantSchedule(value=0.1),
            max_epochs=25,
            random_seed=0,
        ).fit(BINARY)
        engine = BernoulliRBM(
            n_components=3,
            learning_rate=0.1,
            batch_size=40,
            n_iter=25,
            random_state=0,
        ).fit(_BINARY)

        assert np.array_equal(wrapped.weights, engine.components_.T)
        assert np.array_equal(wrapped.visible_bias, engine.intercept_visible_)
        assert np.array_equal(wrapped.hidden_bias, engine.intercept_hidden_)

    def test_the_forward_pass_agrees_with_the_engine_s(self) -> None:
        """The wrapper answers from its weights rather than from the engine,
        and the two arithmetics are the same logistic."""
        wrapped = scikit.RestrictedBoltzmannMachine(
            n_hidden_units=3, max_epochs=25, random_seed=0
        ).fit(BINARY)
        engine = BernoulliRBM(
            n_components=3, learning_rate=0.1, batch_size=40, n_iter=25, random_state=0
        ).fit(_BINARY)

        assert np.allclose(
            block_of(wrapped.transform(BINARY)), engine.transform(_BINARY), atol=1e-12
        )

    def test_a_tolerance_above_the_movement_stops_the_walk_early(self) -> None:
        """The field the engine's ``fit`` could not honour, honoured."""
        model = scikit.RestrictedBoltzmannMachine(
            n_hidden_units=3, max_epochs=50, random_seed=0, tolerance=1.0
        ).fit(BINARY)

        assert model.converged is True
        assert model.epochs_run == 1


class TestTheDecompositionRefusal:
    """More components than rows is padded by numpy and refused here."""

    def test_the_refusal_is_the_library_s_own(self) -> None:
        few = [
            Feature(name, values)
            for name, values in zip(
                "abcd", [[1.0, 2.0], [3.0, 1.0], [0.0, 5.0], [2.0, 2.0]], strict=True
            )
        ]

        assert reference.PrincipalComponentAnalysis(n_components=3).fit(few)

        with pytest.raises(InvalidValuesError):
            scikit.PrincipalComponentAnalysis(n_components=3).fit(few)


#: Two features whose names are neither a component's nor a hidden unit's.
WRONG_NAMES = [Feature("alpha", [1.0, 0.0]), Feature("beta", [0.0, 1.0])]

#: Four binary rows over two visible units, enough to fit a small machine.
SMALL_BITS = [
    Feature("alpha", [1.0, 0.0, 1.0, 0.0]),
    Feature("beta", [0.0, 1.0, 0.0, 1.0]),
]


def refusal_message(call: Any) -> str:
    """The text of the ``InvalidValuesError`` a call raises."""
    with pytest.raises(InvalidValuesError) as raised:
        call()

    return str(raised.value)


class TestTheNameRefusalsUseTheSameNouns:
    """Three questions run through one comparison here, and each keeps its noun.

    The numpy backend writes three refusals for the three questions and this
    backend writes one and hands it the noun, so the two must produce the same
    sentence. A contract test cannot see this, since both backends raise
    ``InvalidValuesError`` whichever noun is printed; what a reader gets is the
    noun, and a message calling a set of hidden units "the fitted features"
    sends them looking for training columns of that name.
    """

    def test_the_fitted_features_are_called_that_on_both(self) -> None:
        rows = [Feature("first", [1.0, 4.0, 2.0]), Feature("second", [2.0, 1.0, 5.0])]

        assert refusal_message(
            lambda: scikit.PrincipalComponentAnalysis().fit(rows).transform([rows[0]])
        ) == refusal_message(
            lambda: (
                reference.PrincipalComponentAnalysis().fit(rows).transform([rows[0]])
            )
        )

    def test_components_are_called_components_on_both(self) -> None:
        rows = [Feature("first", [1.0, 4.0, 2.0]), Feature("second", [2.0, 1.0, 5.0])]
        message = refusal_message(
            lambda: (
                scikit.PrincipalComponentAnalysis()
                .fit(rows)
                .inverse_transform(WRONG_NAMES)
            )
        )

        assert "this model's components" in message
        assert message == refusal_message(
            lambda: (
                reference.PrincipalComponentAnalysis()
                .fit(rows)
                .inverse_transform(WRONG_NAMES)
            )
        )

    @pytest.mark.parametrize("method", ["visible_probabilities", "sample_visible"])
    def test_hidden_units_are_called_hidden_units_on_both(self, method: str) -> None:
        def refuse(package: Any) -> str:
            model = package.RestrictedBoltzmannMachine(
                n_hidden_units=2, max_epochs=3, random_seed=0
            ).fit(SMALL_BITS)

            return refusal_message(lambda: getattr(model, method)(WRONG_NAMES))

        message = refuse(scikit)

        assert "this model's hidden units" in message
        assert message == refuse(reference)


class TestTheEngineKeepsItsOwnVocabulary:
    """An engine warning the library already has words for does not escape."""

    def test_the_empty_group_warning_does_not_reach_the_caller(self) -> None:
        """Six rows holding two distinct points, asked for four groups.

        The engine warns, in its own words, that it found fewer distinct
        clusters than it was told to find. The numpy backend says nothing on
        the same rows, because leaving a group's centre where it is and
        reporting an empty group is what this library does with one. Measured,
        both backends answer sizes ``[0, 0, 3, 3]``, an empty group and an
        inertia of 0.0, so the warning adds nothing the library has not said.
        """
        duplicated = [
            Feature("first", [0.0, 0.0, 5.0, 5.0, 0.0, 5.0]),
            Feature("second", [0.0, 0.0, 5.0, 5.0, 0.0, 5.0]),
        ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wrapped = scikit.KMeans(n_clusters=4, random_seed=0).fit(duplicated)
            expected = reference.KMeans(n_clusters=4, random_seed=0).fit(duplicated)

        assert caught == []
        assert wrapped.clustering.has_an_empty_cluster
        assert sorted(wrapped.clustering.sizes) == sorted(expected.clustering.sizes)
        assert wrapped.inertia == pytest.approx(expected.inertia)


class TestARefitRefusedAfterTheEngineRan:
    """The window compute-into-locals exists to close, on the two calls that reach it.

    Every other refusal in this family lands in a guard at the top of ``fit``,
    where nothing can have been assigned yet. These two land after the engine
    has done its work, and neither can be a contract test, since the numpy
    backend accepts the first call outright and on the second it refuses and
    does not stay intact.
    """

    def test_an_indefinite_gram_matrix_leaves_the_earlier_fit_intact(self) -> None:
        """The engine's own refusal, re-raised, after ``KernelPCA.fit`` ran.

        At ``gamma=1e-4`` the tanh is linear to within rounding on the rings
        and the engine accepts them; at five times the radius it is not, and
        the engine refuses eigenvalues it calls significantly negative. The
        numpy backend clamps them and fits either way.
        """
        model = scikit.KernelPrincipalComponentAnalysis(
            kernel=SigmoidKernel(gamma=1e-4, constant=0.0), n_components=2
        ).fit(RINGS)
        before = block_of(model.transform(RINGS))
        wider = [Feature(feature.name, feature.values * 5.0) for feature in RINGS]

        assert reference.KernelPrincipalComponentAnalysis(
            kernel=SigmoidKernel(gamma=1e-4, constant=0.0), n_components=2
        ).fit(wider)

        with pytest.raises(InvalidValuesError):
            model.fit(wider)

        assert model.is_fitted
        assert np.allclose(block_of(model.transform(RINGS)), before)

    def test_data_with_no_spread_leaves_the_earlier_fit_intact(self) -> None:
        """Both backends refuse this refit and only this one stays intact.

        The wrapper reads the total variance off the rows before it builds
        anything, so nothing has been assigned when the refusal lands. The
        numpy backend records the flat data's column means while preparing
        them and only then meets the refusal from its components, so its
        earlier fit is left with new means and old directions. Measured on
        these three rows, its first component comes back at 1.788854,
        -0.447214, 4.024922 where the fit answered 0.0, -2.236068, 2.236068,
        with no exception in sight. That is a defect in the reference backend
        rather than a divergence this wrapper chose, and it is why the shared
        half of this claim lives in the contract on a shallower refusal.
        """
        rows = [Feature("first", [1.0, 4.0, 2.0]), Feature("second", [2.0, 1.0, 5.0])]
        flat = [Feature("first", [3.0, 3.0, 3.0]), Feature("second", [1.0, 1.0, 1.0])]
        model = scikit.PrincipalComponentAnalysis().fit(rows)
        before = block_of(model.transform(rows))

        with pytest.raises(InvalidValuesError):
            model.fit(flat)

        assert model.is_fitted
        assert np.allclose(block_of(model.transform(rows)), before)

    def test_a_constant_column_is_refused_before_the_scaler_is_kept(self) -> None:
        """The standardizing route's own refusal, which both backends keep."""
        rows = [Feature("first", [1.0, 4.0, 2.0]), Feature("second", [2.0, 1.0, 5.0])]
        constant = [
            Feature("first", [1.0, 2.0, 3.0]),
            Feature("second", [4.0, 4.0, 4.0]),
        ]
        model = scikit.PrincipalComponentAnalysis(standardize=True).fit(rows)
        before = block_of(model.transform(rows))

        with pytest.raises(AllSameValuesError):
            model.fit(constant)

        assert np.allclose(block_of(model.transform(rows)), before)


class TestTheDensityTranslation:
    """``radius`` is the engine's ``eps``, ``min_neighbourhood_size`` its
    ``min_samples``, and ``metric`` goes through the same helper the neighbour
    models use. Nothing carries a scale factor, which is worth pinning anyway
    because the neighbourhood size is the kind of number that quietly means
    "and one more" on one side of a boundary."""

    @pytest.mark.parametrize(
        ("radius", "minimum"), [(0.8, 4), (1.0, 5), (1.5, 5), (2.0, 8)]
    )
    def test_both_backends_find_the_same_core_points(
        self, radius: float, minimum: int
    ) -> None:
        expected = reference.DBSCAN(radius=radius, min_neighbourhood_size=minimum).fit(
            DENSITY_FEATURES
        )
        wrapped = scikit.DBSCAN(radius=radius, min_neighbourhood_size=minimum).fit(
            DENSITY_FEATURES
        )

        assert np.array_equal(
            np.asarray(wrapped.core_indices), np.asarray(expected.core_indices)
        )
        assert np.array_equal(np.asarray(wrapped.labels), np.asarray(expected.labels))
        assert wrapped.n_clusters == expected.n_clusters
        assert wrapped.n_noise == expected.n_noise

    def test_the_neighbourhood_size_counts_the_row_itself_on_both_sides(self) -> None:
        """The off-by-one worth checking. Both this library and the engine
        count the row itself, so a lone pair of rows within the radius is
        enough at a size of two and not at three."""
        pair = [
            Feature("left", np.array([0.0, 0.1])),
            Feature("right", np.array([0.0, 0.0])),
        ]

        assert (
            scikit.DBSCAN(radius=1.0, min_neighbourhood_size=2).fit(pair).n_clusters
            == 1
        )
        assert (
            scikit.DBSCAN(radius=1.0, min_neighbourhood_size=3).fit(pair).n_noise == 2
        )

    def test_a_metric_the_enum_does_not_name_still_reaches_the_engine(self) -> None:
        """``MinkowskiDistance(3)`` becomes the engine's ``minkowski`` at
        ``p=3``, the same route the neighbour wrappers take, and the two
        backends still agree."""
        order_three = MinkowskiDistance(3)
        expected = reference.DBSCAN(
            radius=1.2, min_neighbourhood_size=4, metric=order_three
        ).fit(DENSITY_FEATURES)
        wrapped = scikit.DBSCAN(
            radius=1.2, min_neighbourhood_size=4, metric=order_three
        ).fit(DENSITY_FEATURES)

        assert np.array_equal(np.asarray(wrapped.labels), np.asarray(expected.labels))


class TestTheBorderConventionIsNotTheEngines:
    """The wrapper reads the engine's *core* points and their labels, and then
    labels every row itself. Where a border point goes is the one thing the
    algorithm leaves open, and taking the engine's answer for it would make the
    two backends disagree about rows neither is wrong about."""

    def test_the_labels_are_rebuilt_rather_than_read_through(self) -> None:
        """On this fixture the two answers happen to coincide, which is the
        honest state of it: nothing guarantees they will, and the reason to
        rebuild is that the engine's rule depends on the order the rows
        arrived in while this one does not."""
        engine = EngineDBSCAN(eps=1.0, min_samples=5).fit(np.asarray(_DENSITY_ROWS))
        wrapped = scikit.DBSCAN(radius=1.0, min_neighbourhood_size=5).fit(
            DENSITY_FEATURES
        )

        assert np.array_equal(
            np.asarray(wrapped.core_indices),
            np.asarray(engine.core_sample_indices_),
        )
        assert np.array_equal(
            np.asarray(wrapped.labels), np.asarray(engine.labels_, dtype=float)
        )

    def test_the_engine_cannot_label_a_row_it_never_saw(self) -> None:
        """Which is the other reason the rule lives in this library. The engine
        has no ``predict`` at all, so a fitted one is a labelling of the
        training rows and nothing more."""
        assert not hasattr(EngineDBSCAN, "predict")

        wrapped = scikit.DBSCAN(radius=1.0, min_neighbourhood_size=5).fit(
            DENSITY_FEATURES
        )
        unseen = [
            Feature("left", np.array([0.0, 40.0])),
            Feature("right", np.array([0.0, 40.0])),
        ]

        answered = np.asarray(wrapped.predict(unseen))

        assert answered[0] >= 0.0
        assert answered[1] == -1.0


class TestTheAgglomerativeTranslation:
    """``n_clusters`` and ``linkage`` pass through under their own names, and
    the metric goes by name. What needs pinning is the prefix: the engine
    always builds the whole tree, so its ``distances_`` is longer than this
    model's and only its first ``n - n_clusters`` entries are the merges a
    stop-early fit made."""

    @pytest.mark.parametrize("linkage", list(Linkage))
    @pytest.mark.parametrize("n_clusters", [2, 3, 4])
    def test_both_backends_reach_the_same_grouping_and_heights(
        self, linkage: Linkage, n_clusters: int
    ) -> None:
        expected = reference.AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage
        ).fit(MERGE_FEATURES)
        wrapped = scikit.AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage
        ).fit(MERGE_FEATURES)

        assert all(
            (expected.labels[one] == expected.labels[other])
            == (wrapped.labels[one] == wrapped.labels[other])
            for one in range(len(_MERGE_ROWS))
            for other in range(one + 1, len(_MERGE_ROWS))
        )
        assert np.allclose(
            np.asarray(wrapped.merge_distances),
            np.asarray(expected.merge_distances),
            atol=1e-12,
        )

    def test_the_wrapper_takes_the_prefix_of_the_engine_s_whole_tree(self) -> None:
        """The engine records ``n - 1`` heights whatever it was asked to cut
        at, because it builds the tree first and labels afterwards. Merging is
        greedy, so a stop-early fit's merges are that tree's first few."""
        engine = EngineAgglomerative(
            n_clusters=3, linkage="average", compute_distances=True
        ).fit(np.asarray(_MERGE_ROWS))
        wrapped = scikit.AgglomerativeClustering(
            n_clusters=3, linkage=Linkage.AVERAGE
        ).fit(MERGE_FEATURES)

        assert len(engine.distances_) == len(_MERGE_ROWS) - 1
        assert len(np.asarray(wrapped.merge_distances)) == len(_MERGE_ROWS) - 3
        assert np.allclose(
            np.asarray(wrapped.merge_distances),
            engine.distances_[: len(_MERGE_ROWS) - 3],
        )

    @pytest.mark.parametrize(
        "metric",
        [
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.MANHATTAN,
            DistanceMetric.CHEBYSHEV,
            DistanceMetric.CANBERRA,
        ],
    )
    def test_each_named_metric_reaches_the_engine_meaning_the_same_thing(
        self, metric: DistanceMetric
    ) -> None:
        expected = reference.AgglomerativeClustering(
            n_clusters=3, linkage=Linkage.AVERAGE, metric=metric
        ).fit(MERGE_FEATURES)
        wrapped = scikit.AgglomerativeClustering(
            n_clusters=3, linkage=Linkage.AVERAGE, metric=metric
        ).fit(MERGE_FEATURES)

        assert np.allclose(
            np.asarray(wrapped.merge_distances),
            np.asarray(expected.merge_distances),
            atol=1e-12,
        )


class TestTheMetricRefusalThisBackendAdds:
    """The numpy backend computes the pairwise block itself and takes any
    ``Distance``; the engine takes a metric by name and has no order parameter
    for a general p-norm."""

    def test_the_numpy_backend_accepts_a_distance_object(self) -> None:
        model = reference.AgglomerativeClustering(
            n_clusters=3, linkage=Linkage.AVERAGE, metric=MinkowskiDistance(3)
        ).fit(MERGE_FEATURES)

        assert model.n_clusters == 3

    def test_and_this_one_refuses_it_by_name(self) -> None:
        with pytest.raises(InvalidValuesError):
            scikit.AgglomerativeClustering(
                n_clusters=3, linkage=Linkage.AVERAGE, metric=MinkowskiDistance(3)
            ).fit(MERGE_FEATURES)

    def test_the_callable_route_the_neighbour_wrappers_use_does_not_exist_here(
        self,
    ) -> None:
        """Measured rather than assumed, and it is why the refusal is a refusal
        rather than a slower path. The engine hands the whole block to a
        callable metric instead of a pair of rows, and it has no ``algorithm``
        keyword to ask for the pairwise form."""

        def between_two_rows(left: Any, right: Any) -> float:
            return float(np.abs(np.asarray(left) - np.asarray(right)).max())

        # Through an Any alias, as the wrappers reach their engines, because
        # both calls below are deliberately the wrong shape and that is what
        # is being demonstrated.
        engine_type: Any = EngineAgglomerative

        with pytest.raises(TypeError):
            engine_type(n_clusters=2, linkage="average", metric=between_two_rows).fit(
                np.asarray(_MERGE_ROWS)
            )

        with pytest.raises(TypeError):
            engine_type(n_clusters=2, linkage="average", algorithm="brute")

    def test_a_refused_metric_leaves_the_model_unfitted(self) -> None:
        model = scikit.AgglomerativeClustering(
            n_clusters=3, linkage=Linkage.AVERAGE, metric=MinkowskiDistance(3)
        )

        with pytest.raises(InvalidValuesError):
            model.fit(MERGE_FEATURES)

        with pytest.raises(NotFittedError):
            _ = model.labels
