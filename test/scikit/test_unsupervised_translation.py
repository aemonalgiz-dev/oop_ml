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
from sklearn.cluster import KMeans as EngineKMeans
from sklearn.neural_network import BernoulliRBM

from oop_ml import Feature, scikit
from oop_ml import numpy as reference
from oop_ml.core.exceptions import AllSameValuesError, InvalidValuesError
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
