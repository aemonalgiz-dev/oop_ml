"""Where the scikit-learn classification wrappers translate, and that it holds.

The contract suite checks each backend against a fixture's known answer and
never against the other backend. This file is the other half. Where the two
backends fit the same objective, the unpenalised likelihood, they are asked
to agree on the coefficients themselves, because a wrapper that quietly left
a penalty on would still fit, still score well and still pass a loose
contract. The refusals this backend adds are pinned here too, since they
have no counterpart in the numpy backend and so no place in the contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from oop_ml import Feature, scikit
from oop_ml import numpy as reference
from oop_ml.core.data.row_block import RowBlock
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.kernel.functions import Kernel
from oop_ml.core.tree.criterion import ClassificationCriterion
from oop_ml.core.tree.node import DecisionNode

#: The eight students, whose maximum likelihood fit both backends must reach.
STUDENT_FEATURES = [Feature("hours", [0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0])]
STUDENT_TARGET = Feature("passed", [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])

#: Three overlapping runs along one centred feature.
_POSITIONS = np.array(
    [-5.0, -4.0, -3.0, -2.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0]
)
BAND_FEATURES = [Feature("position", _POSITIONS)]
BAND_TARGET = Feature("band", [0.0] * 5 + [1.0] * 5 + [2.0] * 5)

#: Two clusters with a gap, small enough for the numpy ascent to be stable.
CLUSTER_FEATURES = [
    Feature("across", [0.0, 1.0, 0.0, 1.0, 3.0, 4.0, 3.0, 4.0]),
    Feature("up", [0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0]),
]
CLUSTER_TARGET = Feature("cluster", [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

#: Thirty rows over three classes, the third of them rare enough that a
#: bootstrap resample misses it routinely.
_GENERATOR = np.random.default_rng(3)
_ROWS = _GENERATOR.normal(size=(30, 2))
_RARE = np.array([0.0] * 14 + [1.0] * 14 + [2.0] * 2)
IMBALANCED_FEATURES = [Feature("left", _ROWS[:, 0]), Feature("right", _ROWS[:, 1])]
IMBALANCED_TARGET = Feature("kind", _RARE)

#: Forty rows over three features, the class set by the first two, so a
#: bagged tree has several splits to credit and a third feature to leave at
#: zero.
_THREE = np.random.default_rng(5).normal(size=(40, 3))
THREE_FEATURES = [Feature(name, _THREE[:, index]) for index, name in enumerate("abc")]
THREE_FEATURE_TARGET = Feature(
    "side", (_THREE[:, 0] + 0.5 * _THREE[:, 1] > 0.0).astype(np.float64)
)


class TestTheUnpenalisedLikelihood:
    """``C = inf`` is the translation, and the two backends share a maximum."""

    def test_the_gradient_model_reaches_the_numpy_coefficients(self) -> None:
        expected = reference.LogisticRegression(max_epochs=100_000).fit(
            STUDENT_FEATURES, STUDENT_TARGET
        )
        wrapped = scikit.LogisticRegression(max_epochs=100_000).fit(
            STUDENT_FEATURES, STUDENT_TARGET
        )

        assert wrapped.intercept == pytest.approx(expected.intercept, abs=1e-4)
        assert wrapped.coefficients["hours"] == pytest.approx(
            expected.coefficients["hours"], abs=1e-4
        )

    def test_the_newton_model_reaches_the_numpy_coefficients(self) -> None:
        expected = reference.NewtonLogisticRegression().fit(
            STUDENT_FEATURES, STUDENT_TARGET
        )
        wrapped = scikit.NewtonLogisticRegression().fit(
            STUDENT_FEATURES, STUDENT_TARGET
        )

        assert wrapped.intercept == pytest.approx(expected.intercept, abs=1e-6)
        assert wrapped.coefficients["hours"] == pytest.approx(
            expected.coefficients["hours"], abs=1e-6
        )

    def test_without_an_intercept_the_engine_is_not_handed_a_ones_column(
        self,
    ) -> None:
        expected = reference.NewtonLogisticRegression(fit_intercept=False).fit(
            STUDENT_FEATURES, STUDENT_TARGET
        )
        wrapped = scikit.NewtonLogisticRegression(fit_intercept=False).fit(
            STUDENT_FEATURES, STUDENT_TARGET
        )

        assert wrapped.intercept == 0.0
        assert wrapped.coefficients["hours"] == pytest.approx(
            expected.coefficients["hours"], abs=1e-6
        )

    def test_the_multinomial_model_is_read_against_the_reference_class(
        self,
    ) -> None:
        """The engine's parameters sit anywhere along the ridge; subtracting
        class 0's row lands them on the numpy backend's pinned form."""
        expected = reference.MultinomialLogisticRegression(max_epochs=100_000).fit(
            BAND_FEATURES, BAND_TARGET
        )
        wrapped = scikit.MultinomialLogisticRegression(max_epochs=100_000).fit(
            BAND_FEATURES, BAND_TARGET
        )

        assert np.allclose(wrapped.intercepts, expected.intercepts, atol=1e-3)
        for class_index in range(3):
            assert wrapped.coefficients_for(class_index)["position"] == pytest.approx(
                expected.coefficients_for(class_index)["position"], abs=1e-3
            )
        assert np.allclose(
            np.asarray(wrapped.predict_probabilities(BAND_FEATURES)),
            np.asarray(expected.predict_probabilities(BAND_FEATURES)),
            atol=1e-3,
        )

    def test_a_two_class_multinomial_is_read_from_the_engine_s_binary_fit(
        self,
    ) -> None:
        """With two classes the engine fits one weight vector, which is
        already the difference against class 0."""
        wrapped = scikit.MultinomialLogisticRegression(max_epochs=100_000).fit(
            STUDENT_FEATURES, STUDENT_TARGET
        )
        binary = scikit.LogisticRegression(max_epochs=100_000).fit(
            STUDENT_FEATURES, STUDENT_TARGET
        )

        assert wrapped.n_classes == 2
        assert wrapped.coefficients_for(0)["hours"] == 0.0
        assert wrapped.coefficients_for(1)["hours"] == pytest.approx(
            binary.coefficients["hours"], abs=1e-4
        )
        assert wrapped.intercepts[1] == pytest.approx(binary.intercept, abs=1e-4)

    def test_the_gradient_model_without_an_intercept_reaches_the_numpy_slope(
        self,
    ) -> None:
        """Newton's no-intercept case was pinned and this one was not, and
        a wrapper that dropped ``fit_intercept`` on the way to the L-BFGS
        engine passed every test there was."""
        expected = reference.LogisticRegression(fit_intercept=False).fit(
            STUDENT_FEATURES, STUDENT_TARGET
        )
        wrapped = scikit.LogisticRegression(fit_intercept=False).fit(
            STUDENT_FEATURES, STUDENT_TARGET
        )

        assert wrapped.intercept == 0.0
        assert wrapped.coefficients["hours"] == pytest.approx(
            expected.coefficients["hours"], abs=1e-4
        )

    def test_the_multinomial_model_without_intercepts_reaches_the_numpy_slopes(
        self,
    ) -> None:
        """``fit_intercept`` has to reach the engine, not only zero the
        biases on the way out: with the engine's intercept left free the
        slopes come back at 1.583 and 3.166 where the pinned fit has 0.712
        and 1.423."""
        expected = reference.MultinomialLogisticRegression(
            fit_intercept=False, max_epochs=100_000
        ).fit(BAND_FEATURES, BAND_TARGET)
        wrapped = scikit.MultinomialLogisticRegression(fit_intercept=False).fit(
            BAND_FEATURES, BAND_TARGET
        )

        assert np.array_equal(wrapped.intercepts, np.zeros(3))
        for class_index in range(3):
            assert wrapped.coefficients_for(class_index)["position"] == pytest.approx(
                expected.coefficients_for(class_index)["position"], abs=1e-3
            )


class TestTheStepSizeRefusal:
    """A ``learning_rate`` the engine never reads is refused, not ignored."""

    @pytest.mark.parametrize(
        "model_type",
        [
            scikit.LogisticRegression,
            scikit.MultinomialLogisticRegression,
            scikit.SupportVectorClassifier,
        ],
    )
    def test_a_configured_learning_rate_is_refused(self, model_type: type) -> None:
        with pytest.raises(InvalidValuesError):
            model_type(learning_rate=0.5)

    @pytest.mark.parametrize(
        "model_type",
        [
            scikit.LogisticRegression,
            scikit.MultinomialLogisticRegression,
            scikit.SupportVectorClassifier,
        ],
    )
    def test_the_default_passes_so_a_candidate_can_be_rebuilt(
        self, model_type: type
    ) -> None:
        configured = {
            name: getattr(model_type(), name) for name in model_type.model_fields
        }

        assert model_type(**configured).learning_rate == model_type().learning_rate


class TestTheSupportVectorTranslation:
    """``C = capacity``, the kernel by name, and the boundary in the gap."""

    def test_both_backends_separate_the_clusters_identically(self) -> None:
        expected = reference.SupportVectorClassifier().fit(
            CLUSTER_FEATURES, CLUSTER_TARGET
        )
        wrapped = scikit.SupportVectorClassifier().fit(CLUSTER_FEATURES, CLUSTER_TARGET)

        assert np.array_equal(
            np.asarray(wrapped.predict(CLUSTER_FEATURES)),
            np.asarray(expected.predict(CLUSTER_FEATURES)),
        )
        assert np.array_equal(
            np.sign(wrapped.decision_values(CLUSTER_FEATURES)),
            np.sign(expected.decision_values(CLUSTER_FEATURES)),
        )

    def test_the_multipliers_are_read_back_from_the_dual_coefficients(self) -> None:
        wrapped = scikit.SupportVectorClassifier().fit(CLUSTER_FEATURES, CLUSTER_TARGET)

        engine = wrapped._engine
        assert engine is not None
        for position, magnitude in zip(
            engine.support_, np.abs(engine.dual_coef_).ravel(), strict=True
        ):
            assert wrapped.multipliers[position] == pytest.approx(magnitude)
        assert wrapped.support_vectors.n_vectors == len(engine.support_)

    def test_a_kernel_the_engine_cannot_be_handed_is_refused(self) -> None:
        class SquaredLinearKernel(Kernel):
            @property
            def description(self) -> str:
                return "(a . b) ** 2"

            def _between_blocks(self, left: RowBlock, right: RowBlock) -> np.ndarray:
                return (left.values @ right.values.T) ** 2

        with pytest.raises(InvalidValuesError):
            scikit.SupportVectorClassifier(kernel=SquaredLinearKernel()).fit(
                CLUSTER_FEATURES, CLUSTER_TARGET
            )


class TestTheEnsembleMembers:
    """A bagging engine needs a member that can hand it a scikit-learn prototype."""

    def test_a_member_from_the_numpy_backend_is_refused_at_construction(self) -> None:
        with pytest.raises(InvalidValuesError):
            scikit.BaggingClassifier(base_model=reference.DecisionTreeClassifier())

    def test_a_binary_model_from_the_numpy_backend_is_refused_at_construction(
        self,
    ) -> None:
        with pytest.raises(InvalidValuesError):
            scikit.OneVsRestClassifier(binary_model=reference.LogisticRegression())

    def test_every_member_is_told_the_class_width(self) -> None:
        """A resample of thirty rows misses a two-row class most of the time.
        A member left to infer its own width would answer narrower than its
        siblings; told the width, every one answers three columns."""
        model = scikit.BaggingClassifier(n_members=12, random_seed=0).fit(
            IMBALANCED_FEATURES, IMBALANCED_TARGET
        )

        for member in model.members:
            assert isinstance(member, scikit.DecisionTreeClassifier)
            assert member.n_known_classes == 3
            assert member.n_classes == 3
            assert member.predict_probabilities(IMBALANCED_FEATURES).shape == (30, 3)
        assert model.predict_probabilities(IMBALANCED_FEATURES).shape == (30, 3)
        assert model.out_of_bag_evaluate().n_classes == 3

    def test_a_neighbour_member_keeps_the_rows_it_was_fitted_on(self) -> None:
        model = scikit.BaggingClassifier(
            base_model=scikit.KNearestNeighboursClassifier(n_neighbours=3),
            n_members=4,
            random_seed=0,
        ).fit(CLUSTER_FEATURES, CLUSTER_TARGET)

        for member in model.members:
            assert isinstance(member, scikit.KNearestNeighboursClassifier)
            assert member.n_remembered == 8
        assert model.score(CLUSTER_FEATURES, CLUSTER_TARGET) == pytest.approx(1.0)

    def test_an_adopted_member_carries_the_seed_the_engine_gave_it(self) -> None:
        model = scikit.RandomForestClassifier(
            n_members=4, max_features=1, random_seed=0
        ).fit(IMBALANCED_FEATURES, IMBALANCED_TARGET)

        seeds = []
        for member in model.members:
            assert isinstance(member, scikit.DecisionTreeClassifier)
            seeds.append(member.random_seed)

        assert len(set(seeds)) == 4
        assert seeds == [tree.random_state for tree in model._engine.estimators_]


class TestTheConvertedTree:
    """The engine's leaves come back as classification leaves of full width."""

    def test_a_pure_leaf_reports_its_class_with_certainty(self) -> None:
        model = scikit.DecisionTreeClassifier(max_depth=1).fit(
            CLUSTER_FEATURES, CLUSTER_TARGET
        )

        for row, cluster in zip(
            range(8), np.asarray(CLUSTER_TARGET.values), strict=True
        ):
            leaf = model.root.leaf_for(
                np.array(
                    [CLUSTER_FEATURES[0].values[row], CLUSTER_FEATURES[1].values[row]]
                )
            )
            shares = getattr(leaf, "class_shares")  # noqa: B009
            assert np.allclose(shares, np.eye(2)[int(cluster)])

    def test_the_criterion_reaches_the_engine(self) -> None:
        """Four rows a side, so the root's entropy is one bit exactly, which
        is the engine's unit, and its Gini impurity is one half."""
        by_entropy = scikit.DecisionTreeClassifier(
            criterion=ClassificationCriterion.ENTROPY, max_depth=1
        ).fit(CLUSTER_FEATURES, CLUSTER_TARGET)
        by_gini = scikit.DecisionTreeClassifier(max_depth=1).fit(
            CLUSTER_FEATURES, CLUSTER_TARGET
        )

        entropy_root, gini_root = by_entropy.root, by_gini.root
        assert isinstance(entropy_root, DecisionNode)
        assert isinstance(gini_root, DecisionNode)
        assert entropy_root.impurity == pytest.approx(1.0)
        assert gini_root.impurity == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "ensemble",
        [
            scikit.BaggingClassifier(n_members=5, random_seed=0),
            scikit.RandomForestClassifier(n_members=5, max_features=2, random_seed=0),
        ],
        ids=["bagging", "forest"],
    )
    def test_a_bagged_member_s_importances_are_the_engine_s_own(
        self, ensemble: scikit.BaggingClassifier
    ) -> None:
        """The engine fits a bagged member on every row weighted by how often
        the resample drew it, so a node's distinct-row count and its weighted
        count differ, and only the weighted one matches the impurities.
        Read the wrong one and five members came back off the engine's own
        importances by up to 0.029, with 27 rows at a root that drew 40."""
        model = ensemble.fit(THREE_FEATURES, THREE_FEATURE_TARGET)

        for member, engine in zip(
            model.members, model._engine.estimators_, strict=True
        ):
            assert isinstance(member, scikit.DecisionTreeClassifier)
            assert member.root.n_samples == THREE_FEATURE_TARGET.column.n_samples
            ours = [member.feature_importances[name] for name in ("a", "b", "c")]
            assert np.allclose(ours, engine.feature_importances_, atol=1e-12)
