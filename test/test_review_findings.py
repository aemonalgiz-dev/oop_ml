"""Spec for the audit's behavioural findings, each pinned where it bit.

Cross-cutting like ``test_construction.py``, because these share one genre:
a fit or a configuration that *runs* and answers a different question than the
one asked. Each test's docstring records the failure it guards against; the
placement is together because the lesson is the genre, not any one module.
"""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.clustering.centroids import Centroid, Centroids
from oop_ml.core.clustering.clustering import Clustering
from oop_ml.core.data.dataset import Dataset
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.probabilities import ClassScores
from oop_ml.core.exceptions import (
    AllSameValuesError,
    DivergenceError,
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
)
from oop_ml.core.model_selection.splitting import KFold
from oop_ml.core.pipeline.pipelines import RegressionPipeline
from oop_ml.core.pipeline.steps import PipelineSteps
from oop_ml.numpy.classification.ensembles.random_forest_classifier import (
    RandomForestClassifier,
)
from oop_ml.numpy.classification.neighbours.k_nearest_classifier import (
    KNearestNeighboursClassifier,
)
from oop_ml.numpy.classification.trees.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from oop_ml.numpy.preprocessing.standardization.standardizer import Standardizer
from oop_ml.numpy.regression.ensembles.bagging_regressor import BaggingRegressor
from oop_ml.numpy.regression.ensembles.random_forest_regressor import (
    RandomForestRegressor,
)
from oop_ml.numpy.regression.least_squares.gradient_descent_regression import (
    GradientDescentRegression,
)
from oop_ml.numpy.regression.penalised.ridge_regression import RidgeRegression
from oop_ml.numpy.regression.trees.decision_tree_regressor import DecisionTreeRegressor
from test.fixtures import THREE_CLASSES


class TestDivergenceRefuses:
    """A diverged walk raises at the fit, not as nan three calls later."""

    def test_an_oversized_learning_rate_raises_by_name(self) -> None:
        """The weights overflow to non-finite and numpy raises nothing.

        The fit used to complete, report converged=False honestly, and hand
        back a model whose every prediction was nan -- and score() then
        returned nan silently through every aggregation above it.
        """
        with (
            np.errstate(over="ignore", invalid="ignore"),
            pytest.raises(DivergenceError, match="learning rate"),
        ):
            GradientDescentRegression(learning_rate=10.0, max_epochs=200).fit(
                [Feature("first", [1.0, 2.0, 3.0, 4.0])],
                Feature("outcome", [2.0, 4.0, 6.0, 8.0]),
            )


class TestAdjacentFloatThresholds:
    """A winning cut between two adjacent float64s must route rows."""

    def test_adjacent_values_still_split(self) -> None:
        """(a + nextafter(a)) / 2 rounds down to a, and a < a sends none left.

        The eligibility check admits the cut and the gain is maximal, so the
        degenerate threshold used to win the node and crash growth with
        EmptyValuesError on fully valid data. Adjacent doubles arise from
        computed features -- 0.3 and 0.1 + 0.2 are one ulp apart.
        """
        lower = 1.0
        upper = float(np.nextafter(1.0, 2.0))
        model = DecisionTreeRegressor().fit(
            [Feature("position", [lower, lower, upper, upper])],
            Feature("outcome", [0.0, 0.0, 1.0, 1.0]),
        )

        predictions = np.asarray(model.predict([Feature("position", [lower, upper])]))

        assert predictions == pytest.approx([0.0, 1.0])


class TestForestsRefuseWhatTheyIgnore:
    """A configured base_model must raise, not silently not be the model."""

    def test_the_regression_forest_refuses_a_configured_prototype(self) -> None:
        """The field is inherited from bagging and never read by a forest.

        Accepting it silently meant the caller's tuned prototype simply was
        not the model that fitted -- the plausible-wrong-value failure
        extra='forbid' exists to stop, reintroduced through inheritance.
        """
        with pytest.raises(InvalidValuesError, match="forest"):
            RandomForestRegressor(base_model=DecisionTreeRegressor(max_depth=1))

    def test_the_classification_forest_refuses_identically(self) -> None:
        with pytest.raises(InvalidValuesError, match="forest"):
            RandomForestClassifier(base_model=DecisionTreeClassifier(max_depth=1))

    def test_the_default_still_constructs(self) -> None:
        """A search rebuilding candidates field-by-field passes the default."""
        assert RandomForestRegressor(n_members=3).n_members == 3


class TestBaggingOffsetsMemberSeeds:
    """A seeded member prototype cannot replay one stream into every member."""

    def test_each_member_gets_its_own_seed(self) -> None:
        """All members sharing the prototype's seed is the forest degeneracy
        arrived at through configuration: every restricted tree would draw
        the same features at every node while looking like an ensemble."""
        bagging = BaggingRegressor(
            base_model=DecisionTreeRegressor(random_seed=5, max_features=1)
        )

        first = bagging._prototype(0)
        second = bagging._prototype(3)
        assert isinstance(first, DecisionTreeRegressor)
        assert isinstance(second, DecisionTreeRegressor)

        assert first.random_seed == 5
        assert second.random_seed == 8

    def test_an_unseeded_member_is_left_alone(self) -> None:
        bagging = BaggingRegressor(base_model=DecisionTreeRegressor())

        assert bagging._prototype(3) is bagging.base_model


class TestVestigialCriterionIsGone:
    """The field both trees accepted and neither read."""

    def test_the_trees_refuse_the_dead_field(self) -> None:
        """DecisionTreeClassifier(classification_criterion=ENTROPY) used to
        construct cleanly and split on Gini anyway -- its own criterion field
        governed. extra='forbid' now does its job because the field is gone."""
        dead_field: dict[str, Any] = {"classification_criterion": "gini"}

        with pytest.raises(ValidationError):
            DecisionTreeClassifier(**dead_field)

        with pytest.raises(ValidationError):
            DecisionTreeRegressor(**dead_field)


class TestFailedRefitsLeaveTheOldFit:
    """A refit that raises must not half-replace fitted state."""

    def test_the_nearest_neighbour_classifier_keeps_its_count(self) -> None:
        """_n_classes used to be assigned before feature validation ran."""
        model = KNearestNeighboursClassifier(n_neighbours=3).fit(
            THREE_CLASSES.input_features, THREE_CLASSES.target_feature
        )

        with pytest.raises(NonEqualArrayLengthError):
            model.fit(
                [Feature("first", [1.0, 2.0])],  # wrong length vs target
                Feature("outcome", [0.0, 1.0, 1.0]),
            )

        assert model.n_classes == 3
        assert len(np.asarray(model.predict(THREE_CLASSES.input_features))) == len(
            THREE_CLASSES.target_feature.values
        )

    def test_a_pipeline_survives_a_failing_refit(self) -> None:
        """Fit commits nothing until the steps and the model both succeeded.

        The first version replaced _fitted_model with an unfitted copy before
        any validation ran, so a failing refit left a pipeline that claimed to
        be fitted and crashed on predict.
        """
        pipeline = RegressionPipeline(
            steps=PipelineSteps.of(scaler=Standardizer()),
            model=RidgeRegression(penalty=0.1),
        )
        features = [Feature("first", [1.0, 2.0, 3.0, 4.0])]
        target = Feature("outcome", [2.0, 4.0, 6.0, 8.0])
        pipeline.fit(features, target)
        before = np.asarray(pipeline.predict(features))

        with pytest.raises(AllSameValuesError):
            # a constant column: the standardizer refuses to learn from it
            pipeline.fit([Feature("first", [7.0, 7.0, 7.0, 7.0])], target)

        assert np.allclose(np.asarray(pipeline.predict(features)), before)


class TestSmallContracts:
    """The remaining constructor promises."""

    def test_class_scores_need_at_least_one_class_column(self) -> None:
        """(n, 0) passed the [0, 1] bound vacuously, then most_likely crashed."""
        with pytest.raises(EmptyValuesError):
            ClassScores(np.empty((3, 0)))

    def test_signed_zeros_hash_together(self) -> None:
        """array_equal calls them equal, so the hash contract requires this."""
        positive = Feature("first", [0.0])
        negative = Feature("first", [-0.0])

        assert positive == negative
        assert hash(positive) == hash(negative)
        assert len({positive, negative}) == 1

    def test_a_fractional_cluster_label_is_refused(self) -> None:
        """astype(intp) would have truncated 1.5 to 1, reassigning the row."""
        centroids = Centroids(
            [
                Centroid("cluster_1", np.array([0.0]), ("first",)),
                Centroid("cluster_2", np.array([1.0]), ("first",)),
            ]
        )

        with pytest.raises(InvalidValuesError):
            Clustering(np.array([0.0, 1.5]), centroids, 1.0)

    def test_splits_expose_no_container(self) -> None:
        """The design note said so all along; a docstring-less property
        contradicted it. Iterate the object."""
        splits = KFold(n_folds=2, random_seed=0).split(
            Dataset(
                [Feature("first", [1.0, 2.0, 3.0, 4.0])],
                Feature("outcome", [1.0, 2.0, 3.0, 4.0]),
            )
        )

        assert not hasattr(splits, "splits")
        assert len(list(splits)) == 2
