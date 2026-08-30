"""Spec for the round trip: every registered model, saved and rebuilt, answers
bit-identically.

One parametrized test covers the whole registry, and the parametrization is
itself pinned against ``PERSISTABLE_TYPES`` so a newly registered model that
nobody added a round-trip case for fails the completeness test rather than
shipping unexercised.

``TestLoadingRevalidates`` is the other half of the persistence story: a
document is untrusted input, and a tampered learned value meets the same typed
refusal the equivalent bad data would -- the ordering invariant on principal
components, the label range on a clustering, the finiteness rule on a kernel
matrix all re-run on load, because loading rebuilds through the ordinary
validating constructors rather than trusting the file.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.classification.binary.logistic_regression import LogisticRegression
from oop_ml.classification.binary.newton_logistic_regression import (
    NewtonLogisticRegression,
)
from oop_ml.classification.ensembles.bagging_classifier import BaggingClassifier
from oop_ml.classification.ensembles.random_forest_classifier import (
    RandomForestClassifier,
)
from oop_ml.classification.kernels.support_vector_classifier import (
    SupportVectorClassifier,
)
from oop_ml.classification.multiclass.multinomial_logistic_regression import (
    MultinomialLogisticRegression,
)
from oop_ml.classification.multiclass.one_vs_rest import OneVsRestClassifier
from oop_ml.classification.neighbours.k_nearest_classifier import (
    KNearestNeighboursClassifier,
)
from oop_ml.classification.trees.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from oop_ml.clustering.k_means import KMeans
from oop_ml.core.base.estimator import Fittable
from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.kernel.functions import Kernel, RadialBasisKernel
from oop_ml.decomposition.kernel_principal_component_analysis import (
    KernelPrincipalComponentAnalysis,
)
from oop_ml.decomposition.principal_component_analysis import (
    PrincipalComponentAnalysis,
)
from oop_ml.persistence.document import ModelDocument
from oop_ml.persistence.store import (
    PERSISTABLE_TYPES,
    build_model,
    model_document,
)
from oop_ml.pipeline.pipelines import (
    ClassificationPipeline,
    RegressionPipeline,
)
from oop_ml.pipeline.steps import PipelineSteps
from oop_ml.preprocessing.polynomial.features import PolynomialFeatures
from oop_ml.preprocessing.standardization.standardizer import Standardizer
from oop_ml.regression.ensembles.bagging_regressor import BaggingRegressor
from oop_ml.regression.ensembles.gradient_boosting_regressor import (
    GradientBoostingRegressor,
)
from oop_ml.regression.ensembles.random_forest_regressor import (
    RandomForestRegressor,
)
from oop_ml.regression.kernels.kernel_ridge_regression import (
    KernelRidgeRegression,
)
from oop_ml.regression.least_squares.gradient_descent_regression import (
    GradientDescentRegression,
)
from oop_ml.regression.least_squares.multiple_feature_regression import (
    MultipleLinearRegression,
)
from oop_ml.regression.least_squares.simple_linear_regression import (
    SimpleLinearRegression,
)
from oop_ml.regression.neighbours.k_nearest_regressor import (
    KNearestNeighboursRegressor,
)
from oop_ml.regression.penalised.lasso_regression import LassoRegression
from oop_ml.regression.penalised.ridge_regression import RidgeRegression
from oop_ml.regression.trees.decision_tree_regressor import DecisionTreeRegressor

_GENERATOR = np.random.default_rng(7)
_N_ROWS = 40
_MATRIX = _GENERATOR.normal(size=(_N_ROWS, 3))

FEATURES = [Feature(f"feature_{index}", _MATRIX[:, index]) for index in range(3)]
TARGET = Feature(
    "outcome",
    2.0 * _MATRIX[:, 0] - _MATRIX[:, 1] + _GENERATOR.normal(scale=0.2, size=_N_ROWS),
)
OVERLAPPING_CLASSES = Feature(
    "outcome",
    ((_MATRIX[:, 0] + _GENERATOR.normal(scale=1.5, size=_N_ROWS)) > 0.0).astype(float),
)
THREE_CLASSES_TARGET = Feature(
    "outcome", np.clip(np.floor(_MATRIX[:, 0] + 1.5), 0.0, 2.0)
)

SIMPLE_INPUT = [1.0, 2.0, 3.0, 4.0, 5.0]
SIMPLE_TARGET = [2.1, 4.2, 5.9, 8.1, 9.9]


def regression_answer(model):
    return np.asarray(model.predict(FEATURES))


def probability_answer(model):
    return np.asarray(model.predict_probability(FEATURES))


def probabilities_answer(model):
    return np.asarray(model.predict_probabilities(FEATURES))


def transform_answer(model):
    return np.column_stack([feature.values for feature in model.transform(FEATURES)])


ROUND_TRIPS = {
    "SimpleLinearRegression": (
        lambda: SimpleLinearRegression().fit(SIMPLE_INPUT, SIMPLE_TARGET),
        lambda model: np.asarray(model.predict(SIMPLE_INPUT)),
    ),
    "MultipleLinearRegression": (
        lambda: MultipleLinearRegression().fit(FEATURES, TARGET),
        regression_answer,
    ),
    "GradientDescentRegression": (
        lambda: GradientDescentRegression(learning_rate=0.05, max_epochs=2000).fit(
            FEATURES, TARGET
        ),
        regression_answer,
    ),
    "RidgeRegression": (
        lambda: RidgeRegression(penalty=0.5).fit(FEATURES, TARGET),
        regression_answer,
    ),
    "LassoRegression": (
        lambda: LassoRegression(penalty=0.1).fit(FEATURES, TARGET),
        regression_answer,
    ),
    "KNearestNeighboursRegressor": (
        lambda: KNearestNeighboursRegressor(n_neighbours=3).fit(FEATURES, TARGET),
        regression_answer,
    ),
    "DecisionTreeRegressor": (
        lambda: DecisionTreeRegressor(max_depth=3).fit(FEATURES, TARGET),
        regression_answer,
    ),
    "BaggingRegressor": (
        lambda: BaggingRegressor(n_members=5, random_seed=1).fit(FEATURES, TARGET),
        regression_answer,
    ),
    "RandomForestRegressor": (
        lambda: RandomForestRegressor(n_members=5, max_features=2, random_seed=1).fit(
            FEATURES, TARGET
        ),
        regression_answer,
    ),
    "GradientBoostingRegressor": (
        lambda: GradientBoostingRegressor(n_rounds=5).fit(FEATURES, TARGET),
        regression_answer,
    ),
    "KernelRidgeRegression": (
        lambda: KernelRidgeRegression(
            kernel=RadialBasisKernel(gamma=0.5), penalty=0.1
        ).fit(FEATURES, TARGET),
        regression_answer,
    ),
    "LogisticRegression": (
        lambda: LogisticRegression(max_epochs=500).fit(FEATURES, OVERLAPPING_CLASSES),
        probability_answer,
    ),
    "NewtonLogisticRegression": (
        lambda: NewtonLogisticRegression().fit(FEATURES, OVERLAPPING_CLASSES),
        probability_answer,
    ),
    "MultinomialLogisticRegression": (
        lambda: MultinomialLogisticRegression(max_epochs=500).fit(
            FEATURES, THREE_CLASSES_TARGET
        ),
        probabilities_answer,
    ),
    "OneVsRestClassifier": (
        lambda: OneVsRestClassifier(
            binary_model=LogisticRegression(max_epochs=300)
        ).fit(FEATURES, THREE_CLASSES_TARGET),
        probabilities_answer,
    ),
    "KNearestNeighboursClassifier": (
        lambda: KNearestNeighboursClassifier(n_neighbours=3).fit(
            FEATURES, THREE_CLASSES_TARGET
        ),
        probabilities_answer,
    ),
    "DecisionTreeClassifier": (
        lambda: DecisionTreeClassifier(max_depth=3).fit(FEATURES, THREE_CLASSES_TARGET),
        probabilities_answer,
    ),
    "BaggingClassifier": (
        lambda: BaggingClassifier(n_members=5, random_seed=1).fit(
            FEATURES, THREE_CLASSES_TARGET
        ),
        probabilities_answer,
    ),
    "RandomForestClassifier": (
        lambda: RandomForestClassifier(n_members=5, max_features=1, random_seed=1).fit(
            FEATURES, THREE_CLASSES_TARGET
        ),
        probabilities_answer,
    ),
    "SupportVectorClassifier": (
        lambda: SupportVectorClassifier(
            kernel=RadialBasisKernel(gamma=0.5), max_epochs=500
        ).fit(FEATURES, OVERLAPPING_CLASSES),
        lambda model: model.decision_values(FEATURES),
    ),
    "KMeans": (
        lambda: KMeans(n_clusters=3, random_seed=0).fit(FEATURES),
        lambda model: np.asarray(model.predict(FEATURES)),
    ),
    "PrincipalComponentAnalysis": (
        lambda: PrincipalComponentAnalysis(n_components=2, standardize=True).fit(
            FEATURES
        ),
        transform_answer,
    ),
    "KernelPrincipalComponentAnalysis": (
        lambda: KernelPrincipalComponentAnalysis(
            kernel=RadialBasisKernel(gamma=0.5), n_components=2
        ).fit(FEATURES),
        transform_answer,
    ),
    "Standardizer": (
        lambda: Standardizer().fit(FEATURES),
        transform_answer,
    ),
    "PolynomialFeatures": (
        lambda: PolynomialFeatures(degree=2).fit(FEATURES),
        transform_answer,
    ),
    "RegressionPipeline": (
        lambda: RegressionPipeline(
            steps=PipelineSteps.of(
                terms=PolynomialFeatures(degree=2), scaler=Standardizer()
            ),
            model=RidgeRegression(penalty=0.1),
        ).fit(FEATURES, TARGET),
        regression_answer,
    ),
    "ClassificationPipeline": (
        lambda: ClassificationPipeline(
            steps=PipelineSteps.of(scaler=Standardizer()),
            model=MultinomialLogisticRegression(max_epochs=300),
        ).fit(FEATURES, THREE_CLASSES_TARGET),
        probabilities_answer,
    ),
}


class TestEveryModelRoundTrips:
    """Save, rebuild, and answer bit-identically."""

    @pytest.mark.parametrize("model_name", sorted(ROUND_TRIPS))
    def test_the_rebuilt_model_answers_identically(self, model_name: str) -> None:
        """Exactly equal, not approximately: the rebuilt model holds the same
        numbers and runs the same arithmetic, so any drift is a codec bug.

        The trip goes all the way through ``to_json`` and ``from_json``, not
        just ``model_document`` / ``build_model``, because that is where an
        order-bearing mapping would be reordered by key sorting -- the failure
        an in-memory round trip cannot see.
        """
        build, answer = ROUND_TRIPS[model_name]
        fitted = build()

        text = model_document(fitted).to_json()
        rebuilt = build_model(ModelDocument.from_json(text))

        assert np.array_equal(answer(fitted), answer(rebuilt))

    def test_the_parametrization_covers_the_whole_registry(self) -> None:
        """A newly registered model without a round-trip case fails here,
        rather than shipping unexercised. Kernels are configuration types and
        are exercised inside the models that carry them."""
        fittable_names = {
            name
            for name, registered in PERSISTABLE_TYPES.items()
            if issubclass(registered, Fittable)
        }
        kernel_names = {
            name
            for name, registered in PERSISTABLE_TYPES.items()
            if issubclass(registered, Kernel)
        }

        assert fittable_names - kernel_names == set(ROUND_TRIPS)


class TestWhatSurvivesBeyondPredictions:
    """The fitted self is the whole of it, not just the answers."""

    def test_out_of_bag_scoring_survives_the_trip(self) -> None:
        """The ensembles keep their samples and training rows on purpose:
        dropping either would silently lose a public method on load."""
        fitted = BaggingRegressor(n_members=5, random_seed=1).fit(FEATURES, TARGET)
        rebuilt = build_model(model_document(fitted))

        assert rebuilt.out_of_bag_score() == pytest.approx(fitted.out_of_bag_score())

    def test_the_class_count_survives(self) -> None:
        fitted = DecisionTreeClassifier(max_depth=3).fit(FEATURES, THREE_CLASSES_TARGET)
        rebuilt = build_model(model_document(fitted))

        assert rebuilt.n_classes == fitted.n_classes

    def test_a_kernel_hyperparameter_survives_as_its_own_type(self) -> None:
        """model_dump would have flattened it to a dict; the codec keeps the
        class, so the rebuilt model kernels with the same function."""
        fitted = KernelRidgeRegression(
            kernel=RadialBasisKernel(gamma=0.5), penalty=0.1
        ).fit(FEATURES, TARGET)
        rebuilt = build_model(model_document(fitted))

        assert isinstance(rebuilt.kernel, RadialBasisKernel)
        assert rebuilt.kernel.gamma == pytest.approx(0.5)

    def test_a_pipeline_rebuilds_its_fitted_steps_as_fitted(self) -> None:
        fitted = RegressionPipeline(
            steps=PipelineSteps.of(scaler=Standardizer()),
            model=RidgeRegression(penalty=0.1),
        ).fit(FEATURES, TARGET)
        rebuilt = build_model(model_document(fitted))

        assert rebuilt.fitted_steps["scaler"].transformer.is_fitted
        assert not rebuilt.steps["scaler"].transformer.is_fitted


class TestLoadingRevalidates:
    """A tampered document meets the same guards as bad data."""

    def tampered(self, model, part: str, mutate) -> ModelDocument:
        document = model_document(model)
        learned = document.learned
        learned[part] = mutate(learned[part])

        return ModelDocument(document.model_type, document.hyperparameters, learned)

    def test_reordered_components_are_refused(self) -> None:
        """The ordering invariant re-runs on load: components claiming
        ascending variance are exactly what an unreversed eigh produces, and
        the constructor refuses them from a file as it would from a fit."""
        fitted = PrincipalComponentAnalysis(n_components=2).fit(FEATURES)

        def reverse_components(payload):
            payload["components"] = payload["components"][::-1]
            return payload

        with pytest.raises(InvalidValuesError):
            build_model(self.tampered(fitted, "_components", reverse_components))

    def test_an_out_of_range_cluster_label_is_refused(self) -> None:
        fitted = KMeans(n_clusters=3, random_seed=0).fit(FEATURES)

        def corrupt_labels(payload):
            payload["labels"]["values"][0] = 99
            return payload

        with pytest.raises(InvalidValuesError):
            build_model(self.tampered(fitted, "_clustering", corrupt_labels))

    def test_a_non_finite_kernel_matrix_is_refused(self) -> None:
        fitted = KernelPrincipalComponentAnalysis(
            kernel=RadialBasisKernel(gamma=0.5), n_components=2
        ).fit(FEATURES)

        def poison(payload):
            payload["values"]["values"][0][0] = None
            return payload

        with pytest.raises((InvalidValuesError, TypeError)):
            build_model(self.tampered(fitted, "_training_matrix", poison))

    def test_hyperparameters_are_revalidated_too(self) -> None:
        """A document claiming penalty=-5 goes through the constructor and
        meets pydantic's bound, not a bypass."""
        document = model_document(RidgeRegression(penalty=1.0).fit(FEATURES, TARGET))
        hyperparameters = document.hyperparameters
        hyperparameters["penalty"] = -5.0
        hostile = ModelDocument(document.model_type, hyperparameters, document.learned)

        with pytest.raises(ValidationError):
            build_model(hostile)
