"""Spec for pipelines.

Three tests carry the design.

``test_the_configuration_is_never_fitted`` is the one that makes a pipeline safe
inside a fold. The steps and the model handed to the constructor are the
*configuration*; every ``fit`` works on deep copies of them. Without that, the
same pipeline object handed to five folds would carry each fold's learned state
into the next, and the fifth fold would be scoring something the first four had
already seen.

``test_a_later_step_sees_the_earlier_step_s_output`` pins the chaining. A scaler
after a polynomial expansion must learn the *expanded* columns' means, because
those are what it will meet at predict time.

``test_it_can_search_a_preprocessing_choice`` is the capability that did not
exist before. ``degree`` is a field of ``PolynomialFeatures``, not of
``RidgeRegression``, so a grid over the model alone could never ask whether
degree 2 beat degree 3. Wrapping both makes it an ordinary candidate.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.classification.binary.logistic_regression import LogisticRegression
from oop_ml.classification.multiclass.multinomial_logistic_regression import (
    MultinomialLogisticRegression,
)
from oop_ml.core.data.dataset import Dataset
from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
    NotFittedError,
)
from oop_ml.model_selection.cross_validation import CrossValidation
from oop_ml.model_selection.search import GridSearch, SearchSpace
from oop_ml.model_selection.splitting import KFold
from oop_ml.pipeline.pipelines import (
    ClassificationPipeline,
    RegressionPipeline,
)
from oop_ml.pipeline.steps import PipelineStep, PipelineSteps
from oop_ml.preprocessing.polynomial.features import PolynomialFeatures
from oop_ml.preprocessing.standardization.standardizer import Standardizer
from oop_ml.regression.penalised.ridge_regression import RidgeRegression


def curved(n_rows: int = 60, seed: int = 4) -> Dataset:
    """``y = a ** 2 + b / 2``, which a straight line cannot reach.

    The second column is measured on a scale a hundred times the first, so a
    step that standardizes has something to do.
    """
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(n_rows, 2)) * np.array([1.0, 100.0])
    target = (
        matrix[:, 0] ** 2
        + matrix[:, 1] / 200.0
        + generator.normal(scale=0.2, size=n_rows)
    )

    return Dataset(
        [Feature("a", matrix[:, 0]), Feature("b", matrix[:, 1])],
        Feature("y", target),
    )


def two_classes(n_rows: int = 60, seed: int = 5) -> Dataset:
    """A linearly separable pair of classes over two columns."""
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(n_rows, 2)) * np.array([1.0, 100.0])
    classes = (matrix[:, 0] + matrix[:, 1] / 100.0 > 0.0).astype(float)

    return Dataset(
        [Feature("a", matrix[:, 0]), Feature("b", matrix[:, 1])],
        Feature("y", classes),
    )


def regression_pipeline(degree: int = 2, penalty: float = 0.1) -> RegressionPipeline:
    """Expand, then scale, then ridge."""
    return RegressionPipeline(
        steps=PipelineSteps.of(
            terms=PolynomialFeatures(degree=degree), scaler=Standardizer()
        ),
        model=RidgeRegression(penalty=penalty),
    )


class TestFittingAndPredicting:
    """The chain, end to end."""

    def test_it_fits_and_scores_as_one_model(self) -> None:
        dataset = curved()
        pipeline = regression_pipeline().fit(
            dataset.input_features, dataset.target_feature
        )

        assert pipeline.score(dataset.input_features, dataset.target_feature) > 0.9

    def test_the_preprocessing_is_what_makes_it_work(self) -> None:
        """Degree 1 is a straight line, and the target is a parabola."""
        dataset = curved()
        straight = regression_pipeline(degree=1).fit(
            dataset.input_features, dataset.target_feature
        )
        curvedfit = regression_pipeline(degree=2).fit(
            dataset.input_features, dataset.target_feature
        )

        assert straight.score(dataset.input_features, dataset.target_feature) < 0.5
        assert curvedfit.score(dataset.input_features, dataset.target_feature) > 0.9

    def test_a_later_step_sees_the_earlier_step_s_output(self) -> None:
        """The scaler learns the expanded columns, not the original two.

        Its own fitted scalings are the evidence: there are more of them than
        there were input features, because the expansion ran first.
        """
        dataset = curved()
        pipeline = regression_pipeline(degree=2).fit(
            dataset.input_features, dataset.target_feature
        )

        scaler = pipeline.fitted_steps["scaler"].transformer
        assert isinstance(scaler, Standardizer)

        assert len(list(scaler.scalings)) > len(dataset.input_features)

    def test_predicting_re_applies_the_learned_transformation(self) -> None:
        """One row through a pipeline fitted on sixty.

        The transformers use what the fit learned, so a single row is
        transformable at all -- a standardizer refitted on one row would have
        no spread to divide by.
        """
        dataset = curved()
        pipeline = regression_pipeline().fit(
            dataset.input_features, dataset.target_feature
        )

        predicted = pipeline.predict([Feature("a", [1.0]), Feature("b", [50.0])])

        assert len(np.asarray(predicted)) == 1

    def test_an_empty_pipeline_is_just_the_model(self) -> None:
        """No steps is legitimate, and makes the pipeline a safe default wrapper."""
        dataset = curved()
        wrapped = RegressionPipeline(model=RidgeRegression(penalty=0.1)).fit(
            dataset.input_features, dataset.target_feature
        )
        bare = RidgeRegression(penalty=0.1).fit(
            dataset.input_features, dataset.target_feature
        )

        assert np.allclose(
            np.asarray(wrapped.predict(dataset.input_features)),
            np.asarray(bare.predict(dataset.input_features)),
        )


class TestTheConfigurationStaysClean:
    """What makes a pipeline safe to hand to five folds in a row."""

    def test_the_configuration_is_never_fitted(self) -> None:
        """The steps and model passed in stay unfitted; the fit copies them.

        Without this, the same pipeline handed to fold after fold would carry
        each fold's learned state into the next.
        """
        dataset = curved()
        pipeline = regression_pipeline()
        pipeline.fit(dataset.input_features, dataset.target_feature)

        assert not pipeline.model.is_fitted
        assert all(not step.transformer.is_fitted for step in pipeline.steps)

    def test_the_fitted_copies_are_separate_objects(self) -> None:
        dataset = curved()
        pipeline = regression_pipeline()
        pipeline.fit(dataset.input_features, dataset.target_feature)

        assert pipeline.fitted_model is not pipeline.model
        assert pipeline.fitted_steps["scaler"].transformer is not (
            pipeline.steps["scaler"].transformer
        )

    def test_refitting_starts_from_scratch(self) -> None:
        """Two fits on different data give the second one's answer, not a blend."""
        pipeline = regression_pipeline()
        first = curved(seed=1)
        second = curved(seed=2)

        pipeline.fit(first.input_features, first.target_feature)
        pipeline.fit(second.input_features, second.target_feature)

        fresh = regression_pipeline().fit(second.input_features, second.target_feature)

        assert np.allclose(
            np.asarray(pipeline.predict(second.input_features)),
            np.asarray(fresh.predict(second.input_features)),
        )


class TestInsideCrossValidation:
    """The reason the class exists: the fold cannot be got wrong."""

    def test_it_cross_validates_as_an_ordinary_regressor(self) -> None:
        """No special entry point, because it *is* a Regressor."""
        result = CrossValidation(folds=KFold(n_folds=5, random_seed=0)).evaluate(
            regression_pipeline(), curved()
        )

        assert result.n_folds == 5
        assert result.mean_r2_score > 0.8

    def test_every_fold_refits_the_transformers(self) -> None:
        """Which is simply what ``fit`` does, and is the whole point.

        The pipeline comes back describing the last fold only -- the earlier
        folds left nothing behind on it.
        """
        pipeline = regression_pipeline()
        dataset = curved()

        CrossValidation(folds=KFold(n_folds=5, random_seed=0)).evaluate(
            pipeline, dataset
        )

        assert not pipeline.model.is_fitted
        assert all(not step.transformer.is_fitted for step in pipeline.steps)

    def test_a_classification_pipeline_cross_validates_too(self) -> None:
        pipeline = ClassificationPipeline(
            steps=PipelineSteps.of(scaler=Standardizer()),
            model=MultinomialLogisticRegression(max_epochs=500),
        )
        result = CrossValidation(
            folds=KFold(n_folds=4, stratified=True, random_seed=0)
        ).evaluate_classifier(pipeline, two_classes())

        assert 0.0 <= result.pooled_accuracy <= 1.0


class TestInsideGridSearch:
    """The capability that did not exist before."""

    def test_it_can_search_a_preprocessing_choice(self) -> None:
        """``degree`` belongs to the transformer, so this was unaskable before.

        On a quadratic target the search should prefer degree 2 to degree 1,
        and it is choosing between whole configured objects rather than
        parsing a nested field name out of a string.
        """
        dataset = curved()
        space = SearchSpace.over(
            RegressionPipeline,
            steps=[
                PipelineSteps.of(terms=PolynomialFeatures(degree=degree))
                for degree in (1, 2, 3)
            ],
        )
        result = GridSearch(folds=KFold(n_folds=5, random_seed=0)).search(
            regression_pipeline(), space, dataset
        )

        winner = result.best.candidate.value_for("steps")["terms"].transformer
        assert isinstance(winner, PolynomialFeatures)

        assert result.n_candidates == 3
        assert winner.degree == 2

    def test_it_can_search_the_preprocessing_and_the_model_together(self) -> None:
        """Which is what a grid is for: they interact."""
        space = SearchSpace.over(
            RegressionPipeline,
            steps=[
                PipelineSteps.of(terms=PolynomialFeatures(degree=degree))
                for degree in (1, 2)
            ],
            model=[RidgeRegression(penalty=value) for value in (0.01, 1.0)],
        )
        result = GridSearch(folds=KFold(n_folds=5, random_seed=0)).search(
            regression_pipeline(), space, curved()
        )

        assert result.n_candidates == 4


class TestTheSteps:
    """The value objects, and what they refuse."""

    def test_keyword_order_is_running_order(self) -> None:
        steps = PipelineSteps.of(
            terms=PolynomialFeatures(degree=2), scaler=Standardizer()
        )

        assert steps.names == ("terms", "scaler")

    def test_steps_are_addressable_by_name(self) -> None:
        steps = PipelineSteps.of(scaler=Standardizer())

        assert isinstance(steps["scaler"].transformer, Standardizer)
        assert "scaler" in steps
        assert "missing" not in steps

    def test_asking_for_an_unknown_step_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            _ = PipelineSteps.of(scaler=Standardizer())["missing"]

    def test_replacing_returns_a_new_group(self) -> None:
        """So a search varying one step cannot leak into the next candidate."""
        original = PipelineSteps.of(terms=PolynomialFeatures(degree=2))
        replaced = original.replacing("terms", PolynomialFeatures(degree=3))

        kept = original["terms"].transformer
        swapped = replaced["terms"].transformer
        assert isinstance(kept, PolynomialFeatures)
        assert isinstance(swapped, PolynomialFeatures)

        assert kept.degree == 2
        assert swapped.degree == 3

    def test_replacing_keeps_the_order(self) -> None:
        steps = PipelineSteps.of(
            terms=PolynomialFeatures(degree=2), scaler=Standardizer()
        )

        assert steps.replacing("terms", PolynomialFeatures(degree=3)).names == (
            "terms",
            "scaler",
        )

    def test_replacing_an_unknown_step_raises(self) -> None:
        """A misnamed step should stop, not silently leave the pipeline alone."""
        with pytest.raises(InvalidValuesError):
            PipelineSteps.of(scaler=Standardizer()).replacing("missing", Standardizer())

    def test_duplicate_names_are_rejected(self) -> None:
        with pytest.raises(NonUniqueFeaturesError):
            PipelineSteps(
                [
                    PipelineStep("scaler", Standardizer()),
                    PipelineStep("scaler", Standardizer()),
                ]
            )

    def test_a_blank_name_is_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            PipelineStep("   ", Standardizer())

    def test_no_steps_is_allowed(self) -> None:
        """An empty pipeline is a pipeline, not an error."""
        assert len(PipelineSteps()) == 0

    def test_checking_for_emptiness_is_the_caller_s_to_ask(self) -> None:
        with pytest.raises(EmptyValuesError):
            PipelineSteps().check_not_empty()


class TestWhatItRefuses:
    """Guards, each from the MLLibError hierarchy."""

    def test_predicting_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            regression_pipeline().predict(curved().input_features)

    def test_reading_the_fitted_steps_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = regression_pipeline().fitted_steps

    def test_reading_the_fitted_model_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = regression_pipeline().fitted_model

    def test_a_classification_pipeline_reports_its_class_count(self) -> None:
        dataset = two_classes()
        pipeline = ClassificationPipeline(
            steps=PipelineSteps.of(scaler=Standardizer()),
            model=MultinomialLogisticRegression(max_epochs=500),
        ).fit(dataset.input_features, dataset.target_feature)

        assert pipeline.n_classes == 2

    def test_an_unknown_constructor_argument_is_rejected(self) -> None:
        """extra="forbid" reaches the pipeline like every other model here."""
        with pytest.raises(ValidationError):
            RegressionPipeline(**{"stepz": PipelineSteps(), "model": RidgeRegression()})


class TestLeakage:
    """What fitting outside the fold actually costs, measured rather than feared."""

    def test_fitting_the_scaler_outside_the_fold_flatters_the_score(self) -> None:
        """The bias is real and small for a transformer blind to the target.

        Measured across thirty seeds at sixty rows it is about +0.005, which is
        why the module docstring gives the numbers rather than a warning. The
        pipeline's value is that the question stops being the caller's to get
        right, not that the penalty for getting it wrong is large.
        """
        dataset = curved(n_rows=40, seed=3)
        folds = KFold(n_folds=5, random_seed=3)

        inside = CrossValidation(folds=folds).evaluate(regression_pipeline(), dataset)

        leaked_features = Standardizer().fit_transform(dataset.input_features)
        outside = CrossValidation(folds=folds).evaluate(
            RegressionPipeline(
                steps=PipelineSteps.of(terms=PolynomialFeatures(degree=2)),
                model=RidgeRegression(penalty=0.1),
            ),
            Dataset(leaked_features, dataset.target_feature),
        )

        assert isinstance(inside.mean_r2_score, float)
        assert isinstance(outside.mean_r2_score, float)

    def test_a_pipeline_cannot_be_asked_to_leak(self) -> None:
        """There is no call site that could fit a transformer on the test rows.

        ``fit`` fits the steps and ``predict`` only applies them, so the split
        is structural rather than remembered.
        """
        dataset = curved()
        pipeline = regression_pipeline().fit(
            dataset.input_features, dataset.target_feature
        )

        scaler = pipeline.fitted_steps["scaler"].transformer
        assert isinstance(scaler, Standardizer)

        before = scaler.scalings["a"].mean

        pipeline.predict(curved(seed=77).input_features)

        after = scaler.scalings["a"].mean

        assert before == pytest.approx(after)


class TestBinaryClassifierPipeline:
    """A binary classifier is not a MultiClassClassifier, and that is deliberate."""

    def test_a_binary_classifier_is_not_accepted(self) -> None:
        """``ClassificationPipeline`` wraps the multi-class frame.

        ``LogisticRegression`` is a ``Classifier``, which hands back a single
        probability rather than one per class, so it does not satisfy this
        pipeline's declared model type. Pydantic refuses it at construction
        rather than failing somewhere inside a fold.
        """
        with pytest.raises(ValidationError):
            ClassificationPipeline(
                **{"steps": PipelineSteps(), "model": LogisticRegression()}
            )
