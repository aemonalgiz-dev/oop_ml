"""The K sub-fits must be the K sub-fits the ensemble is made of.

One-vs-rest looks like one estimator and is K of them. The observed route holds
each recoded target beside the model fitted against it, and the agreement test
is that those models predict what the ensemble predicts.

The other thing pinned here is the consequence nobody expects: the K models
were never introduced to each other, so their probabilities do not sum to one.
Holding the sub-fits turns that from a claim in a docstring into something a
caller can check.
"""

import numpy as np
import pytest

from oop_ml import LogisticRegression, OneVsRestClassifier
from oop_ml.classification.multiclass.one_vs_rest_fits import ClassFit, OneVsRestFits
from oop_ml.core.observation import Observation
from test.fixtures import THREE_CLASSES


def fitted() -> OneVsRestClassifier:
    model = OneVsRestClassifier(binary_model=LogisticRegression())
    model.fit(THREE_CLASSES.input_features, THREE_CLASSES.target_feature)

    return model


def observed(model: OneVsRestClassifier) -> OneVsRestFits:
    return model.one_vs_rest_fits(
        THREE_CLASSES.input_features, THREE_CLASSES.target_feature
    )


class TestTheTwoRoutesAgree:
    def test_there_is_one_fit_per_class(self):
        model = fitted()

        assert len(observed(model)) == model.n_classes

    def test_the_sub_models_predict_what_the_ensemble_predicts(self):
        model = fitted()
        features = THREE_CLASSES.input_features

        from_ensemble = model.predict_probabilities(features)
        from_parts = np.column_stack(
            [fit.model.predict_probability(features) for fit in observed(model)]
        )

        assert from_parts == pytest.approx(from_ensemble)

    def test_recording_does_not_refit(self):
        model = fitted()
        before = model.predict(THREE_CLASSES.input_features)

        observed(model)

        assert model.predict(THREE_CLASSES.input_features) == pytest.approx(before)


class TestWhatTheSubFitsShow:
    def test_each_target_is_named_for_the_question_it_encodes(self):
        names = [fit.recoded_target.name for fit in observed(fitted())]

        assert names == ["outcome==0", "outcome==1", "outcome==2"]

    def test_each_target_is_one_against_the_rest(self):
        actual = THREE_CLASSES.target_feature.values

        for fit in observed(fitted()):
            assert fit.recoded_target.values == pytest.approx(
                (actual == fit.class_index).astype(float)
            )

    def test_the_positive_count_is_that_class_only(self):
        actual = THREE_CLASSES.target_feature.values

        for fit in observed(fitted()):
            assert fit.positive_rows == int((actual == fit.class_index).sum())

    def test_each_fit_gets_its_own_model(self):
        # A shared prototype fitted K times would leave every entry holding
        # whichever class went last.
        fits = list(observed(fitted()))
        models = [fit.model for fit in fits]

        assert len({id(model) for model in models}) == len(models)

    def test_the_probabilities_do_not_sum_to_one(self):
        # The consequence of K independent fits, made checkable.
        totals = (
            fitted()
            .predict_probabilities(THREE_CLASSES.input_features)
            .values.sum(axis=1)
        )

        assert not np.allclose(totals, 1.0)


class TestItIsAnObservation:
    def test_it_satisfies_the_protocol(self):
        fits = observed(fitted())

        assert isinstance(fits, Observation)
        assert isinstance(fits, OneVsRestFits)
        assert all(isinstance(fit, ClassFit) for fit in fits)

    def test_result_is_the_fitted_models(self):
        fits = observed(fitted())

        assert fits.result == tuple(fit.model for fit in fits)
