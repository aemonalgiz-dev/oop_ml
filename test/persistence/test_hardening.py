"""Spec for the attacks an adversarial review found in the first cut.

A saved model is untrusted input, and the first implementation trusted it in
five ways a verification pass reproduced end to end. Each is pinned here where
it was found, because the round-trip spec structurally could not reach them:
it never crossed ``to_json`` / ``from_json`` (so key sorting was invisible) and
it only fed documents the library itself wrote (so tampering was untested).
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import InvalidDocumentError
from oop_ml.decomposition.principal_component_analysis import (
    PrincipalComponentAnalysis,
)
from oop_ml.persistence.document import ModelDocument
from oop_ml.persistence.store import build_model, model_document
from oop_ml.regression.neighbours.k_nearest_regressor import (
    KNearestNeighboursRegressor,
)
from oop_ml.regression.penalised.ridge_regression import RidgeRegression

FEATURES = [
    Feature("floor_area", [72.0, 140.0, 96.0, 210.0, 55.0, 88.0]),
    Feature("bathrooms", [1.0, 2.0, 1.0, 3.0, 1.0, 1.0]),
]
TARGET = Feature("price", [310.0, 505.0, 372.0, 690.0, 240.0, 350.0])


def ridge_document() -> ModelDocument:
    return model_document(RidgeRegression(penalty=1.0).fit(FEATURES, TARGET))


def with_learned(document: ModelDocument, part: str, payload) -> ModelDocument:
    learned = document.learned
    learned[part] = payload
    return ModelDocument(document.model_type, document.hyperparameters, learned)


class TestUntrustedArrays:
    """The dtype string is document-controlled, so it is allowlisted."""

    def test_object_dtype_cannot_smuggle_python_objects(self) -> None:
        """dtype='object' would land strings and dicts in a float array with
        every finiteness guard bypassed, the model reporting itself fitted
        until the first predict died on a bare TypeError."""
        knn = model_document(
            KNearestNeighboursRegressor(n_neighbours=3).fit(FEATURES, TARGET)
        )
        hostile = with_learned(
            knn,
            "_remembered_targets",
            {"__kind__": "array", "dtype": "object", "values": ["evil", {"x": 1}]},
        )

        with pytest.raises(InvalidDocumentError, match="dtype"):
            build_model(hostile)

    def test_an_unparseable_dtype_is_a_typed_refusal(self) -> None:
        """np.dtype('pwned') raises a bare numpy TypeError; the allowlist
        turns it into the InvalidDocumentError the boundary promises."""
        hostile = with_learned(
            ridge_document(),
            "_coefficients",
            {"__kind__": "array", "dtype": "pwned", "values": [1.0]},
        )

        with pytest.raises(InvalidDocumentError, match="dtype"):
            build_model(hostile)

    def test_a_malformed_array_payload_is_a_typed_refusal(self) -> None:
        hostile = with_learned(
            ridge_document(),
            "_coefficients",
            {"__kind__": "array", "dtype": "float64", "values": "not a list"},
        )

        with pytest.raises(InvalidDocumentError):
            build_model(hostile)


class TestDepthBound:
    """A nested payload cannot overflow the interpreter stack."""

    def test_a_deeply_nested_tuple_is_refused(self) -> None:
        """600 tuple tags overflow the default recursion limit; the bound
        refuses it as a document rather than crashing as a RecursionError."""
        nested = {"__kind__": "tuple", "items": []}
        cursor = nested
        for _ in range(600):
            child = {"__kind__": "tuple", "items": []}
            cursor["items"].append(child)
            cursor = child

        with pytest.raises(InvalidDocumentError, match="nests deeper"):
            build_model(
                ModelDocument(
                    "SimpleLinearRegression", {}, {"_slope": nested, "_intercept": 0.0}
                )
            )


class TestMalformedShapes:
    """A tag of the wrong shape is a typed refusal, not a bare crash."""

    def test_a_mapping_that_is_not_a_list_of_pairs_is_refused(self) -> None:
        hostile = with_learned(
            ridge_document(),
            "_coefficients",
            {"__kind__": "mapping", "items": [1, 2, 3]},
        )

        with pytest.raises(InvalidDocumentError):
            build_model(hostile)

    def test_a_tuple_whose_items_are_not_a_list_is_refused(self) -> None:
        hostile = with_learned(
            ridge_document(),
            "_coefficients",
            {"__kind__": "tuple", "items": {"not": "a list"}},
        )

        with pytest.raises(InvalidDocumentError):
            build_model(hostile)


class TestOrderBearingMappings:
    """The finding the round-trip spec could not reach: key sorting."""

    def test_pca_transforms_identically_through_json(self) -> None:
        """PrincipalComponentAnalysis keeps its fitted feature order in the
        insertion order of _feature_means. to_json sorts keys, so an
        alphabetised reload centred and stacked the columns in the wrong order
        while the loadings stayed put -- transforming silently wrong. The
        fixture's names are reverse-alphabetical on purpose, so sorting would
        move them.
        """
        generator = np.random.default_rng(3)
        matrix = generator.normal(size=(30, 2))
        features = [
            Feature("zulu", matrix[:, 0] * 100.0),
            Feature("alpha", matrix[:, 1]),
        ]
        fitted = PrincipalComponentAnalysis(n_components=2, standardize=True).fit(
            features
        )

        before = np.column_stack(
            [feature.values for feature in fitted.transform(features)]
        )
        text = model_document(fitted).to_json()
        reloaded = build_model(ModelDocument.from_json(text))
        after = np.column_stack(
            [feature.values for feature in reloaded.transform(features)]
        )

        assert np.array_equal(before, after)

    def test_saving_twice_gives_identical_text(self) -> None:
        """Stability, and the reason it is safe to sort structural keys."""
        first = ridge_document().to_json()
        rebuilt = build_model(ModelDocument.from_json(first))
        second = model_document(rebuilt).to_json()

        assert first == second


class TestLoadedArraysStayFrozen:
    """A bare-array learned part frozen at fit comes back frozen."""

    def test_remembered_targets_are_read_only_after_load(self) -> None:
        fitted = KNearestNeighboursRegressor(n_neighbours=3).fit(FEATURES, TARGET)
        reloaded = build_model(model_document(fitted))

        with pytest.raises(ValueError):
            reloaded._remembered_targets[0] = 999.0
