"""Spec for the document envelope and what it refuses.

The refusals carry the security story, so they are most of the spec. A saved
model is untrusted input: the tests here are the boundary a document crosses
before any model code runs, and each asserts a *typed* refusal -- an unknown
model type, an unreadable version, a payload of the wrong shape -- rather than
whatever error the failure would otherwise surface as three layers down.
"""

import numpy as np
import pytest

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import InvalidDocumentError, NotFittedError
from oop_ml.numpy.persistence.document import FORMAT_VERSION, ModelDocument
from oop_ml.numpy.persistence.store import (
    PERSISTABLE_TYPES,
    build_model,
    load_model,
    model_document,
    save_model,
)
from oop_ml.numpy.regression.penalised.ridge_regression import RidgeRegression

FEATURES = [
    Feature("floor_area", [72.0, 140.0, 96.0, 210.0]),
    Feature("bathrooms", [1.0, 2.0, 1.0, 3.0]),
]
TARGET = Feature("price", [310.0, 505.0, 372.0, 690.0])


def fitted_ridge() -> RidgeRegression:
    return RidgeRegression(penalty=1.0).fit(FEATURES, TARGET)


class TestTheEnvelope:
    """What a document is, and the JSON trip."""

    def test_a_document_survives_json_and_back(self) -> None:
        document = model_document(fitted_ridge())
        parsed = ModelDocument.from_json(document.to_json())

        assert parsed.model_type == "RidgeRegression"
        assert parsed.format_version == FORMAT_VERSION
        assert parsed.learned == document.learned
        assert parsed.hyperparameters == document.hyperparameters

    def test_the_document_is_readable_data(self) -> None:
        """The saved artifact doubles as a teaching document: a person can
        open it and read what the model learned, by name."""
        text = model_document(fitted_ridge()).to_json()

        assert '"floor_area"' in text
        assert '"penalty": 1.0' in text
        assert "_coefficients" in text

    def test_saving_and_loading_through_a_file(self, tmp_path) -> None:
        model = fitted_ridge()
        path = tmp_path / "model.json"

        save_model(model, path)
        rebuilt = load_model(path)

        assert np.array_equal(
            np.asarray(model.predict(FEATURES)), np.asarray(rebuilt.predict(FEATURES))
        )


class TestWhatItRefuses:
    """The boundary, one typed refusal per way in."""

    def test_an_unfitted_model_cannot_be_saved(self) -> None:
        """Unfitted is configuration, and configuration already round-trips
        through its constructor; a document of nothing would imply otherwise."""
        with pytest.raises(NotFittedError):
            model_document(RidgeRegression())

    def test_an_unknown_model_type_is_refused_by_name(self) -> None:
        """The registry is closed: no string in a file causes an import."""
        document = ModelDocument("EvilModel", {}, {"_x": 1.0})

        with pytest.raises(InvalidDocumentError, match="EvilModel"):
            build_model(document)

    def test_a_configuration_type_cannot_pose_as_a_model(self) -> None:
        """Kernels are registered for hyperparameters, not as top-level models."""
        document = ModelDocument("LinearKernel", {}, {"_x": 1.0})

        with pytest.raises(InvalidDocumentError):
            build_model(document)

    def test_a_future_format_version_is_refused_with_both_numbers(self) -> None:
        document = model_document(fitted_ridge())
        stale = ModelDocument(
            document.model_type,
            document.hyperparameters,
            document.learned,
            format_version=FORMAT_VERSION + 1,
        )

        with pytest.raises(InvalidDocumentError, match=str(FORMAT_VERSION + 1)):
            build_model(stale)

    def test_a_missing_learned_part_is_refused_by_name(self) -> None:
        document = model_document(fitted_ridge())
        learned = document.learned
        learned.pop("_coefficients")
        gutted = ModelDocument(document.model_type, document.hyperparameters, learned)

        with pytest.raises(InvalidDocumentError, match="_coefficients"):
            build_model(gutted)

    def test_non_json_text_is_refused(self) -> None:
        with pytest.raises(InvalidDocumentError):
            ModelDocument.from_json("not a document")

    def test_a_json_document_missing_keys_is_refused(self) -> None:
        with pytest.raises(InvalidDocumentError, match="learned"):
            ModelDocument.from_json('{"model_type": "RidgeRegression"}')

    def test_an_unknown_learned_kind_is_refused(self) -> None:
        document = model_document(fitted_ridge())
        learned = document.learned
        learned["_coefficients"] = {"__kind__": "pickle", "payload": "gg=="}
        tampered = ModelDocument(document.model_type, document.hyperparameters, learned)

        with pytest.raises(InvalidDocumentError, match="pickle"):
            build_model(tampered)


class TestTheRegistry:
    """The closed set, pinned so growth is deliberate."""

    def test_every_registered_name_is_the_class_it_names(self) -> None:
        for name, registered in PERSISTABLE_TYPES.items():
            assert registered.__name__ == name

    def test_every_fittable_registered_type_declares_learned_state(self) -> None:
        """A registered model with an empty LEARNED_STATE would save nothing
        and claim success; the declaration is the format contract."""
        for registered in PERSISTABLE_TYPES.values():
            if isinstance(registered, type) and issubclass(registered, Fittable):
                assert registered.LEARNED_STATE, registered.__name__
