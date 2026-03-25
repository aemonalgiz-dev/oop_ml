"""Saving a fitted model and getting the same answers back.

The mechanics live here: the closed registry of types a document may name, the
codec that turns learned values into plain data and back, and the four public
calls -- :func:`model_document`, :func:`build_model`, :func:`save_model`,
:func:`load_model`.

What makes a model persistable
------------------------------
Each model declares ``LEARNED_STATE``: the private attributes that constitute
its fitted self, in document order. That tuple *is* the format contract for the
class -- renaming a private attribute listed there is a format change, which is
exactly why the declaration is explicit rather than reflected: nothing outside
the tuple is ever written, so runtime-only state (a tree's random generator)
stays out of the document by construction.

Restoring goes the other way with the same guarantees the library gives fresh
data: hyperparameters through the pydantic constructor, learned values through
the vocabulary's own validating constructors, and only then is the model marked
fitted. A document is untrusted input and is treated as one.

Why the codec dispatches on type rather than each model writing its own
-----------------------------------------------------------------------
The learned state of twenty-six models is built from about a dozen value
types -- arrays, name-bound coefficients, row blocks, trees, centroids,
components, fitted sub-models. One encoder/decoder pair per *type*, written
once, keeps the per-model cost to the ``LEARNED_STATE`` tuple and makes a new
model persistable the moment its state is made of known parts. A model per
bespoke serializer would be twenty-six copies of the same decisions, and
copies drift.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

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
from oop_ml.core.clustering.centroids import Centroid, Centroids
from oop_ml.core.clustering.clustering import Clustering
from oop_ml.core.data.coefficients import Coefficient, Coefficients
from oop_ml.core.data.dataset import Dataset
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.decomposition.components import (
    PrincipalComponent,
    PrincipalComponents,
)
from oop_ml.core.ensemble.bootstrap import BootstrapSample
from oop_ml.core.exceptions import InvalidDocumentError, NotFittedError
from oop_ml.core.kernel.functions import (
    LinearKernel,
    PolynomialKernel,
    RadialBasisKernel,
    SigmoidKernel,
)
from oop_ml.core.kernel.matrix import KernelMatrix
from oop_ml.core.tree.node import (
    ClassificationLeaf,
    DecisionNode,
    LeafNode,
    TreeNode,
)
from oop_ml.core.tree.split import Split
from oop_ml.decomposition.kernel_principal_component_analysis import (
    KernelComponent,
    KernelComponents,
    KernelPrincipalComponentAnalysis,
)
from oop_ml.decomposition.principal_component_analysis import (
    PrincipalComponentAnalysis,
)
from oop_ml.persistence.document import ModelDocument
from oop_ml.pipeline.pipelines import (
    ClassificationPipeline,
    RegressionPipeline,
)
from oop_ml.pipeline.steps import PipelineStep, PipelineSteps
from oop_ml.preprocessing.polynomial.features import PolynomialFeatures
from oop_ml.preprocessing.polynomial.terms import PolynomialTerm, PolynomialTerms
from oop_ml.preprocessing.standardization.scaling import (
    FeatureScaling,
    FeatureScalings,
)
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

PERSISTABLE_TYPES: dict[str, type[BaseModel]] = {
    persistable.__name__: persistable
    for persistable in (
        # regression
        SimpleLinearRegression,
        MultipleLinearRegression,
        GradientDescentRegression,
        RidgeRegression,
        LassoRegression,
        KNearestNeighboursRegressor,
        DecisionTreeRegressor,
        BaggingRegressor,
        RandomForestRegressor,
        GradientBoostingRegressor,
        KernelRidgeRegression,
        # classification
        LogisticRegression,
        NewtonLogisticRegression,
        MultinomialLogisticRegression,
        OneVsRestClassifier,
        KNearestNeighboursClassifier,
        DecisionTreeClassifier,
        BaggingClassifier,
        RandomForestClassifier,
        SupportVectorClassifier,
        # unsupervised
        KMeans,
        PrincipalComponentAnalysis,
        KernelPrincipalComponentAnalysis,
        # preprocessing and composition
        Standardizer,
        PolynomialFeatures,
        RegressionPipeline,
        ClassificationPipeline,
        # kernels appear inside hyperparameters, never as top-level models
        LinearKernel,
        PolynomialKernel,
        RadialBasisKernel,
        SigmoidKernel,
    )
}
"""Every class a document may name, and the only ones it may.

A dict rather than an import-by-path precisely so the set is closed: a
document naming anything else is refused with the list of what exists, and no
string in a file can ever cause an import.
"""


def model_document(model: Fittable) -> ModelDocument:
    """The fitted model as a document.

    Raises
    ------
    NotFittedError
        If the model has not been fitted; an unfitted model is configuration,
        and configuration alone round-trips through its constructor already.
    InvalidDocumentError
        If the model's type is not registered, or declares no learned state.
    """
    if not model.is_fitted:
        raise NotFittedError(
            f"{type(model).__name__} must be fitted before it can be saved"
        )

    _check_registered(type(model))

    learned_parts = _learned_parts_of(type(model))

    return ModelDocument(
        type(model).__name__,
        _encoded_hyperparameters(model),
        {name: _encoded(getattr(model, name)) for name in learned_parts},
    )


def build_model(document: ModelDocument) -> Any:
    """A fitted model rebuilt from its document, revalidated on the way.

    Raises
    ------
    InvalidDocumentError
        If the document's version or model type is unreadable, if a learned
        part is missing, or if any payload is malformed.
    MLLibError
        Whichever typed error a vocabulary constructor raises when a learned
        value violates its invariants -- the same refusal the equivalent bad
        data would meet anywhere else in the library.
    """
    document.check_readable()

    model_type = _registered_type(document.model_type)
    model = model_type(**_decoded_hyperparameters(document.hyperparameters))

    if not isinstance(model, Fittable):
        raise InvalidDocumentError(
            f"{document.model_type} is a configuration type, not a fittable "
            f"model; it belongs inside another document's hyperparameters"
        )

    learned = document.learned
    for name in _learned_parts_of(model_type):
        if name not in learned:
            raise InvalidDocumentError(
                f"the document for {document.model_type} is missing the "
                f"learned part {name!r}"
            )
        setattr(model, name, _decoded(learned[name]))

    model._mark_fitted()

    return model


def save_model(model: Fittable, path: str | Path) -> None:
    """Write the fitted model's document to ``path`` as JSON."""
    Path(path).write_text(model_document(model).to_json(), encoding="utf-8")


def load_model(path: str | Path) -> Any:
    """Read a document from ``path`` and rebuild its model, revalidating."""
    return build_model(ModelDocument.from_json(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# hyperparameters: pydantic fields, with nested models flattened to data
# ---------------------------------------------------------------------------


def _encoded_hyperparameters(model: BaseModel) -> dict[str, Any]:
    """Every field, with nested models encoded as configuration documents.

    ``model_dump`` is deliberately not used: it recurses a nested model into a
    plain dict that loses its type, which is the exact failure
    ``Candidate.applied_to`` documents. Fields are read with ``getattr`` and
    each nested model keeps its name.
    """
    return {
        name: _encoded_field(getattr(model, name)) for name in type(model).model_fields
    }


def _encoded_field(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value

    if isinstance(value, PipelineSteps):
        return {
            "__kind__": "configured_steps",
            "steps": [
                {"name": step.name, "transformer": _encoded_field(step.transformer)}
                for step in value
            ],
        }

    if isinstance(value, BaseModel):
        _check_registered(type(value))

        return {
            "__kind__": "configured_model",
            "model_type": type(value).__name__,
            "hyperparameters": _encoded_hyperparameters(value),
        }

    return value


def _decoded_hyperparameters(payload: dict[str, Any]) -> dict[str, Any]:
    return {name: _decoded_field(value) for name, value in payload.items()}


def _decoded_field(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__kind__") == "configured_model":
        model_type = _registered_type(value["model_type"])

        return model_type(**_decoded_hyperparameters(value["hyperparameters"]))

    if isinstance(value, dict) and value.get("__kind__") == "configured_steps":
        return PipelineSteps(
            [
                PipelineStep(step["name"], _decoded_field(step["transformer"]))
                for step in value["steps"]
            ]
        )

    return value


# ---------------------------------------------------------------------------
# learned state: one encoder/decoder pair per value type
# ---------------------------------------------------------------------------


def _encoded(value: Any) -> Any:
    """One learned value as plain data, tagged with what it was."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, np.ndarray):
        return {
            "__kind__": "array",
            "dtype": str(value.dtype),
            "values": value.tolist(),
        }

    if isinstance(value, tuple):
        return {"__kind__": "tuple", "items": [_encoded(item) for item in value]}

    if isinstance(value, dict):
        return {
            "__kind__": "mapping",
            "items": {name: _encoded(item) for name, item in value.items()},
        }

    if isinstance(value, Coefficients):
        return {
            "__kind__": "Coefficients",
            "coefficients": [{"name": one.name, "value": one.value} for one in value],
        }

    if isinstance(value, RowBlock):
        return {
            "__kind__": "RowBlock",
            "feature_names": list(value.feature_names),
            "values": _encoded(value.values),
        }

    if isinstance(value, KernelMatrix):
        return {"__kind__": "KernelMatrix", "values": _encoded(value.values)}

    if isinstance(value, BootstrapSample):
        return {
            "__kind__": "BootstrapSample",
            "drawn": _encoded(value.drawn),
            "n_rows": value.n_rows,
        }

    if isinstance(value, Dataset):
        return {
            "__kind__": "Dataset",
            "input_features": [
                {"name": feature.name, "values": _encoded(feature.values)}
                for feature in value.input_features
            ],
            "target_feature": {
                "name": value.target_feature.name,
                "values": _encoded(value.target_feature.values),
            },
        }

    if isinstance(value, FeatureScalings):
        return {
            "__kind__": "FeatureScalings",
            "scalings": [
                {
                    "name": one.name,
                    "mean": one.mean,
                    "standard_deviation": one.standard_deviation,
                }
                for one in value
            ],
        }

    if isinstance(value, PolynomialTerms):
        return {
            "__kind__": "PolynomialTerms",
            "terms": [{"powers": dict(term.powers)} for term in value],
        }

    if isinstance(value, PrincipalComponents):
        return {
            "__kind__": "PrincipalComponents",
            "components": [
                {
                    "name": one.name,
                    "loadings": _encoded(one.loadings),
                    "variance": one.variance,
                }
                for one in value
            ],
            "total_variance": value.total_variance,
        }

    if isinstance(value, KernelComponents):
        return {
            "__kind__": "KernelComponents",
            "components": [
                {
                    "name": one.name,
                    "row_coefficients": _encoded(one.row_coefficients),
                    "variance": one.variance,
                }
                for one in value
            ],
            "total_variance": value.total_variance,
        }

    if isinstance(value, Clustering):
        return {
            "__kind__": "Clustering",
            "labels": _encoded(value.labels),
            "centroids": [
                {
                    "name": one.name,
                    "coordinates": _encoded(one.coordinates),
                    "feature_names": list(one.feature_names),
                }
                for one in value.centroids
            ],
            "inertia": value.inertia,
        }

    if isinstance(value, TreeNode):
        return {"__kind__": "TreeNode", "node": _encoded_node(value)}

    if isinstance(value, PipelineSteps):
        return {
            "__kind__": "fitted_steps",
            "steps": [
                {"name": step.name, "document": _document_payload(step.transformer)}
                for step in value
            ],
        }

    if isinstance(value, Fittable):
        return {"__kind__": "fitted_model", "document": _document_payload(value)}

    raise InvalidDocumentError(
        f"no codec knows how to persist a {type(value).__name__}"
    )


def _decoded(value: Any) -> Any:
    """One plain-data payload back into the value it encodes, revalidated."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if not isinstance(value, dict) or "__kind__" not in value:
        raise InvalidDocumentError(
            f"a learned value must be a primitive or a tagged object; got "
            f"{type(value).__name__}"
        )

    kind = value["__kind__"]

    if kind == "array":
        return np.array(value["values"], dtype=np.dtype(value["dtype"]))

    if kind == "tuple":
        return tuple(_decoded(item) for item in value["items"])

    if kind == "mapping":
        return {name: _decoded(item) for name, item in value["items"].items()}

    if kind == "Coefficients":
        return Coefficients(
            [Coefficient(one["name"], one["value"]) for one in value["coefficients"]]
        )

    if kind == "RowBlock":
        return rows_of(_decoded(value["values"]), tuple(value["feature_names"]))

    if kind == "KernelMatrix":
        return KernelMatrix(_decoded(value["values"]))

    if kind == "BootstrapSample":
        return BootstrapSample(_decoded(value["drawn"]), value["n_rows"])

    if kind == "Dataset":
        return Dataset(
            [
                Feature(one["name"], _decoded(one["values"]))
                for one in value["input_features"]
            ],
            Feature(
                value["target_feature"]["name"],
                _decoded(value["target_feature"]["values"]),
            ),
        )

    if kind == "FeatureScalings":
        return FeatureScalings(
            [
                FeatureScaling(one["name"], one["mean"], one["standard_deviation"])
                for one in value["scalings"]
            ]
        )

    if kind == "PolynomialTerms":
        return PolynomialTerms(
            [PolynomialTerm(term["powers"]) for term in value["terms"]]
        )

    if kind == "PrincipalComponents":
        return PrincipalComponents(
            [
                PrincipalComponent(
                    one["name"], _decoded(one["loadings"]), one["variance"]
                )
                for one in value["components"]
            ],
            value["total_variance"],
        )

    if kind == "KernelComponents":
        return KernelComponents(
            [
                KernelComponent(
                    one["name"], _decoded(one["row_coefficients"]), one["variance"]
                )
                for one in value["components"]
            ],
            value["total_variance"],
        )

    if kind == "Clustering":
        return Clustering(
            _decoded(value["labels"]),
            Centroids(
                [
                    Centroid(
                        one["name"],
                        _decoded(one["coordinates"]),
                        tuple(one["feature_names"]),
                    )
                    for one in value["centroids"]
                ]
            ),
            value["inertia"],
        )

    if kind == "TreeNode":
        return _decoded_node(value["node"])

    if kind == "fitted_steps":
        return PipelineSteps(
            [
                PipelineStep(step["name"], build_model(_document_of(step["document"])))
                for step in value["steps"]
            ]
        )

    if kind == "fitted_model":
        return build_model(_document_of(value["document"]))

    raise InvalidDocumentError(f"unknown learned-value kind {kind!r}")


# ---------------------------------------------------------------------------
# trees: recursive, one shape per node class
# ---------------------------------------------------------------------------


def _encoded_node(node: TreeNode) -> dict[str, Any]:
    if isinstance(node, DecisionNode):
        return {
            "node_type": "decision",
            "split": {
                "feature_index": node.split.feature_index,
                "feature_name": node.split.feature_name,
                "threshold": node.split.threshold,
                "gain": node.split.gain,
            },
            "left": _encoded_node(node.left),
            "right": _encoded_node(node.right),
            "n_samples": node.n_samples,
            "impurity": node.impurity,
        }

    if isinstance(node, ClassificationLeaf):
        return {
            "node_type": "classification_leaf",
            "prediction": node.prediction,
            "class_shares": _encoded(node.class_shares),
            "n_samples": node.n_samples,
            "impurity": node.impurity,
        }

    if isinstance(node, LeafNode):
        return {
            "node_type": "leaf",
            "prediction": node.prediction,
            "n_samples": node.n_samples,
            "impurity": node.impurity,
        }

    raise InvalidDocumentError(f"no codec knows how to persist a {type(node).__name__}")


def _decoded_node(payload: dict[str, Any]) -> TreeNode:
    node_type = payload.get("node_type")

    if node_type == "decision":
        split = payload["split"]

        return DecisionNode(
            Split(
                split["feature_index"],
                split["feature_name"],
                split["threshold"],
                split["gain"],
            ),
            _decoded_node(payload["left"]),
            _decoded_node(payload["right"]),
            payload["n_samples"],
            payload["impurity"],
        )

    if node_type == "classification_leaf":
        return ClassificationLeaf(
            payload["prediction"],
            _decoded(payload["class_shares"]),
            payload["n_samples"],
            payload["impurity"],
        )

    if node_type == "leaf":
        return LeafNode(
            payload["prediction"], payload["n_samples"], payload["impurity"]
        )

    raise InvalidDocumentError(f"unknown tree node type {node_type!r}")


# ---------------------------------------------------------------------------
# plumbing shared by the public calls
# ---------------------------------------------------------------------------


def _document_payload(model: Fittable) -> dict[str, Any]:
    """A nested fitted model's document, as the plain dict JSON will hold."""
    nested = model_document(model)

    return {
        "format_version": nested.format_version,
        "model_type": nested.model_type,
        "hyperparameters": nested.hyperparameters,
        "learned": nested.learned,
    }


def _document_of(payload: dict[str, Any]) -> ModelDocument:
    return ModelDocument(
        payload["model_type"],
        payload["hyperparameters"],
        payload["learned"],
        payload["format_version"],
    )


def _registered_type(name: str) -> type[BaseModel]:
    if name not in PERSISTABLE_TYPES:
        raise InvalidDocumentError(
            f"unknown model type {name!r}; this build knows "
            f"{', '.join(sorted(PERSISTABLE_TYPES))}"
        )

    return PERSISTABLE_TYPES[name]


def _check_registered(model_type: type) -> None:
    registered = PERSISTABLE_TYPES.get(model_type.__name__)

    if registered is not model_type:
        raise InvalidDocumentError(
            f"{model_type.__name__} is not a registered persistable type"
        )


def _learned_parts_of(model_type: type) -> tuple[str, ...]:
    parts = getattr(model_type, "LEARNED_STATE", ())

    if not parts:
        raise InvalidDocumentError(
            f"{model_type.__name__} declares no learned state to persist"
        )

    return parts
