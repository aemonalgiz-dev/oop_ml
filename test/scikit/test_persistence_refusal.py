"""What saving does when handed a model the other backend fitted.

The registry keys on a bare class name, which was unambiguous while one
backend existed and is not now. Every name in it belongs to two classes and
only the from-scratch one is registered, so the refusal has to say which
``Standardizer`` it means or it reads as a statement that no ``Standardizer``
is persistable, which is false.

The refusal itself is correct and is not the interesting part. What is
recorded alongside it, and measured here rather than asserted, is that the
learned state of several wrappers transplants into the from-scratch namesake
and predicts identically. That is the evidence behind the open question in
CLAUDE.md about whether a document should be backend-neutral, and it is
pinned so the answer does not drift while the question is open.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.exceptions import InvalidDocumentError
from oop_ml.numpy import (
    DecisionTreeRegressor as NumpyDecisionTreeRegressor,
)
from oop_ml.numpy import (
    LassoRegression as NumpyLassoRegression,
)
from oop_ml.numpy import (
    LogisticRegression as NumpyLogisticRegression,
)
from oop_ml.numpy import (
    MultipleLinearRegression as NumpyMultipleLinearRegression,
)
from oop_ml.numpy import (
    RidgeRegression as NumpyRidgeRegression,
)
from oop_ml.numpy import (
    Standardizer as NumpyStandardizer,
)
from oop_ml.numpy.persistence.store import save_model
from oop_ml.scikit import (
    DecisionTreeRegressor as ScikitDecisionTreeRegressor,
)
from oop_ml.scikit import (
    LassoRegression as ScikitLassoRegression,
)
from oop_ml.scikit import (
    LogisticRegression as ScikitLogisticRegression,
)
from oop_ml.scikit import (
    MultipleLinearRegression as ScikitMultipleLinearRegression,
)
from oop_ml.scikit import (
    RidgeRegression as ScikitRidgeRegression,
)
from oop_ml.scikit import (
    Standardizer as ScikitStandardizer,
)

ROWS = [
    Feature("area", np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])),
    Feature("baths", np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])),
]
TARGET = Feature("price", np.array([6.0, 9.0, 14.0, 17.0, 22.0, 25.0, 30.0, 33.0]))


def private_of(model: object) -> dict[str, object]:
    """A fitted model's private attributes, narrowed.

    Pydantic types the attribute as optional because it is unset until the
    first private is assigned, and every model here has been fitted, so the
    only readers of this are asking about a model that certainly has some.
    """
    held = getattr(model, "__pydantic_private__", None)
    assert held is not None, f"{type(model).__name__} holds no private state"
    return held


class TestTheRefusal:
    def test_a_scikit_model_is_refused_by_module_rather_than_by_name(
        self, tmp_path: pathlib.Path
    ) -> None:
        model = ScikitStandardizer().fit(ROWS)

        with pytest.raises(InvalidDocumentError) as raised:
            save_model(model, tmp_path / "model.json")

        message = str(raised.value)
        assert "oop_ml.scikit" in message
        assert "oop_ml.numpy" in message

    def test_the_refusal_does_not_claim_the_name_is_unknown(
        self, tmp_path: pathlib.Path
    ) -> None:
        # The wording that prompted this. A Standardizer plainly is a
        # registered persistable type; it is the other one.
        model = ScikitStandardizer().fit(ROWS)

        with pytest.raises(InvalidDocumentError) as raised:
            save_model(model, tmp_path / "model.json")

        assert "is not a registered persistable type" not in str(raised.value)

    def test_the_from_scratch_namesake_still_saves(
        self, tmp_path: pathlib.Path
    ) -> None:
        target = tmp_path / "model.json"

        save_model(NumpyStandardizer().fit(ROWS), target)

        assert target.exists()


class TestWhatTheLearnedStateWouldCarry:
    """Measured, because it is the evidence behind an open format question.

    A document holds what a model learned rather than what computed it, so for
    a model whose predictions follow entirely from its learned state, a fit
    made by either backend describes the same thing. These four transplant
    exactly. The logistic pair do not, because the from-scratch model records
    ``_passes_run``, which is a fact about the walk it took and which a wrapper
    that never walked has no answer for.
    """

    @pytest.mark.parametrize(
        ("scikit_type", "numpy_type"),
        [
            (ScikitRidgeRegression, NumpyRidgeRegression),
            (ScikitLassoRegression, NumpyLassoRegression),
            (ScikitMultipleLinearRegression, NumpyMultipleLinearRegression),
            (ScikitDecisionTreeRegressor, NumpyDecisionTreeRegressor),
        ],
    )
    def test_a_scikit_fit_transplanted_predicts_identically(
        self, scikit_type: type, numpy_type: type
    ) -> None:
        fitted = scikit_type().fit(ROWS, TARGET)
        wanted = np.asarray(fitted.predict(ROWS), dtype=np.float64)

        restored = numpy_type()
        private = private_of(fitted)
        for field in numpy_type.LEARNED_STATE:
            private_of(restored)[field] = private[field]
        private_of(restored)["_fitted"] = True

        assert np.array_equal(np.asarray(restored.predict(ROWS)), wanted)

    def test_the_tree_transplants_because_the_wrapper_converted_it(self) -> None:
        # The wrapper predicts through its engine, and separately walks that
        # engine's tree into this library's own nodes. It is the nodes that
        # travel, which is why a tree transplants where an engine could not.
        fitted = ScikitDecisionTreeRegressor(max_depth=3).fit(ROWS, TARGET)

        held = private_of(fitted)

        assert "_root" in held
        assert "_engine" in held

    def test_the_logistic_walk_records_something_a_wrapper_cannot(self) -> None:
        label = Feature("passed", np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        fitted = ScikitLogisticRegression().fit(ROWS, label)

        declared = set(NumpyLogisticRegression.LEARNED_STATE)
        held = set(private_of(fitted))

        assert "_passes_run" in declared - held
