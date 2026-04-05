"""Spec for the buffer contract: what a caller gets cannot corrupt the model.

Cross-cutting, like ``test_construction.py``, because the promise is the same
everywhere: a validated object's answers do not change because a caller wrote
into an array it handed out or handed in. Two mechanisms carry it -- frozen
buffers (a stray write raises instead of corrupting) and honest ``__array__``
copy semantics.

The ``__array__`` half is numpy-2 specific and worth spelling out. numpy
trusts any object whose ``__array__`` accepts the ``copy`` parameter to honour
it, and adds no copy of its own. The first implementations declared the
parameter and ignored it, so ``np.array(model.predict_probability(...))`` --
whose default is ``copy=True`` -- returned the internal buffer while the
caller believed they held a private copy: measured, mutating that "copy" put a
7.0 inside a validated ``Probabilities``. For frozen buffers the same bug
surfaced differently: writing into the "copy" raised a bare read-only
``ValueError`` from a completely ordinary idiom.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.probabilities import Probabilities
from oop_ml.core.ensemble.bootstrap import BootstrapSample
from oop_ml.core.evaluation.regression import RegressionEvaluation
from oop_ml.core.kernel.functions import PolynomialKernel
from oop_ml.numpy.regression.kernels.kernel_ridge_regression import (
    KernelRidgeRegression,
)


class TestArrayProtocol:
    """np.array(wrapper) means what numpy says it means."""

    def test_the_default_conversion_is_a_genuine_copy(self) -> None:
        """np.array defaults to copy=True, and the wrapper must honour it."""
        probabilities = Probabilities(np.array([0.2, 0.8]))
        converted = np.array(probabilities)

        assert not np.shares_memory(converted, probabilities.values)

        converted[0] = 7.0

        assert float(probabilities.values[0]) == pytest.approx(0.2)

    def test_the_copy_of_a_frozen_buffer_is_writable(self) -> None:
        """The ordinary idiom must not raise a read-only ValueError."""
        predictions = Predictions(np.array([1.0, 2.0, 3.0]))
        converted = np.array(predictions)

        converted[0] = 5.0

        assert float(converted[0]) == pytest.approx(5.0)

    def test_copy_false_with_a_forced_conversion_raises(self) -> None:
        """The protocol's third case: a view was demanded and none exists."""
        probabilities = Probabilities(np.array([0.2, 0.8]))

        with pytest.raises(ValueError, match="copy"):
            probabilities.__array__(dtype=np.float32, copy=False)

    def test_asarray_may_still_share(self) -> None:
        """copy=None allows the view, which is what keeps wrapping cheap."""
        probabilities = Probabilities(np.array([0.2, 0.8]))

        assert np.shares_memory(np.asarray(probabilities), probabilities.values)


class TestFrozenBuffers:
    """A stray write raises instead of silently changing later answers."""

    def test_a_probabilities_buffer_cannot_be_unvalidated(self) -> None:
        """The bounds check would otherwise be revocable after the fact."""
        source = np.array([0.2, 0.8])
        probabilities = Probabilities(source)

        source[0] = 7.0  # the caller's own array stays theirs

        assert float(probabilities.values[0]) == pytest.approx(0.2)

        with pytest.raises(ValueError):
            np.asarray(probabilities)[0] = 7.0

    def test_residuals_cannot_be_rewritten_under_the_metrics(self) -> None:
        """Every metric derives from this buffer, and it is handed out."""
        evaluation = RegressionEvaluation([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])

        with pytest.raises(ValueError):
            evaluation.residuals[0] = 100.0

        assert evaluation.residual_sum_of_squares == pytest.approx(1.0)

    def test_a_bootstrap_draw_cannot_be_redrawn_by_hand(self) -> None:
        """in_bag and the out-of-bag grid are recomputed from these positions."""
        sample = BootstrapSample.draw(10, np.random.default_rng(0))

        with pytest.raises(ValueError):
            sample.drawn[0] = 9


class TestFittedStateStaysFitted:
    """Mutating what a fitted model handed out does not change its answers."""

    def test_kernel_ridge_training_rows_are_a_copy(self) -> None:
        """The internal block is what predict kernels every query against."""
        features = [Feature("first", [1.0, 2.0, 3.0, 4.0])]
        target = Feature("outcome", [1.0, 4.0, 9.0, 16.0])
        model = KernelRidgeRegression(
            kernel=PolynomialKernel(degree=2), penalty=0.01
        ).fit(features, target)

        before = np.array(model.predict(features))

        model.training_rows.values[:] = 999.0

        after = np.array(model.predict(features))

        assert np.allclose(before, after)
