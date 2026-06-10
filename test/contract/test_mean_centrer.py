"""The contract every backend's MeanCentrer keeps.

The height column ``[10, 20, 30, 40, 50]`` has a mean of 30.0, so it lands on
``[-20, -10, 0, 10, 20]``. Its neighbours stay 10 apart, which is the
assertion that separates a centrer from a scaler that also divides: anything
dividing by the column's spread would bring them closer together. The balance
column ``[-3, 1, 5]`` crosses zero, with a mean of 1.0.

Two assertions separate this scaler from its near misses in the family. The
spread it reports is exactly 1.0 on every column, since nothing is divided by
anything; and a constant column is accepted and answers exact zeros, which
every dividing member refuses.

The engine underneath has two switches that sound alike. ``with_mean=False``
is the one the root mean square decline records as not doing what it says,
since the engine computes the mean anyway and divides by the deviation about
it. ``with_std=False`` is this one, and measured over 600 random blocks it
does exactly what it says: ``scale_`` and ``var_`` are both ``None``, and the
engine's transform equals the block minus ``mean_`` to the last bit on every
one of the 57421 entries.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml import Feature
from oop_ml.core.exceptions import InvalidValuesError, NotFittedError

from .harness import provided

_HEIGHTS = [10.0, 20.0, 30.0, 40.0, 50.0]
_BALANCES = [-3.0, 1.0, 5.0, 1.0, 1.0]
FEATURES = [Feature("height", _HEIGHTS), Feature("balance", _BALANCES)]

HEIGHT_MEAN = 30.0
BALANCE_MEAN = 1.0
CENTRED_HEIGHTS = [-20.0, -10.0, 0.0, 10.0, 20.0]
CENTRED_BALANCES = [-4.0, 0.0, 4.0, 0.0, 0.0]


def column_of(features: list[Feature], name: str) -> np.ndarray:
    """One named column out of a transform's output."""
    return next(feature.values for feature in features if feature.name == name)


def test_it_is_constructed_without_arguments(backend: ModuleType) -> None:
    MeanCentrer = provided(backend, "MeanCentrer")

    assert MeanCentrer().is_fitted is False


def test_it_takes_no_hyperparameters_at_all(backend: ModuleType) -> None:
    """Not even the engine's own switch, which would name a different model."""
    MeanCentrer = provided(backend, "MeanCentrer")

    with pytest.raises(ValidationError):
        MeanCentrer(with_std=True)


def test_it_fits_features_and_returns_itself(backend: ModuleType) -> None:
    MeanCentrer = provided(backend, "MeanCentrer")
    model = MeanCentrer()

    assert model.fit(FEATURES) is model


def test_its_scalings_are_addressable_by_feature_name(backend: ModuleType) -> None:
    MeanCentrer = provided(backend, "MeanCentrer")
    scalings = MeanCentrer().fit(FEATURES).scalings

    assert scalings.n_features == 2
    assert scalings["height"].centre == pytest.approx(HEIGHT_MEAN)
    assert scalings["balance"].centre == pytest.approx(BALANCE_MEAN)


def test_the_spread_is_exactly_one_on_every_column(backend: ModuleType) -> None:
    """Exactly, not approximately. It is not read off the column at all."""
    MeanCentrer = provided(backend, "MeanCentrer")
    scalings = MeanCentrer().fit(FEATURES).scalings

    assert scalings["height"].spread == 1.0
    assert scalings["balance"].spread == 1.0


def test_it_subtracts_the_mean_and_divides_by_nothing(backend: ModuleType) -> None:
    MeanCentrer = provided(backend, "MeanCentrer")

    transformed = MeanCentrer().fit(FEATURES).transform(FEATURES)

    assert [feature.name for feature in transformed] == ["height", "balance"]
    assert np.allclose(column_of(transformed, "height"), CENTRED_HEIGHTS)
    assert np.allclose(column_of(transformed, "balance"), CENTRED_BALANCES)


def test_the_gaps_between_values_are_untouched(backend: ModuleType) -> None:
    """Neighbours 10 apart stay 10 apart, which no dividing scaler manages."""
    MeanCentrer = provided(backend, "MeanCentrer")

    centred = column_of(MeanCentrer().fit(FEATURES).transform(FEATURES), "height")

    assert np.allclose(np.diff(centred), np.diff(_HEIGHTS))


def test_a_constant_column_is_accepted_and_answers_exact_zeros(
    backend: ModuleType,
) -> None:
    """Nothing is divided, so there is no zero to divide by. The engine has
    nothing to patch here either, which makes this the one scaler whose two
    backends cannot disagree about a flat column."""
    MeanCentrer = provided(backend, "MeanCentrer")
    flat = [Feature("flat", [7.0, 7.0, 7.0])]

    transformed = MeanCentrer().fit(flat).transform(flat)

    assert list(column_of(transformed, "flat")) == [0.0, 0.0, 0.0]


def test_held_out_rows_are_centred_by_the_training_mean(backend: ModuleType) -> None:
    MeanCentrer = provided(backend, "MeanCentrer")
    model = MeanCentrer().fit(FEATURES)

    held_out = model.transform([Feature("height", [30.0, 60.0])])

    assert np.allclose(column_of(held_out, "height"), [0.0, 30.0])


def test_inverse_transform_undoes_transform(backend: ModuleType) -> None:
    MeanCentrer = provided(backend, "MeanCentrer")
    model = MeanCentrer().fit(FEATURES)

    restored = model.inverse_transform(model.transform(FEATURES))

    assert np.allclose(column_of(restored, "height"), _HEIGHTS)
    assert np.allclose(column_of(restored, "balance"), _BALANCES)


def test_it_accepts_a_subset_of_the_fitted_features_in_any_order(
    backend: ModuleType,
) -> None:
    MeanCentrer = provided(backend, "MeanCentrer")
    model = MeanCentrer().fit(FEATURES)

    only_balance = model.transform([Feature("balance", _BALANCES)])
    reversed_order = model.transform([FEATURES[1], FEATURES[0]])

    assert [feature.name for feature in only_balance] == ["balance"]
    assert [feature.name for feature in reversed_order] == ["balance", "height"]
    assert np.allclose(column_of(reversed_order, "height"), CENTRED_HEIGHTS)


def test_it_refuses_a_feature_it_never_learned(backend: ModuleType) -> None:
    MeanCentrer = provided(backend, "MeanCentrer")
    model = MeanCentrer().fit(FEATURES)

    with pytest.raises(InvalidValuesError):
        model.transform([Feature("weight", _HEIGHTS)])


def test_it_refuses_to_transform_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    MeanCentrer = provided(backend, "MeanCentrer")

    with pytest.raises(NotFittedError):
        MeanCentrer().transform(FEATURES)

    with pytest.raises(NotFittedError):
        _ = MeanCentrer().scalings
