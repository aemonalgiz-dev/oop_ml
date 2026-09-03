"""The contract every backend's RestrictedBoltzmannMachine keeps.

There is no closed-form answer to check a contrastive divergence fit against,
so the known answer here is structural. The rows are two complementary
six-unit patterns with five percent of their bits flipped, and a model that
has learned anything gives the two patterns hidden representations that
differ, rebuilds each on the right side of 0.5 in every unit, and scores both
below a row that mixes them on free energy. Three claims, because each is
weak alone, exactly as the numpy spec argues.

The forward half needs no fit. Whatever the weights, a hidden probability is
a probability, a reconstruction is a block of probabilities, and the shapes
are fixed by the layer widths, and those are asserted on both backends
before any claim about learning is.

Both backends run one full-batch update per epoch at the same rate, and both
spend the first hundred or so epochs breaking the symmetry of their small
initial weights, so the fits here run for five hundred.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.exceptions import InvalidValuesError, NotFittedError
from oop_ml.core.schedule import ConstantSchedule, LinearDecaySchedule

from .harness import provided

FEATURE_NAMES = ("first", "second", "third", "fourth", "fifth", "sixth")
FIRST_PATTERN = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
SECOND_PATTERN = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
MIXED_ROW = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)

N_ROWS = 120
FLIP_CHANCE = 0.05
FIXTURE_SEED = 20260901
EPOCHS = 500


def _generated_block() -> np.ndarray:
    """The training rows, drawn once from a fixed seed so they are a constant."""
    generator = np.random.default_rng(FIXTURE_SEED)
    patterns = np.array([FIRST_PATTERN, SECOND_PATTERN], dtype=np.float64)
    chosen = patterns[generator.integers(0, 2, size=N_ROWS)]
    flipped = (generator.random(chosen.shape) < FLIP_CHANCE).astype(np.float64)

    return np.abs(chosen - flipped)


_ROWS = _generated_block()
FEATURES = [
    Feature(name, _ROWS[:, position]) for position, name in enumerate(FEATURE_NAMES)
]


def as_features(*rows: tuple[float, ...]) -> list[Feature]:
    """The given rows as one feature per visible unit."""
    block = np.array(rows, dtype=np.float64)

    return [
        Feature(name, block[:, position]) for position, name in enumerate(FEATURE_NAMES)
    ]


def matrix_of(features: list[Feature]) -> np.ndarray:
    """A transform's output as a ``(n_rows, n_columns)`` block."""
    return np.column_stack([feature.values for feature in features])


def fitted(backend: ModuleType) -> Any:
    """A three-unit model fitted to the two-pattern rows."""
    RestrictedBoltzmannMachine = provided(backend, "RestrictedBoltzmannMachine")

    return RestrictedBoltzmannMachine(
        n_hidden_units=3, max_epochs=EPOCHS, random_seed=0
    ).fit(FEATURES)


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    RestrictedBoltzmannMachine = provided(backend, "RestrictedBoltzmannMachine")
    model = RestrictedBoltzmannMachine(
        n_hidden_units=3,
        learning_rate=ConstantSchedule(value=0.05),
        n_gibbs_steps=1,
        max_epochs=20,
        random_seed=4,
        tolerance=1e-06,
    )

    assert model.n_hidden_units == 3
    assert model.learning_rate == ConstantSchedule(value=0.05)
    assert model.n_gibbs_steps == 1
    assert model.max_epochs == 20
    assert model.random_seed == 4
    assert model.tolerance == pytest.approx(1e-06)


def test_a_normal_fit_runs_out_of_epochs_rather_than_settling(
    backend: ModuleType,
) -> None:
    """The negative statistic is resampled every epoch, so the walk never
    stops moving; ``converged`` being ``False`` here is not a failed fit."""
    model = fitted(backend)

    assert model.epochs_run == EPOCHS
    assert model.converged is False


def test_a_zero_rate_settles_on_the_first_epoch_and_moves_nothing(
    backend: ModuleType,
) -> None:
    """Both backends start the biases at exactly zero and the weights at
    small normal noise, and a rate of zero scales every change to nothing.
    So the biases stay at zero to the bit, the weights keep their noise, and
    the walk is settled after one epoch, which is all ``converged`` means.

    This is also the claim that catches a wrapper ignoring ``learning_rate``:
    at the engine's default rate the biases move on the first epoch.
    """
    RestrictedBoltzmannMachine = provided(backend, "RestrictedBoltzmannMachine")
    model = RestrictedBoltzmannMachine(
        n_hidden_units=3,
        learning_rate=ConstantSchedule(value=0.0),
        max_epochs=EPOCHS,
        random_seed=0,
    ).fit(FEATURES)

    assert model.converged is True
    assert model.epochs_run == 1
    assert np.array_equal(model.visible_bias, np.zeros(6))
    assert np.array_equal(model.hidden_bias, np.zeros(3))
    assert 0.0 < np.abs(model.weights).max() < 0.1


def test_the_rate_scales_the_first_step(backend: ModuleType) -> None:
    """One epoch is one step, and a step is proportional to the rate.

    Two fits from one seed start from identical weights and draw identical
    chain states, since every draw happens before any change is applied, so
    doubling the rate exactly doubles every change. A wrapper handing the
    engine a rate at the wrong scale, or the engine's own batch cadence, is
    off by a factor here while still training something plausible.
    """
    RestrictedBoltzmannMachine = provided(backend, "RestrictedBoltzmannMachine")

    def after_one_epoch(rate: float) -> Any:
        return RestrictedBoltzmannMachine(
            n_hidden_units=3,
            learning_rate=ConstantSchedule(value=rate),
            max_epochs=1,
            random_seed=0,
        ).fit(FEATURES)

    unmoved = after_one_epoch(0.0)
    small = after_one_epoch(0.05)
    large = after_one_epoch(0.1)

    assert np.max(np.abs(large.visible_bias)) > 0.0
    assert np.max(np.abs(large.hidden_bias)) > 0.0
    assert np.allclose(
        large.weights - unmoved.weights, 2.0 * (small.weights - unmoved.weights)
    )
    assert np.allclose(large.visible_bias, 2.0 * small.visible_bias)
    assert np.allclose(large.hidden_bias, 2.0 * small.hidden_bias)


def test_a_decaying_rate_is_accepted_and_used(backend: ModuleType) -> None:
    """A schedule rather than a number is the whole reason for the field.

    Decaying to zero over the run, the last epoch changes nothing, and the
    fit still differs from the constant-rate one from the same seed.
    """
    RestrictedBoltzmannMachine = provided(backend, "RestrictedBoltzmannMachine")
    decayed = RestrictedBoltzmannMachine(
        n_hidden_units=3,
        learning_rate=LinearDecaySchedule(start=0.1, end=0.0),
        max_epochs=20,
        random_seed=0,
    ).fit(FEATURES)
    constant = RestrictedBoltzmannMachine(
        n_hidden_units=3,
        learning_rate=ConstantSchedule(value=0.1),
        max_epochs=20,
        random_seed=0,
    ).fit(FEATURES)

    assert decayed.epochs_run == 20
    assert np.all(np.isfinite(decayed.weights))
    assert not np.allclose(decayed.weights, constant.weights)


def test_it_fits_features_and_returns_itself(backend: ModuleType) -> None:
    RestrictedBoltzmannMachine = provided(backend, "RestrictedBoltzmannMachine")
    model = RestrictedBoltzmannMachine(n_hidden_units=3, max_epochs=5, random_seed=0)

    assert model.fit(FEATURES) is model


def test_it_learns_one_weight_per_visible_and_hidden_pair(backend: ModuleType) -> None:
    model = fitted(backend)

    assert model.weights.shape == (6, 3)
    assert model.visible_bias.shape == (6,)
    assert model.hidden_bias.shape == (3,)
    assert model.n_visible_units == 6
    assert model.feature_names == FEATURE_NAMES
    assert model.parameters.n_hidden_units == 3


def test_the_learned_arrays_are_read_only(backend: ModuleType) -> None:
    """Every learned buffer, on both routes to it, and not only the weights.

    A stray write raises instead of silently changing every later answer.
    Freezing the weights alone would leave the two biases, which the forward
    pass adds to every score, writeable through an ordinary property read.
    """
    model = fitted(backend)

    for buffer in (
        model.weights,
        model.visible_bias,
        model.hidden_bias,
        model.parameters.weights,
        model.parameters.visible_bias,
        model.parameters.hidden_bias,
    ):
        assert not buffer.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        model.weights[0, 0] = 1.0

    with pytest.raises(ValueError, match="read-only"):
        model.visible_bias[0] = 1.0

    with pytest.raises(ValueError, match="read-only"):
        model.hidden_bias[0] = 1.0


def test_a_refused_refit_leaves_the_earlier_fit_intact(backend: ModuleType) -> None:
    """Compute into locals, assign at the end, checked rather than intended.

    A refit that raises must leave the model as the last successful fit left
    it, rather than half replaced or unfitted. Both backends read every row
    before touching a weight, so the refusal here lands before anything can
    have moved, and this is what says so.
    """
    model = fitted(backend)
    weights = np.array(model.weights)
    unbounded = [
        Feature(name, values)
        for name, values in zip(
            FEATURE_NAMES,
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1.5], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            strict=True,
        )
    ]

    with pytest.raises(InvalidValuesError):
        model.fit(unbounded)

    assert model.is_fitted
    assert model.epochs_run == EPOCHS
    assert model.feature_names == FEATURE_NAMES
    assert np.array_equal(model.weights, weights)


def test_it_transforms_into_one_probability_per_hidden_unit(
    backend: ModuleType,
) -> None:
    model = fitted(backend)

    transformed = model.transform(FEATURES)
    block = matrix_of(transformed)

    assert [feature.name for feature in transformed] == [
        "hidden_1",
        "hidden_2",
        "hidden_3",
    ]
    assert block.shape == (N_ROWS, 3)
    assert np.all((block >= 0.0) & (block <= 1.0))
    assert np.allclose(block, model.hidden_probabilities(FEATURES).values)


def test_the_hidden_layer_tells_the_two_patterns_apart(backend: ModuleType) -> None:
    model = fitted(backend)

    hidden = matrix_of(model.transform(as_features(FIRST_PATTERN, SECOND_PATTERN)))

    assert np.max(np.abs(hidden[0] - hidden[1])) > 0.5


def test_it_rebuilds_both_generating_patterns(backend: ModuleType) -> None:
    model = fitted(backend)
    patterns = np.array([FIRST_PATTERN, SECOND_PATTERN])

    rebuilt = matrix_of(model.reconstruct(as_features(FIRST_PATTERN, SECOND_PATTERN)))

    assert rebuilt.shape == (2, 6)
    assert np.array_equal(rebuilt > 0.5, patterns > 0.5)


def test_reconstruction_error_is_a_small_bounded_mean_square(
    backend: ModuleType,
) -> None:
    error = fitted(backend).reconstruction_error(FEATURES)

    assert isinstance(error, float)
    assert 0.0 <= error < 0.1


def test_free_energy_prefers_the_patterns_to_a_mixed_row(backend: ModuleType) -> None:
    model = fitted(backend)

    energies = np.asarray(
        model.free_energy(as_features(FIRST_PATTERN, SECOND_PATTERN, MIXED_ROW))
    )

    assert energies.shape == (3,)
    assert np.all(np.isfinite(energies))
    assert energies[0] < energies[2]
    assert energies[1] < energies[2]


def test_sampled_states_are_binary(backend: ModuleType) -> None:
    model = fitted(backend)

    hidden = matrix_of(model.sample_hidden(FEATURES))
    visible = matrix_of(model.sample_visible(model.transform(FEATURES)))

    assert hidden.shape == (N_ROWS, 3)
    assert visible.shape == (N_ROWS, 6)
    assert set(np.unique(hidden)) <= {0.0, 1.0}
    assert set(np.unique(visible)) <= {0.0, 1.0}


def test_column_order_does_not_matter(backend: ModuleType) -> None:
    model = fitted(backend)

    forward = matrix_of(model.transform(FEATURES))
    reversed_order = matrix_of(model.transform(FEATURES[::-1]))

    assert np.allclose(forward, reversed_order)


def test_it_refuses_a_value_outside_the_unit_interval(backend: ModuleType) -> None:
    RestrictedBoltzmannMachine = provided(backend, "RestrictedBoltzmannMachine")
    unbounded = [Feature("first", [0.0, 1.5]), Feature("second", [1.0, 0.0])]

    with pytest.raises(InvalidValuesError):
        RestrictedBoltzmannMachine(n_hidden_units=2, max_epochs=2).fit(unbounded)


def test_it_refuses_a_query_over_the_wrong_features(backend: ModuleType) -> None:
    model = fitted(backend)

    with pytest.raises(InvalidValuesError):
        model.transform(FEATURES[:3])


def test_it_refuses_to_transform_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    RestrictedBoltzmannMachine = provided(backend, "RestrictedBoltzmannMachine")

    with pytest.raises(NotFittedError):
        RestrictedBoltzmannMachine(n_hidden_units=3).transform(FEATURES)

    with pytest.raises(NotFittedError):
        _ = RestrictedBoltzmannMachine(n_hidden_units=3).weights

    with pytest.raises(NotFittedError):
        _ = RestrictedBoltzmannMachine(n_hidden_units=3).converged

    with pytest.raises(NotFittedError):
        _ = RestrictedBoltzmannMachine(n_hidden_units=3).epochs_run
