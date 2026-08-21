"""Synthetic datasets, each paired with the coefficients that generated it.

Real data is the wrong teaching tool here. When the answer is unknown a fitted
model can only be compared against other fitted models, and every example turns
into a plausibility argument. Generating the data means the truth is available
to print beside the estimate, so a reader watches recovery happen rather than
taking it on faith.

The truth is carried as :class:`~oop_ml.core.coefficients.Coefficients` -- the same type
the models learn -- so the comparison is like against like rather than a bare
dict against an object.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from oop_ml import Coefficient, Coefficients, Dataset, Feature, FloatArray


class SyntheticRegression:
    """A generated dataset and the coefficients it was generated from.

    Parameters
    ----------
    dataset:
        The features and target a model will be fitted to.
    true_intercept:
        The intercept used to generate the target.
    true_coefficients:
        The per-feature weights used to generate the target. Noise means a fit
        will not reproduce these exactly; it should come close.
    """

    __slots__ = ("_dataset", "_true_coefficients", "_true_intercept")

    def __init__(
        self,
        dataset: Dataset,
        true_intercept: float,
        true_coefficients: Coefficients,
    ) -> None:
        self._dataset = dataset
        self._true_intercept = true_intercept
        self._true_coefficients = true_coefficients

    @property
    def dataset(self) -> Dataset:
        """The features and target, kept together so they cannot drift apart."""
        return self._dataset

    @property
    def true_intercept(self) -> float:
        """The intercept the target was generated from."""
        return self._true_intercept

    @property
    def true_coefficients(self) -> Coefficients:
        """The weights the target was generated from."""
        return self._true_coefficients

    @property
    def input_features(self) -> list[Feature]:
        """Shorthand for ``dataset.input_features``, which examples use a lot."""
        return self._dataset.input_features

    @property
    def target_feature(self) -> Feature:
        """Shorthand for ``dataset.target_feature``."""
        return self._dataset.target_feature

    def __repr__(self) -> str:
        return (
            f"SyntheticRegression(n_samples={self._dataset.n_samples}, "
            f"n_features={self._dataset.n_features})"
        )


def _assembled(
    predictor_columns: Mapping[str, FloatArray],
    intercept: float,
    weights: Mapping[str, float],
    noise: FloatArray,
) -> SyntheticRegression:
    """Build the target from named columns and the weights behind them."""
    target_values = np.full(noise.shape, intercept) + noise
    for name, values in predictor_columns.items():
        target_values = target_values + weights[name] * values

    return SyntheticRegression(
        Dataset(
            [Feature(name, values) for name, values in predictor_columns.items()],
            Feature("y", target_values),
        ),
        intercept,
        Coefficients([Coefficient(name, weights[name]) for name in predictor_columns]),
    )


def straight_line(
    sample_count: int = 30, noise_scale: float = 2.0, random_seed: int = 0
) -> SyntheticRegression:
    """One predictor, one response: ``y = 4 + 2.5x + noise``."""
    generator = np.random.default_rng(random_seed)

    return _assembled(
        {"x": generator.uniform(0.0, 20.0, size=sample_count)},
        intercept=4.0,
        weights={"x": 2.5},
        noise=generator.normal(scale=noise_scale, size=sample_count),
    )


def independent_predictors(
    sample_count: int = 60, noise_scale: float = 1.0, random_seed: int = 1
) -> SyntheticRegression:
    """Three uncorrelated predictors, well conditioned -- the easy case."""
    generator = np.random.default_rng(random_seed)

    return _assembled(
        {
            "rooms": generator.normal(loc=5.0, scale=1.5, size=sample_count),
            "age_years": generator.uniform(0.0, 60.0, size=sample_count),
            "distance_km": generator.uniform(0.5, 25.0, size=sample_count),
        },
        intercept=120.0,
        weights={"rooms": 30.0, "age_years": -1.2, "distance_km": -4.0},
        noise=generator.normal(scale=noise_scale, size=sample_count),
    )


def collinear_predictors(
    sample_count: int = 40, noise_scale: float = 1.5, random_seed: int = 7
) -> SyntheticRegression:
    """Two predictors that nearly duplicate one another.

    ``second`` is ``first`` plus a whisper of noise, so the design matrix is
    close to singular and least squares has almost no information with which to
    split the two weights apart. This is the case a penalty exists for.
    """
    generator = np.random.default_rng(random_seed)
    first = generator.normal(size=sample_count)
    second = 0.95 * first + 0.05 * generator.normal(size=sample_count)

    return _assembled(
        {"first": first, "second": second},
        intercept=1.0,
        weights={"first": 3.0, "second": -2.0},
        noise=generator.normal(scale=noise_scale, size=sample_count),
    )


def sparse_signal(
    sample_count: int = 50, noise_scale: float = 0.5, random_seed: int = 3
) -> SyntheticRegression:
    """Six predictors of which only two drive the target.

    The other four have a true weight of exactly zero. A method that can set
    coefficients to zero should find them; one that only shrinks cannot.
    """
    generator = np.random.default_rng(random_seed)
    weights = {
        "signal_a": 5.0,
        "signal_b": -3.0,
        "noise_c": 0.0,
        "noise_d": 0.0,
        "noise_e": 0.0,
        "noise_f": 0.0,
    }

    return _assembled(
        {name: generator.normal(size=sample_count) for name in weights},
        intercept=0.0,
        weights=weights,
        noise=generator.normal(scale=noise_scale, size=sample_count),
    )


def mixed_units(
    sample_count: int = 50, noise_scale: float = 2.0, random_seed: int = 5
) -> SyntheticRegression:
    """Two predictors whose scales differ by two orders of magnitude.

    ``floor_area_sqm`` runs into the hundreds while ``bathrooms`` runs one to
    four. Nothing is wrong with the data -- but a single learning rate and a
    single penalty both treat those columns as comparable, and they are not.
    """
    generator = np.random.default_rng(random_seed)

    return _assembled(
        {
            "floor_area_sqm": generator.uniform(40.0, 400.0, size=sample_count),
            "bathrooms": generator.integers(1, 5, size=sample_count).astype(float),
        },
        intercept=15.0,
        weights={"floor_area_sqm": 0.8, "bathrooms": 12.0},
        noise=generator.normal(scale=noise_scale, size=sample_count),
    )


def wide_correlated_design(
    sample_count: int = 110,
    feature_count: int = 30,
    noise_scale: float = 5.0,
    random_seed: int = 13,
) -> SyntheticRegression:
    """Many predictors, all driven by one shared factor, on modest data.

    This is the shape where a penalty genuinely earns its place: enough columns
    relative to rows that least squares starts fitting noise, and enough
    correlation between them that it cannot tell which column deserves the
    credit. Held-out score then peaks at some penalty above zero rather than
    falling from it, which is what makes choosing one a real decision.
    """
    generator = np.random.default_rng(random_seed)
    shared_factor = generator.normal(size=sample_count)
    predictor_columns = {
        f"x{index + 1}": 0.9 * shared_factor + 0.3 * generator.normal(size=sample_count)
        for index in range(feature_count)
    }
    drawn_weights = generator.normal(scale=2.0, size=feature_count)

    return _assembled(
        predictor_columns,
        intercept=2.0,
        weights={
            name: float(weight)
            for name, weight in zip(predictor_columns, drawn_weights, strict=True)
        },
        noise=generator.normal(scale=noise_scale, size=sample_count),
    )


def quadratic_curve(
    sample_count: int = 25, noise_scale: float = 3.0, random_seed: int = 11
) -> SyntheticRegression:
    """A genuinely curved relationship: ``y = 5 - 4x + 1.5x^2 + noise``.

    Handed back with the single column ``x``. Recovering the curve is the job of
    :class:`~oop_ml.preprocessing.polynomial_features.PolynomialFeatures`, which is the
    whole point of the example that uses this.
    """
    generator = np.random.default_rng(random_seed)
    predictor = np.linspace(-4.0, 6.0, sample_count)
    noise = generator.normal(scale=noise_scale, size=sample_count)

    return SyntheticRegression(
        Dataset(
            [Feature("x", predictor)],
            Feature("y", 5.0 - 4.0 * predictor + 1.5 * predictor**2 + noise),
        ),
        5.0,
        Coefficients([Coefficient("x", -4.0), Coefficient("x^2", 1.5)]),
    )
