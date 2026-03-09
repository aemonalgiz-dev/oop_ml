"""Synthetic datasets, each paired with the coefficients that generated it.

Real data is the wrong teaching tool here. When the answer is unknown a fitted
model can only be compared against other fitted models, and every example turns
into a plausibility argument. Generating the data means the truth is available
to print beside the estimate, so a reader watches recovery happen rather than
taking it on faith.

The truth is carried as
:class:`~oop_ml.core.data.coefficients.Coefficients` -- the same type the
models learn -- so the comparison is like against like rather than a bare dict against
an object.
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
    :class:`~oop_ml.preprocessing.polynomial.features.PolynomialFeatures`, which is the
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


class SyntheticClassification:
    """Generated labels and the boundary they were generated from.

    The classification counterpart to :class:`SyntheticRegression`. Labels are
    drawn from the true probabilities rather than assigned by thresholding them,
    so the classes overlap the way real ones do and the fitted boundary has
    something to be uncertain about.

    Parameters
    ----------
    dataset:
        The features and the 0/1 target.
    true_intercept, true_coefficients:
        The log-odds surface the labels were drawn from.
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
        """The features and target, kept together."""
        return self._dataset

    @property
    def input_features(self) -> list[Feature]:
        """Shorthand for ``dataset.input_features``."""
        return self._dataset.input_features

    @property
    def target_feature(self) -> Feature:
        """The 0/1 labels."""
        return self._dataset.target_feature

    @property
    def true_intercept(self) -> float:
        """The intercept of the log-odds surface behind the labels."""
        return self._true_intercept

    @property
    def true_coefficients(self) -> Coefficients:
        """The weights of the log-odds surface behind the labels."""
        return self._true_coefficients

    @property
    def positive_rate(self) -> float:
        """Share of rows whose label is 1."""
        return float(np.mean(self._dataset.target_feature.values))

    def __repr__(self) -> str:
        return (
            f"SyntheticClassification(n_samples={self._dataset.n_samples}, "
            f"positive_rate={self.positive_rate:.3f})"
        )


def _sigmoid(linear_predictor: FloatArray) -> FloatArray:
    """The stable sigmoid, so the generators never overflow on extreme inputs."""
    return np.exp(np.minimum(linear_predictor, 0.0)) / (
        1.0 + np.exp(-np.abs(linear_predictor))
    )


def _labelled(
    predictor_columns: Mapping[str, FloatArray],
    intercept: float,
    weights: Mapping[str, float],
    generator: np.random.Generator,
) -> SyntheticClassification:
    """Draw 0/1 labels from the log-odds surface these weights describe."""
    log_odds = np.full(next(iter(predictor_columns.values())).shape, intercept)
    for name, values in predictor_columns.items():
        log_odds = log_odds + weights[name] * values

    labels = (generator.uniform(size=log_odds.shape) < _sigmoid(log_odds)).astype(float)

    return SyntheticClassification(
        Dataset(
            [Feature(name, values) for name, values in predictor_columns.items()],
            Feature("passed", labels),
        ),
        intercept,
        Coefficients([Coefficient(name, weights[name]) for name in predictor_columns]),
    )


def exam_outcomes(
    sample_count: int = 200, random_seed: int = 21
) -> SyntheticClassification:
    """Hours studied and hours slept against passing, with genuine overlap.

    Labels are sampled from the true probability rather than thresholded, so a
    student who studied very little sometimes passes anyway. That overlap is
    what gives the likelihood a finite maximum to find.
    """
    generator = np.random.default_rng(random_seed)

    return _labelled(
        {
            "hours_studied": generator.uniform(0.0, 8.0, size=sample_count),
            "hours_slept": generator.uniform(3.0, 9.0, size=sample_count),
        },
        intercept=-6.0,
        weights={"hours_studied": 0.9, "hours_slept": 0.4},
        generator=generator,
    )


def separable_outcomes(sample_count: int = 40) -> SyntheticClassification:
    """Two classes with a clean gap between them, and no finite fit.

    Every row below four hours fails and every row above five passes, with
    nothing in between. Any boundary drawn in the gap is perfect on this data,
    so the likelihood keeps rising as the coefficients grow and never attains a
    maximum.
    """
    half = sample_count // 2
    hours = np.concatenate(
        [np.linspace(0.5, 4.0, half), np.linspace(5.0, 9.0, sample_count - half)]
    )
    labels = np.concatenate([np.zeros(half), np.ones(sample_count - half)])

    return SyntheticClassification(
        Dataset([Feature("hours_studied", hours)], Feature("passed", labels)),
        float("nan"),
        Coefficients([Coefficient("hours_studied", float("nan"))]),
    )


def rare_event(
    sample_count: int = 600, random_seed: int = 33
) -> SyntheticClassification:
    """One predictor, and a positive class that turns up about 10% of the time.

    The shape where accuracy stops being evidence. Predicting negative for every
    row scores about 0.90 while finding none of the positives at all.

    The coefficient is large enough that fitted probabilities run most of the
    way from 0 to 1, which is what makes a threshold sweep over this data show
    anything. A weak predictor on a rare class puts every probability under the
    default cut, and then the sweep has only one answer to give.
    """
    generator = np.random.default_rng(random_seed)

    return _labelled(
        {"risk_score": generator.normal(size=sample_count)},
        intercept=-4.5,
        weights={"risk_score": 2.8},
        generator=generator,
    )


def iris_like_species(
    sample_count: int = 300, random_seed: int = 4
) -> SyntheticClassification:
    """Three species over two measurements, deliberately unbalanced.

    Roughly 55 / 30 / 15, because a balanced target hides the whole reason
    macro and micro averaging both exist. Two of the three overlap heavily and
    the third is easier, so the per-class recalls come apart and a single
    accuracy figure stops being an answer.

    The classes are whole positions 0, 1, 2 -- the encoding every multi-class
    model in the library expects, so that class ``k`` is column ``k`` of a
    probability matrix with no lookup table to keep in step.
    """
    generator = np.random.default_rng(random_seed)
    shares = (0.55, 0.30, 0.15)
    centres = ((5.0, 3.4), (5.9, 2.8), (6.6, 3.0))
    spreads = (0.45, 0.40, 0.35)

    lengths, widths, species = [], [], []
    for index, (share, centre, spread) in enumerate(
        zip(shares, centres, spreads, strict=True)
    ):
        count = int(round(sample_count * share))
        lengths.append(generator.normal(centre[0], spread, size=count))
        widths.append(generator.normal(centre[1], spread * 0.6, size=count))
        species.append(np.full(count, float(index)))

    order = generator.permutation(sum(len(part) for part in species))

    return SyntheticClassification(
        Dataset(
            [
                Feature("sepal_length", np.concatenate(lengths)[order]),
                Feature("sepal_width", np.concatenate(widths)[order]),
            ],
            Feature("species", np.concatenate(species)[order]),
        ),
        float("nan"),
        Coefficients([Coefficient("sepal_length", float("nan"))]),
    )


def concentric_rings(sample_count: int = 400, random_seed: int = 7) -> Dataset:
    """Two classes, one enclosing the other, so no straight line separates them.

    There are no true coefficients to carry here, which is the point: the
    boundary is a circle, and a circle is not a hyperplane at any intercept.
    Logistic regression on these columns cannot do better than chance, and it
    fails without complaint -- it fits, it converges, and it is wrong. A
    neighbour model needs no boundary at all and reads the shape straight off
    the rows.

    The two rings are deliberately given enough spread to overlap. Cleanly
    separated ones make the point about shape and then hide the one about
    ``k``: every value from 1 to 15 scores a perfect 1.0, and a table of
    identical numbers teaches nothing about a bias-variance trade.

    Returned as a bare :class:`~oop_ml.model_selection.dataset.Dataset` rather
    than a ``SyntheticClassification`` for that reason. There is nothing to
    print in a "truth" column.
    """
    generator = np.random.default_rng(random_seed)

    inner_count = sample_count // 2
    outer_count = sample_count - inner_count

    angles = generator.uniform(0.0, 2.0 * np.pi, size=sample_count)
    radii = np.concatenate(
        [
            generator.normal(loc=1.0, scale=0.70, size=inner_count),
            generator.normal(loc=2.6, scale=0.70, size=outer_count),
        ]
    )
    labels = np.concatenate([np.zeros(inner_count), np.ones(outer_count)])

    return Dataset(
        [
            Feature("horizontal", radii * np.cos(angles)),
            Feature("vertical", radii * np.sin(angles)),
        ],
        Feature("ring", labels),
    )


def temperature_by_hour(sample_count: int = 240, random_seed: int = 21) -> Dataset:
    """A daily temperature cycle, plus a column measured in the wrong units.

    Two predictors. ``hour`` runs 0 to 24 and carries the whole signal, a sine
    wave that no straight line follows. ``pressure_pascals`` runs near 101325
    and carries nothing at all.

    The second column is what makes this worth having. It is pure noise, and it
    is measured in numbers four orders of magnitude larger than the useful one,
    so an unstandardised distance is very nearly a ranking of pressure alone.
    A regressor that weights its inputs would shrink that column toward zero on
    the evidence; a neighbour model has no coefficients to shrink and no way to
    notice, which is why standardising is part of being correct here rather
    than a convenience.
    """
    generator = np.random.default_rng(random_seed)

    hours = generator.uniform(0.0, 24.0, size=sample_count)
    temperature = 14.0 + 9.0 * np.sin((hours - 9.0) * np.pi / 12.0)

    return Dataset(
        [
            Feature("hour", hours),
            Feature(
                "pressure_pascals",
                101325.0 + generator.normal(scale=180.0, size=sample_count),
            ),
        ],
        Feature(
            "temperature", temperature + generator.normal(scale=0.8, size=sample_count)
        ),
    )
