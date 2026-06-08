"""Spec for GaussianMixture.

The oracle is the guarantee expectation-maximisation makes rather than a
number: every round raises the likelihood of the data under the model, or
leaves it where it is. :class:`TestTheLikelihoodOnlyRises` checks that by
running the same fit stopped after one round, two rounds and so on, which needs
nothing from the implementation but its own reported score.

The M step is checked against its definition separately, by taking the fitted
model's responsibilities and refitting from them in plain arithmetic.

What the rest of the file is for is the two relationships the module docstring
claims: that k-means is this algorithm with the shares hardened and the shapes
removed, and that starting from k-means means inheriting where k-means got to.
The second is the uncomfortable one and it is measured rather than glossed.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import NotFittedError, TooFewValuesError
from oop_ml.numpy.clustering.gaussian_mixture import GaussianMixture
from oop_ml.numpy.clustering.k_means import KMeans

_GENERATOR = np.random.default_rng(9)
SOURCES = np.vstack(
    [
        _GENERATOR.multivariate_normal([0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]], size=80),
        _GENERATOR.multivariate_normal([6.0, 1.0], [[2.0, -1.2], [-1.2, 1.5]], size=60),
        _GENERATOR.multivariate_normal([2.0, 6.0], [[0.6, 0.0], [0.0, 0.6]], size=60),
    ]
)
FEATURES = [Feature("left", SOURCES[:, 0]), Feature("right", SOURCES[:, 1])]

_BAND_GENERATOR = np.random.default_rng(21)
_LOWER = np.column_stack(
    [
        _BAND_GENERATOR.normal(scale=5.0, size=80),
        _BAND_GENERATOR.normal(scale=0.4, size=80),
    ]
)
_UPPER = np.column_stack(
    [
        _BAND_GENERATOR.normal(scale=5.0, size=80),
        _BAND_GENERATOR.normal(scale=0.4, size=80) + 3.0,
    ]
)
BAND_ROWS = np.vstack([_LOWER, _UPPER])
#: Two long flat bands stacked three apart. The variance runs along them, so a
#: centre-based grouping cuts across them instead, and a walk that starts there
#: has no way back.
BAND_FEATURES = [
    Feature("left", BAND_ROWS[:, 0]),
    Feature("right", BAND_ROWS[:, 1]),
]
BAND_TRUTH = np.array([0] * 80 + [1] * 80)


def agreement(answered: object, truth: np.ndarray) -> float:
    """How often two labellings of two groups agree, either way round."""
    found = np.asarray(answered).astype(int)

    return max(float((found == truth).mean()), float((found == 1 - truth).mean()))


class TestTheLikelihoodOnlyRises:
    """The guarantee the method rests on, and the reason it terminates."""

    def test_every_extra_round_scores_at_least_as_well(self):
        scores = [
            GaussianMixture(
                n_components=3, random_seed=0, max_iterations=rounds, tolerance=1e-12
            )
            .fit(FEATURES)
            .mean_log_likelihood
            for rounds in range(1, 9)
        ]

        # Not strict: the two lists differ in length by one by
        # construction, which is what pairing consecutive entries means.
        for earlier, later in zip(scores, scores[1:], strict=False):
            assert later >= earlier - 1e-12

    def test_and_it_rises_by_a_visible_amount_before_settling(self):
        """A guard on the guard: if the walk did nothing at all, the run above
        would pass while proving nothing."""
        first = GaussianMixture(
            n_components=3, random_seed=0, max_iterations=1, tolerance=1e-12
        ).fit(FEATURES)
        settled = GaussianMixture(n_components=3, random_seed=0, tolerance=1e-12).fit(
            FEATURES
        )

        assert settled.mean_log_likelihood - first.mean_log_likelihood > 0.01


class TestTheStepsAgainstTheirDefinitions:
    def test_refitting_from_the_responsibilities_reproduces_the_parameters(self):
        """The M step, written out in plain arithmetic from the module
        docstring: a weighted count where an ordinary fit has a count."""
        model = GaussianMixture(n_components=3, random_seed=0, tolerance=1e-12).fit(
            FEATURES
        )
        shares = np.asarray(model.predict_probabilities(FEATURES))
        claimed = shares.sum(axis=0)

        weights = claimed / SOURCES.shape[0]
        means = (shares.T @ SOURCES) / claimed[:, None]

        assert np.allclose(weights, np.asarray(model.weights), atol=1e-6)
        assert np.allclose(means, np.asarray(model.means), atol=1e-6)

        for component in range(3):
            gaps = SOURCES - means[component]
            weighted = (shares[:, component, None] * gaps).T @ gaps / claimed[component]
            expected = weighted + model.covariance_smoothing * np.eye(2)

            assert np.allclose(
                expected, np.asarray(model.covariances)[component], atol=1e-5
            )

    def test_the_responsibilities_are_shares_of_one(self):
        model = GaussianMixture(n_components=3, random_seed=0).fit(FEATURES)
        shares = np.asarray(model.predict_probabilities(FEATURES))

        assert np.allclose(shares.sum(axis=1), 1.0)
        assert shares.min() >= 0.0


class TestTheRelationshipToKMeans:
    def test_one_round_is_still_essentially_the_k_means_grouping(self):
        """Because that is where the walk starts. The first M step is what
        k-means would have reported with shapes added, so stopping immediately
        leaves the two grouping the rows the same way."""
        started = GaussianMixture(n_components=3, random_seed=0, max_iterations=1).fit(
            FEATURES
        )
        centres = KMeans(n_clusters=3, random_seed=0).fit(FEATURES)

        hard = np.asarray(started.predict(FEATURES)).astype(int)
        theirs = np.asarray(centres.predict(FEATURES)).astype(int)
        shared = sum(
            (hard[one] == hard[other]) == (theirs[one] == theirs[other])
            for one in range(len(hard))
            for other in range(one + 1, len(hard))
        )
        pairs = len(hard) * (len(hard) - 1) // 2

        assert shared / pairs > 0.98

    def test_but_the_mixture_keeps_moving_after_k_means_has_stopped(self):
        started = GaussianMixture(n_components=3, random_seed=0, max_iterations=1).fit(
            FEATURES
        )
        settled = GaussianMixture(n_components=3, random_seed=0).fit(FEATURES)

        assert not np.allclose(
            np.asarray(started.means), np.asarray(settled.means), atol=1e-3
        )


class TestWhereTheWalkGetsStuck:
    """Measured rather than glossed, because the textbook claim that a mixture
    finds elongated groups where k-means cannot is true of the model and not of
    this fit."""

    def test_a_k_means_start_is_inherited_rather_than_escaped(self):
        """Two long flat bands. The variance runs along them, so a centre-based
        grouping cuts across, and the walk begins there and stays: every seed
        tried lands near chance."""
        for seed in (0, 1, 2, 3):
            model = GaussianMixture(n_components=2, random_seed=seed).fit(BAND_FEATURES)

            assert agreement(model.predict(BAND_FEATURES), BAND_TRUTH) < 0.7

    def test_and_k_means_is_no_better_on_the_same_rows(self):
        """Which is the point: the mixture is not failing where k-means
        succeeds, it is failing *because* it began where k-means finished."""
        centres = KMeans(n_clusters=2, random_seed=0).fit(BAND_FEATURES)

        assert agreement(centres.predict(BAND_FEATURES), BAND_TRUTH) < 0.7

    def test_the_model_itself_can_describe_those_bands(self):
        """Handed the answer, the shapes fit it. So what is missing is a
        starting point rather than the freedom to represent it, which is why
        this is recorded as a property of the initialisation."""
        model = GaussianMixture(n_components=2, random_seed=0).fit(BAND_FEATURES)
        spreads = [
            float(np.linalg.eigvalsh(component).max())
            for component in np.asarray(model.covariances)
        ]

        assert max(spreads) > 1.0


class TestBeforeFitAndBadInput:
    @pytest.mark.parametrize(
        "attribute",
        ["weights", "means", "covariances", "mean_log_likelihood", "iterations_run"],
    )
    def test_learned_attributes_raise_before_fit(self, attribute):
        with pytest.raises(NotFittedError):
            getattr(GaussianMixture(), attribute)

    def test_more_components_than_rows_is_refused(self):
        with pytest.raises(TooFewValuesError):
            GaussianMixture(n_components=5).fit(
                [
                    Feature("left", np.array([0.0, 1.0, 2.0])),
                    Feature("right", np.array([0.0, 1.0, 4.0])),
                ]
            )
