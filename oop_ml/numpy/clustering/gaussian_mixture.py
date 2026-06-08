"""Group rows by supposing they came from a handful of Gaussians, and asking which.

The fourth clusterer, and the first here that gives a row *shares* of several
groups rather than one group. k-means says row 40 is in cluster 2; this says it
is 0.91 cluster 2 and 0.09 cluster 1, which is a different and often more
honest answer for a row sitting between two clumps.

The model it fits
-----------------
Suppose every row was produced by picking one of ``K`` Gaussians at random,
with probability ``weight_k``, and then drawing from it. That is a *generative*
model of the whole dataset::

    p(x) = sum_k weight_k * N(x | mean_k, covariance_k)

Nothing observes which Gaussian produced which row -- if it did, this would be
:class:`~oop_ml.numpy.classification.generative.quadratic_discriminant_analysis.QuadraticDiscriminantAnalysis`
and the fit would be three counts and some sums. The whole difficulty is that
the labels are missing.

Expectation-maximisation
------------------------
The two halves of the problem are each easy given the other. If the labels were
known, the parameters would be counts and sums; if the parameters were known,
the labels would be a scoring. So do both, alternately, and let each improve
the other:

- **E step.** With the parameters as they stand, work out the *responsibility*
  each component takes for each row: how much of the row's total likelihood
  that component accounts for. That is the same arithmetic
  :func:`~oop_ml.core.gaussian.quadratic_discriminant_scores` does for the
  quadratic discriminant, with the mixing weights standing where the class
  priors stood, followed by the same normalisation. The connection is not a
  coincidence; the discriminant is this model with the responsibilities already
  known.
- **M step.** With the responsibilities as they stand, refit each component as
  if every row belonged to it in that proportion::

      weight_k     = sum_i r_ik / n
      mean_k       = sum_i r_ik x_i / sum_i r_ik
      covariance_k = sum_i r_ik (x_i - mean_k)(x_i - mean_k)' / sum_i r_ik

  Each is the ordinary formula with the count replaced by a weighted count,
  which is what "soft" means here.

Every round raises the likelihood of the data under the model, or leaves it
where it is, so the walk cannot get worse. It can and does settle on a local
maximum rather than the best one, which is why the starting point matters and
why the fit begins from a k-means grouping rather than from noise.

What this says about k-means
-----------------------------
k-means is this algorithm with two things taken away. Harden the
responsibilities so each row belongs entirely to its nearest component, and
force every covariance to be the same multiple of the identity, and the E step
becomes "assign to the nearest centre" and the M step becomes "average the
members". That is Lloyd's algorithm exactly. So k-means was never a separate
idea; it is the special case that drops the shapes and the shares.

Which invites a claim this model does not actually keep. Having a shape per
component, it *can describe* an elongated group where k-means cannot -- but it
starts its walk from a k-means grouping and only ever climbs, so a start that
cut across two long bands is inherited rather than escaped. Measured on two
flat bands stacked three apart, this model, the engine at its own default
initialisation, and k-means itself all land within a few points of chance at
every seed tried; only a different starting point reaches the answer. What the
shapes reliably buy is the soft assignment and a likelihood to compare, not an
escape from where the search began.

What it costs
-------------
A full covariance per component, so ``K p (p + 1) / 2`` numbers on top of the
means, and the same fragility the quadratic discriminant has: a component that
collects too few rows has a covariance that cannot be inverted.
``covariance_smoothing`` adds a small multiple of the identity to every one,
which is what keeps the fit from collapsing onto a single row with zero spread
and infinite likelihood -- a real failure of this model rather than a numerical
one, since that collapse genuinely maximises the objective.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

from oop_ml.core.base.estimator import Clusterer
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.probabilities import ClassScores, ProbabilityMatrix
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.exceptions import TooFewValuesError
from oop_ml.core.gaussian import (
    LOG_TWO_PI,
    log_total_from_log_scores,
    normalised_from_log_scores,
    quadratic_discriminant_scores,
)
from oop_ml.core.types import FloatArray
from oop_ml.numpy.clustering.k_means import KMeans


class GaussianMixture(Clusterer[Sequence[Feature]]):
    """A handful of Gaussians fitted to unlabelled rows by expectation-maximisation.

    Parameters
    ----------
    n_components:
        How many Gaussians the data is supposed to have come from. Given, not
        learned, as for k-means and unlike the density clusterer.
    max_iterations:
        Cap on the alternating rounds.
    tolerance:
        Stop once a round raises the average log likelihood by less than this.
        Larger than most tolerances here, at 1e-3, because the quantity is an
        average log likelihood per row rather than a movement in parameters,
        and it is the engine's default too.
    covariance_smoothing:
        A small multiple of the identity added to every component's covariance.
        Not decoration: without it a component can collapse onto a single row,
        where the likelihood is unbounded, and the fit runs away rather than
        converging.
    random_seed:
        Fixes the k-means grouping the fit starts from, so a fit is
        reproducible. Expectation-maximisation finds a local maximum and which
        one depends on where it began.

    Raises
    ------
    NotFittedError
        From any learned property, or from ``predict``, before ``fit``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_components: int = Field(default=1, ge=1)
    max_iterations: int = Field(default=100, gt=0)
    tolerance: float = Field(default=1e-3, gt=0.0)
    covariance_smoothing: float = Field(default=1e-6, ge=0.0)
    random_seed: int | None = None

    LEARNED_STATE: ClassVar[tuple[str, ...]] = (
        "_feature_names",
        "_weights",
        "_means",
        "_covariances",
        "_mean_log_likelihood",
        "_iterations_run",
        "_converged",
    )

    _feature_names: tuple[str, ...] | None = PrivateAttr(default=None)
    _weights: FloatArray | None = PrivateAttr(default=None)
    _means: FloatArray | None = PrivateAttr(default=None)
    _covariances: FloatArray | None = PrivateAttr(default=None)
    _mean_log_likelihood: float | None = PrivateAttr(default=None)
    _iterations_run: int | None = PrivateAttr(default=None)
    _converged: bool | None = PrivateAttr(default=None)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The columns this model was fitted on, in order."""
        self._check_fitted()
        assert self._feature_names is not None
        return self._feature_names

    @property
    def weights(self) -> FloatArray:
        """How much of the data each component accounts for, summing to one.

        The mixing proportions, and the thing that stands where a classifier's
        class priors stand. A component whose weight has collapsed toward zero
        is one the data did not need.
        """
        self._check_fitted()
        assert self._weights is not None
        return self._weights

    @property
    def means(self) -> FloatArray:
        """Each component's centre, ``(n_components, n_features)``."""
        self._check_fitted()
        assert self._means is not None
        return self._means

    @property
    def covariances(self) -> FloatArray:
        """Each component's shape, ``(n_components, n_features, n_features)``.

        What k-means does not have, and the whole of why this can find an
        elongated or tilted group where that cannot.
        """
        self._check_fitted()
        assert self._covariances is not None
        return self._covariances

    @property
    def mean_log_likelihood(self) -> float:
        """The average log likelihood of a training row under the fitted model.

        What the walk is climbing, per row so that fits on differently sized
        datasets are comparable. Rises every round by construction, which is
        the guarantee expectation-maximisation makes and inertia's fall is the
        k-means version of.
        """
        self._check_fitted()
        assert self._mean_log_likelihood is not None
        return self._mean_log_likelihood

    @property
    def iterations_run(self) -> int:
        """How many rounds of the two steps the last fit took."""
        self._check_fitted()
        assert self._iterations_run is not None
        return self._iterations_run

    @property
    def converged(self) -> bool:
        """Whether the last fit stopped on ``tolerance`` rather than the cap."""
        self._check_fitted()
        assert self._converged is not None
        return self._converged

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Alternate the two steps until the likelihood stops rising.

        Parameters
        ----------
        input_values:
            One or more columns, all the same length, and at least as many rows
            as components.

        Returns
        -------
        Self
            This model, so calls can chain.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If the features disagree in length.
        TooFewValuesError
            If there are fewer rows than components, since a component with no
            rows has no centre.
        CollinearFeaturesError
            If a component's covariance comes out singular, which
            ``covariance_smoothing`` above zero prevents.
        """
        feature_set = FeatureSet(input_values)
        names = tuple(feature.name for feature in feature_set)
        rows = rows_of(feature_set.feature_matrix, names)

        if rows.n_rows < self.n_components:
            raise TooFewValuesError(
                f"{self.n_components} components were asked for and "
                f"{rows.n_rows} row(s) were supplied. A component with no rows "
                "has no centre to describe"
            )

        responsibilities = self._grouped_by_k_means(input_values, rows.n_rows)
        weights, means, covariances = self._refitted(rows.values, responsibilities)

        previous = -np.inf
        mean_log_likelihood = previous
        self._iterations_run = 0
        self._converged = False

        for _ in range(self.max_iterations):
            responsibilities, mean_log_likelihood = self._responsibilities(
                rows.values, weights, means, covariances
            )
            weights, means, covariances = self._refitted(rows.values, responsibilities)

            self._iterations_run += 1
            if mean_log_likelihood - previous < self.tolerance:
                self._converged = True
                break
            previous = mean_log_likelihood

        self._feature_names = names
        self._weights = weights
        self._means = means
        self._covariances = covariances
        self._mean_log_likelihood = float(mean_log_likelihood)

        self._mark_fitted()
        return self

    def predict_probabilities(self, input_values: Sequence[Feature]) -> ClassScores:
        """Each component's share of the responsibility for each row.

        The answer k-means cannot give. A row between two clumps comes back
        divided between them rather than assigned to whichever is marginally
        nearer, which is both more honest and the thing to look at when a
        grouping is being trusted.

        Returns
        -------
        ProbabilityMatrix
            ``(n_rows, n_components)``, rows summing to one.
        """
        block = self._matched_rows(input_values)
        assert self._weights is not None
        assert self._means is not None
        assert self._covariances is not None

        shares, _ = self._responsibilities(
            block.values, self._weights, self._means, self._covariances
        )

        return ProbabilityMatrix(shares)

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The component most responsible for each row.

        The hard answer, for callers that want one. ``predict_probabilities``
        is where the extra information is.
        """
        return Predictions.already_checked(
            np.argmax(
                np.asarray(self.predict_probabilities(input_values)), axis=1
            ).astype(np.float64)
        )

    def _grouped_by_k_means(
        self, input_values: Sequence[Feature], n_rows: int
    ) -> FloatArray:
        """One-hot responsibilities from a k-means grouping, as a starting point.

        The engine this library's other backend wraps starts the same way, and
        the reason is the module docstring's: the walk finds a local maximum,
        so where it begins decides which. Starting from k-means also makes the
        relationship between the two models concrete -- the first M step here
        is exactly what k-means would have reported, with shapes added.
        """
        grouping = KMeans(
            n_clusters=self.n_components, random_seed=self.random_seed
        ).fit(input_values)
        labels = np.asarray(grouping.predict(input_values), dtype=np.int64)

        responsibilities = np.zeros((n_rows, self.n_components), dtype=np.float64)
        responsibilities[np.arange(n_rows), labels] = 1.0

        return responsibilities

    def _responsibilities(
        self,
        rows: FloatArray,
        weights: FloatArray,
        means: FloatArray,
        covariances: FloatArray,
    ) -> tuple[FloatArray, float]:
        """The E step, and the likelihood it implies.

        The scores are the quadratic discriminant's, with the mixing weights
        where the priors were, so the two models share one implementation of
        the arithmetic. Dividing them by their own total gives the
        responsibilities; the total itself, summed in logs, is the likelihood
        of the row under the whole mixture, which is the number the walk
        climbs.

        The constant the discriminant drops has to come back for that second
        use. It cancels in the division, so the responsibilities never needed
        it, but a likelihood compared against anything else does.
        """
        scores = quadratic_discriminant_scores(rows, means, covariances, weights)
        per_row = log_total_from_log_scores(scores) - 0.5 * rows.shape[1] * LOG_TWO_PI

        return normalised_from_log_scores(scores), float(per_row.mean())

    def _refitted(
        self, rows: FloatArray, responsibilities: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """The M step: every summary, with a weighted count in place of a count.

        Returns the three together because they are one answer, not three: they
        are the parameters of one mixture and no caller here wants a subset.
        """
        n_rows, n_features = rows.shape
        claimed = responsibilities.sum(axis=0)

        # A component that has been abandoned entirely would divide by zero,
        # and the answer for it is that it keeps whatever it had; the smoothing
        # floor below is what stops the covariance going with it.
        safe = np.maximum(claimed, np.finfo(np.float64).tiny)

        weights = claimed / n_rows
        means = (responsibilities.T @ rows) / safe[:, None]

        covariances = np.empty(
            (self.n_components, n_features, n_features), dtype=np.float64
        )
        smoothing = self.covariance_smoothing * np.eye(n_features)
        for component in range(self.n_components):
            gaps = rows - means[component]
            weighted = responsibilities[:, component, None] * gaps
            covariances[component] = (weighted.T @ gaps) / safe[component] + smoothing

        return weights, means, covariances

    def _matched_rows(self, input_values: Sequence[Feature]) -> RowBlock:
        """The query columns, in the order the fit saw them.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If a fitted feature is missing.
        """
        self._check_fitted()
        assert self._feature_names is not None

        matched = FeatureSet.matching(self._feature_names, input_values)
        return rows_of(matched.feature_matrix, self._feature_names)
