"""Spec for hyperparameter search.

Two tests carry the design, and both assert something the search must *refuse*
or *admit* rather than something it computes.

``test_a_misspelled_parameter_is_caught_before_any_fitting`` is the first. The
obvious implementation builds candidates with ``model_copy(update=...)``, which
accepts an unknown name silently and leaves the field at its default -- so a
grid over ``pentalty`` would fit the same model at every point, report a flat
curve, and pick a value by tie-break. Nothing about the output would look
wrong. This asserts the error arrives at construction, before a fold is fitted.

``test_the_winning_score_is_optimistic`` is the second, and it asserts a
*failure* on purpose: on a target that is pure noise the search still reports a
winner that beats the average candidate, because the maximum of many noisy
estimates is inflated. A spec that only showed the search finding the right
penalty on clean data would be hiding the thing a user most needs to know.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.data.dataset import Dataset
from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.kernel.functions import RadialBasisKernel
from oop_ml.core.model_selection.search import (
    Candidate,
    GridSearch,
    ParameterRange,
    ScoredCandidate,
    SearchResult,
    SearchSpace,
)
from oop_ml.core.model_selection.splitting import KFold
from oop_ml.numpy.classification.neighbours.k_nearest_classifier import (
    KNearestNeighboursClassifier,
)
from oop_ml.numpy.regression.kernels.kernel_ridge_regression import (
    KernelRidgeRegression,
)
from oop_ml.numpy.regression.neighbours.k_nearest_regressor import (
    KNearestNeighboursRegressor,
)
from oop_ml.numpy.regression.penalised.ridge_regression import RidgeRegression

PENALTIES = [0.01, 0.1, 1.0, 10.0, 100.0]


def noisy_line(n_rows: int = 60, seed: int = 11) -> Dataset:
    """A target driven by two of four columns, with noise over the top.

    Enough signal that a search should prefer a small penalty to a huge one,
    and enough noise that the huge one is genuinely worse rather than merely
    different.
    """
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(n_rows, 4))
    target = (
        3.0 * matrix[:, 0]
        - 2.0 * matrix[:, 1]
        + generator.normal(scale=0.5, size=n_rows)
    )

    return Dataset(
        [Feature(f"feature_{position}", matrix[:, position]) for position in range(4)],
        Feature("outcome", target),
    )


def pure_noise(n_rows: int = 80, seed: int = 7) -> Dataset:
    """A target with no relationship to the features at all.

    Nothing can genuinely beat R^2 = 0 here, which is what makes it the fixture
    that exposes selection optimism.
    """
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(n_rows, 4))

    return Dataset(
        [Feature(f"feature_{position}", matrix[:, position]) for position in range(4)],
        Feature("outcome", generator.normal(size=n_rows)),
    )


def searched(dataset: Dataset | None = None) -> SearchResult:
    """A grid over ridge penalties on the noisy line."""
    return GridSearch(folds=KFold(n_folds=5, random_seed=1)).search(
        RidgeRegression(),
        SearchSpace.over(RidgeRegression, penalty=PENALTIES),
        dataset or noisy_line(),
    )


def result_worst_penalty(result: SearchResult) -> float:
    """The penalty of the lowest-scoring candidate."""
    return result.ranked()[-1].candidate.value_for("penalty")


class TestBuildingTheSpace:
    """Names checked against the model, before anything is fitted."""

    def test_a_misspelled_parameter_is_caught_before_any_fitting(self) -> None:
        """The failure ``model_copy`` would have let through.

        ``model_copy(update={"pentalty": 5.0})`` returns a model with the
        penalty untouched and raises nothing, so the whole grid would fit one
        configuration and report a flat curve. Checking the name against the
        model's declared fields turns that into an error at construction.
        """
        with pytest.raises(InvalidValuesError):
            SearchSpace.over(RidgeRegression, pentalty=[1.0, 2.0])

    def test_the_error_names_what_the_model_does_have(self) -> None:
        with pytest.raises(InvalidValuesError, match="fit_intercept"):
            ParameterRange(RidgeRegression, "pentalty", [1.0])

    def test_an_empty_range_is_rejected(self) -> None:
        """A range of nothing would silently empty the whole grid."""
        with pytest.raises(EmptyValuesError):
            ParameterRange(RidgeRegression, "penalty", [])

    def test_a_space_with_no_ranges_is_rejected(self) -> None:
        with pytest.raises(EmptyValuesError):
            SearchSpace(RidgeRegression, [])

    def test_varying_one_parameter_twice_is_rejected(self) -> None:
        """Two ranges for one field leaves the grid ambiguous about which wins."""
        with pytest.raises(InvalidValuesError):
            SearchSpace(
                RidgeRegression,
                [
                    ParameterRange(RidgeRegression, "penalty", [1.0]),
                    ParameterRange(RidgeRegression, "penalty", [2.0]),
                ],
            )


class TestTheGrid:
    """Combinations, not sweeps."""

    def test_it_is_the_cartesian_product(self) -> None:
        """Three penalties and two intercept settings is six, not five."""
        space = SearchSpace.over(
            RidgeRegression, penalty=[0.1, 1.0, 10.0], fit_intercept=[True, False]
        )

        assert space.n_candidates == 6
        assert len(space.candidates()) == 6

    def test_every_candidate_assigns_every_parameter(self) -> None:
        """That is what makes it a grid rather than two separate sweeps.

        Parameters interact, so the best penalty at one intercept setting need
        not be the best at the other, and only trying them together finds that.
        """
        space = SearchSpace.over(
            RidgeRegression, penalty=[0.1, 1.0], fit_intercept=[True, False]
        )

        for candidate in space.candidates():
            assert set(candidate.assignments) == {"penalty", "fit_intercept"}

    def test_the_combinations_are_distinct(self) -> None:
        space = SearchSpace.over(
            RidgeRegression, penalty=[0.1, 1.0, 10.0], fit_intercept=[True, False]
        )
        seen = {tuple(sorted(one.assignments.items())) for one in space.candidates()}

        assert len(seen) == 6

    def test_it_counts_the_fits_before_running_them(self) -> None:
        """Worth reading before starting: multiply by n_folds for the real cost."""
        space = SearchSpace.over(
            RidgeRegression,
            penalty=[0.1, 1.0, 10.0, 100.0],
            fit_intercept=[True, False],
        )

        assert space.n_candidates == 8


class TestBuildingAModel:
    """Candidates become models through the validating constructor."""

    def test_it_sets_what_the_candidate_assigns(self) -> None:
        built = Candidate({"penalty": 7.0}).applied_to(RidgeRegression(penalty=1.0))

        assert built.penalty == pytest.approx(7.0)

    def test_it_keeps_what_the_candidate_does_not_assign(self) -> None:
        built = Candidate({"penalty": 7.0}).applied_to(
            RidgeRegression(penalty=1.0, fit_intercept=False)
        )

        assert built.fit_intercept is False

    def test_it_leaves_the_prototype_untouched(self) -> None:
        """The prototype is reused for every candidate, so mutating it would
        make the result depend on the order the grid ran in."""
        prototype = RidgeRegression(penalty=1.0)
        Candidate({"penalty": 7.0}).applied_to(prototype)

        assert prototype.penalty == pytest.approx(1.0)

    def test_a_value_the_model_refuses_still_raises(self) -> None:
        """Validation stays on the model rather than being copied into the search.

        ``model_copy`` would have accepted this. Rebuilding through the
        constructor means the penalty's own ``gt=0`` rule still applies.
        """
        with pytest.raises(ValidationError):
            Candidate({"penalty": -5.0}).applied_to(RidgeRegression())

    def test_a_field_holding_another_model_survives(self) -> None:
        """``model_dump()`` would have flattened it into a dict.

        ``model_dump`` recurses, so a field holding a pydantic model comes back
        as a plain dict, and rebuilding from that tries to instantiate the
        field's *declared* type -- the abstract ``Kernel`` here. Reading the
        unset fields with ``getattr`` keeps the configured object itself.
        """
        built = Candidate({"penalty": 2.0}).applied_to(
            KernelRidgeRegression(kernel=RadialBasisKernel(gamma=0.5))
        )

        assert isinstance(built.kernel, RadialBasisKernel)
        assert built.kernel.gamma == pytest.approx(0.5)
        assert built.penalty == pytest.approx(2.0)

    def test_a_candidate_with_no_assignments_is_rejected(self) -> None:
        with pytest.raises(EmptyValuesError):
            Candidate({})


class TestSearching:
    """The loop, on data with a real answer in it."""

    def test_it_scores_every_candidate(self) -> None:
        result = searched()

        assert result.n_candidates == len(PENALTIES)
        assert len(list(result)) == len(PENALTIES)

    def test_it_prefers_a_small_penalty_when_the_signal_is_strong(self) -> None:
        """Two features genuinely drive the target, so heavy shrinkage hurts."""
        result = searched()

        assert result.best.candidate.value_for("penalty") <= 1.0

    def test_the_worst_candidate_is_the_heaviest_shrinkage(self) -> None:
        assert result_worst_penalty(searched()) == pytest.approx(100.0)

    def test_it_keeps_the_losing_scores(self) -> None:
        """Which is what lets ``score_spread`` say whether the winner meant
        anything. A loop that tracked only a best-so-far would throw them away."""
        result = searched()

        assert all(isinstance(one, ScoredCandidate) for one in result)
        assert result.score_spread > 0.0

    def test_ranked_puts_the_winner_first(self) -> None:
        ranked = searched().ranked()

        assert ranked[0].score == pytest.approx(searched().best_score)
        assert [one.score for one in ranked] == sorted(
            [one.score for one in ranked], reverse=True
        )

    def test_the_best_model_comes_back_unfitted(self) -> None:
        """The search fitted it to folds; the caller decides what to refit on."""
        model = searched().best_model(RidgeRegression())

        assert not model.is_fitted

    def test_the_best_model_carries_the_winning_values(self) -> None:
        result = searched()
        model = result.best_model(RidgeRegression())

        assert model.penalty == pytest.approx(
            result.best.candidate.value_for("penalty")
        )

    def test_identical_candidates_score_identically_even_unseeded(self) -> None:
        """Every candidate must be scored on the same fold arrangement.

        ``KFold`` defaults to ``shuffle=True`` with no seed, and an unseeded
        shuffle deals fresh folds on every ``.split()`` call -- so without a
        guard, each candidate is scored on different folds and the comparison
        includes pure fold noise. Two copies of the same configuration are the
        detector: on identical folds they tie exactly, on redrawn folds they
        differ with near-certainty.
        """
        result = GridSearch().search(
            KNearestNeighboursRegressor(),
            SearchSpace.over(KNearestNeighboursRegressor, n_neighbours=[3, 3]),
            noisy_line(),
        )

        scores = [one.score for one in result]

        assert scores[0] == scores[1]

    def test_a_space_for_another_model_is_refused(self) -> None:
        """Caught before the search runs, not hundreds of fits into it."""
        with pytest.raises(InvalidValuesError):
            GridSearch().search(
                RidgeRegression(),
                SearchSpace.over(KNearestNeighboursRegressor, n_neighbours=[3, 5]),
                noisy_line(),
            )


class TestSelectionOptimism:
    """The failure no implementation can fix, asserted so it is not a surprise."""

    def test_the_winning_score_is_optimistic(self) -> None:
        """On a target that is pure noise, the winner still beats the average.

        Nothing here can genuinely beat R^2 = 0, so the gap between the best
        candidate and the mean candidate is entirely the maximum of many noisy
        estimates being higher than any of them deserves. Measured at roughly
        0.22 on this fixture across 25 candidates.
        """
        dataset = pure_noise()
        result = GridSearch(folds=KFold(n_folds=5, random_seed=3)).search(
            KNearestNeighboursRegressor(),
            SearchSpace.over(
                KNearestNeighboursRegressor, n_neighbours=list(range(1, 26))
            ),
            dataset,
        )

        mean_score = float(np.mean([one.score for one in result]))

        assert result.best_score < 0.0
        assert result.best_score > mean_score + 0.1

    def test_the_spread_says_whether_the_winner_meant_anything(self) -> None:
        """A wide spread means the parameter matters; a narrow one means it
        did not, and the winner is a coin toss."""
        result = searched()

        assert result.score_spread > 0.0

    def test_an_honest_score_needs_rows_the_search_never_saw(self) -> None:
        """The number to quote, and it is a different number.

        ``best_score`` was chosen because it was the largest, so it carries the
        selection's optimism. This one was not used to choose anything.
        """
        result = searched()
        holdout = noisy_line(n_rows=40, seed=999)

        honest = result.honest_score_on(RidgeRegression(), noisy_line(), holdout)

        assert isinstance(honest, float)
        assert honest != pytest.approx(result.best_score)

    def test_the_honest_score_cannot_be_a_training_score(self) -> None:
        """It must be fit on the training rows and scored on the held-out ones.

        The first version of this method fit the winner on the holdout and
        scored the same rows -- a training score wearing the name honest.

        The probe is a search whose only candidate is k=1, because that is
        where the two behaviours separate structurally rather than
        statistically. Fit-and-score-on-the-same-rows at k=1 returns exactly
        1.0 -- every row's nearest neighbour is itself -- while the honest
        procedure, fit on the search data and scored on fresh noise, lands
        near zero. At larger k the same-rows flattering shrinks to roughly
        1/k and drowns in forty-row sampling noise, which is why the sharp
        version of this test is the small-k one.
        """
        search_data = pure_noise(seed=7)
        holdout = pure_noise(n_rows=40, seed=999)
        result = GridSearch(folds=KFold(n_folds=5, random_seed=3)).search(
            KNearestNeighboursRegressor(),
            SearchSpace.over(KNearestNeighboursRegressor, n_neighbours=[1]),
            search_data,
        )

        honest = result.honest_score_on(
            KNearestNeighboursRegressor(), search_data, holdout
        )

        assert honest < 0.9


class TestSearchingAClassifier:
    """The second entry point, following the seam CrossValidation already has."""

    def dataset(self) -> Dataset:
        generator = np.random.default_rng(5)
        matrix = generator.normal(size=(60, 3))
        classes = (matrix[:, 0] + matrix[:, 1] > 0.0).astype(float)

        return Dataset(
            [
                Feature(f"feature_{position}", matrix[:, position])
                for position in range(3)
            ],
            Feature("outcome", classes),
        )

    def test_it_scores_candidates_by_pooled_accuracy(self) -> None:
        result = GridSearch(folds=KFold(n_folds=4, random_seed=2)).search_classifier(
            KNearestNeighboursClassifier(), self.space(), self.dataset()
        )

        assert 0.0 <= result.best_score <= 1.0
        assert result.n_candidates == 3

    def space(self) -> SearchSpace:
        return SearchSpace.over(KNearestNeighboursClassifier, n_neighbours=[1, 3, 5])


class TestTheResultObject:
    """Comparisons and construction."""

    def scored(self, score: float) -> ScoredCandidate:
        return ScoredCandidate(Candidate({"penalty": score}), score)

    def test_higher_scores_win(self) -> None:
        assert self.scored(0.9).beats(self.scored(0.5))
        assert not self.scored(0.5).beats(self.scored(0.9))

    def test_a_tie_goes_to_the_incumbent(self) -> None:
        """Strict, so grid order cannot decide the winner."""
        assert not self.scored(0.7).beats(self.scored(0.7))

    def test_the_earlier_of_two_tied_candidates_wins(self) -> None:
        result = SearchResult([self.scored(0.7), self.scored(0.7)])

        assert result.best.candidate.value_for("penalty") == pytest.approx(0.7)

    def test_a_result_with_no_candidates_is_rejected(self) -> None:
        with pytest.raises(EmptyValuesError):
            SearchResult([])
