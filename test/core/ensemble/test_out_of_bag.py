"""Spec for the out-of-bag estimate -- red until ``out_of_bag_estimate`` lands.

Two of these carry the argument.

``test_matches_a_hand_built_average`` is the oracle, and it is written from the
definition rather than from the implementation: build the in-bag grid
independently, walk it row by row in plain Python, and average the members that
missed each row. If the fast route and this loop ever disagree, one of them is
wrong, and a test that reused the implementation's own machinery could not tell.

``test_is_pessimistic_against_the_training_score`` pins the direction of the
bias. The out-of-bag number must land below the training score, because the
training score is measured on rows every member memorised. It is the whole
reason the estimate is worth having, and an implementation that quietly let
in-bag members vote would sail past every shape assertion in this file while
failing this one.
"""

import numpy as np
import pytest

from oop_ml.core.exceptions import NotFittedError
from oop_ml.numpy.regression.ensembles.bagging_regressor import BaggingRegressor
from oop_ml.numpy.regression.trees.decision_tree_regressor import DecisionTreeRegressor
from test.fixtures import DOMINATED_SIGNAL, ENSEMBLE_MEMBERS


@pytest.fixture
def fitted() -> BaggingRegressor:
    return BaggingRegressor(n_members=ENSEMBLE_MEMBERS, random_seed=0).fit(
        DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
    )


class TestShape:
    """What the estimate reports about itself."""

    def test_covers_one_entry_per_training_row(self, fitted: BaggingRegressor) -> None:
        estimate = fitted.out_of_bag_estimate()

        assert len(estimate) == DOMINATED_SIGNAL.n_samples
        assert estimate.predictions.shape == (DOMINATED_SIGNAL.n_samples,)
        assert estimate.covered.shape == (DOMINATED_SIGNAL.n_samples,)
        assert estimate.judges.shape == (DOMINATED_SIGNAL.n_samples,)

    def test_covered_and_uncovered_account_for_every_row(
        self, fitted: BaggingRegressor
    ) -> None:
        estimate = fitted.out_of_bag_estimate()

        assert estimate.n_covered + estimate.n_uncovered == len(estimate)

    def test_a_row_is_covered_exactly_when_it_has_judges(
        self, fitted: BaggingRegressor
    ) -> None:
        estimate = fitted.out_of_bag_estimate()

        assert np.array_equal(estimate.covered, estimate.judges > 0)

    def test_at_twenty_members_every_row_is_covered(
        self, fitted: BaggingRegressor
    ) -> None:
        """The chance a row is in-bag for all twenty is (1 - 1/e)^20, ~1e-4."""
        assert fitted.out_of_bag_estimate().n_uncovered == 0

    def test_judges_average_about_a_third_of_the_members(
        self, fitted: BaggingRegressor
    ) -> None:
        """Each row is judged by roughly ``0.368 * n_members``, not by all."""
        estimate = fitted.out_of_bag_estimate()

        assert estimate.mean_judges == pytest.approx(ENSEMBLE_MEMBERS / np.e, rel=0.15)
        assert estimate.judges.max() < ENSEMBLE_MEMBERS


class TestCorrectness:
    """Checked against the definition, computed independently."""

    def test_matches_a_hand_built_average(self, fitted: BaggingRegressor) -> None:
        drawn = [set(sample.drawn.tolist()) for sample in fitted.samples]
        member_predictions = np.array(
            [
                member.predict(DOMINATED_SIGNAL.input_features)
                for member in fitted.members
                if isinstance(member, DecisionTreeRegressor)
            ]
        )

        expected = []
        for row in range(DOMINATED_SIGNAL.n_samples):
            missed = [
                position for position, rows in enumerate(drawn) if row not in rows
            ]
            expected.append(
                float(np.mean([member_predictions[one][row] for one in missed]))
                if missed
                else np.nan
            )

        estimate = fitted.out_of_bag_estimate()
        covered = estimate.covered

        assert np.allclose(estimate.predictions[covered], np.array(expected)[covered])

    def test_no_member_that_drew_a_row_judges_it(
        self, fitted: BaggingRegressor
    ) -> None:
        """The one thing that would make the estimate dishonest.

        Recomputed here as a count rather than by re-deriving the prediction,
        so it fails loudly if the mask is ever inverted.
        """
        estimate = fitted.out_of_bag_estimate()
        expected = np.array([sample.out_of_bag.size for sample in fitted.samples])
        in_bag = np.array([sample.in_bag for sample in fitted.samples])

        assert np.array_equal(estimate.judges, (~in_bag).sum(axis=0))
        assert expected.sum() == estimate.judges.sum()

    def test_differs_from_the_full_ensemble_prediction(
        self, fitted: BaggingRegressor
    ) -> None:
        """If it matched, every member would be voting on every row."""
        full = fitted.predict(DOMINATED_SIGNAL.input_features)

        assert not np.allclose(fitted.out_of_bag_estimate().predictions, full)


class TestAsAScore:
    """What the number is worth."""

    def test_is_pessimistic_against_the_training_score(
        self, fitted: BaggingRegressor
    ) -> None:
        training = fitted.score(
            DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
        )

        assert fitted.out_of_bag_score() < training

    def test_lands_near_a_genuine_held_out_score(
        self, fitted: BaggingRegressor
    ) -> None:
        """The claim that makes it a substitute for a split.

        A wide band on purpose. The point is that it estimates the held-out
        number rather than the training one, not that it reproduces it.
        """
        held_out = fitted.score(
            DOMINATED_SIGNAL.held_out_features, DOMINATED_SIGNAL.held_out_target
        )

        assert fitted.out_of_bag_score() == pytest.approx(held_out, abs=0.15)

    def test_evaluate_pairs_the_predictions_with_the_covered_truth(
        self, fitted: BaggingRegressor
    ) -> None:
        estimate = fitted.out_of_bag_estimate()
        evaluation = fitted.out_of_bag_evaluate()

        assert evaluation.n_samples == estimate.n_covered

    def test_score_is_the_evaluations_r2(self, fitted: BaggingRegressor) -> None:
        assert fitted.out_of_bag_score() == pytest.approx(
            fitted.out_of_bag_evaluate().r2_score
        )


class TestUnfitted:
    """Nothing is readable before a fit."""

    @pytest.mark.parametrize(
        "call",
        ["out_of_bag_estimate", "out_of_bag_evaluate", "out_of_bag_score"],
    )
    def test_calling_before_fit_raises(self, call: str) -> None:
        with pytest.raises(NotFittedError):
            getattr(BaggingRegressor(), call)()

    def test_reading_the_samples_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = BaggingRegressor().samples
