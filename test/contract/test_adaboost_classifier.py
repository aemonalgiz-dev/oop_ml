"""The contract every backend's AdaBoostClassifier keeps.

The third ensemble shape, and the one whose members are fitted on the same rows
under changing weights rather than on resamples or residuals. What the contract
asserts is what the method claims: that combining learners too weak to be worth
anything alone produces something worth a good deal, and that the sequence of
weighted errors says why.

What is deliberately not asserted
----------------------------------
Which stumps get chosen. A weak learner asked for one question often finds two
equally good ones, and the two backends break that tie differently, after which
the reweighting sends them down different paths. Measured across two, four and
five classes the two agree on every voice and every margin to 1.8e-15; on one
three-class fixture the voices part by 0.265 and the ensembles end up scoring
0.900 and 0.875. So the numbers here are about the ensemble's behaviour and not
about its members.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml import Feature
from oop_ml.core.exceptions import DivergenceError, NotFittedError

from .harness import provided

_GENERATOR = np.random.default_rng(2)
_ROWS = np.vstack(
    [_GENERATOR.normal(loc=[index * 2.0, index], size=(40, 2)) for index in range(3)]
)
#: Three clouds along a diagonal. No single stump comes close, which is what
#: makes the ensemble's score mean something.
FEATURES = [Feature("first", _ROWS[:, 0]), Feature("second", _ROWS[:, 1])]
TARGET = Feature("group", np.concatenate([np.full(40, float(k)) for k in range(3)]))
N_CLASSES = 3

#: Above this a learner is no better than guessing uniformly among three
#: classes, and the walk stops rather than giving it a negative voice.
GUESSING = 1.0 - 1.0 / N_CLASSES

_LINE = np.linspace(-3.0, 3.0, 60)
SEPARABLE = [Feature("first", _LINE), Feature("second", np.zeros(60))]
SEPARABLE_TARGET = Feature("group", (_LINE > 0.0).astype(float))


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")

    model = AdaBoostClassifier(n_members=10, learning_rate=0.5, max_depth=2)

    assert model.n_members == 10
    assert model.learning_rate == pytest.approx(0.5)
    assert model.max_depth == 2


def test_its_learner_is_a_stump_by_default(backend: ModuleType) -> None:
    """One question, two leaves. Weak on purpose: boost something that already
    fits the rows and the first round has nothing to reweight, so every later
    round repeats it."""
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")

    assert AdaBoostClassifier().max_depth == 1
    assert AdaBoostClassifier().n_members == 50


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [("n_members", 0), ("learning_rate", 0.0), ("max_depth", 0)],
)
def test_it_refuses_a_setting_that_describes_no_ensemble(
    backend: ModuleType, field_name: str, invalid: float
) -> None:
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")

    with pytest.raises(ValidationError):
        AdaBoostClassifier(**{field_name: invalid})


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")
    model = AdaBoostClassifier(n_members=20, random_seed=0)

    assert model.fit(FEATURES, TARGET) is model


def test_many_weak_learners_beat_one_of_them(backend: ModuleType) -> None:
    """The whole claim. A single stump on these three clouds is barely better
    than naming the largest class; fifty of them, each fitted on what the last
    found hard, are not."""
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")
    DecisionTreeClassifier = provided(backend, "DecisionTreeClassifier")

    boosted = AdaBoostClassifier(n_members=50, random_seed=0).fit(FEATURES, TARGET)
    alone = DecisionTreeClassifier(max_depth=1).fit(FEATURES, TARGET)

    assert boosted.score(FEATURES, TARGET) > 0.85
    assert alone.score(FEATURES, TARGET) < 0.7


def test_every_member_earns_a_positive_voice(backend: ModuleType) -> None:
    """A member that would have earned a negative one ends the walk instead,
    since an ensemble that listened to it would be learning from a liar."""
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")
    model = AdaBoostClassifier(n_members=30, random_seed=0).fit(FEATURES, TARGET)

    voices = np.asarray(model.member_voices)

    assert len(voices) == model.n_members_fitted
    assert (voices > 0.0).all()


def test_the_errors_stay_near_the_guessing_bar_rather_than_falling(
    backend: ModuleType,
) -> None:
    """The counter-intuitive part, and the sign the method is working. Each
    round hands the next learner a deliberately harder problem, so the errors
    do not improve; they sit just under the bar at which a learner would be
    discarded."""
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")
    model = AdaBoostClassifier(n_members=30, random_seed=0).fit(FEATURES, TARGET)

    errors = np.asarray(model.member_errors)

    assert (errors < GUESSING).all()
    assert float(errors[-5:].mean()) > float(errors[:5].mean())


def test_the_vote_shares_are_shares(backend: ModuleType) -> None:
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")
    model = AdaBoostClassifier(n_members=20, random_seed=0).fit(FEATURES, TARGET)

    shares = np.asarray(model.vote_shares(FEATURES))

    assert shares.shape == (len(_ROWS), N_CLASSES)
    assert np.allclose(shares.sum(axis=1), 1.0)
    assert shares.min() >= 0.0


def test_a_margin_is_a_share_less_what_the_others_took(backend: ModuleType) -> None:
    """``share - (1 - share) / (K - 1)``, which is the form the method's own
    convergence argument is written in, and which cannot reorder the classes."""
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")
    model = AdaBoostClassifier(n_members=20, random_seed=0).fit(FEATURES, TARGET)

    shares = np.asarray(model.vote_shares(FEATURES))
    margins = np.asarray(model.margins(FEATURES))

    assert np.allclose(margins, shares - (1.0 - shares) / (N_CLASSES - 1), atol=1e-9)
    assert np.array_equal(np.argmax(margins, axis=1), np.argmax(shares, axis=1))


def test_the_prediction_is_the_loudest_class(backend: ModuleType) -> None:
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")
    model = AdaBoostClassifier(n_members=20, random_seed=0).fit(FEATURES, TARGET)

    assert np.array_equal(
        np.asarray(model.predict(FEATURES)),
        np.argmax(np.asarray(model.vote_shares(FEATURES)), axis=1).astype(float),
    )


def test_a_class_no_member_picked_still_gets_a_probability(
    backend: ModuleType,
) -> None:
    """Which is what the softmax is for. The shares are hard votes, so a class
    nobody chose has a share of exactly zero, and a probability of exactly zero
    is a claim no later evidence could revise."""
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")
    model = AdaBoostClassifier(n_members=20, random_seed=0).fit(FEATURES, TARGET)

    shares = np.asarray(model.vote_shares(FEATURES))
    chances = np.asarray(model.predict_probabilities(FEATURES))

    assert shares.min() < 1e-9
    assert chances.min() > 0.0
    assert np.allclose(chances.sum(axis=1), 1.0)


def test_a_smaller_learning_rate_quietens_every_member(backend: ModuleType) -> None:
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")

    loud = AdaBoostClassifier(n_members=5, random_seed=0).fit(FEATURES, TARGET)
    quiet = AdaBoostClassifier(n_members=5, learning_rate=0.1, random_seed=0).fit(
        FEATURES, TARGET
    )

    assert float(np.asarray(quiet.member_voices)[0]) < float(
        np.asarray(loud.member_voices)[0]
    )


def test_a_learner_with_nothing_left_to_learn_ends_the_walk(
    backend: ModuleType,
) -> None:
    """One stump separates this fixture perfectly, so the first round is wrong
    about nothing, no row is reweighted, and every later round would repeat
    it. Stopping is the honest outcome rather than fifty copies."""
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")
    model = AdaBoostClassifier(n_members=50, random_seed=0).fit(
        SEPARABLE, SEPARABLE_TARGET
    )

    assert model.n_members_fitted == 1
    assert model.score(SEPARABLE, SEPARABLE_TARGET) == 1.0


def test_it_refuses_a_first_learner_that_cannot_beat_guessing(
    backend: ModuleType,
) -> None:
    """With nothing informative in the columns there is no ensemble to keep,
    and one that reported itself fitted would answer with a constant."""
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")
    noise = [Feature("flat", np.zeros(30)), Feature("also_flat", np.zeros(30))]
    alternating = Feature("group", np.tile([0.0, 1.0, 2.0], 10))

    with pytest.raises(DivergenceError):
        AdaBoostClassifier(n_members=10).fit(noise, alternating)


def test_it_matches_query_columns_by_name(backend: ModuleType) -> None:
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")
    model = AdaBoostClassifier(n_members=10, random_seed=0).fit(FEATURES, TARGET)

    reversed_order = [FEATURES[1], FEATURES[0]]

    assert np.array_equal(
        np.asarray(model.predict(reversed_order)), np.asarray(model.predict(FEATURES))
    )


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")

    with pytest.raises(NotFittedError):
        AdaBoostClassifier().predict(FEATURES)


@pytest.mark.parametrize(
    "summary", ["member_voices", "member_errors", "n_members_fitted", "n_classes"]
)
def test_it_refuses_to_report_its_fit_before_fit(
    backend: ModuleType, summary: str
) -> None:
    AdaBoostClassifier = provided(backend, "AdaBoostClassifier")

    with pytest.raises(NotFittedError):
        getattr(AdaBoostClassifier(), summary)
