"""Check the learning signal, sampling boundary, and terminal return convention."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.numpy.modern import SoftmaxPolicy, discounted_returns


@pytest.mark.parametrize("reward,baseline", [(1.0, 0.0), (-1.0, 0.0), (0.4, 0.4)])
def test_update_matches_the_sampled_log_probability_derivative(reward, baseline):
    logits = np.array([0.2, -0.3, 0.5])
    original = logits.copy()
    action = 1
    policy = SoftmaxPolicy(logits)
    update = policy.learn_from(action, reward, 0.1, baseline)
    numerical = np.empty(3)
    for index in range(3):
        plus, minus = logits.copy(), logits.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = (
            (reward - baseline)
            * (
                np.log(SoftmaxPolicy(plus).probabilities[action])
                - np.log(SoftmaxPolicy(minus).probabilities[action])
            )
            / 2e-6
        )
    assert_allclose(update.gradient, numerical, atol=1e-9)
    assert_allclose(logits, original)
    assert_allclose(update.after.sum(), 1)
    assert np.sign(update.after[action] - update.before[action]) == np.sign(
        reward - baseline
    )


def test_worked_first_update_and_owned_parameter_storage():
    logits = np.zeros(3)
    policy = SoftmaxPolicy(logits)
    logits[:] = [9, 0, 0]
    update = policy.learn_from(1, 1)
    assert_allclose(update.before, [1 / 3] * 3)
    assert_allclose(update.gradient, [-1 / 3, 2 / 3, -1 / 3])
    assert_allclose(update.after, [0.3220434644, 0.3559130712, 0.3220434644])
    update.after[:] = 0
    assert_allclose(policy.probabilities.sum(), 1)


def test_seeded_bandit_learns_from_only_the_chosen_payoff():
    rewards = np.array([0.2, 1.0, -1.0])
    first = SoftmaxPolicy(np.zeros(3)).train_bandit(rewards, 200, seed=0)
    second = SoftmaxPolicy(np.zeros(3)).train_bandit(rewards, 200, seed=0)
    assert [row.action for row in first.updates] == [
        row.action for row in second.updates
    ]
    assert_allclose(first.final, second.final)
    assert first.final[1] > 0.9
    assert first.counts.sum() == 200
    assert_allclose(first.counts, np.bincount([row.action for row in first.updates]))
    for row in first.updates:
        assert row.return_value == rewards[row.action]
        independent = SoftmaxPolicy(np.log(row.before))
        expected = independent.learn_from(row.action, row.return_value)
        assert_allclose(row.after, expected.after)


def test_zero_episodes_preserves_initial_policy_and_has_no_observations():
    result = SoftmaxPolicy(np.zeros(3)).train_bandit(np.array([0.2, 1, -1]), 0)
    assert_allclose(result.initial, result.final)
    assert result.counts.tolist() == [0, 0, 0]
    assert result.updates == ()


@pytest.mark.parametrize(
    "gamma,expected", [(0, [-1, -1, 5]), (1, [3, 4, 5]), (0.9, [2.15, 3.5, 5])]
)
def test_returns_include_only_present_and_future_rewards(gamma, expected):
    rewards = np.array([-1.0, -1.0, 5.0])
    assert_allclose(discounted_returns(rewards, gamma), expected)
    assert_allclose(rewards, [-1, -1, 5])


@pytest.mark.parametrize("gamma", [-0.1, 1.1, np.nan])
def test_discount_rejects_invalid_values(gamma):
    with pytest.raises(InvalidValuesError):
        discounted_returns(np.ones(3), gamma)


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": -1},
        {"action": 3},
        {"action": True},
        {"action": 1.5},
        {"return_value": np.inf},
        {"baseline": np.nan},
        {"learning_rate": 0},
        {"return_value": 1e308, "learning_rate": 1e308},
    ],
)
def test_rejected_update_does_not_mutate_the_policy(arguments):
    policy = SoftmaxPolicy(np.zeros(3))
    kwargs = {"action": 0, "return_value": 1.0, **arguments}
    with pytest.raises(InvalidValuesError):
        policy.learn_from(**kwargs)
    assert_allclose(policy.probabilities, [1 / 3] * 3)


@pytest.mark.parametrize("kwargs", [{"episodes": -1}, {"episodes": 2.5}, {"seed": -1}])
def test_bandit_rejects_invalid_counts(kwargs):
    with pytest.raises(InvalidValuesError):
        SoftmaxPolicy(np.zeros(3)).train_bandit(np.ones(3), **kwargs)


def test_policy_and_environment_validate_shapes_and_finiteness():
    with pytest.raises(ShapeMismatchError):
        SoftmaxPolicy(np.zeros((2, 3)))
    with pytest.raises(ShapeMismatchError):
        SoftmaxPolicy(np.zeros(3)).train_bandit(np.ones(2))
    with pytest.raises(InvalidValuesError):
        SoftmaxPolicy(np.array([np.nan, 0]))
    with pytest.raises(InvalidValuesError):
        discounted_returns(np.array([]))
