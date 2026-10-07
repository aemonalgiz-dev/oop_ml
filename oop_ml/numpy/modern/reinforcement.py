"""Inspectable policy-gradient updates in a one-state, one-step environment.

The bandit simulator knows the payoff table; the learning rule receives only
the sampled action and its observed reward. There is no correct-action label,
value network, PPO clipping, or language-model training hidden in this example.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.types import FloatArray
from oop_ml.numpy.modern._validation import count, finite, positive
from oop_ml.numpy.modern.language import TokenDistribution


def discounted_returns(rewards: FloatArray, gamma: float = 1.0) -> FloatArray:
    """Sum rewards from each decision onward in a completed episode.

    The final supplied reward is terminal, so there is no bootstrapped value after
    it. Gamma discounts each additional step into the future, not past rewards.
    """
    values = finite(rewards, 1, "rewards")
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise InvalidValuesError("gamma must be in [0, 1]")
    result = np.empty_like(values)
    future = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        for index in range(len(values) - 1, -1, -1):
            future = values[index] + gamma * future
            result[index] = future
    return finite(result, 1, "discounted returns")


@dataclass(frozen=True)
class PolicyUpdate:
    action: int
    return_value: float
    advantage: float
    before: FloatArray
    gradient: FloatArray
    after: FloatArray


@dataclass(frozen=True)
class BanditRun:
    initial: FloatArray
    final: FloatArray
    counts: NDArray[np.int_]
    updates: tuple[PolicyUpdate, ...]


class SoftmaxPolicy:
    """Learn action logits directly for a single observation.

    This is a tabular policy for one state, not a neural network. Each update uses
    the gradient of the sampled action's log probability, weighted by its return
    minus a baseline. A multi-state problem needs a policy conditioned on state.
    """

    def __init__(self, logits: FloatArray) -> None:
        self._logits = finite(logits, 1, "logits")

    @property
    def probabilities(self) -> FloatArray:
        """Return a fresh normalized distribution without exposing parameters."""
        return TokenDistribution.from_logits(self._logits).probabilities

    def choose(self, generator: np.random.Generator) -> int:
        """Sample from the current policy, keeping the random stream explicit."""
        return TokenDistribution.from_logits(self._logits).sample(generator)

    def learn_from(
        self,
        action: int,
        return_value: float,
        learning_rate: float = 0.1,
        baseline: float = 0.0,
    ) -> PolicyUpdate:
        """Apply one REINFORCE ascent step using a freshly sampled action.

        The caller must sample with the current policy before this update.
        A baseline may depend on the observation, but must not depend on the
        sampled action. Treat it as constant when taking this derivative.
        """
        action = count(action, "action", minimum=0)
        if action >= len(self._logits):
            raise InvalidValuesError("action is outside the policy")
        rate = positive(learning_rate, "learning_rate")
        observed, reference = finite([return_value, baseline], 1, "return and baseline")
        before = self.probabilities
        direction = -before.copy()
        direction[action] += 1
        with np.errstate(over="ignore", invalid="ignore"):
            advantage = float(observed - reference)
            gradient = advantage * direction
            updated = self._logits + rate * gradient
        # Validate before mutation, so a rejected update leaves the policy intact.
        gradient = finite(gradient, 1, "policy gradient")
        updated = finite(updated, 1, "updated logits")
        self._logits = updated
        return PolicyUpdate(
            action, float(observed), advantage, before, gradient, self.probabilities
        )

    def train_bandit(
        self,
        rewards: FloatArray,
        episodes: int = 60,
        learning_rate: float = 0.1,
        seed: int = 0,
    ) -> BanditRun:
        """Run fresh one-step episodes with a fixed payoff for each action.

        This simulator indexes the reward table only after sampling an action.
        It passes that single payoff to learn_from, using a zero baseline.
        Training continues from this policy's current parameters; the random
        stream restarts at seed for each call. Zero episodes only inspects it.
        """
        payoffs = finite(rewards, 1, "rewards")
        if payoffs.shape != self._logits.shape:
            raise ShapeMismatchError("supply one reward per action")
        episodes = count(episodes, "episodes", minimum=0)
        rate = positive(learning_rate, "learning_rate")
        generator = np.random.default_rng(count(seed, "seed", minimum=0))
        initial = self.probabilities
        counts = np.zeros(len(payoffs), dtype=int)
        updates = []
        for _ in range(episodes):
            action = self.choose(generator)
            reward = float(payoffs[action])
            updates.append(self.learn_from(action, reward, rate))
            counts[action] += 1
        return BanditRun(initial, self.probabilities, counts, tuple(updates))
