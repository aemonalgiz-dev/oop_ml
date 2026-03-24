"""Break one column on purpose and see how much the score suffers.

Mean decrease in impurity reads a fitted tree's own bookkeeping and asks which
features the *split search* liked. That is cheap and it is biased: a continuous
column offers hundreds of candidate thresholds where a binary one offers a
single threshold, so it wins splits more often on chance alone and scores
higher for it. The parity fixture in the tree specs shows the extreme case,
where a pure-noise continuous column outscores two real binary features and
takes the root.

Permutation importance asks a different question. Shuffle a single column,
leaving every other column and the target where they are, score the fitted
model again, and measure how far the score fell. The shuffle destroys whatever
relationship that column had with the target while preserving its distribution
exactly, so the drop is attributable to the column and to nothing else.

It is worth being precise about what that buys, because it is easy to overclaim.
Permutation is not a lie detector for high-cardinality columns. Measured on the
lone parity tree, which scores 0.537, both measures name the noise column and
both are correct: that tree built itself out of noise and genuinely relies on
it. The two only diverge on a model that works. On the forest, which recovers
parity at 0.993, impurity still gives the noise column roughly 0.52 while
permutation gives it 0.018.

Three consequences worth knowing before reading the numbers.

**It measures reliance, not information.** A column the model never consulted
scores zero even if it would have predicted the target perfectly on its own.
That is the right answer to "what is this model leaning on" and the wrong
answer to "what matters in the world".

**Correlated columns hide each other.** Shuffle one of two columns carrying the
same signal and the model leans on the other, so the drop is small and both
look unimportant. This is a real failure and it is worse here than under mean
decrease in impurity, which at least splits the credit between them.

**It costs a scoring pass per feature per repeat**, where the impurity version
costs a tree walk. On a wide dataset that is the difference between free and
expensive, which is why both exist.

Which data you permute decides what you learn. Passing the training rows tells
you what the model leaned on while memorising; passing held-out rows tells you
what it leans on when it is actually being useful. Neither is wrong, and they
disagree most on exactly the overfitted models where the question matters most.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from oop_ml.core.data.feature import Feature
from oop_ml.core.importance.importances import (
    FeatureContribution,
    FeatureImportances,
)


@runtime_checkable
class Scorable(Protocol):
    """Anything fitted that can score itself against features and a target.

    A structural protocol rather than a base class, because permutation
    importance genuinely does not care what it is measuring. Every regressor,
    classifier and ensemble in the library satisfies it already, and so would
    a model from somewhere else.
    """

    def score(self, input_values: Sequence[Feature], target_values: Feature) -> float:
        """How well this model does on the given rows. Higher is better."""
        ...


class PermutationImportance(BaseModel):
    """Importance measured by breaking each column in turn.

    Parameters
    ----------
    n_repeats:
        How many times each column is shuffled. One shuffle is one draw from a
        noisy quantity, and a column that happens to shuffle into a
        near-original arrangement will understate its own importance. The
        default of five is enough to stop a single unlucky draw dominating
        without making the whole thing five times more expensive than it needs
        to be.
    random_seed:
        Fixes every shuffle, so a measurement is reproducible.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_repeats: int = Field(default=5, ge=1)
    random_seed: int | None = None

    def measure(
        self,
        model: Scorable,
        input_values: Sequence[Feature],
        target_values: Feature,
    ) -> FeatureImportances:
        """Score the model, then break each column in turn and score again.

        For every feature, shuffle *only that column* ``n_repeats`` times,
        leaving the other columns and the target untouched, and average how far
        the score fell each time. That average drop is the feature's raw score;
        hand the lot to ``FeatureImportances.from_scores`` and let it normalise.

        Shuffle the column's values rather than replacing them, because the
        point is to destroy the relationship with the target while leaving the
        column's own distribution exactly as it was. Substituting noise would
        change two things at once and the drop would no longer be attributable.

        Draw every shuffle from one generator seeded once, the way the bagging
        frame draws its resamples, so the columns differ from each other while
        the measurement as a whole stays reproducible.

        A drop can come out negative when a column is useless and the model was
        slightly better off without it. Clamp those to zero rather than letting
        them cancel a real contribution elsewhere; ``from_scores`` will refuse
        a negative anyway.

        Parameters
        ----------
        model:
            Already fitted. This never refits anything, which is what makes it
            cheap enough to be worth doing at all.
        input_values, target_values:
            The rows to measure on. Training rows say what the model leaned on
            while memorising; held-out rows say what it leans on when it is
            being useful.

        Returns
        -------
        FeatureImportances
            One share per supplied feature, summing to 1.

        Raises
        ------
        InvalidValuesError
            If no feature earned anything, which means the model scored no
            worse with every column broken in turn than it did intact.
        """
        model_score = model.score(
            input_values=input_values,
            target_values=target_values,
        )

        generator = np.random.default_rng(self.random_seed)
        feature_contributions: list[FeatureContribution] = []

        for index, feature in enumerate(input_values):
            drops = []

            for _ in range(self.n_repeats):
                shuffled = self._shuffled(input_values, index, generator)
                drops.append(model_score - model.score(shuffled, target_values))

            feature_contributions.append(
                FeatureContribution(feature.name, max(0.0, sum(drops) / len(drops)))
            )

        return FeatureImportances.from_contributions(feature_contributions)

    def _shuffled(
        self,
        input_values: Sequence[Feature],
        position: int,
        generator: np.random.Generator,
    ) -> list[Feature]:
        """The same columns, with the one at ``position`` permuted.

        A new list of new ``Feature`` objects rather than a mutation, because
        a ``Column`` is frozen on purpose and the caller's data must come back
        untouched however many times this is called.
        """
        shuffled = generator.permutation(input_values[position].values)

        return [
            Feature(feature.name, shuffled) if index == position else feature
            for index, feature in enumerate(input_values)
        ]
