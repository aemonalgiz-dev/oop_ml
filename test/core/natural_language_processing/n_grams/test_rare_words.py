"""Spec for the rare-word substitution, which is what ``minimum_count`` buys.

A model with a closed vocabulary has to decide what a word it was never taught
becomes, and the answer is the unknown marker. That answer is only worth
anything if the marker was *counted* during the fit, because a marker with a
count of zero gives every unseen word a probability of zero and puts the model
back where it started.

So the claim these tests hold is narrow and load-bearing: a corpus is allowed
to arrive at the counting step with the marker already substituted in, and the
count that comes back says how often the substitution happened.

The framing markers are a different case and stay refused. ``padded`` adds
those itself, so a caller supplying one would have it counted twice and would
shift every context by a position.
"""

import pytest

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.natural_language_processing.n_grams.counts import NGramCounts
from oop_ml.core.natural_language_processing.n_grams.grams import (
    SENTENCE_END,
    SENTENCE_START,
    UNKNOWN_WORD,
)
from oop_ml.core.natural_language_processing.n_grams.language_model import (
    NGramLanguageModel,
)
from oop_ml.core.natural_language_processing.n_grams.smoothing.base import (
    MaximumLikelihood,
)

# Counted by hand: the 4, sat 3, on 3, cat 2, mat 2, a 2, and then rug, dog
# and rare once each. So a threshold of two replaces exactly those three
# and leaves every other word standing.
CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "a dog sat on a rare mat",
]


class TestCountingAnAlreadySubstitutedSentence:
    def test_accepts_the_unknown_marker(self):
        counts = NGramCounts.from_sentences([("the", UNKNOWN_WORD, "mat")], order=2)

        assert counts.count_of((UNKNOWN_WORD,)) == 1

    def test_counts_the_marker_as_an_ordinary_word_in_context(self):
        counts = NGramCounts.from_sentences([("the", UNKNOWN_WORD, "mat")], order=2)

        assert counts.count_of(("the", UNKNOWN_WORD)) == 1
        assert counts.count_of((UNKNOWN_WORD, "mat")) == 1

    @pytest.mark.parametrize("marker", [SENTENCE_START, SENTENCE_END])
    def test_refuses_a_framing_marker(self, marker):
        with pytest.raises(InvalidValuesError, match="reserved"):
            NGramCounts.from_sentences([("the", marker, "mat")], order=2)


class TestFittingWithAThreshold:
    def test_fits_rather_than_raising(self):
        model = NGramLanguageModel(order=2, minimum_count=2).fit(CORPUS)

        assert model.is_fitted

    def test_drops_the_rare_word_from_the_vocabulary(self):
        model = NGramLanguageModel(order=2, minimum_count=2).fit(CORPUS)

        assert "rare" not in model.vocabulary
        assert "mat" in model.vocabulary

    def test_the_marker_carries_the_count_of_what_it_replaced(self):
        model = NGramLanguageModel(order=2, minimum_count=2).fit(CORPUS)

        assert model.counts.count_of((UNKNOWN_WORD,)) == 3

    def test_an_unseen_word_is_possible_rather_than_impossible(self):
        """The whole point of the threshold, stated as the thing that changes.

        Asked under maximum likelihood deliberately, because add-one smoothing
        gives an unseen word a probability whatever the counts say and would
        hide the difference being tested. Without a counted marker the model
        answers zero, which makes any held-out text containing an unseen word
        impossible however good the rest of it is.
        """
        thresholded = NGramLanguageModel(
            order=2, minimum_count=2, smoothing=MaximumLikelihood()
        ).fit(CORPUS)
        everything = NGramLanguageModel(
            order=2, minimum_count=1, smoothing=MaximumLikelihood()
        ).fit(CORPUS)

        assert thresholded.probability_of("hedgehog", ("the",)) > 0.0
        assert everything.probability_of("hedgehog", ("the",)) == 0.0
