"""Spec for SuperBytePairEncoding -- merges that may span several words.

Pinned on a corpus of ``of the`` three times, worked by hand in the module
docstring, and against plain byte pair encoding on Sennrich's corpus, whose
merges it must reproduce wherever there is no boundary to cross.
"""

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
    VocabularyTooSmallError,
)
from oop_ml.core.natural_language_processing.tokenization.encoding import Encoding
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    apply_merges,
    with_pair_merged,
)
from oop_ml.core.natural_language_processing.tokenization.subword.super_byte_pair_encoding import (
    SuperBytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from test.core.natural_language_processing.fixtures import (
    SENNRICH_ALPHABET,
    SENNRICH_CORPUS,
    SENNRICH_WORD_COUNTS,
)

OF_THE_CORPUS = ["of the", "of the", "of the"]

# The five symbols of ``o f</w>`` and ``t h e</w>``, in codepoint order.
OF_THE_ALPHABET = ["e</w>", "f</w>", "h", "o", "t"]

# Every word of Sennrich's corpus as its own text, so no pair ever crosses a
# boundary and the second stage can only continue the first.
SENNRICH_ONE_WORD_PER_TEXT = SENNRICH_CORPUS[0].split()

# The unknown token plus eleven symbols, then ten merges.
TEN_MERGES = len(SENNRICH_ALPHABET) + 1 + 10


def fit_of_the(vocabulary_size: int, transition_size: int) -> SuperBytePairEncoding:
    return SuperBytePairEncoding(
        vocabulary_size=vocabulary_size, transition_size=transition_size
    ).fit(OF_THE_CORPUS)


class TestFit:
    def test_stage_one_merges_within_words_and_stage_two_across_them(self):
        tokenizer = fit_of_the(vocabulary_size=10, transition_size=9)

        assert [merge.merged for merge in tokenizer.word_merges] == [
            "he</w>",
            "of</w>",
            "the</w>",
        ]
        assert [merge.merged for merge in tokenizer.superword_merges] == [
            "of</w>the</w>"
        ]
        assert tokenizer.n_word_merges == 3
        assert tokenizer.n_superword_merges == 1
        assert tokenizer.n_merges == 4

    def test_every_merge_carries_the_count_that_chose_it(self):
        assert [merge.score for merge in fit_of_the(10, 9).merges] == [
            3.0,
            3.0,
            3.0,
            3.0,
        ]

    def test_a_superword_token_holds_a_marker_before_its_end(self):
        tokenizer = fit_of_the(vocabulary_size=10, transition_size=9)

        assert tokenizer.superword_tokens == ("of</w>the</w>",)
        assert "</w>" in "of</w>the</w>"[:-4]

    def test_the_vocabulary_is_unknown_then_alphabet_then_both_stages_merges(self):
        vocabulary = fit_of_the(vocabulary_size=10, transition_size=9).vocabulary

        assert list(vocabulary) == [
            "[UNK]",
            *OF_THE_ALPHABET,
            "he</w>",
            "of</w>",
            "the</w>",
            "of</w>the</w>",
        ]
        assert vocabulary.unknown_token == "[UNK]"

    def test_a_stage_two_merge_may_cross_a_boundary_and_stop_mid_word(self):
        """Stopping stage one after ``of</w>`` leaves ``of</w> t he</w>``; the
        candidates ``of</w> t`` and ``t he</w>`` tie at 3 and the smaller wins."""
        tokenizer = fit_of_the(vocabulary_size=10, transition_size=8)

        assert [merge.merged for merge in tokenizer.word_merges] == ["he</w>", "of</w>"]
        assert [merge.merged for merge in tokenizer.superword_merges] == [
            "of</w>t",
            "of</w>the</w>",
        ]
        assert tokenizer.superword_tokens == ("of</w>t", "of</w>the</w>")

    def test_stage_two_is_given_whatever_room_stage_one_left(self):
        """Stage one exhausts the words after three merges however large the
        transition, and stage two still fills the vocabulary to the target."""
        tokenizer = SuperBytePairEncoding(
            vocabulary_size=10, transition_size=9, minimum_pair_frequency=1
        ).fit(OF_THE_CORPUS)
        roomier = SuperBytePairEncoding(
            vocabulary_size=11, transition_size=10, minimum_pair_frequency=1
        ).fit(OF_THE_CORPUS)

        assert tokenizer.vocabulary.n_tokens == 10
        assert roomier.n_word_merges == 3
        assert roomier.n_superword_merges == 1
        assert roomier.vocabulary.n_tokens == 10

    def test_with_nothing_to_cross_it_is_plain_byte_pair_encoding(self):
        """One word per text: lifting the boundary changes nothing, whatever the
        transition, and the merges are Sennrich's ten in Sennrich's order."""
        plain = BytePairEncoding(vocabulary_size=TEN_MERGES).fit(SENNRICH_CORPUS)
        lifted = SuperBytePairEncoding(
            vocabulary_size=TEN_MERGES, transition_size=15
        ).fit(SENNRICH_ONE_WORD_PER_TEXT)

        assert lifted.merges == plain.merges
        assert lifted.vocabulary == plain.vocabulary
        assert lifted.superword_tokens == ()
        assert lifted.n_word_merges == 3
        assert lifted.n_superword_merges == 7

    def test_a_transition_one_below_the_target_gives_stage_two_one_merge(self):
        plain = BytePairEncoding(vocabulary_size=TEN_MERGES).fit(SENNRICH_CORPUS)
        lifted = SuperBytePairEncoding(
            vocabulary_size=TEN_MERGES, transition_size=TEN_MERGES - 1
        ).fit(SENNRICH_ONE_WORD_PER_TEXT)

        assert (
            lifted.word_merges
            == BytePairEncoding(vocabulary_size=TEN_MERGES - 1)
            .fit(SENNRICH_CORPUS)
            .merges
        )
        assert lifted.n_superword_merges == 1
        assert lifted.merges == plain.merges

    def test_stage_one_reproduces_byte_pair_encodings_merges_exactly(self):
        plain = BytePairEncoding(vocabulary_size=20).fit(SENNRICH_CORPUS)
        lifted = SuperBytePairEncoding(vocabulary_size=22, transition_size=20).fit(
            SENNRICH_CORPUS
        )

        assert lifted.word_merges == plain.merges

    def test_on_one_text_the_superwords_are_the_repeated_words(self):
        """After eight within-word merges the text reads ``low</w>`` five times in a
        row and ``newest</w>`` six times in a row, so the most frequent adjacent
        pairs are a word beside itself: ``newest</w> newest</w>`` occurs 5 times
        and ``low</w> low</w>`` 4."""
        lifted = SuperBytePairEncoding(vocabulary_size=22, transition_size=20).fit(
            SENNRICH_CORPUS
        )

        assert [
            (merge.merged, int(merge.score)) for merge in lifted.superword_merges
        ] == [("newest</w>newest</w>", 5), ("low</w>low</w>", 4)]
        assert lifted.superword_tokens == ("newest</w>newest</w>", "low</w>low</w>")

    def test_the_two_merge_routes_agree_on_every_training_word(self):
        """Fit replays every merge over every occurrence, in learned order, to
        recover the stage-one spellings; encoding merges the lowest-ranked pair one
        occurrence at a time. Without dropout the two must land on one spelling."""
        word_merges = (
            SuperBytePairEncoding(vocabulary_size=22, transition_size=20)
            .fit(SENNRICH_CORPUS)
            .word_merges
        )

        for word in SENNRICH_WORD_COUNTS:
            symbols = (*word[:-1], word[-1] + "</w>")
            replayed = symbols
            for merge in word_merges:
                replayed = with_pair_merged(replayed, merge)

            assert apply_merges(symbols, word_merges) == replayed

    def test_a_transition_below_the_alphabet_is_refused(self):
        with pytest.raises(VocabularyTooSmallError):
            fit_of_the(vocabulary_size=10, transition_size=5)

    def test_a_transition_of_exactly_the_alphabet_learns_no_word_merges(self):
        tokenizer = fit_of_the(vocabulary_size=8, transition_size=6)

        assert tokenizer.n_word_merges == 0
        assert tokenizer.n_superword_merges == 2

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            SuperBytePairEncoding(vocabulary_size=10, transition_size=8).fit("of the")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            SuperBytePairEncoding(vocabulary_size=10, transition_size=8).fit(["  ", ""])

    def test_fit_returns_self(self):
        tokenizer = SuperBytePairEncoding(vocabulary_size=10, transition_size=9)

        assert tokenizer.fit(OF_THE_CORPUS) is tokenizer


class TestEncode:
    def test_a_recurring_bigram_becomes_one_token(self):
        assert fit_of_the(10, 9).encode("of the").texts == ("of</w>the</w>",)

    def test_ids_are_vocabulary_positions(self):
        tokenizer = fit_of_the(10, 9)

        assert tokenizer.encode("of the of the").ids == tokenizer.vocabulary.ids_of(
            ("of</w>the</w>", "of</w>the</w>")
        )

    def test_the_words_in_the_other_order_stay_apart(self):
        assert fit_of_the(10, 9).encode("the of").texts == ("the</w>", "of</w>")

    def test_a_symbol_the_corpus_never_used_is_unknown(self):
        assert fit_of_the(10, 9).encode("ox").texts == ("o", "[UNK]")

    def test_a_blank_text_encodes_to_nothing(self):
        assert fit_of_the(10, 9).encode("   ") == Encoding([])

    def test_stage_one_merges_applied_to_a_whole_text_never_cross_a_boundary(self):
        """``e`` ending one word and ``s`` starting the next: byte pair encoding's
        own guarantee, kept when every merge is applied to the text at once."""
        tokenizer = SuperBytePairEncoding(
            vocabulary_size=100, transition_size=99, minimum_pair_frequency=1
        ).fit(["e s e s e s ex"])

        assert "es" not in tokenizer.vocabulary
        assert tokenizer.encode("es").texts == ("e", "s</w>")

    def test_before_fit_raises_not_fitted(self):
        tokenizer = SuperBytePairEncoding(vocabulary_size=10, transition_size=9)

        with pytest.raises(NotFittedError):
            tokenizer.encode("of the")
        with pytest.raises(NotFittedError):
            _ = tokenizer.vocabulary
        with pytest.raises(NotFittedError):
            _ = tokenizer.merges
        with pytest.raises(NotFittedError):
            _ = tokenizer.n_word_merges


class TestDecode:
    def test_a_superword_token_decodes_to_its_words_with_the_space_restored(self):
        tokenizer = fit_of_the(10, 9)

        assert tokenizer.decode(tokenizer.encode("of the").ids) == "of the"

    def test_round_trips_text_mixing_superwords_and_words(self):
        tokenizer = fit_of_the(10, 9)

        assert (
            tokenizer.decode(tokenizer.encode("the of the of").ids) == "the of the of"
        )

    def test_a_custom_marker_is_honoured(self):
        tokenizer = SuperBytePairEncoding(
            vocabulary_size=10, transition_size=9, end_of_word_marker="_"
        ).fit(OF_THE_CORPUS)

        assert tokenizer.encode("of the").texts == ("of_the_",)
        assert tokenizer.superword_tokens == ("of_the_",)
        assert tokenizer.decode(tokenizer.encode("of the").ids) == "of the"


class TestConstruction:
    @pytest.mark.parametrize("transition_size", [10, 11])
    def test_a_transition_not_below_the_target_is_refused(self, transition_size):
        with pytest.raises(ValidationError):
            SuperBytePairEncoding(vocabulary_size=10, transition_size=transition_size)

    @pytest.mark.parametrize(
        "keywords",
        [
            {"vocabulary_size": 1, "transition_size": 0},
            {"vocabulary_size": 10, "transition_size": 1},
            {"vocabulary_size": 10, "transition_size": 8, "minimum_pair_frequency": 0},
            {"vocabulary_size": 10, "transition_size": 8, "end_of_word_marker": ""},
            {"vocabulary_size": 10, "transition_size": 8, "unknown_token": ""},
        ],
    )
    def test_out_of_range_hyperparameters_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            SuperBytePairEncoding(**keywords)

    def test_the_transition_is_required(self):
        with pytest.raises(ValidationError):
            SuperBytePairEncoding(vocabulary_size=10)  # type: ignore[call-arg]

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            SuperBytePairEncoding(vocabulary_size=10, transition_size=8, vocab_size=10)  # type: ignore[call-arg]

    def test_the_vocabulary_is_a_vocabulary(self):
        assert isinstance(fit_of_the(10, 9).vocabulary, Vocabulary)
