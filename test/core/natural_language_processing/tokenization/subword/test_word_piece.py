"""Spec for WordPiece -- likelihood-scored merges, longest-match cutting.

Pinned against Sennrich et al.'s toy corpus under the continuation spelling,
whose symbol counts and first merges are worked by hand in the module docstring.
"""

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueTokensError,
    NotFittedError,
    VocabularyTooSmallError,
)
from oop_ml.core.natural_language_processing.tokenization.encoding import Encoding
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.word_piece import (
    WORD_INITIAL_MARKER,
    WordPiece,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.network.purpose import PassPurpose
from test.core.natural_language_processing.fixtures import SENNRICH_CORPUS

# The word-initial symbols, then the continuation symbols, each group in
# codepoint order. ``w`` is in both: ``widest`` opens with it, ``low`` and
# ``newest`` continue through it.
WORDPIECE_ALPHABET = [
    "l",
    "n",
    "w",
    "##d",
    "##e",
    "##i",
    "##o",
    "##r",
    "##s",
    "##t",
    "##w",
]

# Every merge the corpus admits, in learned order, as the vocabulary shows it.
# The first three are worked in the module docstring; the rest follow from the
# same rule. Six pairs tie at 1/17 for the sixth merge and ``##e ##r`` is the
# lexicographically smallest of them in the internal spelling.
WORDPIECE_MERGED_PIECES = [
    "##id",
    "wid",
    "lo",
    "##st",
    "low",
    "##er",
    "lower",
    "##est",
    "##ew",
    "new",
    "newest",
    "widest",
]

# The unknown token plus eleven symbols, then ten merges.
TEN_MERGES = len(WORDPIECE_ALPHABET) + 1 + 10


def fit_ten_merges() -> WordPiece:
    return WordPiece(vocabulary_size=TEN_MERGES).fit(SENNRICH_CORPUS)


class TestFit:
    def test_the_vocabulary_is_unknown_then_word_initial_then_continuation_then_merges(
        self,
    ):
        vocabulary = fit_ten_merges().vocabulary

        assert list(vocabulary)[:12] == ["[UNK]", *WORDPIECE_ALPHABET]
        assert list(vocabulary)[12:] == WORDPIECE_MERGED_PIECES[:10]
        assert vocabulary.unknown_token == "[UNK]"

    def test_the_first_merge_is_the_pair_its_parts_least_explain(self):
        """``##i ##d`` at 3 / (3 * 3), where frequency would take ``##e ##s`` at 9."""
        assert list(fit_ten_merges().vocabulary)[12] == "##id"
        assert (
            list(BytePairEncoding(vocabulary_size=13).fit(SENNRICH_CORPUS).vocabulary)[
                12
            ]
            == "es"
        )

    def test_the_first_three_merges_are_id_wid_lo(self):
        assert list(fit_ten_merges().vocabulary)[12:15] == ["##id", "wid", "lo"]

    def test_a_tie_between_a_word_initial_pair_and_a_continuation_pair_goes_to_the_continuation(
        self,
    ):
        """``w ##i`` and ``##i ##d`` both score 1/3; ``▁`` sorts after ``i``."""
        vocabulary = fit_ten_merges().vocabulary

        assert list(vocabulary)[12] == "##id"
        assert "wi" not in vocabulary

    def test_reaches_the_requested_size_exactly(self):
        assert fit_ten_merges().vocabulary.n_tokens == TEN_MERGES
        assert fit_ten_merges().n_merges == 10

    def test_twelve_merges_exhaust_the_corpus(self):
        tokenizer = WordPiece(vocabulary_size=100, minimum_pair_frequency=1).fit(
            SENNRICH_CORPUS
        )

        assert tokenizer.n_merges == 12
        assert list(tokenizer.vocabulary)[12:] == WORDPIECE_MERGED_PIECES

    def test_the_minimum_frequency_filters_the_candidates_rather_than_ending_the_walk(
        self,
    ):
        """Ten merges at a minimum of 3: the two count-2 pairs, ``##e ##r`` and
        ``low ##er``, are never candidates, and every count-3-or-more merge is
        still learned around them.

        This is where likelihood scoring differs from frequency scoring. The
        sixth-best pair by likelihood is ``##e ##r`` at count 2, tying ``##e
        ##st`` at count 9; a rule that tested the minimum on the *winner* and
        stopped would have learned five merges and quit with ``##est``, ``new``
        and ``newest`` unmerged. Under frequency scoring the two rules agree,
        since the winner always has the largest count. Pinned so the semantics
        cannot drift back.
        """
        tokenizer = WordPiece(vocabulary_size=100, minimum_pair_frequency=3).fit(
            SENNRICH_CORPUS
        )

        assert tokenizer.n_merges == 10
        assert list(tokenizer.vocabulary)[12:] == [
            piece for piece in WORDPIECE_MERGED_PIECES if piece not in ("##er", "lower")
        ]

    def test_ties_go_the_same_way_whatever_the_text_order(self):
        reversed_corpus = [" ".join(reversed(SENNRICH_CORPUS[0].split()))]

        assert (
            WordPiece(vocabulary_size=TEN_MERGES).fit(reversed_corpus).vocabulary
            == fit_ten_merges().vocabulary
        )

    def test_counts_across_several_texts(self):
        many_texts = WordPiece(vocabulary_size=TEN_MERGES).fit(
            SENNRICH_CORPUS[0].split()
        )

        assert many_texts.vocabulary == fit_ten_merges().vocabulary

    def test_a_vocabulary_below_the_alphabet_is_refused(self):
        with pytest.raises(VocabularyTooSmallError):
            WordPiece(vocabulary_size=11).fit(SENNRICH_CORPUS)

    def test_exactly_the_alphabet_learns_no_merges(self):
        tokenizer = WordPiece(vocabulary_size=12).fit(SENNRICH_CORPUS)

        assert tokenizer.n_merges == 0
        assert tokenizer.encode("low").texts == ("l", "##o", "##w")

    def test_a_word_containing_the_internal_marker_is_refused(self):
        with pytest.raises(InvalidValuesError):
            WordPiece(vocabulary_size=50).fit([f"{WORD_INITIAL_MARKER}low low"])

    def test_a_word_containing_the_continuation_prefix_is_refused(self):
        with pytest.raises(InvalidValuesError):
            WordPiece(vocabulary_size=50).fit(["a##b a##b"])

    def test_an_unknown_token_colliding_with_a_piece_is_refused(self):
        with pytest.raises(NonUniqueTokensError):
            WordPiece(vocabulary_size=TEN_MERGES, unknown_token="l").fit(
                SENNRICH_CORPUS
            )

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            WordPiece(vocabulary_size=5).fit("low lower")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            WordPiece(vocabulary_size=5).fit(["  ", ""])

    def test_fit_returns_self(self):
        tokenizer = WordPiece(vocabulary_size=TEN_MERGES)

        assert tokenizer.fit(SENNRICH_CORPUS) is tokenizer

    def test_a_one_character_word_is_a_word_initial_symbol_only(self):
        tokenizer = WordPiece(vocabulary_size=3).fit(["a a a"])

        assert list(tokenizer.vocabulary) == ["[UNK]", "a"]


class TestEncode:
    def test_a_word_never_seen_is_cut_by_longest_match(self):
        assert fit_ten_merges().encode("lowest").texts == ("low", "##est")

    def test_longest_match_takes_the_longest_prefix_not_the_first(self):
        """``wid`` is preferred over ``w`` at the start of ``widest``."""
        assert fit_ten_merges().encode("widest").texts == ("wid", "##est")

    def test_ids_are_vocabulary_positions(self):
        tokenizer = fit_ten_merges()

        assert tokenizer.encode("lowest").ids == tokenizer.vocabulary.ids_of(
            ("low", "##est")
        )

    def test_a_training_word_becomes_one_token_once_fully_merged(self):
        tokenizer = WordPiece(vocabulary_size=100, minimum_pair_frequency=1).fit(
            SENNRICH_CORPUS
        )

        assert tokenizer.encode("lower newest").texts == ("lower", "newest")

    def test_a_character_the_corpus_never_used_makes_the_whole_word_unknown(self):
        encoding = fit_ten_merges().encode("lox low")

        assert encoding.texts == ("[UNK]", "low")
        assert encoding.ids[0] == 0

    def test_a_known_character_in_an_unseen_position_makes_the_whole_word_unknown(self):
        """``o`` never opens a word in the corpus, so ``old`` cannot start."""
        tokenizer = fit_ten_merges()

        assert "##o" in tokenizer.vocabulary
        assert tokenizer.encode("old").texts == ("[UNK]",)

    def test_a_word_longer_than_the_limit_is_unknown_outright(self):
        tokenizer = WordPiece(vocabulary_size=TEN_MERGES, max_word_characters=5).fit(
            SENNRICH_CORPUS
        )

        assert tokenizer.encode("lowest").texts == ("[UNK]",)
        assert tokenizer.encode("lower").texts == ("lower",)

    def test_a_word_containing_the_continuation_prefix_is_unknown(self):
        assert fit_ten_merges().encode("lo##w").texts == ("[UNK]",)

    def test_a_blank_text_encodes_to_nothing(self):
        assert fit_ten_merges().encode("   ") == Encoding([])

    def test_the_purpose_changes_nothing(self):
        tokenizer = fit_ten_merges()

        assert tokenizer.encode("lowest", PassPurpose.TRAINING) == tokenizer.encode(
            "lowest"
        )

    def test_a_custom_continuation_prefix_is_honoured(self):
        tokenizer = WordPiece(vocabulary_size=TEN_MERGES, continuation_prefix="@@").fit(
            SENNRICH_CORPUS
        )

        assert tokenizer.encode("lowest").texts == ("low", "@@est")
        assert "##est" not in tokenizer.vocabulary

    def test_before_fit_raises_not_fitted(self):
        tokenizer = WordPiece(vocabulary_size=TEN_MERGES)

        with pytest.raises(NotFittedError):
            tokenizer.encode("low")
        with pytest.raises(NotFittedError):
            _ = tokenizer.vocabulary
        with pytest.raises(NotFittedError):
            _ = tokenizer.n_merges


class TestDecode:
    def test_round_trips_a_word_the_corpus_never_held(self):
        tokenizer = fit_ten_merges()

        assert tokenizer.decode(tokenizer.encode("lowest").ids) == "lowest"

    def test_continuations_join_their_word_and_words_are_separated_by_spaces(self):
        tokenizer = fit_ten_merges()

        assert tokenizer.decode(tokenizer.encode("low lower newest").ids) == (
            "low lower newest"
        )

    def test_a_lone_continuation_piece_decodes_to_its_text(self):
        tokenizer = fit_ten_merges()

        assert tokenizer.decode([tokenizer.vocabulary.id_of("##est")]) == "est"

    def test_an_unknown_word_decodes_as_the_unknown_token_between_spaces(self):
        tokenizer = fit_ten_merges()

        assert tokenizer.decode(tokenizer.encode("lox low").ids) == "[UNK] low"

    def test_a_custom_prefix_is_removed_on_the_way_back(self):
        tokenizer = WordPiece(vocabulary_size=TEN_MERGES, continuation_prefix="@@").fit(
            SENNRICH_CORPUS
        )

        assert tokenizer.decode(tokenizer.encode("lowest lower").ids) == "lowest lower"


class TestConstruction:
    @pytest.mark.parametrize(
        "keywords",
        [
            {"vocabulary_size": 1},
            {"vocabulary_size": 10, "minimum_pair_frequency": 0},
            {"vocabulary_size": 10, "continuation_prefix": ""},
            {"vocabulary_size": 10, "unknown_token": ""},
            {"vocabulary_size": 10, "max_word_characters": 0},
        ],
    )
    def test_out_of_range_hyperparameters_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            WordPiece(**keywords)

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            WordPiece(vocab_size=10)  # type: ignore[call-arg]

    def test_the_defaults_are_berts(self):
        tokenizer = WordPiece(vocabulary_size=10)

        assert tokenizer.continuation_prefix == "##"
        assert tokenizer.unknown_token == "[UNK]"
        assert tokenizer.max_word_characters == 100

    def test_the_vocabulary_is_a_vocabulary(self):
        assert isinstance(fit_ten_merges().vocabulary, Vocabulary)

    def test_the_internal_marker_never_reaches_the_vocabulary(self):
        assert not any(
            WORD_INITIAL_MARKER in token for token in fit_ten_merges().vocabulary
        )
