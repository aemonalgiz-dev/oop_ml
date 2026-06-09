"""Spec for MorphemeConstrainedBytePairEncoding -- merges that stop at a morph.

Pinned against plain byte pair encoding on the one corpus where they part:
``undo untie redo retie``, each ten times, where plain merging joins prefix to
root and constrained merging cannot.
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
from oop_ml.core.natural_language_processing.tokenization.morphology.constrained_merges import (
    MorphemeConstrainedBytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.words import Words
from oop_ml.core.network.purpose import PassPurpose
from test.core.natural_language_processing.tokenization.morphology.fixtures import (
    UNDO_CORPUS,
    UNDO_MORPHS,
    UNDO_TABLE,
    MorphTable,
)

# The five merges both tokenizers learn, each at count 20, in tie-rule order.
WITHIN_MORPH_MERGES = [
    ("d", "o</w>", 20),
    ("i", "e</w>", 20),
    ("r", "e", 20),
    ("t", "ie</w>", 20),
    ("u", "n", 20),
]

# The four plain byte pair encoding goes on to learn, each at count 10.
ACROSS_MORPH_MERGES = [
    ("re", "do</w>", 10),
    ("re", "tie</w>", 10),
    ("un", "do</w>", 10),
    ("un", "tie</w>", 10),
]

MORPHS = ("un", "re", "do", "tie")


class NothingSplitter(PreTokenizer):
    """Finds no morph in any word, which must leave the word whole."""

    def _words_of(self, text: str) -> Words:
        return Words([])


def fit_constrained(**keywords: object) -> MorphemeConstrainedBytePairEncoding:
    return MorphemeConstrainedBytePairEncoding(
        vocabulary_size=30,
        morph_splitter=UNDO_TABLE,
        minimum_pair_frequency=1,
        **keywords,  # type: ignore[arg-type]
    ).fit(UNDO_CORPUS)


def fit_plain() -> BytePairEncoding:
    # Byte fallback off, because the claim is equivalence to the published
    # method, and this tokenizer has no fallback of its own to match.
    return BytePairEncoding(
        vocabulary_size=30, minimum_pair_frequency=1, byte_fallback=False
    ).fit(UNDO_CORPUS)


def merge_triples(tokenizer: BytePairEncoding | MorphemeConstrainedBytePairEncoding):
    return [(merge.left, merge.right, int(merge.score)) for merge in tokenizer.merges]


class TestWhereItPartsFromPlainBytePairEncoding:
    def test_plain_merging_joins_the_prefix_to_the_root(self):
        plain = fit_plain()

        assert merge_triples(plain) == WITHIN_MORPH_MERGES + ACROSS_MORPH_MERGES
        assert "undo</w>" in plain.vocabulary
        assert plain.encode("undo").texts == ("undo</w>",)

    def test_constrained_merging_learns_only_the_within_morph_merges(self):
        constrained = fit_constrained()

        assert merge_triples(constrained) == WITHIN_MORPH_MERGES
        assert constrained.encode("undo").texts == ("un", "do</w>")

    def test_no_piece_spans_two_morphs(self):
        for token in fit_constrained().vocabulary:
            if token == "[UNK]":
                continue
            assert any(token.removesuffix("</w>") in morph for morph in MORPHS), token

    def test_stops_when_every_morph_is_one_symbol_though_more_were_asked_for(self):
        constrained = fit_constrained()

        assert constrained.n_merges == 5
        assert constrained.vocabulary.n_tokens == 15

    def test_its_vocabulary_is_a_prefix_of_the_plain_one(self):
        assert list(fit_constrained().vocabulary) == list(fit_plain().vocabulary)[:15]

    def test_the_default_minimum_frequency_learns_the_same_five(self):
        constrained = MorphemeConstrainedBytePairEncoding(
            vocabulary_size=30, morph_splitter=UNDO_TABLE
        ).fit(UNDO_CORPUS)

        assert merge_triples(constrained) == WITHIN_MORPH_MERGES


class TestTheMarker:
    def test_only_the_last_morph_of_a_word_carries_it(self):
        vocabulary = fit_constrained().vocabulary

        assert "un" in vocabulary
        assert "un</w>" not in vocabulary
        assert "n</w>" not in vocabulary
        assert "do</w>" in vocabulary
        assert "do" not in vocabulary

    def test_a_prefix_standing_alone_has_no_marked_last_symbol(self):
        """``n`` never ended a word in the corpus, so ``n</w>`` is unknown."""
        assert fit_constrained().encode("un").texts == ("u", "[UNK]")


class TestFit:
    def test_the_vocabulary_is_unknown_then_alphabet_then_merges(self):
        vocabulary = fit_constrained().vocabulary

        assert list(vocabulary)[:10] == [
            "[UNK]",
            "d",
            "e",
            "e</w>",
            "i",
            "n",
            "o</w>",
            "r",
            "t",
            "u",
        ]
        assert list(vocabulary)[10:] == ["do</w>", "ie</w>", "re", "tie</w>", "un"]

    def test_a_splitter_leaving_words_whole_reproduces_byte_pair_encoding(self):
        same = MorphemeConstrainedBytePairEncoding(
            vocabulary_size=30,
            morph_splitter=WhitespacePreTokenizer(),
            minimum_pair_frequency=1,
        ).fit(UNDO_CORPUS)
        plain = fit_plain()

        assert same.merges == plain.merges
        assert same.vocabulary == plain.vocabulary
        assert same.encode("undo retie") == plain.encode("undo retie")

    def test_a_splitter_finding_no_morph_leaves_the_word_whole(self):
        nothing = MorphemeConstrainedBytePairEncoding(
            vocabulary_size=30,
            morph_splitter=NothingSplitter(),
            minimum_pair_frequency=1,
        ).fit(UNDO_CORPUS)

        assert nothing.morphs_of("undo") == ("undo",)
        assert nothing.merges == fit_plain().merges

    def test_a_vocabulary_below_the_alphabet_is_refused(self):
        """Nine symbols plus the unknown token is the floor."""
        with pytest.raises(VocabularyTooSmallError):
            MorphemeConstrainedBytePairEncoding(
                vocabulary_size=9, morph_splitter=UNDO_TABLE
            ).fit(UNDO_CORPUS)

    def test_exactly_the_alphabet_learns_no_merges(self):
        tokenizer = MorphemeConstrainedBytePairEncoding(
            vocabulary_size=10, morph_splitter=UNDO_TABLE
        ).fit(UNDO_CORPUS)

        assert tokenizer.n_merges == 0
        assert tokenizer.encode("undo").texts == ("u", "n", "d", "o</w>")

    def test_fit_returns_self(self):
        tokenizer = MorphemeConstrainedBytePairEncoding(
            vocabulary_size=30, morph_splitter=UNDO_TABLE
        )

        assert tokenizer.fit(UNDO_CORPUS) is tokenizer

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            MorphemeConstrainedBytePairEncoding(
                vocabulary_size=30, morph_splitter=UNDO_TABLE
            ).fit("undo untie")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            MorphemeConstrainedBytePairEncoding(
                vocabulary_size=30, morph_splitter=UNDO_TABLE
            ).fit(["  "])


class TestMorphsOf:
    def test_reads_the_splitter_and_needs_no_fit(self):
        tokenizer = MorphemeConstrainedBytePairEncoding(
            vocabulary_size=30, morph_splitter=UNDO_TABLE
        )

        assert tokenizer.morphs_of("undo") == UNDO_MORPHS["undo"]
        assert tokenizer.morphs_of("retie") == ("re", "tie")

    def test_a_word_the_splitter_does_not_know_stays_whole(self):
        assert fit_constrained().morphs_of("zebra") == ("zebra",)

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            fit_constrained().morphs_of(["undo"])  # type: ignore[arg-type]


class TestEncode:
    @pytest.mark.parametrize(
        ("word", "pieces"),
        [
            ("undo", ("un", "do</w>")),
            ("untie", ("un", "tie</w>")),
            ("redo", ("re", "do</w>")),
            ("retie", ("re", "tie</w>")),
            ("tie", ("tie</w>",)),
        ],
    )
    def test_each_morph_is_merged_on_its_own(self, word, pieces):
        assert fit_constrained().encode(word).texts == pieces

    def test_an_unseen_word_is_left_whole_by_the_table_and_still_cannot_cross(self):
        """Only within-morph merges exist, so even a whole unseen word cannot
        acquire a piece spanning two morphs."""
        assert fit_constrained().encode("unredo").texts == ("un", "re", "do</w>")

    def test_ids_are_vocabulary_positions(self):
        tokenizer = fit_constrained()

        assert tokenizer.encode("untie").ids == tokenizer.vocabulary.ids_of(
            ("un", "tie</w>")
        )

    def test_a_blank_text_encodes_to_nothing(self):
        assert fit_constrained().encode("  ") == Encoding([])

    def test_the_purpose_changes_nothing(self):
        tokenizer = fit_constrained()

        assert tokenizer.encode("undo", PassPurpose.TRAINING) == tokenizer.encode(
            "undo"
        )

    def test_before_fit_raises_not_fitted(self):
        tokenizer = MorphemeConstrainedBytePairEncoding(
            vocabulary_size=30, morph_splitter=UNDO_TABLE
        )

        with pytest.raises(NotFittedError):
            tokenizer.encode("undo")
        with pytest.raises(NotFittedError):
            _ = tokenizer.vocabulary
        with pytest.raises(NotFittedError):
            _ = tokenizer.merges


class TestDecode:
    def test_round_trips_the_corpus_words(self):
        tokenizer = fit_constrained()

        assert tokenizer.decode(tokenizer.encode("undo retie untie").ids) == (
            "undo retie untie"
        )

    def test_a_custom_marker_is_honoured(self):
        tokenizer = fit_constrained(end_of_word_marker="_")

        assert tokenizer.encode("undo").texts == ("un", "do_")
        assert tokenizer.decode(tokenizer.encode("undo redo").ids) == "undo redo"


class TestConstruction:
    @pytest.mark.parametrize(
        "keywords",
        [
            {"vocabulary_size": 1},
            {"vocabulary_size": 10, "minimum_pair_frequency": 0},
            {"vocabulary_size": 10, "end_of_word_marker": ""},
            {"vocabulary_size": 10, "unknown_token": ""},
        ],
    )
    def test_out_of_range_hyperparameters_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            MorphemeConstrainedBytePairEncoding(morph_splitter=UNDO_TABLE, **keywords)

    def test_the_splitter_is_required(self):
        with pytest.raises(ValidationError):
            MorphemeConstrainedBytePairEncoding(vocabulary_size=10)  # type: ignore[call-arg]

    def test_the_splitter_must_be_a_pre_tokenizer(self):
        with pytest.raises(ValidationError):
            MorphemeConstrainedBytePairEncoding(
                vocabulary_size=10,
                morph_splitter=UNDO_MORPHS,  # type: ignore[arg-type]
            )

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            MorphemeConstrainedBytePairEncoding(
                vocabulary_size=10,
                morph_splitter=UNDO_TABLE,
                splitter=UNDO_TABLE,  # type: ignore[call-arg]
            )

    def test_the_table_double_itself_cuts_as_stated(self):
        table = MorphTable(morphs_by_word={"undo": ("un", "do")})

        assert table.split("undo").texts == ("un", "do")
        assert [(word.start, word.end) for word in table.split("undo")] == [
            (0, 2),
            (2, 4),
        ]
        assert table.split("zebra").texts == ("zebra",)
