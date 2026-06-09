"""Spec for fast vocabulary transfer -- which old ids each new token is made of.

Pinned on byte pair vocabularies of different sizes over Sennrich's corpus,
whose merges are known, so that every mapping can be written down by hand.
"""

from collections.abc import Sequence

import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
    UnknownTokenError,
)
from oop_ml.core.natural_language_processing.tokenization.characters.byte_symbols import (
    BYTE_SYMBOLS,
)
from oop_ml.core.natural_language_processing.tokenization.subword.byte_level import (
    ByteLevelBytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.vocabulary_transfer import (
    TokenMapping,
    TokenMappings,
    VocabularyTransfer,
)
from oop_ml.core.natural_language_processing.tokenization.subword.word_piece import (
    WordPiece,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import Tokenizer
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.network.purpose import PassPurpose
from test.core.natural_language_processing.fixtures import (
    SENNRICH_ALPHABET,
    SENNRICH_CORPUS,
)

# The unknown token plus eleven symbols, then two, three or ten merges.
TWO_MERGES = len(SENNRICH_ALPHABET) + 1 + 2
THREE_MERGES = len(SENNRICH_ALPHABET) + 1 + 3
TEN_MERGES = len(SENNRICH_ALPHABET) + 1 + 10


def fit_byte_pair(
    vocabulary_size: int, unknown_token: str = "[UNK]"
) -> BytePairEncoding:
    return BytePairEncoding(
        vocabulary_size=vocabulary_size, unknown_token=unknown_token
    ).fit(SENNRICH_CORPUS)


class MarkerTokens(Tokenizer):
    """A vocabulary whose first token is a bare end-of-word marker."""

    @property
    def vocabulary(self) -> Vocabulary:
        return Vocabulary(["</w>", "a</w>"])

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        return tuple(character + "</w>" for character in text)

    def _text_from(self, pieces: Sequence[str]) -> str:
        return "".join(pieces).replace("</w>", " ").rstrip(" ")


class TestTokenMapping:
    def test_carries_the_token_its_id_and_the_source_ids(self):
        mapping = TokenMapping("low</w>", 7, [3, 10])

        assert mapping.target_token == "low</w>"
        assert mapping.target_id == 7
        assert mapping.source_ids == (3, 10)
        assert mapping.n_source_ids == 2
        assert mapping.is_mapped

    def test_no_source_ids_means_unmapped(self):
        mapping = TokenMapping("</w>", 0, [])

        assert not mapping.is_mapped
        assert mapping.n_source_ids == 0

    def test_an_empty_token_is_refused(self):
        with pytest.raises(EmptyValuesError):
            TokenMapping("", 0, [1])

    @pytest.mark.parametrize("target_id, source_ids", [(-1, [1]), (0, [1, -2])])
    def test_a_negative_id_is_refused(self, target_id, source_ids):
        with pytest.raises(InvalidValuesError):
            TokenMapping("a", target_id, source_ids)

    def test_equality_is_by_value(self):
        assert TokenMapping("a", 1, [2]) == TokenMapping("a", 1, (2,))
        assert TokenMapping("a", 1, [2]) != TokenMapping("a", 1, [3])
        assert TokenMapping("a", 1, [2]).__eq__("a") is NotImplemented


class TestTokenMappings:
    def test_iterates_in_target_id_order_and_answers_by_token(self):
        mappings = TokenMappings([TokenMapping("a", 0, [5]), TokenMapping("b", 1, [])])

        assert [mapping.target_token for mapping in mappings] == ["a", "b"]
        assert len(mappings) == 2
        assert mappings.n_mappings == 2
        assert mappings.for_token("b") == TokenMapping("b", 1, [])
        assert "a" in mappings
        assert "c" not in mappings

    def test_reports_the_unmapped_tokens(self):
        mappings = TokenMappings(
            [
                TokenMapping("a", 0, [5]),
                TokenMapping("b", 1, []),
                TokenMapping("c", 2, []),
            ]
        )

        assert mappings.n_unmapped == 2
        assert mappings.unmapped_tokens == ("b", "c")

    def test_a_token_not_in_the_target_raises_unknown_token(self):
        with pytest.raises(UnknownTokenError):
            TokenMappings([TokenMapping("a", 0, [5])]).for_token("z")

    def test_no_mappings_is_refused(self):
        with pytest.raises(EmptyValuesError):
            TokenMappings([])

    def test_a_token_mapped_twice_is_refused(self):
        with pytest.raises(InvalidValuesError):
            TokenMappings([TokenMapping("a", 0, [5]), TokenMapping("a", 1, [6])])


class TestBetweenSharedTokens:
    def test_every_token_of_a_smaller_vocabulary_over_the_same_corpus_maps_to_itself(
        self,
    ):
        source = fit_byte_pair(TEN_MERGES)
        target = fit_byte_pair(THREE_MERGES)

        mappings = VocabularyTransfer.between(source, target)

        assert mappings.n_mappings == THREE_MERGES
        assert mappings.n_unmapped == 0
        for mapping in mappings:
            assert mapping.source_ids == (
                source.vocabulary.id_of(mapping.target_token),
            )

    def test_a_shared_inner_piece_maps_by_spelling_not_by_re_encoding(self):
        """``lo`` is in both; re-encoding its text would invent an ``o</w>``."""
        source = fit_byte_pair(TEN_MERGES)
        target = fit_byte_pair(THREE_MERGES)

        mapping = VocabularyTransfer.between(source, target).for_token("lo")

        assert mapping.source_ids == (source.vocabulary.id_of("lo"),)

    def test_the_unknown_token_maps_to_the_source_unknown_whatever_either_is_called(
        self,
    ):
        source = fit_byte_pair(TEN_MERGES)
        target = fit_byte_pair(THREE_MERGES, unknown_token="<unk>")

        mapping = VocabularyTransfer.between(source, target).for_token("<unk>")

        assert mapping.source_ids == (0,)
        assert source.vocabulary.token_of(0) == "[UNK]"


class TestBetweenNewTokens:
    def test_a_merged_target_token_maps_to_the_source_pieces_that_spell_it(self):
        """``low</w>`` is the seventh of ten merges; a three-merge source spells the
        word ``low`` as ``lo`` and ``w</w>``."""
        source = fit_byte_pair(THREE_MERGES)
        target = fit_byte_pair(TEN_MERGES)

        mappings = VocabularyTransfer.between(source, target)

        assert mappings.for_token("low</w>").source_ids == source.vocabulary.ids_of(
            ("lo", "w</w>")
        )
        assert mappings.for_token("ewest</w>").source_ids == source.vocabulary.ids_of(
            ("e", "w", "est</w>")
        )
        assert mappings.n_unmapped == 0

    def test_an_inner_piece_the_source_lacks_is_re_read_as_a_whole_word(self):
        """``ew`` never ends a word, but its text does: the source spells the word
        ``ew`` as ``e`` and ``w</w>``, a marker the piece never carried."""
        source = fit_byte_pair(THREE_MERGES)
        target = fit_byte_pair(TEN_MERGES)

        mapping = VocabularyTransfer.between(source, target).for_token("ew")

        assert mapping.source_ids == source.vocabulary.ids_of(("e", "w</w>"))

    def test_an_inner_piece_whose_last_letter_never_ends_a_word_reaches_the_unknown(
        self,
    ):
        """No word of the corpus ends in ``o``, so ``o</w>`` is not a symbol and
        the text ``lo`` re-encodes as ``l`` followed by the unknown token."""
        source = fit_byte_pair(TWO_MERGES)
        target = fit_byte_pair(TEN_MERGES)

        mapping = VocabularyTransfer.between(source, target).for_token("lo")

        assert "lo" not in source.vocabulary
        assert mapping.source_ids == (source.vocabulary.id_of("l"), 0)

    def test_across_conventions_a_final_piece_lands_and_a_word_initial_one_gains_a_marker(
        self,
    ):
        source = fit_byte_pair(TEN_MERGES)
        target = WordPiece(vocabulary_size=22).fit(SENNRICH_CORPUS)

        mappings = VocabularyTransfer.between(source, target)

        assert mappings.for_token("##est").source_ids == (
            source.vocabulary.id_of("est</w>"),
        )
        assert mappings.for_token("low").source_ids == (
            source.vocabulary.id_of("low</w>"),
        )


class TestBetweenUnmappedTokens:
    def test_a_token_that_is_only_a_marker_decodes_to_nothing_and_is_unmapped(self):
        source = fit_byte_pair(TEN_MERGES)

        mappings = VocabularyTransfer.between(source, MarkerTokens())

        assert not mappings.for_token("</w>").is_mapped
        assert mappings.for_token("</w>").source_ids == ()
        assert mappings.unmapped_tokens == ("</w>",)
        assert mappings.for_token("a</w>").is_mapped

    def test_a_byte_symbol_naming_whitespace_finds_no_word_in_the_source(self):
        source = fit_byte_pair(TEN_MERGES)
        target = ByteLevelBytePairEncoding(vocabulary_size=266).fit(SENNRICH_CORPUS)

        mappings = VocabularyTransfer.between(source, target)

        assert not mappings.for_token(BYTE_SYMBOLS[ord(" ")]).is_mapped
        assert not mappings.for_token(BYTE_SYMBOLS[ord("\t")]).is_mapped
        assert mappings.for_token("low").is_mapped

    def test_exactly_the_whitespace_bytes_are_unmapped(self):
        """Ten of them: tab through carriage return, the four separators 0x1c to
        0x1f, and the space. Every other byte decodes to something the whitespace
        pre-tokenizer counts as a word, even if only the replacement character."""
        source = fit_byte_pair(TEN_MERGES)
        target = ByteLevelBytePairEncoding(vocabulary_size=266).fit(SENNRICH_CORPUS)
        whitespace_symbols = {
            BYTE_SYMBOLS[value]
            for value in range(256)
            if bytes([value]).decode("utf-8", errors="replace").isspace()
        }

        mappings = VocabularyTransfer.between(source, target)

        assert set(mappings.unmapped_tokens) == whitespace_symbols
        assert mappings.n_unmapped == 10

    def test_a_closed_source_spells_the_unknown_token_as_text(self):
        """A byte-level source has no unknown token to hand over, so ``[UNK]`` is
        five characters of text like any other."""
        source = ByteLevelBytePairEncoding(vocabulary_size=266).fit(SENNRICH_CORPUS)
        target = fit_byte_pair(THREE_MERGES)

        mapping = VocabularyTransfer.between(source, target).for_token("[UNK]")

        assert mapping.source_ids == source.encode("[UNK]").ids
        assert mapping.n_source_ids == 5


class TestBetweenGuards:
    def test_an_unfitted_source_raises_not_fitted(self):
        with pytest.raises(NotFittedError):
            VocabularyTransfer.between(
                BytePairEncoding(vocabulary_size=TEN_MERGES),
                fit_byte_pair(THREE_MERGES),
            )

    def test_an_unfitted_target_raises_not_fitted(self):
        with pytest.raises(NotFittedError):
            VocabularyTransfer.between(
                fit_byte_pair(TEN_MERGES),
                BytePairEncoding(vocabulary_size=THREE_MERGES),
            )

    def test_the_mappings_cover_the_target_in_id_order(self):
        source = fit_byte_pair(THREE_MERGES)
        target = fit_byte_pair(TEN_MERGES)

        mappings = VocabularyTransfer.between(source, target)

        assert [mapping.target_id for mapping in mappings] == list(range(TEN_MERGES))
        assert [mapping.target_token for mapping in mappings] == list(target.vocabulary)
