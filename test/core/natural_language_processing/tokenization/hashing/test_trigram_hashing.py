"""Spec for TrigramHashTokenizer -- T-FREE's tokenizer-side, a word as the set of
buckets its character trigrams hash to."""

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.hashing.trigram_hashing import (
    HASH_MULTIPLIERS,
    TextActivations,
    TrigramHashTokenizer,
    WordActivations,
    overlap,
    polynomial_hash,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import Tokenizer

# Worked in the module docstring: 95 * 1000003^2 + 99 * 1000003 + 97.
HASH_OF_UNDERSCORE_C_A = 95000669001249

# At 8192 buckets under one hash. The two shared buckets are _ca and cat.
BUCKETS_OF_CAT = (2002, 5665, 6052)
BUCKETS_OF_CATS = (1676, 2002, 5665, 6072)


def tokenizer_8192() -> TrigramHashTokenizer:
    return TrigramHashTokenizer(n_buckets=8192)


class TestPolynomialHash:
    def test_matches_the_hand_worked_value(self):
        assert polynomial_hash("_ca", 1000003) == HASH_OF_UNDERSCORE_C_A
        assert HASH_OF_UNDERSCORE_C_A == 95 * 1000003**2 + 99 * 1000003 + 97

    def test_a_single_character_is_its_codepoint(self):
        assert polynomial_hash("a", 1000003) == 97

    def test_the_empty_string_is_zero(self):
        assert polynomial_hash("", 1000003) == 0

    def test_is_a_fixed_function_of_the_text(self):
        assert polynomial_hash("cat", 1000003) == polynomial_hash("cat", 1000003)
        assert polynomial_hash("cat", 1000003) != polynomial_hash("cat", 1000033)
        assert polynomial_hash("cat", 1000003) != polynomial_hash("act", 1000003)

    def test_the_multipliers_are_eight_distinct_primes(self):
        assert len(HASH_MULTIPLIERS) == 8
        assert len(set(HASH_MULTIPLIERS)) == 8
        for multiplier in HASH_MULTIPLIERS:
            assert all(
                multiplier % divisor for divisor in range(2, int(multiplier**0.5) + 1)
            )


class TestWordActivations:
    def test_carries_the_word_and_its_sorted_buckets(self):
        activations = WordActivations("cat", [2, 5, 9])

        assert activations.word == "cat"
        assert activations.bucket_ids == (2, 5, 9)
        assert activations.n_active == 3
        assert list(activations) == [2, 5, 9]
        assert len(activations) == 3
        assert 5 in activations
        assert 4 not in activations

    def test_needs_a_word(self):
        with pytest.raises(EmptyValuesError):
            WordActivations("", [1])

    def test_needs_at_least_one_bucket(self):
        with pytest.raises(EmptyValuesError):
            WordActivations("cat", [])

    @pytest.mark.parametrize("bucket_ids", [[3, 2], [2, 2], [-1, 0]])
    def test_buckets_are_a_strictly_increasing_set_of_positions(self, bucket_ids):
        with pytest.raises(InvalidValuesError):
            WordActivations("cat", bucket_ids)

    def test_equality_is_by_value(self):
        assert WordActivations("cat", [1]) == WordActivations("cat", [1])
        assert WordActivations("cat", [1]) != WordActivations("cat", [2])
        assert WordActivations("cat", [1]) != WordActivations("dog", [1])
        assert WordActivations("cat", [1]) != ("cat", (1,))
        assert hash(WordActivations("cat", [1])) == hash(WordActivations("cat", [1]))


class TestOverlap:
    def test_counts_the_buckets_both_words_light(self):
        assert (
            overlap(WordActivations("a", [1, 2, 3]), WordActivations("b", [2, 3, 4]))
            == 2
        )

    def test_a_word_overlaps_itself_completely(self):
        activations = WordActivations("cat", [1, 2, 3])

        assert overlap(activations, activations) == 3

    def test_disjoint_words_overlap_in_nothing(self):
        assert overlap(WordActivations("a", [1]), WordActivations("b", [2])) == 0


class TestTextActivations:
    def test_reads_off_the_words(self):
        activations = TextActivations(
            [WordActivations("the", [1, 2]), WordActivations("cat", [3])]
        )

        assert activations.words == ("the", "cat")
        assert activations.bucket_ids == ((1, 2), (3,))
        assert activations.n_words == 2
        assert len(activations) == 2
        assert activations[1] == WordActivations("cat", [3])
        assert [word.word for word in activations] == ["the", "cat"]

    def test_may_be_empty(self):
        assert TextActivations([]).n_words == 0
        assert TextActivations([]).words == ()

    def test_equality_is_by_value(self):
        assert TextActivations([WordActivations("a", [1])]) == TextActivations(
            [WordActivations("a", [1])]
        )
        assert TextActivations([]) != TextActivations([WordActivations("a", [1])])
        assert TextActivations([]) != []


class TestTrigrams:
    def test_a_word_is_wrapped_and_cut_into_overlapping_trigrams(self):
        assert tokenizer_8192().trigrams_of("word") == ("_wo", "wor", "ord", "rd_")

    @pytest.mark.parametrize("word", ["a", "at", "cat", "cats", "hashing", "日本語"])
    def test_a_word_of_n_characters_has_n_trigrams(self, word):
        assert len(tokenizer_8192().trigrams_of(word)) == len(word)

    def test_a_one_character_word_has_one_trigram_holding_both_markers(self):
        assert tokenizer_8192().trigrams_of("a") == ("_a_",)

    def test_repeated_trigrams_are_reported_as_they_occur(self):
        assert tokenizer_8192().trigrams_of("aaaa") == ("_aa", "aaa", "aaa", "aa_")

    def test_a_custom_marker_is_honoured(self):
        tokenizer = TrigramHashTokenizer(n_buckets=8192, boundary_marker=" ")

        assert tokenizer.trigrams_of("cat") == (" ca", "cat", "at ")

    def test_an_empty_word_is_refused(self):
        with pytest.raises(EmptyValuesError):
            tokenizer_8192().trigrams_of("")

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            tokenizer_8192().trigrams_of(["cat"])  # type: ignore[arg-type]


class TestActivations:
    def test_cat_lights_the_pinned_buckets(self):
        assert tokenizer_8192().activations_of("cat") == WordActivations(
            "cat", BUCKETS_OF_CAT
        )

    def test_cat_and_cats_share_the_buckets_of_their_shared_trigrams(self):
        tokenizer = tokenizer_8192()
        cat = tokenizer.activations_of("cat")
        cats = tokenizer.activations_of("cats")

        assert cats == WordActivations("cats", BUCKETS_OF_CATS)
        assert overlap(cat, cats) == 2
        assert set(cat.bucket_ids) & set(cats.bucket_ids) == {2002, 5665}

    def test_the_shared_buckets_are_the_shared_trigrams(self):
        tokenizer = tokenizer_8192()

        assert polynomial_hash("_ca", 1000003) % 8192 in BUCKETS_OF_CAT
        assert polynomial_hash("cat", 1000003) % 8192 in BUCKETS_OF_CAT
        assert tokenizer.activations_of("_ca"[1:]).word == "ca"

    def test_the_same_word_gives_the_same_buckets_across_constructions(self):
        assert TrigramHashTokenizer(n_buckets=8192).activations_of(
            "hashing"
        ) == TrigramHashTokenizer(n_buckets=8192).activations_of("hashing")

    def test_a_word_is_a_set_so_a_repeated_trigram_lights_one_bucket(self):
        activations = tokenizer_8192().activations_of("aaaa")

        assert activations.n_active == 3
        assert activations.bucket_ids == (3243, 3245, 4507)

    def test_the_buckets_are_sorted(self):
        for word in ("hashing", "tokenizer", "zzzyyyxxx"):
            bucket_ids = tokenizer_8192().activations_of(word).bucket_ids

            assert bucket_ids == tuple(sorted(bucket_ids))

    @pytest.mark.parametrize("n_buckets", [2, 7, 8192])
    def test_every_bucket_is_in_range(self, n_buckets):
        tokenizer = TrigramHashTokenizer(n_buckets=n_buckets)

        for word in ("the", "quick", "brown", "fox", "日本語", "a"):
            assert all(
                0 <= bucket_id < n_buckets
                for bucket_id in tokenizer.activations_of(word).bucket_ids
            )

    def test_more_hash_functions_light_more_buckets_including_the_first_hashs(self):
        one = TrigramHashTokenizer(n_buckets=8192).activations_of("cat")
        two = TrigramHashTokenizer(n_buckets=8192, n_hash_functions=2).activations_of(
            "cat"
        )

        assert set(one.bucket_ids) <= set(two.bucket_ids)
        assert 3 < two.n_active <= 6

    def test_the_hash_spreads_the_trigrams(self):
        """19,683 trigrams over 26 letters and the marker into 8,192 buckets."""
        alphabet = "abcdefghijklmnopqrstuvwxyz_"
        loads: dict[int, int] = {}
        for first in alphabet:
            for second in alphabet:
                for third in alphabet:
                    bucket = polynomial_hash(first + second + third, 1000003) % 8192
                    loads[bucket] = loads.get(bucket, 0) + 1

        assert len(loads) == 8122
        assert max(loads.values()) == 5


class TestEncode:
    def test_one_activation_set_per_word_the_pre_tokenizer_finds(self):
        activations = tokenizer_8192().encode("the cat  sat")

        assert activations.words == ("the", "cat", "sat")
        assert activations[1] == tokenizer_8192().activations_of("cat")

    def test_a_blank_text_has_no_words(self):
        assert tokenizer_8192().encode("   ") == TextActivations([])
        assert tokenizer_8192().encode("") == TextActivations([])

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            tokenizer_8192().encode(["the", "cat"])  # type: ignore[arg-type]

    def test_the_lookup_a_decoder_would_run_finds_the_nearest_candidate(self):
        """T-FREE recovers a word by nearest neighbour over a caller's dictionary."""
        tokenizer = tokenizer_8192()
        candidates = [tokenizer.activations_of(word) for word in ("dog", "cat", "cats")]
        query = tokenizer.encode("cats")[0]

        ranked = sorted(candidates, key=lambda candidate: -overlap(query, candidate))

        assert [candidate.word for candidate in ranked] == ["cats", "cat", "dog"]

    def test_is_not_a_tokenizer_and_does_not_decode(self):
        tokenizer = tokenizer_8192()

        assert not isinstance(tokenizer, Tokenizer)
        assert not hasattr(tokenizer, "decode")
        assert not hasattr(tokenizer, "vocabulary")


class TestConstruction:
    def test_the_defaults_are_one_hash_whitespace_words_and_an_underscore(self):
        tokenizer = tokenizer_8192()

        assert tokenizer.n_hash_functions == 1
        assert tokenizer.boundary_marker == "_"
        assert tokenizer.multipliers == (1000003,)
        assert tokenizer.pre_tokenizer.split("a b").texts == ("a", "b")

    def test_eight_hashes_are_the_most_that_can_be_asked_for(self):
        assert TrigramHashTokenizer(n_buckets=8192, n_hash_functions=8).multipliers == (
            HASH_MULTIPLIERS
        )

    def test_more_hashes_than_multipliers_is_refused_at_construction(self):
        with pytest.raises(ValidationError):
            TrigramHashTokenizer(n_buckets=8192, n_hash_functions=9)

    @pytest.mark.parametrize(
        "keywords",
        [
            {},
            {"n_buckets": 1},
            {"n_buckets": 8192, "n_hash_functions": 0},
            {"n_buckets": 8192, "boundary_marker": ""},
            {"n_buckets": 8192, "boundary_marker": "__"},
        ],
    )
    def test_out_of_range_hyperparameters_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            TrigramHashTokenizer(**keywords)

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            TrigramHashTokenizer(n_buckets=8192, vocabulary_size=8192)  # type: ignore[call-arg]
