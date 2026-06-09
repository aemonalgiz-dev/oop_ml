"""Spec for HashedCharacterTokenizer -- CANINE's hash-based character ids."""

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.hashing.hashed_characters import (
    HASH_PRIMES,
    HashedCharacter,
    HashedCharacters,
    HashedCharacterTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import Tokenizer

# CANINE's eight primes, in the order its released code applies them.
CANINE_PRIMES = (31, 43, 59, 61, 73, 97, 103, 113)

# (97 + 1) * prime for each, all below 16384 so the modulus changes nothing.
BUCKETS_OF_A = (3038, 4214, 5782, 5978, 7154, 9506, 10094, 11074)


def is_prime(value: int) -> bool:
    return value > 1 and all(
        value % divisor for divisor in range(2, int(value**0.5) + 1)
    )


class TestHashedCharacter:
    def test_carries_the_codepoint_and_its_buckets(self):
        hashed = HashedCharacter(97, [3, 5])

        assert hashed.codepoint == 97
        assert hashed.character == "a"
        assert hashed.bucket_ids == (3, 5)
        assert hashed.n_hash_functions == 2

    @pytest.mark.parametrize("codepoint", [-1, 0x110000])
    def test_refuses_a_codepoint_outside_unicode(self, codepoint):
        with pytest.raises(InvalidValuesError):
            HashedCharacter(codepoint, [0])

    def test_needs_at_least_one_bucket(self):
        with pytest.raises(EmptyValuesError):
            HashedCharacter(97, [])

    def test_refuses_a_negative_bucket(self):
        with pytest.raises(InvalidValuesError):
            HashedCharacter(97, [1, -1])

    def test_equality_is_by_value(self):
        assert HashedCharacter(97, [1]) == HashedCharacter(97, [1])
        assert HashedCharacter(97, [1]) != HashedCharacter(97, [2])
        assert HashedCharacter(97, [1]) != HashedCharacter(98, [1])
        assert HashedCharacter(97, [1]) != (97, (1,))
        assert hash(HashedCharacter(97, [1])) == hash(HashedCharacter(97, [1]))


class TestHashedCharacters:
    def test_reads_off_the_characters(self):
        hashed = HashedCharacters(
            [HashedCharacter(104, [1, 2]), HashedCharacter(105, [3, 4])]
        )

        assert hashed.codepoints == (104, 105)
        assert hashed.bucket_ids == ((1, 2), (3, 4))
        assert hashed.text == "hi"
        assert hashed.n_characters == 2
        assert len(hashed) == 2
        assert hashed[1] == HashedCharacter(105, [3, 4])
        assert [character.character for character in hashed] == ["h", "i"]

    def test_may_be_empty(self):
        assert HashedCharacters([]).text == ""
        assert HashedCharacters([]).n_characters == 0

    def test_equality_is_by_value(self):
        assert HashedCharacters([HashedCharacter(1, [1])]) == HashedCharacters(
            [HashedCharacter(1, [1])]
        )
        assert HashedCharacters([]) != HashedCharacters([HashedCharacter(1, [1])])
        assert HashedCharacters([]) != []


class TestThePrimes:
    def test_the_first_eight_are_canines(self):
        assert HASH_PRIMES[:8] == CANINE_PRIMES
        assert HashedCharacterTokenizer().primes == CANINE_PRIMES

    def test_there_are_thirty_two_distinct_primes(self):
        assert len(HASH_PRIMES) == 32
        assert len(set(HASH_PRIMES)) == 32
        assert all(is_prime(prime) for prime in HASH_PRIMES)

    def test_asking_for_fewer_takes_a_prefix(self):
        assert HashedCharacterTokenizer(n_hash_functions=3).primes == (31, 43, 59)


class TestEncode:
    def test_the_letter_a_lands_where_canine_puts_it(self):
        assert HashedCharacterTokenizer().bucket_ids_of(ord("a")) == BUCKETS_OF_A

    def test_a_character_hashes_the_same_every_time(self):
        first = HashedCharacterTokenizer().encode("aa")
        second = HashedCharacterTokenizer().encode("a")

        assert first.bucket_ids[0] == first.bucket_ids[1] == BUCKETS_OF_A
        assert second.bucket_ids == (BUCKETS_OF_A,)

    def test_one_hashed_character_per_character(self):
        hashed = HashedCharacterTokenizer().encode("hi there")

        assert hashed.n_characters == 8
        assert hashed.codepoints == tuple(ord(character) for character in "hi there")

    def test_each_character_gets_n_hash_functions_buckets(self):
        hashed = HashedCharacterTokenizer(n_hash_functions=3).encode("abc")

        assert all(len(bucket_ids) == 3 for bucket_ids in hashed.bucket_ids)

    @pytest.mark.parametrize("n_buckets", [2, 7, 16384])
    def test_every_bucket_is_in_range(self, n_buckets):
        tokenizer = HashedCharacterTokenizer(n_buckets=n_buckets)
        text = "".join(chr(codepoint) for codepoint in range(0, 0x3000, 7))

        for bucket_ids in tokenizer.encode(text).bucket_ids:
            assert all(0 <= bucket_id < n_buckets for bucket_id in bucket_ids)

    @pytest.mark.parametrize(
        "text", ["hello world", "naïve café", "日本語", "😀 emoji", ""]
    )
    def test_the_codepoints_round_trip_the_text(self, text):
        assert HashedCharacterTokenizer().encode(text).text == text

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            HashedCharacterTokenizer().encode(b"hi")  # type: ignore[arg-type]

    def test_a_codepoint_outside_unicode_is_refused(self):
        with pytest.raises(InvalidValuesError):
            HashedCharacterTokenizer().bucket_ids_of(0x110000)

    def test_is_not_a_tokenizer_and_has_no_vocabulary(self):
        tokenizer = HashedCharacterTokenizer()

        assert not isinstance(tokenizer, Tokenizer)
        assert not hasattr(tokenizer, "vocabulary")
        assert not hasattr(tokenizer, "decode")


class TestCollisions:
    """Multiplicative hashing modulo a shared bucket count: what it separates."""

    def test_characters_closer_than_n_buckets_differ_in_every_bucket(self):
        tokenizer = HashedCharacterTokenizer()
        buckets_of_a, buckets_of_b = tokenizer.encode("ab").bucket_ids

        assert all(
            first != second
            for first, second in zip(buckets_of_a, buckets_of_b, strict=True)
        )

    def test_characters_congruent_modulo_n_buckets_collide_in_every_bucket(self):
        tokenizer = HashedCharacterTokenizer()
        capital_a, far_ideograph = tokenizer.encode(
            "A" + chr(ord("A") + 16384)
        ).bucket_ids

        assert (
            capital_a
            == far_ideograph
            == (2046, 2838, 3894, 4026, 4818, 6402, 6798, 7458)
        )

    @pytest.mark.parametrize("n_hash_functions", [1, 8])
    def test_eight_hashes_separate_no_more_than_one_when_a_prime_is_coprime(
        self, n_hash_functions
    ):
        """Every hash is a permutation of the residues, so all are functions of
        ``codepoint mod n_buckets`` and the tuple count is the bucket count."""
        tokenizer = HashedCharacterTokenizer(n_hash_functions=n_hash_functions)
        distinct = {
            tokenizer.bucket_ids_of(codepoint) for codepoint in range(2 * 16384)
        }

        assert len(distinct) == 16384

    def test_a_second_hash_helps_only_when_the_first_prime_shares_a_factor(self):
        """93 is 3 * 31: the first hash collapses to three classes, the second
        (prime 43, coprime) still tells all 93 residues apart."""
        one_hash = HashedCharacterTokenizer(n_hash_functions=1, n_buckets=93)
        two_hashes = HashedCharacterTokenizer(n_hash_functions=2, n_buckets=93)

        assert len({one_hash.bucket_ids_of(codepoint) for codepoint in range(186)}) == 3
        assert (
            len({two_hashes.bucket_ids_of(codepoint) for codepoint in range(186)}) == 93
        )
        assert two_hashes.bucket_ids_of(0) == (31, 43)
        assert two_hashes.bucket_ids_of(3) == (31, 79)


class TestConstruction:
    def test_the_defaults_are_canines(self):
        tokenizer = HashedCharacterTokenizer()

        assert tokenizer.n_hash_functions == 8
        assert tokenizer.n_buckets == 16384

    def test_thirty_two_hashes_are_the_most_that_can_be_asked_for(self):
        tokenizer = HashedCharacterTokenizer(n_hash_functions=32)

        assert len(tokenizer.encode("a").bucket_ids[0]) == 32

    def test_more_hashes_than_primes_is_refused_at_construction(self):
        with pytest.raises(ValidationError):
            HashedCharacterTokenizer(n_hash_functions=33)

    @pytest.mark.parametrize(
        "keywords",
        [{"n_hash_functions": 0}, {"n_buckets": 1}, {"n_buckets": 0}],
    )
    def test_out_of_range_hyperparameters_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            HashedCharacterTokenizer(**keywords)

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            HashedCharacterTokenizer(num_buckets=8)  # type: ignore[call-arg]
