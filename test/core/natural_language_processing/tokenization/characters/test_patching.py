"""Spec for byte patching -- fixed blocks, and blocks cut where the next byte is
hard to predict.

The entropies are pinned against closed forms written from Shannon's
definition over hand-derived context counts, never against the implementation.
"""

import math

import pytest
from pydantic import ValidationError

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
)
from oop_ml.core.natural_language_processing.tokenization.characters.patching import (
    START_OF_TEXT,
    EntropyPatcher,
    FixedSizePatcher,
    Patch,
    Patches,
    PatchingRule,
    patches_starting_at,
)


def entropy_of_one_outcome(n_observations: int, smoothing: float) -> float:
    """A context seen ``n_observations`` times, always with the same next byte.

    ``p_seen = (n + s) / (n + 256 s)`` and 255 others at ``s / (n + 256 s)``.
    """
    total = n_observations + 256 * smoothing
    p_seen = (n_observations + smoothing) / total
    p_other = smoothing / total
    return -p_seen * math.log2(p_seen) - 255 * p_other * math.log2(p_other)


def entropy_of_two_equal_outcomes(n_each: int, smoothing: float) -> float:
    """A context followed by two different bytes ``n_each`` times each."""
    total = 2 * n_each + 256 * smoothing
    p_seen = (n_each + smoothing) / total
    p_other = smoothing / total
    return -2 * p_seen * math.log2(p_seen) - 254 * p_other * math.log2(p_other)


# Closed forms for the corpus ["aab"] at order 1, smoothing 1. The start was
# followed by a once; a was followed by a once and b once; b was never seen.
ENTROPY_START_OF_AAB = (2 / 257) * math.log2(257 / 2) + (255 / 257) * math.log2(257)
ENTROPY_AFTER_A_OF_AAB = 2 * (2 / 258) * math.log2(258 / 2) + (254 / 258) * math.log2(
    258
)
ENTROPY_UNSEEN = 8.0

# The alignment fixture: twenty copies, order 2, smoothing 0.001.
SENTENCE = "the cat sat on the mat"
SENTENCE_CORPUS = [SENTENCE] * 20
ALIGNMENT_SMOOTHING = 0.001


def fit_sentence(threshold: float, rule: PatchingRule = PatchingRule.GLOBAL_THRESHOLD):
    return EntropyPatcher(
        threshold=threshold, smoothing=ALIGNMENT_SMOOTHING, rule=rule
    ).fit(SENTENCE_CORPUS)


class TestPatch:
    def test_carries_its_bytes_and_span(self):
        patch = Patch([104, 105], 4, 6)

        assert patch.byte_ids == (104, 105)
        assert patch.start == 4
        assert patch.end == 6
        assert patch.n_bytes == 2
        assert list(patch) == [104, 105]
        assert len(patch) == 2

    def test_text_decodes_the_bytes(self):
        assert Patch(list(b"hi"), 0, 2).text == "hi"

    def test_text_of_a_cut_character_is_the_replacement_character(self):
        first_byte, second_byte = "é".encode()

        assert Patch([first_byte], 0, 1).text == "�"
        assert Patch([second_byte], 1, 2).text == "�"

    def test_needs_at_least_one_byte(self):
        with pytest.raises(EmptyValuesError):
            Patch([], 0, 0)

    @pytest.mark.parametrize("byte_id", [-1, 256])
    def test_refuses_a_value_that_is_not_a_byte(self, byte_id):
        with pytest.raises(InvalidValuesError):
            Patch([byte_id], 0, 1)

    def test_refuses_a_negative_start(self):
        with pytest.raises(InvalidValuesError):
            Patch([1], -1, 0)

    @pytest.mark.parametrize(("start", "end"), [(0, 1), (0, 3), (2, 2)])
    def test_refuses_a_span_that_does_not_match_the_count(self, start, end):
        with pytest.raises(InvalidValuesError):
            Patch([1, 2], start, end)

    def test_equality_is_by_value(self):
        assert Patch([1, 2], 0, 2) == Patch([1, 2], 0, 2)
        assert Patch([1, 2], 0, 2) != Patch([1, 2], 2, 4)
        assert Patch([1, 2], 0, 2) != (1, 2)
        assert hash(Patch([1, 2], 0, 2)) == hash(Patch([1, 2], 0, 2))


class TestPatches:
    def test_reads_off_the_patches(self):
        patches = Patches(
            [Patch([1, 2, 3], 0, 3), Patch([4, 5], 3, 5), Patch([6], 5, 6)]
        )

        assert patches.n_patches == 3
        assert patches.n_bytes == 6
        assert patches.byte_ids == (1, 2, 3, 4, 5, 6)
        assert patches.boundaries == (0, 3, 5)
        assert patches.sizes == (3, 2, 1)
        assert len(patches) == 3
        assert patches[1] == Patch([4, 5], 3, 5)
        assert [patch.n_bytes for patch in patches] == [3, 2, 1]

    def test_may_be_empty(self):
        assert Patches([]).n_patches == 0
        assert Patches([]).byte_ids == ()
        assert Patches([]).boundaries == ()

    def test_must_begin_at_offset_zero(self):
        with pytest.raises(InvalidValuesError):
            Patches([Patch([1], 1, 2)])

    def test_refuses_a_gap(self):
        with pytest.raises(InvalidValuesError):
            Patches([Patch([1], 0, 1), Patch([3], 2, 3)])

    def test_refuses_an_overlap(self):
        with pytest.raises(InvalidValuesError):
            Patches([Patch([1, 2], 0, 2), Patch([2, 3], 1, 3)])

    def test_equality_is_by_value(self):
        assert Patches([Patch([1], 0, 1)]) == Patches([Patch([1], 0, 1)])
        assert Patches([Patch([1], 0, 1)]) != Patches([])
        assert Patches([]) != []


class TestPatchesStartingAt:
    def test_cuts_at_the_starts(self):
        patches = patches_starting_at(b"abcdef", [0, 2, 5])

        assert patches.sizes == (2, 3, 1)
        assert [patch.text for patch in patches] == ["ab", "cde", "f"]

    def test_no_bytes_and_no_starts_is_no_patches(self):
        assert patches_starting_at(b"", []) == Patches([])

    def test_no_bytes_with_a_start_is_refused(self):
        with pytest.raises(InvalidValuesError):
            patches_starting_at(b"", [0])

    def test_a_first_start_after_zero_is_refused(self):
        with pytest.raises(InvalidValuesError):
            patches_starting_at(b"abc", [1])

    def test_no_starts_at_all_is_refused_when_there_are_bytes(self):
        with pytest.raises(InvalidValuesError):
            patches_starting_at(b"abc", [])

    @pytest.mark.parametrize("starts", [[0, 2, 2], [0, 2, 1], [0, 3], [0, 4]])
    def test_starts_must_increase_within_the_bytes(self, starts):
        with pytest.raises(InvalidValuesError):
            patches_starting_at(b"abc", starts)


class TestFixedSizePatcher:
    def test_ten_bytes_at_size_four_are_four_four_two(self):
        patches = FixedSizePatcher(patch_size=4).patch("0123456789")

        assert patches.sizes == (4, 4, 2)
        assert patches.boundaries == (0, 4, 8)
        assert [patch.text for patch in patches] == ["0123", "4567", "89"]

    def test_a_size_that_divides_leaves_no_short_patch(self):
        assert FixedSizePatcher(patch_size=5).patch("0123456789").sizes == (5, 5)

    def test_a_size_at_or_past_the_length_is_one_patch(self):
        assert FixedSizePatcher(patch_size=10).patch("0123456789").n_patches == 1
        assert FixedSizePatcher(patch_size=11).patch("0123456789").n_patches == 1

    def test_size_one_is_the_byte_tokenizer(self):
        patches = FixedSizePatcher(patch_size=1).patch("abc")

        assert patches.sizes == (1, 1, 1)
        assert patches.byte_ids == tuple(b"abc")

    def test_the_bytes_are_the_texts_utf8_bytes_in_order(self):
        assert FixedSizePatcher(patch_size=3).patch("naïve").byte_ids == tuple(
            "naïve".encode()
        )

    def test_patches_are_bytes_and_cut_through_a_character(self):
        patches = FixedSizePatcher(patch_size=1).patch("é")

        assert patches.n_patches == 2
        assert [patch.text for patch in patches] == ["�", "�"]

    def test_an_empty_text_has_no_patches(self):
        assert FixedSizePatcher(patch_size=4).patch("") == Patches([])

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            FixedSizePatcher(patch_size=4).patch(b"0123")  # type: ignore[arg-type]

    @pytest.mark.parametrize("keywords", [{"patch_size": 0}, {"patch_size": -4}, {}])
    def test_an_out_of_range_or_missing_size_is_refused(self, keywords):
        with pytest.raises(ValidationError):
            FixedSizePatcher(**keywords)

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            FixedSizePatcher(patch_size=4, stride=2)  # type: ignore[call-arg]


class TestEntropyPatcherFit:
    def test_counts_one_context_per_distinct_preceding_window(self):
        assert EntropyPatcher(order=1, threshold=1.0).fit(["aab"]).n_contexts == 2
        assert EntropyPatcher(order=2, threshold=1.0).fit(["abab"]).n_contexts == 4

    def test_the_start_of_every_text_is_its_own_context(self):
        """Two texts share the start context; one text of the same bytes does not."""
        two_texts = EntropyPatcher(order=1, threshold=1.0).fit(["ab", "ab"])
        one_text = EntropyPatcher(order=1, threshold=1.0).fit(["abab"])

        assert two_texts.n_contexts == 2
        assert one_text.n_contexts == 3

    def test_the_sentinel_is_not_a_byte(self):
        assert START_OF_TEXT == 256

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            EntropyPatcher(threshold=1.0).fit("aab")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            EntropyPatcher(threshold=1.0).fit(["  ", ""])

    def test_fit_returns_self(self):
        patcher = EntropyPatcher(threshold=1.0)

        assert patcher.fit(["aab"]) is patcher

    def test_is_fittable_and_not_fitted_until_fit(self):
        patcher = EntropyPatcher(threshold=1.0)

        assert isinstance(patcher, Fittable)
        assert not patcher.is_fitted
        assert patcher.fit(["aab"]).is_fitted

    def test_before_fit_raises_not_fitted(self):
        patcher = EntropyPatcher(threshold=1.0)

        with pytest.raises(NotFittedError):
            patcher.entropies_of("aab")
        with pytest.raises(NotFittedError):
            patcher.patch("aab")
        with pytest.raises(NotFittedError):
            _ = patcher.n_contexts


class TestEntropies:
    def test_matches_the_closed_forms_on_aab(self):
        entropies = (
            EntropyPatcher(order=1, threshold=1.0).fit(["aab"]).entropies_of("aabb")
        )

        assert entropies == pytest.approx(
            (
                ENTROPY_START_OF_AAB,
                ENTROPY_AFTER_A_OF_AAB,
                ENTROPY_AFTER_A_OF_AAB,
                ENTROPY_UNSEEN,
            ),
            abs=1e-9,
        )

    def test_the_closed_forms_are_what_the_docstring_says(self):
        assert pytest.approx(7.99784, abs=5e-6) == ENTROPY_START_OF_AAB
        assert pytest.approx(7.99572, abs=5e-6) == ENTROPY_AFTER_A_OF_AAB

    def test_order_two_conditions_on_two_bytes(self):
        """Every context of ``abab`` was seen once with one continuation; the
        context ``ba`` at the start of a text was never seen."""
        patcher = EntropyPatcher(order=2, threshold=1.0).fit(["abab"])
        once = entropy_of_one_outcome(1, 1.0)

        assert patcher.entropies_of("abab") == pytest.approx((once,) * 4, abs=1e-9)
        assert patcher.entropies_of("ba") == pytest.approx((once, 8.0), abs=1e-9)

    def test_an_unseen_context_is_exactly_eight_bits(self):
        """Only the first byte has a context the corpus shares, the start."""
        entropies = EntropyPatcher(threshold=1.0).fit(["aab"]).entropies_of("xyz")

        assert entropies[0] == pytest.approx(entropy_of_one_outcome(1, 1.0), abs=1e-9)
        assert entropies[1:] == (8.0, 8.0)

    def test_the_first_byte_is_scored_under_the_start_context(self):
        patcher = EntropyPatcher(order=1, threshold=1.0).fit(["ba"])

        assert patcher.entropies_of("b") == pytest.approx(
            (entropy_of_one_outcome(1, 1.0),), abs=1e-9
        )
        assert patcher.entropies_of("ab")[1] == 8.0

    def test_one_entropy_per_byte_not_per_character(self):
        assert len(EntropyPatcher(threshold=1.0).fit(["aab"]).entropies_of("é字")) == 5

    def test_an_empty_text_has_no_entropies(self):
        assert EntropyPatcher(threshold=1.0).fit(["aab"]).entropies_of("") == ()

    def test_every_entropy_lies_between_zero_and_eight_bits(self):
        patcher = EntropyPatcher(threshold=1.0).fit(SENTENCE_CORPUS)

        for entropy in patcher.entropies_of("the cat sat on a hat, mostly"):
            assert 0.0 <= entropy <= 8.0

    def test_add_one_smoothing_needs_about_2294_observations_to_believe_a_continuation(
        self,
    ):
        """``(n + 1) / (n + 256) = 0.9`` at ``n = 2294``."""
        assert pytest.approx(0.9, abs=1e-4) == (2294 + 1) / (2294 + 256)
        assert entropy_of_one_outcome(2294, 1.0) < entropy_of_one_outcome(20, 1.0)

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            EntropyPatcher(threshold=1.0).fit(["aab"]).entropies_of(b"aab")  # type: ignore[arg-type]


class TestWordAlignment:
    """Twenty copies of ``the cat sat on the mat`` at order 2 and smoothing 0.001.

    Position 1 is ``h`` after ``(start, t)``, a context seen 20 times with one
    continuation; position 2 is ``e`` after ``th``, seen 40 times; position 4
    is ``c`` after ``e␣``, which was followed by ``c`` 20 times and ``m`` 20.
    """

    def test_inside_a_word_the_entropy_is_low_and_at_a_word_start_high(self):
        entropies = fit_sentence(0.5).entropies_of(SENTENCE)

        assert entropies[1] == pytest.approx(
            entropy_of_one_outcome(20, ALIGNMENT_SMOOTHING), abs=1e-9
        )
        assert entropies[2] == pytest.approx(
            entropy_of_one_outcome(40, ALIGNMENT_SMOOTHING), abs=1e-9
        )
        assert entropies[4] == pytest.approx(
            entropy_of_two_equal_outcomes(20, ALIGNMENT_SMOOTHING), abs=1e-9
        )

    def test_the_numbers_are_what_the_docstring_says(self):
        assert entropy_of_one_outcome(20, ALIGNMENT_SMOOTHING) == pytest.approx(
            0.198, abs=5e-4
        )
        assert entropy_of_one_outcome(40, ALIGNMENT_SMOOTHING) == pytest.approx(
            0.106, abs=5e-4
        )
        assert entropy_of_two_equal_outcomes(20, ALIGNMENT_SMOOTHING) == pytest.approx(
            1.099, abs=5e-4
        )

    def test_patches_align_with_the_words_whose_start_was_uncertain(self):
        patches = fit_sentence(0.5).patch(SENTENCE)

        assert [patch.text for patch in patches] == [
            "the ",
            "cat ",
            "sat ",
            "on the ",
            "mat",
        ]

    def test_a_word_always_followed_by_the_same_word_is_not_cut_from_it(self):
        """``on`` led to ``the`` every time, so no boundary falls before it."""
        patches = fit_sentence(0.5).patch(SENTENCE)

        assert 15 not in patches.boundaries
        assert patches.boundaries == (0, 4, 8, 12, 19)

    def test_the_two_rules_agree_where_every_rise_crosses_the_threshold(self):
        assert fit_sentence(0.5, PatchingRule.RELATIVE_INCREASE).patch(
            SENTENCE
        ) == fit_sentence(0.5).patch(SENTENCE)


class TestPatchRules:
    def test_the_first_byte_always_starts_a_patch(self):
        for rule in PatchingRule:
            patches = EntropyPatcher(threshold=7.9, rule=rule).fit(["aab"]).patch("aab")

            assert patches.boundaries[0] == 0

    def test_global_threshold_cuts_every_position_above_it(self):
        patches = EntropyPatcher(order=1, threshold=7.996).fit(["aab"]).patch("aabb")

        assert patches.boundaries == (0, 3)

    def test_the_rules_differ_on_a_plateau(self):
        """Unseen text is 8 bits everywhere: every byte a patch, or one patch."""
        by_level = EntropyPatcher(threshold=1.0).fit(["aab"]).patch("xyz")
        by_rise = (
            EntropyPatcher(threshold=1.0, rule=PatchingRule.RELATIVE_INCREASE)
            .fit(["aab"])
            .patch("xyz")
        )

        assert by_level.sizes == (1, 1, 1)
        assert by_rise.sizes == (3,)

    def test_the_rules_differ_on_the_sentence_at_a_low_threshold(self):
        by_level = fit_sentence(0.15).patch(SENTENCE)
        by_rise = fit_sentence(0.15, PatchingRule.RELATIVE_INCREASE).patch(SENTENCE)

        assert by_level.n_patches > by_rise.n_patches
        assert by_rise.boundaries == (0, 4, 8, 12, 19)

    def test_a_higher_threshold_gives_fewer_patches(self):
        counts = [
            fit_sentence(threshold).patch(SENTENCE).n_patches
            for threshold in (0.1, 0.15, 0.5, 1.0, 2.0)
        ]

        assert counts == sorted(counts, reverse=True)
        assert counts[0] > counts[-1]

    def test_a_threshold_no_entropy_can_exceed_gives_one_patch(self):
        assert fit_sentence(8.0).patch(SENTENCE).n_patches == 1

    def test_the_patches_carry_exactly_the_texts_bytes(self):
        assert fit_sentence(0.5).patch(SENTENCE).byte_ids == tuple(SENTENCE.encode())

    def test_an_empty_text_has_no_patches(self):
        assert fit_sentence(0.5).patch("") == Patches([])

    @pytest.mark.parametrize("rule", list(PatchingRule))
    @pytest.mark.parametrize("threshold", [0.15, 0.5, 1.0])
    def test_patch_agrees_with_the_rule_applied_to_the_observed_entropies(
        self, rule, threshold
    ):
        patcher = fit_sentence(threshold, rule)
        entropies = patcher.entropies_of(SENTENCE)

        if rule is PatchingRule.GLOBAL_THRESHOLD:
            expected = [0] + [
                position
                for position in range(1, len(entropies))
                if entropies[position] > threshold
            ]
        else:
            expected = [0] + [
                position
                for position in range(1, len(entropies))
                if entropies[position] - entropies[position - 1] > threshold
            ]

        assert list(patcher.patch(SENTENCE).boundaries) == expected


class TestEntropyPatcherConstruction:
    @pytest.mark.parametrize(
        "keywords",
        [
            {},
            {"threshold": 0.0},
            {"threshold": -1.0},
            {"threshold": 1.0, "order": 0},
            {"threshold": 1.0, "smoothing": 0.0},
            {"threshold": 1.0, "rule": "monotone"},
        ],
    )
    def test_out_of_range_hyperparameters_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            EntropyPatcher(**keywords)

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            EntropyPatcher(threshold=1.0, window=3)  # type: ignore[call-arg]

    def test_the_defaults_are_order_two_add_one_and_the_global_rule(self):
        patcher = EntropyPatcher(threshold=1.0)

        assert patcher.order == 2
        assert patcher.smoothing == 1.0
        assert patcher.rule is PatchingRule.GLOBAL_THRESHOLD

    def test_the_rule_is_a_closed_enum_of_two(self):
        assert [rule.value for rule in PatchingRule] == [
            "global_threshold",
            "relative_increase",
        ]
