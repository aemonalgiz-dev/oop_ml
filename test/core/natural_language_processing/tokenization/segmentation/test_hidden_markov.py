"""Spec for HiddenMarkovSegmenter -- B/M/E/S tagging decoded by constrained Viterbi.

The smoothed tables are pinned against the hand-counted fractions in
``fixtures.py``, derived on paper for the two-sentence corpus ``ab / c`` and
``abc`` before the implementation existed.
"""

import math

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.hidden_markov import (
    ADMISSIBLE_SUCCESSORS,
    FINAL_TAGS,
    INITIAL_TAGS,
    BoundaryTag,
    HiddenMarkovSegmenter,
    tags_of_word,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word
from test.core.natural_language_processing.tokenization.segmentation.fixtures import (
    BOUNDARY_BEHAVIOUR_CORPUS,
    CJK_SEGMENTED_CORPUS,
    CONTRADICTORY_CJK_CORPUS,
    HAND_EMISSIONS,
    HAND_INITIAL,
    HAND_TRANSITIONS,
    LIFE_READING,
    MAJORITY_READING,
    MINORITY_READING,
    MINORITY_SENTENCE,
    NOVEL_SEGMENTATION,
    NOVEL_SENTENCE,
    RESEARCH_SENTENCE,
    TWO_SENTENCE_CORPUS,
)


def two_sentence_model() -> HiddenMarkovSegmenter:
    return HiddenMarkovSegmenter().fit(TWO_SENTENCE_CORPUS)


def cut_at_boundaries(text: str, tags: tuple[BoundaryTag, ...]) -> tuple[str, ...]:
    """An independent cutter: one word ends after every E and every S.

    Written against the non-whitespace characters, since whitespace carries
    no tag, and it is the oracle the agreement test compares ``split`` to.
    """
    characters = [character for character in text if not character.isspace()]
    assert len(characters) == len(tags)
    words: list[str] = []
    current = ""
    for character, tag in zip(characters, tags, strict=True):
        current += character
        if tag in (BoundaryTag.END, BoundaryTag.SINGLE):
            words.append(current)
            current = ""
    assert current == ""
    return tuple(words)


class TestTagsOfWord:
    def test_a_single_character_is_s(self):
        assert tags_of_word("a") == (BoundaryTag.SINGLE,)

    def test_two_characters_are_b_e(self):
        assert tags_of_word("ab") == (BoundaryTag.BEGIN, BoundaryTag.END)

    def test_longer_words_fill_the_middle_with_m(self):
        assert tags_of_word("abcd") == (
            BoundaryTag.BEGIN,
            BoundaryTag.MIDDLE,
            BoundaryTag.MIDDLE,
            BoundaryTag.END,
        )

    def test_the_structural_constants_describe_a_word(self):
        assert INITIAL_TAGS == (BoundaryTag.BEGIN, BoundaryTag.SINGLE)
        assert FINAL_TAGS == (BoundaryTag.END, BoundaryTag.SINGLE)
        assert ADMISSIBLE_SUCCESSORS[BoundaryTag.BEGIN] == (
            BoundaryTag.MIDDLE,
            BoundaryTag.END,
        )
        assert ADMISSIBLE_SUCCESSORS[BoundaryTag.END] == (
            BoundaryTag.BEGIN,
            BoundaryTag.SINGLE,
        )
        assert (
            sum(len(successors) for successors in ADMISSIBLE_SUCCESSORS.values()) == 8
        )


class TestHandCountedProbabilities:
    @pytest.mark.parametrize(("tag", "expected"), list(HAND_INITIAL.items()))
    def test_initial_tags_are_the_smoothed_ratios(self, tag, expected):
        model = two_sentence_model()

        assert model.initial_log_probability(BoundaryTag(tag)) == pytest.approx(
            math.log(expected)
        )

    def test_the_two_initial_probabilities_sum_to_one(self):
        model = two_sentence_model()

        assert sum(
            math.exp(model.initial_log_probability(tag)) for tag in INITIAL_TAGS
        ) == pytest.approx(1.0)

    @pytest.mark.parametrize("tag", [BoundaryTag.MIDDLE, BoundaryTag.END])
    def test_a_run_cannot_start_mid_word(self, tag):
        assert two_sentence_model().initial_log_probability(tag) == -math.inf

    @pytest.mark.parametrize(
        ("previous", "following", "expected"),
        [
            (previous, following, expected)
            for (previous, following), expected in HAND_TRANSITIONS.items()
        ],
    )
    def test_transitions_are_the_smoothed_ratios(self, previous, following, expected):
        model = two_sentence_model()

        assert model.transition_log_probability(
            BoundaryTag(previous), BoundaryTag(following)
        ) == pytest.approx(math.log(expected))

    @pytest.mark.parametrize(
        ("previous", "following"),
        [
            (BoundaryTag.BEGIN, BoundaryTag.BEGIN),
            (BoundaryTag.BEGIN, BoundaryTag.SINGLE),
            (BoundaryTag.MIDDLE, BoundaryTag.SINGLE),
            (BoundaryTag.END, BoundaryTag.MIDDLE),
            (BoundaryTag.END, BoundaryTag.END),
            (BoundaryTag.SINGLE, BoundaryTag.END),
        ],
    )
    def test_inadmissible_transitions_are_impossible_not_smoothed(
        self, previous, following
    ):
        assert two_sentence_model().transition_log_probability(previous, following) == (
            -math.inf
        )

    @pytest.mark.parametrize(
        ("tag", "character", "expected"),
        [
            (tag, character, expected)
            for (tag, character), expected in HAND_EMISSIONS.items()
        ],
    )
    def test_emissions_are_smoothed_over_the_alphabet_plus_one_slot(
        self, tag, character, expected
    ):
        model = two_sentence_model()

        assert model.emission_log_probability(BoundaryTag(tag), character) == (
            pytest.approx(math.log(expected))
        )

    def test_a_character_of_the_alphabet_never_seen_under_a_tag_gets_the_floor(self):
        """b is in the alphabet and z is not; under B neither was emitted."""
        model = two_sentence_model()

        assert model.emission_log_probability(
            BoundaryTag.BEGIN, "b"
        ) == model.emission_log_probability(BoundaryTag.BEGIN, "z")

    def test_emissions_over_the_alphabet_and_the_unseen_slot_sum_to_one(self):
        model = two_sentence_model()

        for tag in BoundaryTag:
            total = sum(
                math.exp(model.emission_log_probability(tag, character))
                for character in model.alphabet
            ) + math.exp(model.emission_log_probability(tag, "☃"))
            assert total == pytest.approx(1.0)

    def test_the_alphabet_is_every_character_in_codepoint_order(self):
        model = two_sentence_model()

        assert model.alphabet == ("a", "b", "c")
        assert model.n_characters == 3

    def test_smaller_smoothing_trusts_the_counts_more(self):
        """Initial B was seen 2 of 2 times: 3/4 at smoothing 1, 21/22 at 0.1."""
        model = HiddenMarkovSegmenter(smoothing=0.1).fit(TWO_SENTENCE_CORPUS)

        assert model.initial_log_probability(BoundaryTag.BEGIN) == pytest.approx(
            math.log(2.1 / 2.2)
        )


class TestSegmentation:
    def test_recovers_the_segmentation_it_was_fit_on(self):
        model = HiddenMarkovSegmenter().fit(CJK_SEGMENTED_CORPUS)

        for sentence in CJK_SEGMENTED_CORPUS:
            assert model.split("".join(sentence)).texts == tuple(sentence)

    def test_reads_the_research_sentence_as_the_corpus_taught(self):
        model = HiddenMarkovSegmenter().fit(CJK_SEGMENTED_CORPUS)

        assert model.split(RESEARCH_SENTENCE).texts == LIFE_READING

    def test_a_corpus_that_contradicts_itself_is_resolved_by_the_majority(self):
        """研究 is a word three times and 研究生 once; smoothing decides which wins.

        At the default the model does not memorise the minority sentence, and
        that is the smoothing doing its job rather than a failure to fit. At
        0.5 the majority still wins; at 0.1 the counts are trusted enough that
        the minority sentence comes back as written.
        """
        for smoothing in (1.0, 0.5):
            model = HiddenMarkovSegmenter(smoothing=smoothing).fit(
                CONTRADICTORY_CJK_CORPUS
            )
            assert model.split(MINORITY_SENTENCE).texts == MAJORITY_READING

        memorising = HiddenMarkovSegmenter(smoothing=0.1).fit(CONTRADICTORY_CJK_CORPUS)

        assert memorising.split(MINORITY_SENTENCE).texts == MINORITY_READING

    def test_segments_words_it_never_saw_from_their_characters_behaviour(self):
        """a, c, e only ever began a word and b, d, f only ever ended one."""
        model = HiddenMarkovSegmenter().fit(BOUNDARY_BEHAVIOUR_CORPUS)

        assert model.split(NOVEL_SENTENCE).texts == NOVEL_SEGMENTATION

    def test_a_single_character_run_is_a_single(self):
        model = two_sentence_model()

        assert model.split("a").texts == ("a",)
        assert model.tags_of("a") == (BoundaryTag.SINGLE,)

    def test_the_last_tag_is_always_e_or_s(self):
        model = two_sentence_model()

        for text in ["a", "ab", "abc", "zzzz", "cba"]:
            assert model.tags_of(text)[-1] in FINAL_TAGS

    def test_every_decoded_sequence_is_admissible(self):
        model = HiddenMarkovSegmenter().fit(BOUNDARY_BEHAVIOUR_CORPUS)

        for text in ["adcfeb", "abcdefabcdef", "zzz", "aaaa", "bbbb"]:
            tags = model.tags_of(text)
            assert tags[0] in INITIAL_TAGS
            for previous, following in zip(tags, tags[1:], strict=False):
                assert following in ADMISSIBLE_SUCCESSORS[previous]

    def test_whitespace_ends_a_run_and_offsets_are_true(self):
        model = HiddenMarkovSegmenter().fit(CJK_SEGMENTED_CORPUS)

        assert list(model.split("研究 生命起源")) == [
            Word("研究", 0, 2),
            Word("生命", 3, 5),
            Word("起源", 5, 7),
        ]

    def test_a_blank_text_has_no_words_and_no_tags(self):
        model = two_sentence_model()

        assert model.split("  ").n_words == 0
        assert model.tags_of("  ") == ()

    def test_unseen_characters_are_segmented_rather_than_refused(self):
        model = two_sentence_model()

        assert "".join(model.split("xyz").texts) == "xyz"


class TestAgreementBetweenTagsAndWords:
    @pytest.mark.parametrize(
        "text",
        [
            "abc",
            "ab c",
            "abcabc",
            "adcfeb",
            "研究生命起源",
            "研究生命 起源 很好",
            "a",
            "zzzz",
        ],
    )
    def test_split_is_exactly_a_cut_after_every_e_and_s(self, text):
        model = HiddenMarkovSegmenter().fit(
            [*TWO_SENTENCE_CORPUS, *BOUNDARY_BEHAVIOUR_CORPUS, *CJK_SEGMENTED_CORPUS]
        )

        assert model.split(text).texts == cut_at_boundaries(text, model.tags_of(text))

    def test_tags_after_a_space_restart_the_chain(self):
        model = HiddenMarkovSegmenter().fit(CJK_SEGMENTED_CORPUS)

        tags = model.tags_of("研究 生命")

        assert tags[2] in INITIAL_TAGS


class TestFit:
    def test_returns_self(self):
        model = HiddenMarkovSegmenter()

        assert model.fit(TWO_SENTENCE_CORPUS) is model
        assert model.is_fitted

    def test_a_single_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            HiddenMarkovSegmenter().fit("ab c")  # type: ignore[arg-type]

    def test_a_sentence_that_is_one_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            HiddenMarkovSegmenter().fit(["ab c"])  # type: ignore[list-item]

    def test_no_sentences_is_refused(self):
        with pytest.raises(EmptyValuesError):
            HiddenMarkovSegmenter().fit([])

    def test_an_empty_sentence_is_refused(self):
        with pytest.raises(EmptyValuesError):
            HiddenMarkovSegmenter().fit([["ab"], []])

    def test_an_empty_word_is_refused(self):
        with pytest.raises(EmptyValuesError):
            HiddenMarkovSegmenter().fit([["ab", ""]])

    def test_a_word_with_whitespace_is_refused(self):
        with pytest.raises(InvalidValuesError):
            HiddenMarkovSegmenter().fit([["a b"]])

    def test_a_failed_fit_leaves_the_model_unfitted(self):
        model = HiddenMarkovSegmenter()

        with pytest.raises(EmptyValuesError):
            model.fit([])
        assert not model.is_fitted


class TestNotFitted:
    def test_everything_learned_is_guarded(self):
        model = HiddenMarkovSegmenter()

        with pytest.raises(NotFittedError):
            model.split("ab")
        with pytest.raises(NotFittedError):
            model.tags_of("ab")
        with pytest.raises(NotFittedError):
            _ = model.alphabet
        with pytest.raises(NotFittedError):
            _ = model.n_characters
        with pytest.raises(NotFittedError):
            model.initial_log_probability(BoundaryTag.BEGIN)
        with pytest.raises(NotFittedError):
            model.transition_log_probability(BoundaryTag.BEGIN, BoundaryTag.END)
        with pytest.raises(NotFittedError):
            model.emission_log_probability(BoundaryTag.BEGIN, "a")


class TestConstruction:
    @pytest.mark.parametrize("smoothing", [0.0, -1.0])
    def test_non_positive_smoothing_is_refused(self, smoothing):
        with pytest.raises(ValidationError):
            HiddenMarkovSegmenter(smoothing=smoothing)

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            HiddenMarkovSegmenter(alpha=1.0)  # type: ignore[call-arg]

    def test_the_default_smoothing_is_one(self):
        assert HiddenMarkovSegmenter().smoothing == 1.0
