"""Spec for PointwiseSegmenter -- one boundary decision per gap, by an averaged perceptron.

The one-gap and two-gap fits are worked by hand in the module docstring and
here: seven features at window one, so a single mistake moves the score to
exactly 7.0, and two gaps in one epoch pin the averaging arithmetic.
"""

import random

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
    TooFewValuesError,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.pointwise import (
    BIAS_FEATURE,
    CharacterType,
    PointwiseSegmenter,
    character_type_of,
    gap_features,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word
from test.core.natural_language_processing.tokenization.segmentation.fixtures import (
    CJK_POINTWISE_CORPUS,
    SEPARABLE_CORPUS,
    SEPARABLE_N_FEATURES_AT_WINDOW_ONE,
)

# The seven features of the only gap in "ab" at window one, in the order the
# function lists them: bias, then unigrams and types by offset, then bigrams.
AB_GAP_FEATURES = (
    "bias",
    "c[-1]=a",
    "t[-1]=letter",
    "c[0]=b",
    "t[0]=letter",
    "c[-1..0]=ab",
    "t[-1..0]=letter,letter",
)


def separable_model(random_seed: int = 0) -> PointwiseSegmenter:
    return PointwiseSegmenter(window=1, epochs=10, random_seed=random_seed).fit(
        SEPARABLE_CORPUS
    )


class TestCharacterType:
    @pytest.mark.parametrize(
        ("character", "expected"),
        [
            ("研", CharacterType.HAN),
            ("あ", CharacterType.HIRAGANA),
            ("ア", CharacterType.KATAKANA),
            ("ー", CharacterType.KATAKANA),
            ("a", CharacterType.LETTER),
            ("Z", CharacterType.LETTER),
            ("é", CharacterType.LETTER),
            ("7", CharacterType.DIGIT),
            ("３", CharacterType.DIGIT),
            (",", CharacterType.PUNCTUATION),
            ("。", CharacterType.PUNCTUATION),
            ("€", CharacterType.OTHER),
            (" ", CharacterType.OTHER),
        ],
    )
    def test_reads_the_type_from_the_unicode_name_and_category(
        self, character, expected
    ):
        assert character_type_of(character) is expected

    def test_the_type_is_a_string_enum_so_it_can_sit_in_a_feature(self):
        assert f"{character_type_of('研')}" == "han"


class TestGapFeatures:
    def test_the_only_gap_of_ab_has_exactly_seven_features_at_window_one(self):
        assert gap_features("ab", 0, 1) == AB_GAP_FEATURES

    def test_the_bias_is_always_present(self):
        assert BIAS_FEATURE in gap_features("研究生", 1, 3)

    def test_offsets_name_the_characters_around_the_gap(self):
        """The gap after 研 in 研究生: c[-1] is 研, c[0] is 究, c[1] is 生."""
        features = gap_features("研究生", 0, 2)

        assert "c[-1]=研" in features
        assert "c[0]=究" in features
        assert "c[1]=生" in features
        assert "c[0..1]=究生" in features
        assert "c[-1..0]=研究" in features
        assert "c[-2]=" not in "".join(features)

    def test_positions_off_the_run_contribute_nothing(self):
        at_start = gap_features("abcd", 0, 2)
        in_the_middle = gap_features("abcd", 1, 2)

        assert not any(feature.startswith("c[-2]") for feature in at_start)
        assert any(feature.startswith("c[-2]") for feature in in_the_middle)

    def test_a_wider_window_adds_features(self):
        assert len(gap_features("abcdef", 2, 1)) < len(gap_features("abcdef", 2, 3))

    def test_type_features_generalise_across_characters(self):
        assert "t[-1..0]=han,han" in gap_features("研究", 0, 1)
        assert "t[-1..0]=katakana,han" in gap_features("ア研", 0, 1)

    @pytest.mark.parametrize("gap", [-1, 1, 5])
    def test_a_gap_outside_the_run_is_refused(self, gap):
        with pytest.raises(InvalidValuesError):
            gap_features("ab", gap, 1)

    def test_a_window_below_one_is_refused(self):
        with pytest.raises(InvalidValuesError):
            gap_features("ab", 0, 0)


class TestHandWorkedFits:
    def test_one_gap_one_epoch_scores_seven(self):
        """Score zero is a mistake; +1 on seven features; the average of one step is itself."""
        model = PointwiseSegmenter(window=1, epochs=1).fit([["a", "b"]])

        assert model.n_features == 7
        assert model.gap_scores("ab") == (7.0,)
        assert model.n_updates_by_epoch == (1,)
        assert all(model.weight_of(feature) == 1.0 for feature in AB_GAP_FEATURES)

    def test_one_gap_labelled_no_boundary_scores_minus_seven(self):
        model = PointwiseSegmenter(window=1, epochs=1).fit([["ab"]])

        assert model.gap_scores("ab") == (-7.0,)
        assert model.split("ab").texts == ("ab",)

    def test_two_gaps_one_epoch_pin_the_averaging(self):
        """ab is a boundary, cd is not; the two gaps share exactly four features
        (bias and the three type features). Visiting ab first: step one scores
        zero and adds +1 to ab's seven; step two scores +4 on the shared four,
        a mistake, and adds -1 to cd's seven. The average of the two weight
        vectors is 1.0 on ab's own three features, 0.5 on the shared four and
        -0.5 on cd's own three, so ab scores 3 + 2 = 5.0 and cd scores
        2 - 1.5 = 0.5."""
        order = [0, 1]
        random.Random(0).shuffle(order)
        assert order == [0, 1], "seed 0 must visit the ab gap first for the arithmetic"

        model = PointwiseSegmenter(window=1, epochs=1, random_seed=0).fit(
            [["a", "b"], ["cd"]]
        )

        assert model.weight_of("c[-1]=a") == 1.0
        assert model.weight_of("c[-1..0]=ab") == 1.0
        assert model.weight_of(BIAS_FEATURE) == 0.5
        assert model.weight_of("t[0]=letter") == 0.5
        assert model.weight_of("c[0]=d") == -0.5
        assert model.gap_scores("ab") == (5.0,)
        assert model.gap_scores("cd") == (0.5,)
        assert model.n_updates_by_epoch == (2,)

    def test_a_feature_never_seen_has_weight_zero(self):
        model = PointwiseSegmenter(window=1, epochs=1).fit([["a", "b"]])

        assert model.weight_of("c[-1]=z") == 0.0


class TestFit:
    def test_separable_data_is_learned_to_zero_training_errors(self):
        model = separable_model()

        assert model.n_updates_by_epoch[-1] == 0
        for sentence in SEPARABLE_CORPUS:
            assert model.split("".join(sentence)).texts == tuple(sentence)

    def test_n_features_is_the_hand_counted_sixteen(self):
        assert separable_model().n_features == SEPARABLE_N_FEATURES_AT_WINDOW_ONE

    def test_the_seed_makes_two_fits_identical(self):
        first = PointwiseSegmenter(random_seed=7).fit(CJK_POINTWISE_CORPUS)
        second = PointwiseSegmenter(random_seed=7).fit(CJK_POINTWISE_CORPUS)

        assert first.n_updates_by_epoch == second.n_updates_by_epoch
        assert first.gap_scores("研究生命起源") == second.gap_scores("研究生命起源")

    def test_epochs_are_counted(self):
        assert (
            len(PointwiseSegmenter(epochs=3).fit([["a", "b"]]).n_updates_by_epoch) == 3
        )

    def test_returns_self(self):
        model = PointwiseSegmenter()

        assert model.fit([["a", "b"]]) is model
        assert model.is_fitted

    def test_a_corpus_with_no_gap_is_refused(self):
        with pytest.raises(TooFewValuesError):
            PointwiseSegmenter().fit([["a"], ["b"]])

    def test_a_single_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            PointwiseSegmenter().fit("ab cd")  # type: ignore[arg-type]

    def test_a_sentence_that_is_one_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            PointwiseSegmenter().fit(["ab cd"])  # type: ignore[list-item]

    def test_an_empty_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            PointwiseSegmenter().fit([])

    def test_a_failed_fit_leaves_the_model_unfitted(self):
        model = PointwiseSegmenter()

        with pytest.raises(TooFewValuesError):
            model.fit([["a"]])
        assert not model.is_fitted


class TestSegmentation:
    def test_a_held_out_sentence_of_seen_contexts_is_segmented(self):
        model = separable_model()

        assert model.split("cdab").texts == ("cd", "ab")
        assert model.split("abcdabcd").texts == ("ab", "cd", "ab", "cd")

    def test_a_cjk_gap_never_seen_as_a_pair_is_decided_by_its_unigrams(self):
        """命起 never occurs in the corpus, but 命 ends words and 起 begins them."""
        model = PointwiseSegmenter(random_seed=1).fit(CJK_POINTWISE_CORPUS)

        assert model.n_updates_by_epoch[-1] == 0
        assert model.split("生命起源").texts == ("生命", "起源")

    def test_whitespace_ends_a_run_and_offsets_are_true(self):
        model = separable_model()

        assert list(model.split("ab cdab")) == [
            Word("ab", 0, 2),
            Word("cd", 3, 5),
            Word("ab", 5, 7),
        ]

    def test_a_single_character_run_has_no_gap_and_is_one_word(self):
        model = separable_model()

        assert model.split("a").texts == ("a",)
        assert model.gap_scores("a") == ()

    def test_a_blank_text_has_no_words(self):
        assert separable_model().split("  ").n_words == 0


class TestAgreementBetweenScoresAndWords:
    @pytest.mark.parametrize(
        "text", ["abcd", "cdab", "ab cdab", "abcdabcdab", "a", "zzzz"]
    )
    def test_split_cuts_exactly_at_the_positive_scores(self, text):
        model = separable_model()

        scores = iter(model.gap_scores(text))
        expected: list[str] = []
        for run in text.split():
            current = run[0]
            for character in run[1:]:
                if next(scores) > 0:
                    expected.append(current)
                    current = ""
                current += character
            expected.append(current)

        assert model.split(text).texts == tuple(expected)

    def test_one_score_per_gap_within_runs_and_none_across_whitespace(self):
        model = separable_model()

        assert len(model.gap_scores("abcd")) == 3
        assert len(model.gap_scores("ab cd")) == 2
        assert len(model.gap_scores("a b c")) == 0


class TestNotFitted:
    def test_everything_learned_is_guarded(self):
        model = PointwiseSegmenter()

        with pytest.raises(NotFittedError):
            model.split("ab")
        with pytest.raises(NotFittedError):
            model.gap_scores("ab")
        with pytest.raises(NotFittedError):
            _ = model.n_features
        with pytest.raises(NotFittedError):
            _ = model.n_updates_by_epoch
        with pytest.raises(NotFittedError):
            model.weight_of(BIAS_FEATURE)


class TestConstruction:
    @pytest.mark.parametrize(
        "keywords",
        [{"window": 0}, {"epochs": 0}, {"window": -1}],
    )
    def test_out_of_range_hyperparameters_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            PointwiseSegmenter(**keywords)

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            PointwiseSegmenter(iterations=5)  # type: ignore[call-arg]

    def test_the_defaults(self):
        model = PointwiseSegmenter()

        assert model.window == 3
        assert model.epochs == 10
        assert model.random_seed is None
