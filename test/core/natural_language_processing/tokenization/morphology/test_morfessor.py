"""Spec for MorfessorBaseline -- morphs chosen to shorten a two-part code.

The two-word worked example in the module docstring is pinned here against the
hand formulas, and the twelve-word inflection corpus is pinned at the repeat
count where the search succeeds and at the one where it is measurably stuck.
"""

import math

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
)
from oop_ml.core.natural_language_processing.tokenization.encoding import Encoding
from oop_ml.core.natural_language_processing.tokenization.morphology.morfessor import (
    DescriptionLength,
    MorfessorBaseline,
    Morph,
    Morphs,
    MorphSegmentation,
    Segmentations,
    WordSegmentation,
    description_length,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.network.purpose import PassPurpose
from test.core.natural_language_processing.tokenization.morphology.fixtures import (
    INFLECTION_CORPUS,
    INFLECTION_WORDS,
    UNRELATED_CORPUS,
    UNRELATED_WORDS,
    inflection_corpus,
)

# The lexicon the search finds on the inflection corpus at three repeats: each
# stem in four words, each suffix in three.
SIX_MORPHS_AT_THREE = Morphs(
    [
        Morph("walk", 12),
        Morph("talk", 12),
        Morph("play", 12),
        Morph("s", 9),
        Morph("ed", 9),
        Morph("ing", 9),
    ]
)
SIX_MORPHS_AT_FIVE = Morphs(
    [
        Morph("walk", 20),
        Morph("talk", 20),
        Morph("play", 20),
        Morph("s", 15),
        Morph("ed", 15),
        Morph("ing", 15),
    ]
)

# 36 stem tokens and 27 suffix tokens.
TOKENS_AT_THREE = 63


def fit_inflections(**keywords: object) -> MorfessorBaseline:
    return MorfessorBaseline(random_seed=0, **keywords).fit(INFLECTION_CORPUS)  # type: ignore[arg-type]


def whole_words(words: list[str], repeats: int) -> Morphs:
    return Morphs([Morph(word, repeats) for word in words])


class TestDescriptionLength:
    """The module docstring's two-word example, against the hand formulas."""

    def test_two_words_left_whole(self):
        cost = description_length(Morphs([Morph("walk", 1), Morph("walks", 1)]))

        assert cost.corpus_cost == pytest.approx(2 * math.log(2), rel=1e-12)
        assert cost.spelling_cost == pytest.approx(
            11 * math.log(11) - 8 * math.log(2) - 2 * math.log(2), rel=1e-12
        )
        assert cost.count_cost == pytest.approx(0.0, abs=1e-12)
        assert cost.total == pytest.approx(20.8317, abs=5e-5)

    def test_two_words_split_into_a_shared_stem_and_a_suffix(self):
        cost = description_length(Morphs([Morph("walk", 2), Morph("s", 1)]))

        assert cost.corpus_cost == pytest.approx(
            3 * math.log(3) - 2 * math.log(2), rel=1e-12
        )
        assert cost.spelling_cost == pytest.approx(
            7 * math.log(7) - 2 * math.log(2), rel=1e-12
        )
        assert cost.count_cost == pytest.approx(math.log(2), rel=1e-12)
        assert cost.total == pytest.approx(14.8378, abs=5e-5)

    def test_the_split_is_shorter_by_the_docstrings_figure(self):
        whole = description_length(Morphs([Morph("walk", 1), Morph("walks", 1)]))
        split = description_length(Morphs([Morph("walk", 2), Morph("s", 1)]))

        assert whole.total - split.total == pytest.approx(5.9939, abs=5e-5)

    def test_lexicon_cost_is_spelling_plus_counts(self):
        cost = DescriptionLength(1.5, 0.25, 4.0, 1.0)

        assert cost.lexicon_cost == pytest.approx(1.75)
        assert cost.total == pytest.approx(5.75)

    def test_the_weight_scales_only_the_corpus_part(self):
        morphs = Morphs([Morph("walk", 2), Morph("s", 1)])
        plain = description_length(morphs, 1.0)
        doubled = description_length(morphs, 2.0)

        assert doubled.lexicon_cost == plain.lexicon_cost
        assert doubled.corpus_cost == plain.corpus_cost
        assert doubled.total == pytest.approx(
            plain.lexicon_cost + 2 * plain.corpus_cost
        )

    @pytest.mark.parametrize(
        "morphs",
        [Morphs([Morph("walk", 7)]), Morphs([Morph("a", 1), Morph("b", 1)])],
    )
    def test_one_type_or_one_token_per_type_costs_nothing_to_count(self, morphs):
        """C(N - 1, 0) and C(N - 1, N - 1) are both one way."""
        assert description_length(morphs).count_cost == pytest.approx(0.0, abs=1e-12)

    def test_a_single_type_used_once_costs_only_its_spelling(self):
        cost = description_length(Morphs([Morph("ab", 1)]))

        assert cost.corpus_cost == pytest.approx(0.0, abs=1e-12)
        assert cost.spelling_cost == pytest.approx(3 * math.log(3), rel=1e-12)

    @pytest.mark.parametrize("weight", [0.0, -1.0, math.inf, math.nan])
    def test_a_non_positive_weight_is_refused(self, weight):
        with pytest.raises(InvalidValuesError):
            description_length(Morphs([Morph("a", 1)]), weight)
        with pytest.raises(InvalidValuesError):
            DescriptionLength(1.0, 1.0, 1.0, weight)

    def test_a_negative_or_non_finite_cost_is_refused(self):
        with pytest.raises(InvalidValuesError):
            DescriptionLength(-0.1, 0.0, 0.0, 1.0)
        with pytest.raises(InvalidValuesError):
            DescriptionLength(0.0, math.nan, 0.0, 1.0)

    def test_equality_is_on_all_four_figures(self):
        assert DescriptionLength(1.0, 2.0, 3.0, 1.0) == DescriptionLength(
            1.0, 2.0, 3.0, 1.0
        )
        assert DescriptionLength(1.0, 2.0, 3.0, 1.0) != DescriptionLength(
            1.0, 2.0, 3.0, 2.0
        )


class TestMorphs:
    def test_sorted_by_spelling_whatever_the_order_given(self):
        morphs = Morphs([Morph("s", 3), Morph("ed", 3), Morph("walk", 4)])

        assert morphs.texts == ("ed", "s", "walk")
        assert [morph.text for morph in morphs] == ["ed", "s", "walk"]
        assert morphs == Morphs([Morph("walk", 4), Morph("s", 3), Morph("ed", 3)])

    def test_counts_are_addressable_by_spelling(self):
        morphs = Morphs([Morph("walk", 4), Morph("s", 3)])

        assert morphs.count_of("walk") == 4
        assert morphs["s"] == 3
        assert morphs.count_of("ing") == 0
        assert "walk" in morphs
        assert "ing" not in morphs

    def test_total_is_over_tokens_and_length_over_types(self):
        morphs = Morphs([Morph("walk", 4), Morph("s", 3)])

        assert morphs.total == 7
        assert morphs.n_morphs == 2
        assert len(morphs) == 2

    def test_needs_at_least_one_morph(self):
        with pytest.raises(EmptyValuesError):
            Morphs([])

    def test_a_spelling_listed_twice_is_refused(self):
        with pytest.raises(InvalidValuesError):
            Morphs([Morph("walk", 1), Morph("walk", 2)])

    def test_a_morph_needs_a_spelling_and_a_use(self):
        with pytest.raises(EmptyValuesError):
            Morph("", 1)
        with pytest.raises(InvalidValuesError):
            Morph("walk", 0)


class TestSegmentations:
    def test_morphs_must_spell_the_word(self):
        with pytest.raises(InvalidValuesError):
            WordSegmentation("walked", 1, ["walk", "s"])

    def test_a_word_cut_into_nothing_is_refused(self):
        with pytest.raises(EmptyValuesError):
            WordSegmentation("walked", 1, [])

    def test_a_count_below_one_is_refused(self):
        with pytest.raises(InvalidValuesError):
            WordSegmentation("walked", 0, ["walk", "ed"])

    def test_recounts_morph_tokens_as_occurrences_times_word_count(self):
        segmentations = Segmentations(
            [
                WordSegmentation("walked", 3, ["walk", "ed"]),
                WordSegmentation("walks", 2, ["walk", "s"]),
                WordSegmentation("ss", 1, ["s", "s"]),
            ]
        )

        assert segmentations.morphs == Morphs(
            [Morph("walk", 5), Morph("ed", 3), Morph("s", 4)]
        )

    def test_addressable_by_word(self):
        segmentation = WordSegmentation("walked", 3, ["walk", "ed"])
        segmentations = Segmentations([segmentation])

        assert segmentations["walked"] == segmentation
        assert "walked" in segmentations
        assert "talked" not in segmentations
        assert segmentations.n_words == 1
        assert list(segmentations) == [segmentation]

    def test_a_word_the_fit_never_saw_is_refused(self):
        segmentations = Segmentations([WordSegmentation("walked", 3, ["walk", "ed"])])

        with pytest.raises(InvalidValuesError):
            _ = segmentations["talked"]

    def test_a_word_segmented_twice_is_refused(self):
        with pytest.raises(InvalidValuesError):
            Segmentations(
                [
                    WordSegmentation("walked", 3, ["walk", "ed"]),
                    WordSegmentation("walked", 3, ["walked"]),
                ]
            )

    def test_needs_at_least_one_word(self):
        with pytest.raises(EmptyValuesError):
            Segmentations([])


class TestFit:
    def test_finds_the_three_stems_and_the_three_suffixes(self):
        assert fit_inflections().morphs == SIX_MORPHS_AT_THREE

    def test_walked_and_talked_share_the_past_morph(self):
        model = fit_inflections()

        assert model.segmentations["walked"].morphs == ("walk", "ed")
        assert model.segmentations["talked"].morphs == ("talk", "ed")
        assert model.best_segmentation("walked").morphs == ("walk", "ed")
        assert model.best_segmentation("talked").morphs == ("talk", "ed")

    def test_every_training_word_is_stem_then_suffix_or_a_bare_stem(self):
        model = fit_inflections()

        assert model.segmentations.n_words == 12
        assert model.segmentations["walk"].morphs == ("walk",)
        assert model.segmentations["playing"].morphs == ("play", "ing")
        assert model.segmentations["walks"].count == 3

    def test_converges_on_the_second_epoch(self):
        model = fit_inflections()

        assert model.epochs_run == 2
        assert model.converged

    def test_the_cost_is_that_of_the_six_morph_lexicon(self):
        model = fit_inflections()

        assert model.cost == pytest.approx(185.4634, abs=5e-5)
        assert model.cost == pytest.approx(
            description_length(SIX_MORPHS_AT_THREE).total, rel=1e-12
        )

    def test_the_search_shortened_the_code_from_the_unsplit_start(self):
        start = description_length(whole_words(INFLECTION_WORDS, 3)).total

        assert start == pytest.approx(301.0812, abs=5e-5)
        assert fit_inflections().cost < start

    @pytest.mark.parametrize("seed", range(8))
    def test_every_seed_reaches_the_same_lexicon(self, seed):
        model = MorfessorBaseline(random_seed=seed).fit(INFLECTION_CORPUS)

        assert model.morphs == SIX_MORPHS_AT_THREE

    def test_one_epoch_already_reaches_it_but_cannot_know_it_has(self):
        model = fit_inflections(max_epochs=1)

        assert model.epochs_run == 1
        assert not model.converged
        assert model.morphs == SIX_MORPHS_AT_THREE

    def test_unrelated_words_are_left_whole(self):
        model = MorfessorBaseline(random_seed=0).fit(UNRELATED_CORPUS)

        assert model.morphs == whole_words(UNRELATED_WORDS, 3)
        assert model.epochs_run == 1
        assert model.converged
        assert all(segmentation.n_morphs == 1 for segmentation in model.segmentations)

    @pytest.mark.parametrize(
        ("corpus", "weight"),
        [
            (INFLECTION_CORPUS, 1.0),
            (INFLECTION_CORPUS, 0.5),
            (inflection_corpus(5), 1.0),
            (inflection_corpus(5), 0.5),
            (UNRELATED_CORPUS, 1.0),
        ],
    )
    def test_the_accounts_agree_with_the_segmentations_recounted(self, corpus, weight):
        """Two routes to one lexicon: the dictionaries adjusted through every
        add and remove of the search, and the morphs counted afresh from the
        final per-word segmentations. Equal, and their costs equal."""
        model = MorfessorBaseline(random_seed=0, corpus_weight=weight).fit(corpus)

        assert model.segmentations.morphs == model.morphs
        assert model.cost == pytest.approx(
            description_length(model.segmentations.morphs, weight).total, rel=1e-12
        )

    def test_the_same_seed_learns_the_same_lexicon(self):
        first = MorfessorBaseline(random_seed=3).fit(INFLECTION_CORPUS)
        second = MorfessorBaseline(random_seed=3).fit(INFLECTION_CORPUS)

        assert first.morphs == second.morphs
        assert first.cost == second.cost
        assert list(first.segmentations) == list(second.segmentations)
        assert first.vocabulary == second.vocabulary

    def test_fit_returns_self(self):
        model = MorfessorBaseline(random_seed=0)

        assert model.fit(INFLECTION_CORPUS) is model

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            MorfessorBaseline().fit("walk walks")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            MorfessorBaseline().fit(["  ", ""])


class TestWhereTheGreedySearchIsStuck:
    """Five repeats instead of three, and the search cannot take its first step."""

    def test_every_word_is_left_whole(self):
        model = MorfessorBaseline(random_seed=0).fit(inflection_corpus(5))

        assert model.morphs == whole_words(INFLECTION_WORDS, 5)
        assert model.cost == pytest.approx(367.2274, abs=5e-5)
        assert model.epochs_run == 1
        assert model.converged

    def test_although_the_six_morph_lexicon_is_over_a_hundred_nats_shorter(self):
        stuck = MorfessorBaseline(random_seed=0).fit(inflection_corpus(5))
        six = description_length(SIX_MORPHS_AT_FIVE).total

        assert six == pytest.approx(262.9413, abs=5e-5)
        assert stuck.cost - six > 100

    def test_because_the_cheapest_first_split_is_uphill(self):
        """``plays`` into ``play`` and ``s`` is the best first move at five
        repeats and costs +1.7333; at three the same move is -2.4630."""
        for repeats, expected_delta in ((5, 1.7333), (3, -2.4630)):
            counts = {word: repeats for word in INFLECTION_WORDS}
            del counts["plays"]
            counts["play"] += repeats
            counts["s"] = repeats
            whole = description_length(whole_words(INFLECTION_WORDS, repeats)).total
            after = description_length(
                Morphs([Morph(text, count) for text, count in counts.items()])
            ).total

            assert after - whole == pytest.approx(expected_delta, abs=5e-5)

    def test_a_lower_corpus_weight_lets_the_search_through(self):
        model = MorfessorBaseline(random_seed=0, corpus_weight=0.5).fit(
            inflection_corpus(5)
        )

        assert model.morphs == SIX_MORPHS_AT_FIVE
        assert model.cost == pytest.approx(169.4115, abs=5e-5)

    def test_a_higher_corpus_weight_stops_the_three_repeat_search_too(self):
        model = fit_inflections(corpus_weight=2.0)

        assert model.morphs == whole_words(INFLECTION_WORDS, 3)
        assert model.cost == pytest.approx(390.5378, abs=5e-5)


class TestVocabulary:
    def test_is_unknown_then_every_morph_plain_then_every_morph_marked(self):
        vocabulary = fit_inflections().vocabulary

        assert list(vocabulary) == [
            "[UNK]",
            "ed",
            "ing",
            "play",
            "s",
            "talk",
            "walk",
            "ed</w>",
            "ing</w>",
            "play</w>",
            "s</w>",
            "talk</w>",
            "walk</w>",
        ]
        assert vocabulary.unknown_token == "[UNK]"
        assert isinstance(vocabulary, Vocabulary)

    def test_a_custom_marker_is_honoured(self):
        model = fit_inflections(end_of_word_marker="_")

        assert "walk_" in model.vocabulary
        assert model.encode("walked").texts == ("walk", "ed_")
        assert model.decode(model.encode("walked talks").ids) == "walked talks"


class TestBestSegmentation:
    @pytest.mark.parametrize(
        ("word", "morphs", "cost"),
        [
            ("walked", ("walk", "ed"), math.log(63 / 12) + math.log(63 / 9)),
            (
                "walkings",
                ("walk", "ing", "s"),
                math.log(63 / 12) + 2 * math.log(63 / 9),
            ),
            ("s", ("s",), math.log(63 / 9)),
            ("splay", ("s", "play"), math.log(63 / 9) + math.log(63 / 12)),
            (
                "jumping",
                ("j", "u", "m", "p", "ing"),
                4 * (7 * math.log(63) + 1) + math.log(63 / 9),
            ),
            (
                "walkxed",
                ("walk", "x", "ed"),
                math.log(63 / 12) + (7 * math.log(63) + 1) + math.log(63 / 9),
            ),
            ("zzz", ("z", "z", "z"), 3 * (3 * math.log(63) + 1)),
        ],
    )
    def test_the_cheapest_cut_and_its_cost_by_hand(self, word, morphs, cost):
        model = fit_inflections()
        segmentation = model.best_segmentation(word)

        assert model.morphs.total == TOKENS_AT_THREE
        assert segmentation.morphs == morphs
        assert segmentation.cost == pytest.approx(cost, rel=1e-12)
        assert segmentation.word == word

    @pytest.mark.parametrize(
        "word",
        ["walked", "walkings", "s", "splay", "jumping", "walkxed", "zzz", "playing"],
    )
    def test_the_viterbi_total_agrees_with_the_piecewise_sum(self, word):
        model = fit_inflections()
        segmentation = model.best_segmentation(word)

        assert model.cost_of_segmentation(segmentation.morphs) == pytest.approx(
            segmentation.cost, rel=1e-12
        )

    def test_an_unknown_character_costs_more_than_any_all_morph_cut(self):
        """The penalty is ``len log N + 1``; a whole-word cut into known
        morphs costs at most ``len log N``."""
        model = fit_inflections()

        assert model.cost_of_segmentation(["x"]) > 1 * math.log(TOKENS_AT_THREE)
        assert model.cost_of_segmentation(["walk", "x"]) > 5 * math.log(TOKENS_AT_THREE)

    def test_the_viterbi_beats_every_other_cut_of_the_word(self):
        model = fit_inflections()
        best = model.best_segmentation("walkings")

        for first in range(1, 8):
            for second in range(first + 1, 8):
                pieces = [
                    "walkings"[:first],
                    "walkings"[first:second],
                    "walkings"[second:],
                ]
                if all(piece in model.morphs or len(piece) == 1 for piece in pieces):
                    assert model.cost_of_segmentation(pieces) >= best.cost

    def test_an_unknown_piece_longer_than_one_character_is_refused(self):
        with pytest.raises(InvalidValuesError):
            fit_inflections().cost_of_segmentation(["walk", "xy"])

    def test_an_empty_segmentation_is_refused(self):
        with pytest.raises(EmptyValuesError):
            fit_inflections().cost_of_segmentation([])
        with pytest.raises(EmptyValuesError):
            fit_inflections().cost_of_segmentation(["walk", ""])

    def test_an_empty_word_is_refused(self):
        with pytest.raises(EmptyValuesError):
            fit_inflections().best_segmentation("")

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            fit_inflections().best_segmentation(["walk"])  # type: ignore[arg-type]

    def test_a_segmentation_must_spell_its_word(self):
        with pytest.raises(InvalidValuesError):
            MorphSegmentation("walked", ["walk", "s"], 1.0)
        with pytest.raises(EmptyValuesError):
            MorphSegmentation("walked", [], 1.0)
        with pytest.raises(InvalidValuesError):
            MorphSegmentation("walked", ["walk", "ed"], -1.0)


class TestEncode:
    def test_a_training_word_is_stem_then_marked_suffix(self):
        assert fit_inflections().encode("walked").texts == ("walk", "ed</w>")

    def test_an_unseen_word_is_spelled_from_known_morphs(self):
        assert fit_inflections().encode("talkings").texts == ("talk", "ing", "s</w>")

    def test_the_unknown_token_appears_once_per_uncovered_character(self):
        assert fit_inflections().encode("jumping walkxed").texts == (
            "[UNK]",
            "[UNK]",
            "[UNK]",
            "[UNK]",
            "ing</w>",
            "walk",
            "[UNK]",
            "ed</w>",
        )

    def test_ids_are_vocabulary_positions(self):
        model = fit_inflections()
        encoding = model.encode("walkings")

        assert encoding.ids == model.vocabulary.ids_of(("walk", "ing", "s</w>"))

    def test_a_blank_text_encodes_to_nothing(self):
        assert fit_inflections().encode("   ") == Encoding([])

    def test_the_purpose_changes_nothing(self):
        model = fit_inflections()

        assert model.encode("walkings", PassPurpose.TRAINING) == model.encode(
            "walkings"
        )

    def test_before_fit_raises_not_fitted(self):
        model = MorfessorBaseline()

        with pytest.raises(NotFittedError):
            model.encode("walk")
        with pytest.raises(NotFittedError):
            _ = model.vocabulary
        with pytest.raises(NotFittedError):
            _ = model.morphs
        with pytest.raises(NotFittedError):
            _ = model.segmentations
        with pytest.raises(NotFittedError):
            _ = model.cost
        with pytest.raises(NotFittedError):
            _ = model.epochs_run
        with pytest.raises(NotFittedError):
            _ = model.converged
        with pytest.raises(NotFittedError):
            model.best_segmentation("walk")
        with pytest.raises(NotFittedError):
            model.cost_of_segmentation(["walk"])


class TestDecode:
    def test_round_trips_seen_and_unseen_words(self):
        model = fit_inflections()

        assert model.decode(model.encode("walked talking plays").ids) == (
            "walked talking plays"
        )
        assert model.decode(model.encode("talkings").ids) == "talkings"

    def test_an_unknown_last_character_loses_its_word_boundary(self):
        """Documented rather than hidden, as for byte pair encoding."""
        model = fit_inflections()

        assert model.decode(model.encode("walkx talk").ids) == "walk[UNK]talk"


class TestConstruction:
    @pytest.mark.parametrize(
        "keywords",
        [
            {"corpus_weight": 0.0},
            {"corpus_weight": -1.0},
            {"convergence_threshold": 0.0},
            {"max_epochs": 0},
            {"end_of_word_marker": ""},
            {"unknown_token": ""},
        ],
    )
    def test_out_of_range_hyperparameters_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            MorfessorBaseline(**keywords)

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            MorfessorBaseline(epochs=3)  # type: ignore[call-arg]

    def test_defaults_are_the_documented_ones(self):
        model = MorfessorBaseline()

        assert model.corpus_weight == 1.0
        assert model.convergence_threshold == 0.005
        assert model.max_epochs == 10
        assert model.random_seed is None
