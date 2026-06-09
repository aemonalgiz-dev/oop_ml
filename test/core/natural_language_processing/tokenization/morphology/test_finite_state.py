"""Spec for FiniteStateAnalyzer -- lexicons walked as a transducer.

The toy English morphology lives in ``fixtures.py`` in two versions, the naive
one that fails ``baked`` and the repaired one with an allomorph, and both
halves of that story are pinned.
"""

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
)
from oop_ml.core.natural_language_processing.tokenization.morphology.constrained_merges import (
    MorphemeConstrainedBytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.morphology.finite_state import (
    END,
    Analyses,
    Analysis,
    FiniteStateAnalyzer,
    Lexicon,
    LexiconEntry,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.words import Word, Words
from test.core.natural_language_processing.tokenization.morphology.fixtures import (
    NAIVE_ENGLISH,
    ONES_AND_TWOS,
    REPAIRED_ENGLISH,
    VERB_SUFFIXES,
)

WALK_VERB = LexiconEntry("walk", "walk+V", "VerbSuffix")
PAST = LexiconEntry("ed", "+PAST", END)


class TestLexiconEntry:
    def test_carries_surface_analysis_and_continuation(self):
        assert WALK_VERB.surface == "walk"
        assert WALK_VERB.analysis == "walk+V"
        assert WALK_VERB.continuation == "VerbSuffix"
        assert not WALK_VERB.leads_to_end
        assert PAST.leads_to_end

    def test_an_empty_surface_is_refused(self):
        """A zero morpheme is written as a second entry continuing to END."""
        with pytest.raises(EmptyValuesError):
            LexiconEntry("", "+PRES", END)

    def test_an_empty_continuation_is_refused(self):
        with pytest.raises(EmptyValuesError):
            LexiconEntry("walk", "walk+V", "")

    def test_an_empty_analysis_is_allowed(self):
        assert LexiconEntry("o", "", "Root").analysis == ""

    def test_a_non_string_analysis_is_refused(self):
        with pytest.raises(InvalidValuesError):
            LexiconEntry("walk", None, END)  # type: ignore[arg-type]

    def test_equality_is_on_all_three(self):
        assert LexiconEntry("walk", "walk+V", "VerbSuffix") == WALK_VERB
        assert LexiconEntry("walk", "walk+N", "VerbSuffix") != WALK_VERB
        assert LexiconEntry("walk", "walk+V", END) != WALK_VERB


class TestLexicon:
    def test_names_its_entries_and_knows_where_they_lead(self):
        assert VERB_SUFFIXES.name == "VerbSuffix"
        assert VERB_SUFFIXES.n_entries == 3
        assert len(VERB_SUFFIXES) == 3
        assert [entry.surface for entry in VERB_SUFFIXES] == ["s", "ed", "ing"]
        assert VERB_SUFFIXES.continuations == frozenset({END})

    def test_the_same_surface_with_different_analyses_is_ambiguity_not_a_repeat(self):
        lexicon = Lexicon(
            "Root",
            [LexiconEntry("walk", "walk+V", END), LexiconEntry("walk", "walk+N", END)],
        )

        assert lexicon.n_entries == 2

    def test_an_identical_entry_twice_is_refused(self):
        with pytest.raises(InvalidValuesError):
            Lexicon("Root", [WALK_VERB, WALK_VERB])

    def test_needs_a_name(self):
        with pytest.raises(EmptyValuesError):
            Lexicon("", [WALK_VERB])

    def test_cannot_be_named_end(self):
        with pytest.raises(InvalidValuesError):
            Lexicon(END, [WALK_VERB])

    def test_needs_an_entry(self):
        with pytest.raises(EmptyValuesError):
            Lexicon("Root", [])


class TestAnalysis:
    def test_reads_morphs_surface_and_tags_off_the_path(self):
        analysis = Analysis([WALK_VERB, PAST])

        assert analysis.morphs == ("walk", "ed")
        assert analysis.surface == "walked"
        assert analysis.tags == "walk+V +PAST"
        assert analysis.n_morphemes == 2
        assert list(analysis) == [WALK_VERB, PAST]

    def test_an_empty_analysis_leaves_no_gap_in_the_tags(self):
        analysis = Analysis([WALK_VERB, LexiconEntry("o", "", "Link"), PAST])

        assert analysis.tags == "walk+V +PAST"
        assert analysis.morphs == ("walk", "o", "ed")

    def test_needs_an_entry(self):
        with pytest.raises(EmptyValuesError):
            Analysis([])


class TestAnalyses:
    def test_orders_by_morpheme_count_then_tags_then_morphs(self):
        one = Analysis([LexiconEntry("walks", "walks+N", END)])
        two_noun = Analysis(
            [LexiconEntry("walk", "walk+N", "N"), LexiconEntry("s", "+PL", END)]
        )
        two_verb = Analysis(
            [LexiconEntry("walk", "walk+V", "V"), LexiconEntry("s", "+3SG", END)]
        )

        analyses = Analyses("walks", [two_verb, one, two_noun])

        assert list(analyses) == [one, two_noun, two_verb]
        assert analyses.shortest == one
        assert analyses[2] == two_verb
        assert analyses.n_analyses == 3

    def test_no_analyses_is_a_legitimate_answer(self):
        analyses = Analyses("xyz", [])

        assert not analyses.is_analysable
        assert analyses.n_analyses == 0
        assert len(analyses) == 0

    def test_shortest_of_nothing_is_refused(self):
        with pytest.raises(EmptyValuesError):
            _ = Analyses("xyz", []).shortest

    def test_an_analysis_must_spell_the_word(self):
        with pytest.raises(InvalidValuesError):
            Analyses("walks", [Analysis([WALK_VERB, PAST])])

    def test_needs_a_word(self):
        with pytest.raises(EmptyValuesError):
            Analyses("", [])


class TestConstruction:
    def test_the_root_must_be_a_lexicon(self):
        with pytest.raises(ValidationError):
            FiniteStateAnalyzer(root="Stems", lexicons=NAIVE_ENGLISH.lexicons)

    def test_two_lexicons_may_not_share_a_name(self):
        with pytest.raises(ValidationError):
            FiniteStateAnalyzer(
                root="Root",
                lexicons=(NAIVE_ENGLISH.lexicons[0], NAIVE_ENGLISH.lexicons[0]),
            )

    def test_every_continuation_must_name_a_lexicon_or_end(self):
        with pytest.raises(ValidationError):
            FiniteStateAnalyzer(
                root="Root",
                lexicons=(Lexicon("Root", [LexiconEntry("walk", "walk+V", "Suffix")]),),
            )

    def test_needs_a_lexicon(self):
        with pytest.raises(ValidationError):
            FiniteStateAnalyzer(root="Root", lexicons=())

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            FiniteStateAnalyzer(
                root="Root",
                lexicons=NAIVE_ENGLISH.lexicons,
                start="Root",  # type: ignore[call-arg]
            )

    def test_a_lexicon_must_be_a_lexicon(self):
        with pytest.raises(ValidationError):
            FiniteStateAnalyzer(root="Root", lexicons=("Root",))  # type: ignore[arg-type]

    def test_counts_its_states(self):
        assert NAIVE_ENGLISH.n_lexicons == 3
        assert REPAIRED_ENGLISH.n_lexicons == 5
        assert isinstance(NAIVE_ENGLISH, PreTokenizer)


class TestAnalyse:
    def test_walked_is_walk_plus_past(self):
        analyses = NAIVE_ENGLISH.analyse("walked")

        assert analyses.n_analyses == 1
        assert analyses.shortest.morphs == ("walk", "ed")
        assert analyses.shortest.tags == "walk+V +PAST"
        assert analyses.word == "walked"

    def test_walks_is_ambiguous_and_the_noun_wins_the_tie(self):
        """Both readings have two morphemes; ``walk+N +PL`` sorts before
        ``walk+V +3SG`` because ``N`` sorts before ``V``."""
        analyses = NAIVE_ENGLISH.analyse("walks")

        assert analyses.n_analyses == 2
        assert [analysis.tags for analysis in analyses] == ["walk+N +PL", "walk+V +3SG"]
        assert analyses.shortest.tags == "walk+N +PL"

    def test_a_bare_stem_reaches_the_end_through_its_own_entry(self):
        analyses = NAIVE_ENGLISH.analyse("walk")

        assert [analysis.tags for analysis in analyses] == ["walk+N", "walk+V"]
        assert all(analysis.n_morphemes == 1 for analysis in analyses)

    @pytest.mark.parametrize("word", ["walkeds", "xyz", "wal", "swalk", "ed"])
    def test_a_word_no_path_spells_has_no_analysis(self, word):
        assert not NAIVE_ENGLISH.analyse(word).is_analysable

    def test_an_empty_word_is_refused(self):
        with pytest.raises(EmptyValuesError):
            NAIVE_ENGLISH.analyse("")

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            NAIVE_ENGLISH.analyse(["walked"])  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("length", "n_analyses"),
        [(1, 1), (2, 2), (3, 3), (4, 5), (5, 8), (6, 13), (7, 21)],
    )
    def test_enumerates_every_path_and_the_count_is_fibonacci(self, length, n_analyses):
        assert ONES_AND_TWOS.analyse("a" * length).n_analyses == n_analyses

    def test_the_paths_are_the_compositions_into_ones_and_twos(self):
        assert sorted(analysis.tags for analysis in ONES_AND_TWOS.analyse("aaa")) == [
            "1 1 1",
            "1 2",
            "2 1",
        ]


class TestAllomorphy:
    def test_the_naive_lexicon_fails_baked_and_accepts_bakeed(self):
        assert not NAIVE_ENGLISH.analyse("baked").is_analysable
        assert NAIVE_ENGLISH.analyse("bakeed").shortest.tags == "bake+V +PAST"

    def test_the_allomorph_repairs_both(self):
        baked = REPAIRED_ENGLISH.analyse("baked")

        assert baked.n_analyses == 1
        assert baked.shortest.morphs == ("bak", "ed")
        assert baked.shortest.tags == "bake+V +PAST"
        assert not REPAIRED_ENGLISH.analyse("bakeed").is_analysable

    def test_the_full_stem_still_takes_the_consonant_suffix(self):
        assert REPAIRED_ENGLISH.analyse("bakes").shortest.morphs == ("bake", "s")
        assert REPAIRED_ENGLISH.analyse("baking").shortest.morphs == ("bak", "ing")
        assert not REPAIRED_ENGLISH.analyse("baks").is_analysable

    def test_the_repair_changes_nothing_for_the_regular_stems(self):
        for word in ("walked", "walks", "walk", "talking"):
            assert list(REPAIRED_ENGLISH.analyse(word)) == list(
                NAIVE_ENGLISH.analyse(word)
            )


class TestSegment:
    def test_cuts_by_the_shortest_analysis_with_offsets(self):
        assert REPAIRED_ENGLISH.segment("walked") == Words(
            [Word("walk", 0, 4), Word("ed", 4, 6)]
        )
        assert REPAIRED_ENGLISH.segment("baked") == Words(
            [Word("bak", 0, 3), Word("ed", 3, 5)]
        )

    def test_an_unanalysable_word_is_left_whole(self):
        assert REPAIRED_ENGLISH.segment("zzz") == Words([Word("zzz", 0, 3)])

    def test_an_empty_word_is_refused(self):
        with pytest.raises(EmptyValuesError):
            REPAIRED_ENGLISH.segment("")


class TestAsPreTokenizer:
    def test_split_segments_each_whitespace_word_in_place(self):
        words = REPAIRED_ENGLISH.split("walked talks zzz bakes")

        assert [(word.text, word.start, word.end) for word in words] == [
            ("walk", 0, 4),
            ("ed", 4, 6),
            ("talk", 7, 11),
            ("s", 11, 12),
            ("zzz", 13, 16),
            ("bake", 17, 21),
            ("s", 21, 22),
        ]

    def test_a_blank_text_has_no_words(self):
        assert REPAIRED_ENGLISH.split("   ") == Words([])

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            REPAIRED_ENGLISH.split(["walked"])  # type: ignore[arg-type]


class TestAsMorphSplitter:
    """The analyser handed to the constrained merger as its boundary rule."""

    WORDS = [
        "walked",
        "walking",
        "talked",
        "talking",
        "baked",
        "baking",
        "walks",
        "bakes",
    ]
    CORPUS = [" ".join(word for word in WORDS for _ in range(5))]
    MORPHS = ("walk", "talk", "bake", "bak", "ed", "ing", "s")

    def fit(self) -> MorphemeConstrainedBytePairEncoding:
        return MorphemeConstrainedBytePairEncoding(
            vocabulary_size=60,
            morph_splitter=REPAIRED_ENGLISH,
            minimum_pair_frequency=1,
        ).fit(self.CORPUS)

    def test_no_piece_spans_a_stem_and_its_suffix(self):
        for token in self.fit().vocabulary:
            if token == "[UNK]":
                continue
            assert any(token.removesuffix("</w>") in morph for morph in self.MORPHS), (
                token
            )

    def test_stems_and_suffixes_become_whole_pieces(self):
        tokenizer = self.fit()

        assert tokenizer.encode("walked").texts == ("walk", "ed</w>")
        assert tokenizer.encode("baked").texts == ("bak", "ed</w>")
        assert tokenizer.encode("bakes").texts == ("bake", "s</w>")

    def test_the_allomorph_is_a_piece_of_its_own(self):
        vocabulary = self.fit().vocabulary

        assert "bak" in vocabulary
        assert "bake" in vocabulary
        assert "baked" not in vocabulary
