"""Spec for MeanPooling and SmoothInverseFrequency -- documents from word vectors.

Pinned against a hand-built seven-word table in four dimensions and six
two-word documents, worked in the module docstring and in ``fixtures.py``.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
    TooFewValuesError,
    UndefinedMetricError,
    UnknownTokenError,
)
from oop_ml.core.natural_language_processing.embeddings.documents.pooling import (
    MeanPooling,
    PoolingWeighting,
    SmoothInverseFrequency,
    WordVectorPooling,
    smooth_inverse_frequency_weights,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import WordEmbeddings
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from test.core.natural_language_processing.embeddings.documents.fixtures import (
    ANIMAL_DOCUMENTS,
    FINANCE_DOCUMENTS,
    IDF_MEAN_MARGIN,
    PROBABILITY_OF_A_TOPIC_WORD,
    PROBABILITY_OF_THE,
    SIF_MARGIN,
    SIF_WEIGHTS_ONLY_MARGIN,
    TOPIC_CORPUS,
    TOPIC_IDF_OF_A_TOPIC_WORD,
    TOPIC_IDF_OF_THE,
    TOPIC_TABLE,
    UNIFORM_MEAN_MARGIN,
    cosine,
    separation_margin,
    topic_embeddings,
)

THE, CAT, DOG, PET, STOCK, BOND, MARKET = (
    TOPIC_TABLE[position] for position in range(7)
)
IDF = PoolingWeighting.INVERSE_DOCUMENT_FREQUENCY
SMOOTHING = 1e-3


def fit_uniform() -> MeanPooling:
    return MeanPooling(embeddings=topic_embeddings()).fit(TOPIC_CORPUS)


def fit_idf() -> MeanPooling:
    return MeanPooling(embeddings=topic_embeddings(), weighting=IDF).fit(TOPIC_CORPUS)


def fit_sif(remove_first_component: bool = True) -> SmoothInverseFrequency:
    return SmoothInverseFrequency(
        embeddings=topic_embeddings(), remove_first_component=remove_first_component
    ).fit(TOPIC_CORPUS)


def sif_weight(probability: float) -> float:
    """The paper's formula, written out."""
    return SMOOTHING / (SMOOTHING + probability)


class TestMeanPoolingUniform:
    def test_is_the_plain_mean_of_the_word_vectors(self):
        """(2 * the + cat + dog) / 4 = (0.5, 0, 1, 0)."""
        vector = fit_uniform().transform(["the cat the dog"]).vector_of(0)

        assert np.allclose(vector, [0.5, 0.0, 1.0, 0.0])
        assert np.allclose(vector, (2 * THE + CAT + DOG) / 4)

    def test_a_repeated_word_counts_each_time(self):
        vector = fit_uniform().transform(["cat cat dog"]).vector_of(0)

        assert np.allclose(vector, (2 * CAT + DOG) / 3)

    def test_a_one_word_document_is_that_word_vector(self):
        assert np.allclose(fit_uniform().transform(["stock"]).vector_of(0), STOCK)

    def test_order_does_not_matter(self):
        vectors = fit_uniform().transform(["dog bites man", "man bites dog"])

        assert np.array_equal(vectors.vector_of(0), vectors.vector_of(1))

    def test_separates_the_two_topics(self):
        vectors = fit_uniform().transform(TOPIC_CORPUS)
        within = [vectors.similarity(0, 1), vectors.similarity(3, 4)]
        between = [vectors.similarity(0, 3), vectors.similarity(2, 5)]

        assert min(within) > max(between)
        assert separation_margin(vectors.vectors) == pytest.approx(
            UNIFORM_MEAN_MARGIN, abs=1e-4
        )

    def test_the_common_word_pulls_the_topics_together(self):
        """Doc 0 is (0.5, 0, 1, 0) and doc 3 is (0, 0.5, 1, 0): cosine 1 / 1.25."""
        vectors = fit_uniform().transform(TOPIC_CORPUS)

        assert vectors.similarity(0, 3) == pytest.approx(0.8)

    def test_a_document_of_unseen_words_is_the_zero_vector(self):
        vectors = fit_uniform().transform(["zebra quokka", "the cat"])

        assert np.array_equal(vectors.vector_of(0), np.zeros(4))
        with pytest.raises(UndefinedMetricError):
            vectors.similarity(0, 1)

    def test_the_dimension_is_the_embeddings_dimension(self):
        assert fit_uniform().transform(["cat"]).dimension == 4

    def test_fit_is_still_required(self):
        with pytest.raises(NotFittedError):
            MeanPooling(embeddings=topic_embeddings()).transform(["the cat"])

    def test_fit_transform_matches_fit_then_transform(self):
        assert MeanPooling(embeddings=topic_embeddings()).fit_transform(
            TOPIC_CORPUS
        ) == (fit_uniform().transform(TOPIC_CORPUS))


class TestMeanPoolingInverseDocumentFrequency:
    def test_a_word_in_every_training_document_weighs_exactly_one(self):
        pooling = fit_idf()

        assert (
            pooling.inverse_document_frequencies[
                pooling.embeddings.vocabulary.id_of("the")
            ]
            == TOPIC_IDF_OF_THE
        )
        assert TOPIC_IDF_OF_THE == 1.0

    def test_a_topic_word_weighs_the_worked_value(self):
        """In two of six documents: log(7 / 3) + 1."""
        pooling = fit_idf()
        weight = pooling.inverse_document_frequencies[
            pooling.embeddings.vocabulary.id_of("cat")
        ]

        assert weight == pytest.approx(TOPIC_IDF_OF_A_TOPIC_WORD)
        assert weight == pytest.approx(1.8473, abs=1e-4)

    def test_a_word_in_no_training_document_weighs_the_smoothed_maximum(self):
        """Six documents, none holding ``market``: log(7 / 1) + 1."""
        pooling = MeanPooling(embeddings=topic_embeddings(), weighting=IDF).fit(
            TOPIC_CORPUS[:4]
        )
        market = pooling.embeddings.vocabulary.id_of("market")

        assert pooling.inverse_document_frequencies[market] == pytest.approx(
            np.log(5.0) + 1.0
        )

    def test_differs_from_the_uniform_mean_when_a_word_is_in_every_document(self):
        """Weights (1, 1.8473, 1, 1.8473): (2 * the + 1.8473 * (cat + dog)) / 5.6946."""
        weight = TOPIC_IDF_OF_A_TOPIC_WORD
        expected = (2 * TOPIC_IDF_OF_THE * THE + weight * (CAT + DOG)) / (
            2 * TOPIC_IDF_OF_THE + 2 * weight
        )
        vector = fit_idf().transform(["the cat the dog"]).vector_of(0)

        assert np.allclose(vector, expected)
        assert np.allclose(vector, [0.6488, 0.0, 0.7024, 0.0], atol=1e-4)
        assert not np.allclose(
            vector, fit_uniform().transform(["the cat the dog"]).vector_of(0)
        )

    def test_the_topic_share_rises_and_the_common_word_share_falls(self):
        uniform = fit_uniform().transform(["the cat the dog"]).vector_of(0)
        weighted = fit_idf().transform(["the cat the dog"]).vector_of(0)

        assert weighted[0] / weighted[2] > uniform[0] / uniform[2]

    def test_agrees_with_the_uniform_mean_when_every_word_has_the_same_idf(self):
        """Fit on ``cat dog`` and ``stock bond``: every word is in one of two
        documents, so the weights are equal and a weighted mean is a mean."""
        pooling = MeanPooling(embeddings=topic_embeddings(), weighting=IDF).fit(
            ["cat dog", "stock bond"]
        )

        assert np.allclose(pooling.transform(["cat dog"]).vector_of(0), (CAT + DOG) / 2)
        assert np.allclose(
            pooling.transform(["cat stock bond"]).vector_of(0), (CAT + STOCK + BOND) / 3
        )

    def test_separates_better_than_the_plain_mean(self):
        margin = separation_margin(fit_idf().transform(TOPIC_CORPUS).vectors)

        assert margin == pytest.approx(IDF_MEAN_MARGIN, abs=1e-4)
        assert margin > UNIFORM_MEAN_MARGIN

    def test_a_document_of_unseen_words_is_the_zero_vector(self):
        assert np.array_equal(fit_idf().transform(["zebra"]).vector_of(0), np.zeros(4))

    def test_the_idf_is_frozen(self):
        assert fit_idf().inverse_document_frequencies.flags.writeable is False

    def test_the_idf_is_learned_under_the_uniform_weighting_too(self):
        assert np.array_equal(
            fit_uniform().inverse_document_frequencies,
            fit_idf().inverse_document_frequencies,
        )

    def test_the_weighting_accepts_its_string_value(self):
        pooling = MeanPooling(
            embeddings=topic_embeddings(),
            weighting="inverse_document_frequency",  # type: ignore[arg-type]
        )

        assert pooling.weighting is IDF

    def test_before_fit_raises_not_fitted(self):
        pooling = MeanPooling(embeddings=topic_embeddings(), weighting=IDF)

        with pytest.raises(NotFittedError):
            _ = pooling.inverse_document_frequencies


class TestWordVectorPoolingFrame:
    @pytest.mark.parametrize("model_type", [MeanPooling, SmoothInverseFrequency])
    def test_minimum_count_other_than_one_is_refused(self, model_type):
        with pytest.raises(ValidationError):
            model_type(embeddings=topic_embeddings(), minimum_count=2)

    @pytest.mark.parametrize("model_type", [MeanPooling, SmoothInverseFrequency])
    def test_the_default_minimum_count_constructs(self, model_type):
        assert (
            model_type(embeddings=topic_embeddings(), minimum_count=1).minimum_count
            == 1
        )

    @pytest.mark.parametrize("model_type", [MeanPooling, SmoothInverseFrequency])
    def test_embeddings_are_required(self, model_type):
        with pytest.raises(ValidationError):
            model_type()

    @pytest.mark.parametrize("model_type", [MeanPooling, SmoothInverseFrequency])
    def test_embeddings_must_be_word_embeddings(self, model_type):
        with pytest.raises(ValidationError):
            model_type(embeddings=TOPIC_TABLE)

    @pytest.mark.parametrize("model_type", [MeanPooling, SmoothInverseFrequency])
    def test_an_unknown_keyword_is_refused(self, model_type):
        with pytest.raises(ValidationError):
            model_type(embeddings=topic_embeddings(), dimension=4)

    @pytest.mark.parametrize("model_type", [MeanPooling, SmoothInverseFrequency])
    def test_fit_returns_self(self, model_type):
        model = model_type(embeddings=topic_embeddings())

        assert model.fit(TOPIC_CORPUS) is model

    @pytest.mark.parametrize("model_type", [MeanPooling, SmoothInverseFrequency])
    def test_transform_on_a_single_string_is_refused(self, model_type):
        model = model_type(embeddings=topic_embeddings()).fit(TOPIC_CORPUS)

        with pytest.raises(InvalidValuesError):
            model.transform("the cat the dog")  # type: ignore[arg-type]

    @pytest.mark.parametrize("model_type", [MeanPooling, SmoothInverseFrequency])
    def test_fit_on_a_single_string_is_refused(self, model_type):
        with pytest.raises(InvalidValuesError):
            model_type(embeddings=topic_embeddings()).fit("the cat the dog")  # type: ignore[arg-type]

    @pytest.mark.parametrize("model_type", [MeanPooling, SmoothInverseFrequency])
    def test_a_blank_corpus_is_refused(self, model_type):
        with pytest.raises(EmptyValuesError):
            model_type(embeddings=topic_embeddings()).fit(["", "  "])

    @pytest.mark.parametrize("model_type", [MeanPooling, SmoothInverseFrequency])
    def test_the_frame_is_shared(self, model_type):
        assert issubclass(model_type, WordVectorPooling)

    def test_the_frame_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            WordVectorPooling(embeddings=topic_embeddings())  # type: ignore[abstract]

    def test_the_weighting_enum_has_two_members(self):
        assert [member.value for member in PoolingWeighting] == [
            "uniform",
            "inverse_document_frequency",
        ]


class TestSmoothInverseFrequencyWeights:
    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.5, 0.001996007984031936),
            (1e-3, 0.5),
            (1e-6, 0.999000999000999),
        ],
    )
    def test_the_papers_weight_at_a_of_one_thousandth(self, probability, expected):
        weight = smooth_inverse_frequency_weights(SMOOTHING, np.array([probability]))

        assert weight[0] == pytest.approx(expected, rel=1e-12)
        assert weight[0] == pytest.approx(sif_weight(probability), rel=1e-12)

    def test_the_weights_are_rounded_as_the_docstring_says(self):
        weights = smooth_inverse_frequency_weights(
            SMOOTHING, np.array([0.5, 1e-3, 1e-6])
        )

        assert np.allclose(weights, [0.001996, 0.5, 0.999001], atol=5e-7)

    def test_a_never_seen_word_weighs_exactly_one(self):
        assert smooth_inverse_frequency_weights(SMOOTHING, np.array([0.0]))[0] == 1.0

    def test_a_rarer_word_weighs_more(self):
        weights = smooth_inverse_frequency_weights(
            SMOOTHING, np.array([0.5, 0.1, 0.01])
        )

        assert weights[0] < weights[1] < weights[2]

    @pytest.mark.parametrize("smoothing", [0.0, -1e-3])
    def test_a_non_positive_smoothing_is_refused(self, smoothing):
        with pytest.raises(InvalidValuesError):
            smooth_inverse_frequency_weights(smoothing, np.array([0.5]))

    @pytest.mark.parametrize("probability", [-0.1, 1.1])
    def test_a_probability_outside_the_unit_interval_is_refused(self, probability):
        with pytest.raises(InvalidValuesError):
            smooth_inverse_frequency_weights(SMOOTHING, np.array([probability]))


class TestSmoothInverseFrequencyFit:
    def test_word_probabilities_are_the_corpus_shares(self):
        sif = fit_sif()
        vocabulary = sif.embeddings.vocabulary

        assert sif.word_probabilities[vocabulary.id_of("the")] == PROBABILITY_OF_THE
        assert sif.word_probabilities[vocabulary.id_of("cat")] == pytest.approx(
            PROBABILITY_OF_A_TOPIC_WORD
        )
        assert sif.word_probabilities.sum() == pytest.approx(1.0)

    def test_weight_of_agrees_with_the_formula(self):
        sif = fit_sif()

        assert sif.weight_of("the") == pytest.approx(
            sif_weight(PROBABILITY_OF_THE), rel=1e-12
        )
        assert sif.weight_of("cat") == pytest.approx(
            sif_weight(PROBABILITY_OF_A_TOPIC_WORD), rel=1e-12
        )
        assert sif.weight_of("the") == pytest.approx(0.001996, abs=5e-7)

    def test_word_weights_are_the_formula_over_the_probabilities(self):
        sif = fit_sif()

        assert np.array_equal(
            sif.word_weights,
            smooth_inverse_frequency_weights(SMOOTHING, sif.word_probabilities),
        )

    def test_the_common_word_is_down_weighted_against_a_topic_word(self):
        sif = fit_sif()

        assert sif.weight_of("cat") / sif.weight_of("the") == pytest.approx(
            (SMOOTHING + PROBABILITY_OF_THE) / (SMOOTHING + PROBABILITY_OF_A_TOPIC_WORD)
        )

    def test_weight_of_a_word_without_a_vector_is_refused(self):
        with pytest.raises(UnknownTokenError):
            fit_sif().weight_of("zebra")

    def test_a_vocabulary_word_the_corpus_never_used_weighs_one(self):
        sif = SmoothInverseFrequency(embeddings=topic_embeddings()).fit(
            TOPIC_CORPUS[:3]
        )

        assert sif.word_probabilities[sif.embeddings.vocabulary.id_of("market")] == 0.0
        assert sif.weight_of("market") == 1.0

    def test_the_smoothing_moves_the_weights(self):
        heavy = SmoothInverseFrequency(
            embeddings=topic_embeddings(), smoothing=1.0
        ).fit(TOPIC_CORPUS)

        assert heavy.weight_of("the") == pytest.approx(1.0 / 1.5)

    def test_the_first_component_is_a_unit_vector(self):
        assert np.linalg.norm(fit_sif().first_component) == pytest.approx(1.0)

    def test_the_first_component_has_its_largest_entry_positive(self):
        component = fit_sif().first_component

        assert component[np.argmax(np.abs(component))] > 0.0

    def test_the_worked_first_component(self):
        """The average topic direction plus the direction of ``the``."""
        assert np.allclose(
            fit_sif().first_component, [0.6384, 0.6384, 0.4299, 0.0], atol=1e-4
        )

    def test_the_first_component_is_uncentred(self):
        """On one training document the component is that document's direction;
        centring one row would leave nothing to decompose."""
        sif = SmoothInverseFrequency(embeddings=topic_embeddings()).fit(
            ["the cat the dog"]
        )
        raw = (
            fit_sif_without_removal_on(["the cat the dog"])
            .transform(["the cat the dog"])
            .vector_of(0)
        )

        assert np.allclose(sif.first_component, raw / np.linalg.norm(raw))

    def test_the_first_component_is_learned_even_when_not_removed(self):
        assert np.allclose(
            fit_sif(remove_first_component=False).first_component,
            fit_sif().first_component,
        )

    def test_the_learned_arrays_are_frozen(self):
        sif = fit_sif()

        assert sif.word_probabilities.flags.writeable is False
        assert sif.word_weights.flags.writeable is False
        assert sif.first_component.flags.writeable is False

    def test_a_corpus_with_no_word_that_has_a_vector_is_refused(self):
        with pytest.raises(TooFewValuesError):
            SmoothInverseFrequency(embeddings=topic_embeddings()).fit(
                ["zebra quokka", "emu"]
            )

    def test_a_corpus_whose_every_vector_is_zero_is_refused(self):
        zeros = WordEmbeddings(Vocabulary(["a", "b"]), np.zeros((2, 3)))

        with pytest.raises(InvalidValuesError):
            SmoothInverseFrequency(embeddings=zeros).fit(["a b", "b a"])

    def test_before_fit_raises_not_fitted(self):
        sif = SmoothInverseFrequency(embeddings=topic_embeddings())

        with pytest.raises(NotFittedError):
            sif.transform(TOPIC_CORPUS)
        with pytest.raises(NotFittedError):
            _ = sif.word_probabilities
        with pytest.raises(NotFittedError):
            _ = sif.word_weights
        with pytest.raises(NotFittedError):
            _ = sif.first_component
        with pytest.raises(NotFittedError):
            sif.weight_of("the")


def fit_sif_without_removal_on(corpus: list[str]) -> SmoothInverseFrequency:
    return SmoothInverseFrequency(
        embeddings=topic_embeddings(), remove_first_component=False
    ).fit(corpus)


class TestSmoothInverseFrequencyTransform:
    def test_every_training_vector_is_orthogonal_to_the_first_component(self):
        sif = fit_sif()
        vectors = sif.transform(TOPIC_CORPUS)

        assert np.abs(vectors.vectors @ sif.first_component).max() < 1e-10

    def test_new_texts_are_orthogonal_to_it_too(self):
        sif = fit_sif()
        vectors = sif.transform(["the cat", "stock market the the", "pet bond"])

        assert np.abs(vectors.vectors @ sif.first_component).max() < 1e-10

    def test_without_removal_a_one_word_document_is_its_weighted_vector(self):
        sif = fit_sif(remove_first_component=False)

        assert np.allclose(
            sif.transform(["cat"]).vector_of(0), sif.weight_of("cat") * CAT
        )

    def test_without_removal_is_the_weighted_mean_by_hand(self):
        """(2 * w(the) * the + w(cat) * cat + w(dog) * dog) / 4, divided by the
        word count and not by the total weight."""
        common = sif_weight(PROBABILITY_OF_THE)
        topic = sif_weight(PROBABILITY_OF_A_TOPIC_WORD)
        expected = (2 * common * THE + topic * CAT + topic * DOG) / 4

        vector = (
            fit_sif(remove_first_component=False)
            .transform(["the cat the dog"])
            .vector_of(0)
        )

        assert np.allclose(vector, expected)
        assert np.allclose(vector, [0.005929, 0.0, 0.001996, 0.0], atol=5e-7)

    def test_removal_subtracts_the_projection(self):
        """Two routes: the removing transform, and the raw one less ``u (u . v)``."""
        removing = fit_sif()
        raw = fit_sif(remove_first_component=False).transform(TOPIC_CORPUS).vectors
        component = removing.first_component
        by_hand = raw - np.outer(raw @ component, component)

        assert np.allclose(
            removing.transform(TOPIC_CORPUS).vectors, by_hand, atol=1e-15
        )

    def test_a_common_word_document_is_shorter_than_a_topic_word_document(self):
        vectors = fit_sif(remove_first_component=False).transform(["the", "cat"])

        assert np.linalg.norm(vectors.vector_of(0)) < np.linalg.norm(
            vectors.vector_of(1)
        )

    def test_the_weights_alone_separate_better_than_the_plain_mean(self):
        margin = separation_margin(
            fit_sif(remove_first_component=False).transform(TOPIC_CORPUS).vectors
        )

        assert margin == pytest.approx(SIF_WEIGHTS_ONLY_MARGIN, abs=1e-4)
        assert margin > UNIFORM_MEAN_MARGIN

    def test_separates_at_least_as_well_as_mean_pooling(self):
        vectors = fit_sif().transform(TOPIC_CORPUS)
        margin = separation_margin(vectors.vectors)

        assert margin == pytest.approx(SIF_MARGIN, abs=1e-4)
        assert margin >= separation_margin(
            fit_uniform().transform(TOPIC_CORPUS).vectors
        )
        assert margin >= separation_margin(fit_idf().transform(TOPIC_CORPUS).vectors)

    def test_the_two_topics_land_opposite_on_this_symmetric_fixture(self):
        vectors = fit_sif().transform(TOPIC_CORPUS)

        assert vectors.similarity(0, 3) == pytest.approx(-1.0, abs=1e-9)
        for first in ANIMAL_DOCUMENTS:
            for second in FINANCE_DOCUMENTS:
                assert vectors.similarity(first, second) < -0.98
        assert vectors.similarity(0, 1) > 0.98

    def test_the_result_agrees_with_the_definition_of_cosine(self):
        vectors = fit_sif().transform(TOPIC_CORPUS)

        assert vectors.similarity(0, 1) == pytest.approx(
            cosine(vectors.vector_of(0), vectors.vector_of(1))
        )

    def test_a_document_of_unseen_words_is_zero_even_after_removal(self):
        vectors = fit_sif().transform(["zebra quokka", "the cat"])

        assert np.array_equal(vectors.vector_of(0), np.zeros(4))
        with pytest.raises(UndefinedMetricError):
            vectors.similarity(0, 1)

    def test_the_dimension_is_the_embeddings_dimension(self):
        assert fit_sif().transform(["cat"]).dimension == 4

    def test_fit_transform_matches_fit_then_transform(self):
        assert SmoothInverseFrequency(embeddings=topic_embeddings()).fit_transform(
            TOPIC_CORPUS
        ) == fit_sif().transform(TOPIC_CORPUS)

    def test_a_blank_batch_is_refused(self):
        with pytest.raises(EmptyValuesError):
            fit_sif().transform(["", " "])


class TestSmoothInverseFrequencyConstruction:
    @pytest.mark.parametrize(
        "keywords",
        [
            {"smoothing": 0.0},
            {"smoothing": -1e-3},
            {"minimum_count": 2},
            {"remove_first_component": "maybe"},
            {"a": 1e-3},
            {"alpha": 1e-3},
        ],
    )
    def test_out_of_range_or_unknown_keywords_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            SmoothInverseFrequency(embeddings=topic_embeddings(), **keywords)

    def test_the_defaults_are_the_papers(self):
        sif = SmoothInverseFrequency(embeddings=topic_embeddings())

        assert sif.smoothing == 1e-3
        assert sif.remove_first_component is True
        assert sif.minimum_count == 1
        assert sif.embeddings == topic_embeddings()
