"""Every configurable object rejects a hyperparameter it does not have.

This exists because the gap it closes cost a real bug. An example was written
with ``TrainTestSplitter(testing_share=0.3)``; the field is called
``test_fraction``, pydantic's default is to ignore unknown keys, and so the
split silently used the default fraction instead. Nothing raised, the script
ran, and the printed numbers were quietly answering a different question.

The failure mode is worse than a typo usually is, because the wrong value is
always the default -- which is by construction a plausible number. A model
constructed with ``alpha=`` instead of ``penalty=`` does not blow up, it fits
an unpenalised model and reports respectable-looking coefficients.

One parametrized test over every constructible class, so a new estimator cannot
be added without inheriting the guard.
"""

import pytest
from pydantic import ValidationError

from oop_ml import (
    BaggingClassifier,
    BaggingRegressor,
    BagOfWords,
    ByteTokenizer,
    CharacterTokenizer,
    CrossValidation,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    FastText,
    GloVe,
    GradientBoostingRegressor,
    GradientDescentRegression,
    HashedCharacterTokenizer,
    HiddenMarkovSegmenter,
    KFold,
    KNearestNeighboursClassifier,
    KNearestNeighboursRegressor,
    LassoRegression,
    LatentSemanticAnalysis,
    LogisticRegression,
    MorfessorBaseline,
    MosesPreTokenizer,
    MultinomialLogisticRegression,
    MultipleLinearRegression,
    NewtonLogisticRegression,
    OneVsRestClassifier,
    ParagraphVectors,
    PennTreebankPreTokenizer,
    PointwiseMutualInformationEmbeddings,
    PointwiseSegmenter,
    PolynomialFeatures,
    RandomForestClassifier,
    RandomForestRegressor,
    RandomIndexing,
    RidgeRegression,
    RowShuffler,
    RuleAndExceptionPreTokenizer,
    SimpleLinearRegression,
    Standardizer,
    TrainTestSplitter,
    UnicodeWordPreTokenizer,
    WhitespacePreTokenizer,
    Word2Vec,
)

# The tokenizers that construct with no arguments join the list, so the
# same guard covers the natural language processing package. The ones whose
# size or dictionary is required (BytePairEncoding, MaximumMatchingSegmenter,
# ...) inherit the forbid rule from the same two bases, which
# test/core/natural_language_processing/tokenization/test_tokenizer.py pins.
CONSTRUCTIBLE = [
    Word2Vec,
    FastText,
    GloVe,
    ParagraphVectors,
    LatentSemanticAnalysis,
    PointwiseMutualInformationEmbeddings,
    RandomIndexing,
    BagOfWords,
    WhitespacePreTokenizer,
    PennTreebankPreTokenizer,
    MosesPreTokenizer,
    RuleAndExceptionPreTokenizer,
    UnicodeWordPreTokenizer,
    HiddenMarkovSegmenter,
    PointwiseSegmenter,
    CharacterTokenizer,
    ByteTokenizer,
    MorfessorBaseline,
    HashedCharacterTokenizer,
    SimpleLinearRegression,
    MultipleLinearRegression,
    RidgeRegression,
    LassoRegression,
    GradientDescentRegression,
    KNearestNeighboursRegressor,
    DecisionTreeRegressor,
    BaggingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    LogisticRegression,
    NewtonLogisticRegression,
    MultinomialLogisticRegression,
    OneVsRestClassifier,
    KNearestNeighboursClassifier,
    DecisionTreeClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    Standardizer,
    PolynomialFeatures,
    RowShuffler,
    TrainTestSplitter,
    KFold,
    CrossValidation,
]


@pytest.mark.parametrize(
    "configurable",
    CONSTRUCTIBLE,
    ids=[cls.__name__ for cls in CONSTRUCTIBLE],
)
def test_an_unknown_hyperparameter_is_rejected(configurable):
    with pytest.raises(ValidationError):
        configurable(definitely_not_a_real_hyperparameter=1)


# OneVsRestClassifier is the one that has no sensible default: it wraps a
# binary model, and there is no obvious binary model to reach for on its
# behalf. Requiring it is the right call, so it sits out the next test.
DEFAULT_CONSTRUCTIBLE = [
    configurable
    for configurable in CONSTRUCTIBLE
    if configurable is not OneVsRestClassifier
]


@pytest.mark.parametrize(
    "configurable",
    DEFAULT_CONSTRUCTIBLE,
    ids=[cls.__name__ for cls in DEFAULT_CONSTRUCTIBLE],
)
def test_the_defaults_still_construct(configurable):
    # The other half of the guarantee: forbidding extras must not have made
    # any of these impossible to build with no arguments at all.
    assert configurable() is not None


def test_the_one_that_requires_an_argument_still_takes_it():
    assert OneVsRestClassifier(binary_model=LogisticRegression()) is not None


class TestTheSpecificCasesThatMotivatedThis:
    """Each typo below is deliberate, so pyright is told to allow it.

    Worth noticing that pyright flags all three without being asked -- which
    means a project running it would have caught the original bug before the
    script was ever executed. The runtime guard is for everyone else, and for
    hyperparameter names arriving from a config file, where no type checker
    is looking.
    """

    def test_the_splitter_typo_that_started_it(self):
        with pytest.raises(ValidationError):
            TrainTestSplitter(testing_share=0.3)  # pyright: ignore[reportCallIssue]

    def test_a_scikit_learn_habit_does_not_silently_pass(self):
        # alpha is scikit-learn's name for what this library calls penalty.
        # Accepting it silently would fit an unpenalised model and say nothing.
        with pytest.raises(ValidationError):
            RidgeRegression(alpha=2.0)  # pyright: ignore[reportCallIssue]

    def test_the_american_spelling_does_not_silently_pass(self):
        with pytest.raises(ValidationError):
            KNearestNeighboursRegressor(n_neighbors=9)  # pyright: ignore[reportCallIssue]
