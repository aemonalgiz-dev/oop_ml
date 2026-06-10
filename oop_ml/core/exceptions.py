"""Exception hierarchy shared across the library.

Every error the library raises derives from :class:`MLLibError`, so callers can
write ``except MLLibError`` to trap anything this package throws, or catch a
specific subclass when they want to react to one failure mode.
"""


class MLLibError(Exception):
    """Base class for every error raised by this library."""


class NotFittedError(MLLibError):
    """Raised when a fitted attribute or ``predict`` is used before ``fit``."""


class EmptyValuesError(MLLibError):
    """Raised when an input array is empty but values are required."""


class TooFewValuesError(MLLibError):
    """Raised when an input has fewer samples than an estimator needs."""


class NonEqualArrayLengthError(MLLibError):
    """Raised when two arrays that must align have different lengths."""


class InvalidValuesError(MLLibError):
    """Raised when input cannot be coerced to a finite 1-D float array."""


class AllSameValuesError(MLLibError):
    """Raised when an input has no variance (all values identical)."""


class UndefinedMetricError(MLLibError):
    """Raised when a metric is mathematically undefined for the given input."""


class NonUniqueFeaturesError(MLLibError):
    """Raised when an input has a non-unique set of feature names."""


class NonBinaryLabelsError(MLLibError):
    """Raised when a binary classifier is handed a target that is not 0 or 1."""


class CollinearFeaturesError(MLLibError):
    """Raised when the features are linear combinations of each other.

    A column that is (nearly) a sum or multiple of other columns makes
    ``X.T X`` singular, so the normal equations have no unique solution --
    infinitely many coefficient vectors fit equally well, and no solver can
    choose among them. Given a name of its own so that it routes with the rest
    of the hierarchy rather than escaping as a bare ``numpy.linalg.LinAlgError``,
    which for years was the one reachable failure outside it.
    """


class InvalidDocumentError(MLLibError):
    """Raised when a saved model document cannot be trusted or read.

    An unknown model type, a format version this build does not speak, a
    missing learned part, a payload of the wrong shape. Distinct from the
    data errors because the remedy is different: bad data means fixing the
    input, a bad document means the file is from a different build, a
    different library, or a hand that edited it.
    """


class DivergenceError(MLLibError):
    """Raised when an iterative fit's weights overflow to non-finite values.

    A learning rate too large for the data makes every step bigger than the
    last, and the walk overflows to inf and then nan without numpy raising
    anything -- a fit that "completes" and then answers nan to every question.
    Raised at the source so the failure names its cause (lower the learning
    rate) instead of surfacing as nan predictions three calls later.
    """


class SingularHessianError(MLLibError):
    """Raised when a second-order solver's Hessian has no unique solution.

    The numerical face of separation. Once every ``p (1 - p)`` weight has
    underflowed, ``X.T W X`` is the zero matrix and there is no Newton step to
    take. Given a name of its own so that it routes with the rest of the
    hierarchy rather than escaping as a bare ``numpy.linalg.LinAlgError``.
    """


class SingleClassError(MLLibError):
    """Raised when a classifier's target contains only one of the two classes.

    Separate from :class:`NonBinaryLabelsError` because the labels are perfectly
    valid; there is simply nothing to discriminate between, and a boundary
    fitted against them would be meaningless rather than merely wrong.
    """


class NonUniqueStationaryDistributionError(MLLibError):
    """Raised when a Markov chain has more than one stationary distribution.

    That happens exactly when the chain holds more than one closed group of
    states, a group a walk can enter and never leave. Each closed group has its
    own stationary distribution, and every mixture of them is stationary too,
    so "the" stationary distribution names nothing. A reducible chain with a
    single closed group is not this case: it has exactly one, with no share at
    all on the states it drains away from. Named separately from
    :class:`UndefinedMetricError` because the remedy is specific: the counts
    are fine, and it is the chain they describe that splits in two.
    """


class ShapeMismatchError(MLLibError):
    """Raised when two widths that have to agree do not.

    The error this library exists to raise *early*. A network's shape is a
    chain of integer equalities, each layer's output count matching the next
    layer's input count, and every one of those numbers is known before any
    data arrives. So a disagreement is a fact about the architecture rather
    than a discovery made part-way through a training run, and it should
    surface at construction, where the mistake is, instead of hours later
    inside a matrix multiply, where the mistake merely lands.
    """


class UnknownTokenError(MLLibError):
    """Raised when a token is looked up in a vocabulary that has no room for it.

    A closed vocabulary answers every lookup: a token it holds gets its own id,
    and one it does not gets the id of the unknown token. This is raised only
    when there is no unknown token to fall back on, which is the case for the
    byte-level vocabularies that are closed by construction and so never
    needed one, and for a caller decoding an id no vocabulary of that size
    contains. Named separately from :class:`InvalidValuesError` because the
    remedy is different: the text is fine, and it is the vocabulary that was
    built without a way to say "something else".
    """


class NonUniqueTokensError(MLLibError):
    """Raised when a vocabulary is handed the same token twice.

    Two positions for one string would mean two ids for one token, and the
    encoder would have to choose one of them silently. The sibling of
    :class:`NonUniqueFeaturesError`, named separately because a token is not a
    feature and a message about features would send a reader to the wrong
    object.
    """


class VocabularyTooSmallError(MLLibError):
    """Raised when a tokenizer is asked for fewer tokens than its alphabet.

    Every subword vocabulary begins with the symbols the corpus is spelled in,
    because a symbol that has no token cannot be encoded at all. A requested
    size below that count is not a small vocabulary, it is one that cannot
    represent its own training data, and the fit refuses rather than dropping
    letters. Raised at ``fit`` rather than at construction because the
    alphabet is a fact about the corpus, and the corpus arrives at ``fit``.
    """
