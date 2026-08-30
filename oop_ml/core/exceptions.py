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
