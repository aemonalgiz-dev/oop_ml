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


class SingleClassError(MLLibError):
    """Raised when a classifier's target contains only one of the two classes.

    Separate from :class:`NonBinaryLabelsError` because the labels are perfectly
    valid; there is simply nothing to discriminate between, and a boundary
    fitted against them would be meaningless rather than merely wrong.
    """
