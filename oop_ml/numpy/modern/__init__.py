"""Small, inspectable NumPy operations behind modern generative models.

These are deliberately bounded teaching implementations. The decoder implements
inference, the adversarial example trains a scalar distribution, and the
reinforcement example learns a one-state policy. The remaining objects expose
individual operations. None downloads or wraps a pretrained model.
"""

from oop_ml.numpy.modern.adaptation import LowRankAdapter, TermOverlapRetriever
from oop_ml.numpy.modern.generation import (
    AdversarialGaussian,
    DiagonalGaussian,
    GaussianDiffusion,
    LinearAutoencoder,
)
from oop_ml.numpy.modern.language import (
    CausalLanguageModel,
    TokenCrossEntropy,
    TokenDistribution,
)
from oop_ml.numpy.modern.reinforcement import SoftmaxPolicy, discounted_returns

__all__ = [
    "AdversarialGaussian",
    "CausalLanguageModel",
    "DiagonalGaussian",
    "GaussianDiffusion",
    "LinearAutoencoder",
    "LowRankAdapter",
    "SoftmaxPolicy",
    "TermOverlapRetriever",
    "TokenCrossEntropy",
    "TokenDistribution",
    "discounted_returns",
]
