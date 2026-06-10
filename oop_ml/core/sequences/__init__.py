"""The vocabulary a model of sequences answers with.

Here rather than beside the Markov chain for the persistence format's reason:
the codec that writes a fitted chain lives in ``oop_ml.core.persistence``, and
core never imports a backend, so a value the codec has to rebuild has to live
beneath it. :class:`~oop_ml.core.generative.boltzmann.BoltzmannParameters` is in
core for the same reason.
"""

from oop_ml.core.sequences.transitions import (
    StateDistribution,
    TransitionCounts,
    TransitionMatrix,
)

__all__ = ["StateDistribution", "TransitionCounts", "TransitionMatrix"]
