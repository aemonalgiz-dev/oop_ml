"""Models whose data is an ordered sequence of named states.

Why sequences are a directory of their own
------------------------------------------
The library's layout rule is that the directory names the task, and a sequence
is a task none of the others is. Regression and classification read rows that
could be shuffled without changing the answer; clustering and decomposition do
the same without a target. Here the order is the data. Shuffle a sequence of
weather readings and a Markov chain fitted to it learns a different chain,
because what it learns is which state follows which.

The input differs too. Every other model here reads named numeric columns,
while a chain reads a sequence of sequences of state names, so there is no
:class:`~oop_ml.core.data.feature.Feature` anywhere in it.

A hidden Markov model, where the states are not observed and only something
they emit is, would belong here as well. The chain comes first because it is
the part a hidden Markov model is built on.
"""

from oop_ml.numpy.sequences.markov_chain import MarkovChain

__all__ = ["MarkovChain"]
