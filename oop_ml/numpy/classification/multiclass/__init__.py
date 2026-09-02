"""More than two classes, by joint fit or by a committee of binary ones.

Softmax gives every class its own weight vector and normalises across them, so
what comes back is a distribution because it could not be anything else.
One-vs-rest fits an independent binary model per class and compares their
answers, which needs no new mathematics and offers no such guarantee: its
probabilities do not sum to one, and this package reports them raw rather than
tidying them into looking as though they do.
"""
