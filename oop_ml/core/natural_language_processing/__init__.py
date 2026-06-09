"""Natural language processing: the vocabulary a text model is built from.

Inside :mod:`oop_ml.core` because it is shared for the reason everything in
``core`` is shared: a token id has to mean one thing whichever backend reads
it, so nothing here belongs to a backend and nothing here imports one. It is a
domain built *on* the frame rather than part of it -- a tokenizer is a
``Fittable``, and raises the same ``NotFittedError`` -- and the frame never
reaches up for it, which ``test/test_layering.py`` checks. The package is pure
Python: tokenization is string handling, and the arithmetic a model does with
the ids it produces lives with the model.
"""
