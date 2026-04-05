"""Predicting a quantity from whoever is nearby.

The first regressor here that fits nothing. There is no line, no surface and no
coefficient to read back -- the answer for a row is the average of the answers
for the rows most like it, which makes the model as flexible as the data is
dense and as unreliable as the data is sparse.
"""
