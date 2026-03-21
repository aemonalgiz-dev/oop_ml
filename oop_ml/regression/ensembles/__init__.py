"""Predicting a quantity from many models rather than one.

Both families, and they pull in opposite directions. Bagging and forests
average independent members to cut variance, and want deep members. Gradient
boosting adds dependent members to cut bias, and wants shallow ones.
"""
