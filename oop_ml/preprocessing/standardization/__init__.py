"""Rescaling every feature to mean 0 and standard deviation 1.

The transformer and the thing it learns. A Standardizer holds one
FeatureScaling per column -- a centre and a spread bound to the name they came
from -- and the arithmetic lives on those rather than being repeated wherever a
column needs shifting.

Splitting fit from transform is the point rather than a convenience: the
statistics have to come from the training rows alone, or the held-out rows have
already told the model something about themselves.
"""
