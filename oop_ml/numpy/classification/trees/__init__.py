"""Predicting a class by which box a row falls in.

The boundary is a union of axis-aligned rectangles, which is a different
vocabulary from the hyperplane a logistic model draws -- better where the truth
encloses a region, worse where it runs at an angle to every axis.

Any number of classes is the same code, because counting rows in a leaf does not
care how many candidates are on the ballot.
"""
