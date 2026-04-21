"""What a fitted preprocessing step learned, shared by every backend.

The steps themselves belong to a backend, since one standardises by hand and
another asks an engine to. What they hand back does not: a caller who fits a
``Standardizer`` from either backend gets the same ``FeatureScalings``, and a
type that came from one backend's package would make that claim untrue.
"""
