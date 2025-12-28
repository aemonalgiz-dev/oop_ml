"""Shared frame: validated data types, learned-parameter containers, base classes.

Nothing here is specific to a model family. ``Column`` is the one place raw
input is coerced; everything downstream takes validated types and re-checks
nothing.
"""
