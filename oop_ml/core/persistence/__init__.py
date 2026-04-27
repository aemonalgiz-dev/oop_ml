"""Saving a fitted model as data, shared by every backend.

The format is here rather than under a backend because a document is a
statement about a fitted model, and every backend fits models. What is *not*
here is which models exist: each backend owns a registry of its own types and
registers it, so this package never imports a backend and the arrow keeps
pointing one way.
"""
