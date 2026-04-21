"""What a generative model holds, shared by every backend.

A restricted Boltzmann machine's fitted self is a weight block and two bias
vectors, and it is the same triple whichever backend learned it, so the type
that carries it belongs here rather than under one of them.
"""
