"""Computer vision, the half of it that was built before anything was learned.

A second domain package beside
:mod:`oop_ml.core.natural_language_processing`, inside ``core`` for the same
reason that one is: a picture has to mean one thing whichever backend reads
it. The frame never imports it and neither domain imports the other, both of
which ``test_layering.py`` holds.

The foundation
--------------
:mod:`picture` is the coercion boundary, a validated grid of brightness.
:mod:`filtering` is the sweep every method here is built from, a small grid of
weights carried across a large picture, and the closed enum of what to read
where it hangs off the edge. :mod:`edges` is two such sweeps read together as
a rate of change, which is what classical vision describes things with, since
an edge survives a change in lighting where a brightness does not.

The four families, and the question each answers
------------------------------------------------
:mod:`matching` asks where a known picture sits inside a larger one, by
carrying it across and scoring every position. The oldest answer there is, and
the one that breaks as soon as the thing turns or changes size.

:mod:`oriented_gradients` asks what a patch looks like, answering with which
directions its edges point in rather than with its pixels. Built to hold still
under a change in lighting, which it does, and not under a rotation, which it
does not.

:mod:`keypoints` asks which few places in a picture are worth describing at
all, and answers with the corners, because a position along an edge cannot be
told from its neighbours along that edge while a corner can.

:mod:`cascade` asks whether a window holds the thing being looked for, cheaply
enough to ask of every window. Three separable ideas: a table that makes the
sum over any rectangle four lookups whatever its size, features built from
those sums, and stages ordered so the cheap ones run on everything and the
expensive ones on almost nothing.

What is deliberately absent
---------------------------
Anything learned by gradient descent, which is the network package's business,
and colour, which none of these methods reads. :meth:`Picture.from_channels`
collapses a colour array to brightness so a caller has one obvious way in.
"""
