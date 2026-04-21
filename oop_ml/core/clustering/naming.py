"""What a cluster is called.

Numbered from one behind this prefix, and shared because both backends label
the columns they hand back with it.

Worth remembering when reading those labels: a cluster's number means nothing
across two fits. A classifier's 0 is the class the data called 0, forever. A
clusterer's 0 is whichever group this particular seeding numbered first, so
two correct fits can agree completely about which rows belong together and
disagree about every label.
"""

from __future__ import annotations

CLUSTER_NAME_PREFIX = "cluster"
"""How groups are named: ``cluster_1``, ``cluster_2``, and so on.

One-indexed to match the components, and a name rather than a bare position so
that a caller reading a report is looking at ``cluster_3`` rather than at the
number ``2``, which in this library usually means a class.
"""
