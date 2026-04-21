"""What a principal component is called.

A component is not a feature and cannot borrow a feature's name. Naming them
after the features would make transformed data indistinguishable from raw
data, and the two guards that check a caller passed the right columns would
both go blind at once. So components are numbered from one behind this prefix,
and the prefix lives here because a caller reading a transformed column's name
and a backend writing it have to agree, whichever backend that is.
"""

from __future__ import annotations

COMPONENT_NAME_PREFIX = "component"
"""How components are named: ``component_1``, ``component_2``, and so on.

One-indexed, because "the first principal component" is what the literature and
every caller says, and a ``component_0`` would make the two disagree.
"""
