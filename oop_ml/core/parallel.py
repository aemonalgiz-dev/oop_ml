"""Running independent work across cores, when it is worth it.

Several fits here repeat an independent computation: k-means restarts the whole
algorithm from different seeds, cross-validation fits a model per fold, a grid
search fits one per candidate. These are embarrassingly parallel -- no restart
reads another's state -- and until now they ran one after another.

Threads, not async, and not processes
--------------------------------------
The work is CPU-bound numpy, so ``async`` is the wrong tool entirely: an
``async def`` that never awaits does not yield the event loop, it holds it for
the whole solve. What actually parallelises CPU-bound numpy is threads, because
numpy releases the GIL inside its BLAS kernels -- a matrix multiply on one
thread genuinely runs beside a matrix multiply on another. Processes would work
too but cost a pickle of every argument and result across the boundary; threads
share memory and cost nothing to hand a numpy array to.

The catch, and why the worker count is small
--------------------------------------------
BLAS is *already* threading each individual multiply, so stacking a wide outer
pool on top of it oversubscribes the cores and the two layers contend.
Measured on k-means, ten restarts over 10000 rows: four workers cut the wall
clock in half, and eight bought nothing beyond four. So the default ceiling is
deliberately low, and a caller that knows its inner work is single-threaded can
raise it.

Determinism is preserved
------------------------
:func:`parallel_map` returns results in input order regardless of which thread
finished first, so a fit that reduces them deterministically -- keeping the
lowest-inertia restart under a strict tie rule, say -- gets the identical answer
it got serially. Parallelism changes the timing, never the result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

_Result = TypeVar("_Result")

DEFAULT_MAX_WORKERS = 4
"""How many threads to use at most.

Low on purpose: BLAS threads each numpy kernel underneath, so a wider pool
oversubscribes the cores. Four was the measured plateau for k-means; past it
the outer and inner threading contend and the wall clock stops improving.
"""

PARALLEL_THRESHOLD = 2
"""The fewest independent tasks worth a thread pool.

Below this the pool's own start-up costs more than it saves, so the work runs
serially. One task is never worth a pool; the threshold is expressed in task
count rather than data size because the tasks here are whole model fits, each
already large enough that a handful of them clears the overhead.
"""


def parallel_map(
    work: Callable[[_Result], object],
    items: Sequence[_Result],
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list:
    """Apply ``work`` to every item, across threads when there are enough.

    Results come back in the order of ``items``, never in completion order, so
    the caller can reduce them deterministically. With fewer than
    :data:`PARALLEL_THRESHOLD` items, or a single worker, the work runs serially
    and no pool is created -- the parallel path is an optimisation for the case
    that has enough independent work to fill it, not a new default that taxes
    the small case.
    """
    if len(items) < PARALLEL_THRESHOLD or max_workers <= 1:
        return [work(item) for item in items]

    workers = min(max_workers, len(items))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(work, items))
