"""Parallel map returns in input order, so reductions stay deterministic.

The whole reason a fit can run its restarts across threads and still be
reproducible: results come back keyed to input position, never to which thread
finished first, so a downstream reduction sees the same sequence it saw serial.
"""

import time

from oop_ml.core.parallel import PARALLEL_THRESHOLD, parallel_map


def test_results_are_in_input_order_not_completion_order() -> None:
    def slow_for_small_indices(index: int) -> int:
        # Earlier indices finish later, so completion order reverses input
        # order; the result must still come back in input order.
        time.sleep((10 - index) * 0.002)
        return index * index

    results = parallel_map(slow_for_small_indices, list(range(10)))

    assert results == [index * index for index in range(10)]


def test_a_single_item_runs_serially() -> None:
    assert parallel_map(lambda value: value + 1, [41]) == [42]


def test_below_the_threshold_runs_serially() -> None:
    calls: list[int] = []

    def record(value: int) -> int:
        calls.append(value)
        return value

    parallel_map(record, list(range(PARALLEL_THRESHOLD - 1)))

    assert calls == list(range(PARALLEL_THRESHOLD - 1))


def test_one_worker_forces_the_serial_path() -> None:
    assert parallel_map(lambda value: value * 2, [1, 2, 3, 4], max_workers=1) == [
        2,
        4,
        6,
        8,
    ]
