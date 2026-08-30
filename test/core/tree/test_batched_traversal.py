"""The batched leaf assignment agrees with the single-row route.

``predict`` walks the tree with ``assign_leaves`` -- one array comparison per
split, partitioning row indices -- while the observed single-row path walks it
with ``leaf_for``. They are two implementations of one question ("which leaf
does this row reach"), and this pins that they meet, the way every fast/slow
pair in this library carries an agreement test.
"""

import numpy as np

from oop_ml.core.data.feature import Feature
from oop_ml.regression.trees.decision_tree_regressor import DecisionTreeRegressor


def test_batched_and_single_row_reach_the_same_leaves() -> None:
    generator = np.random.default_rng(3)
    matrix = generator.normal(size=(300, 6))
    target = matrix[:, 0] * 2 + np.sin(matrix[:, 1] * 3)
    features = [Feature(f"f{index}", matrix[:, index]) for index in range(6)]

    model = DecisionTreeRegressor(max_depth=6).fit(features, Feature("y", target))
    root = model.root

    query = generator.normal(size=(120, 6))
    query_rows = np.column_stack([query[:, index] for index in range(6)])

    single = [root.leaf_for(row).prediction for row in query_rows]

    batched: list = [root] * query_rows.shape[0]
    root.assign_leaves(
        query_rows, np.arange(query_rows.shape[0], dtype=np.intp), batched
    )
    batched_predictions = [leaf.prediction for leaf in batched]

    assert single == batched_predictions
