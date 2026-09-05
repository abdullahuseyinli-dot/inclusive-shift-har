from __future__ import annotations

import pytest

from inclusive_shift_har.artifacts.parallel_test_gate import partition_nodes


def test_shards_are_complete_disjoint_and_deterministic() -> None:
    nodes = [f"tests/test_gate.py::test_case[{index}]" for index in range(23)]
    shards = partition_nodes(nodes, 4)
    assert shards == partition_nodes(nodes, 4)
    assert sorted(node for shard in shards for node in shard) == sorted(nodes)
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    assert partition_nodes(nodes[:2], 4) == [[nodes[0]], [nodes[1]]]


@pytest.mark.parametrize("nodes,workers", [([], 1), (["same", "same"], 2), (["test"], 0)])
def test_shards_reject_missing_duplicate_or_invalid_collection(
    nodes: list[str], workers: int
) -> None:
    with pytest.raises(ValueError):
        partition_nodes(nodes, workers)
