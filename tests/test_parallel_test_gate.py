from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from inclusive_shift_har.artifacts.parallel_test_gate import partition_nodes, write_shard_arguments
from inclusive_shift_har.manifests.canonical import sha256_file


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


def test_pytest_argument_file_preserves_space_parameter_and_is_create_only(tmp_path: Path) -> None:
    node = "tests/test_cohort_gap.py::test_cross_cohort_report_rejects_mutated_input_self_hashes[1-record_sha256-target statistics self-hash]"
    path = tmp_path / "selected tests.args"
    receipt = write_shard_arguments(path, [node])
    assert path.read_text(encoding="utf-8").splitlines() == [node]
    assert receipt["sha256"] == sha256_file(path)
    with pytest.raises(FileExistsError):
        write_shard_arguments(path, [node])
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", f"@{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert node in result.stdout
    assert "1 test collected" in result.stdout


@pytest.mark.parametrize("nodes", [[], ["--ignore=tests"], ["tests/a::test_x\n--ignore=tests"]])
def test_argument_file_rejects_options_and_line_injection(tmp_path: Path, nodes: list[str]) -> None:
    with pytest.raises(ValueError):
        write_shard_arguments(tmp_path / "rejected.args", nodes)
