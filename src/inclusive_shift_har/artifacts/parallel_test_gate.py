"""Auditably shard the complete collected pytest suite without omitting nodes."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from inclusive_shift_har.artifacts.publication_checkpoint import record_command
from inclusive_shift_har.experiments.cross_dataset_har import _write_json_create_only
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def partition_nodes(nodes: list[str], workers: int) -> list[list[str]]:
    """Round-robin by ordered node ID; no outcomes inform shard assignment."""

    if workers < 1 or not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("a nonempty unique test collection and positive worker count are required")
    shards = [nodes[index::workers] for index in range(min(workers, len(nodes)))]
    if sorted(node for shard in shards for node in shard) != sorted(nodes):
        raise AssertionError("test collection changed during partition")
    return shards


def run_full_suite(output: Path, root: Path, workers: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    collection = record_command(
        output / "collection", [sys.executable, "-m", "pytest", "--collect-only", "-q"], root
    )
    if collection["exit_code"] != 0:
        raise ValueError("full pytest collection failed; shard launch prohibited")
    nodes = [
        line.strip()
        for line in (output / "collection/stdout.log").read_text(encoding="utf-8").splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    shards = partition_nodes(nodes, workers)
    plan = {"collected_nodes": nodes, "shards": shards, "collected_count": len(nodes)}
    plan["record_sha256"] = canonical_json_sha256(plan)
    _write_json_create_only(output / "collection_plan.json", plan)
    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"--junitxml={output / f'shard_{index:02d}.xml'}",
            *shard,
        ]
        for index, shard in enumerate(shards)
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(record_command, output / f"shard_{index:02d}", command, root)
            for index, command in enumerate(commands)
        ]
        records = [future.result() for future in futures]
    reports: list[dict[str, Any]] = []
    for index, shard in enumerate(shards):
        path = output / f"shard_{index:02d}.xml"
        if not path.is_file():
            reports.append({"expected_tests": len(shard), "status": "MISSING_REPORT"})
            continue
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        counts = {
            key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        reports.append({**counts, "expected_tests": len(shard), "sha256": sha256_file(path)})
    passed = all(record["exit_code"] == 0 for record in records) and all(
        report.get("tests") == report["expected_tests"]
        and report.get("failures") == 0
        and report.get("errors") == 0
        for report in reports
    )
    summary = {
        "status": "PASS" if passed else "FAILED_PRESERVED",
        "collected_test_count": len(nodes),
        "reported_test_count": sum(int(report.get("tests", 0)) for report in reports),
        "no_collected_tests_omitted_or_duplicated": sum(len(shard) for shard in shards)
        == len(nodes),
        "plan_sha256": plan["record_sha256"],
        "shard_reports": reports,
    }
    summary["record_sha256"] = canonical_json_sha256(summary)
    _write_json_create_only(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    result = run_full_suite(args.output.resolve(), args.repository_root.resolve(), args.workers)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
