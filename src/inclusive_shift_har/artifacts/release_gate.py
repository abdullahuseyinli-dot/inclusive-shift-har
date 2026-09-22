"""Git-object release security scans and exact-candidate attestations.

The scanners in this module never walk raw-data directories. Repository content
is read through Git object IDs, and only blobs whose paths already pass the
forbidden-path/extension policy are inspected for disguised binary payloads.
Every generated record is create-only and canonically self-hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
)

SCHEMA_VERSION = "1.0.0"
REPORT_KIND = "final_release_gate_report"
SCAN_KIND = "git_object_release_scan"
INDEX_SCAN_KIND = "git_index_release_scan"
SECRET_KIND = "gitleaks_secret_scan_attestation"
LICENSE_KIND = "python_license_audit"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
TAG_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
PACKAGE_IDENTITY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*@[a-z0-9][a-z0-9.+_-]*$")
LICENSE_EXCEPTION_SCOPE = (
    "transitive_linux_xgboost_gpu_runtime_installed_from_pypi_not_redistributed"
)
LICENSE_REVIEW_KIND = "dependency_license_exception_review"
LICENSE_REVIEW_STATUS = "approved_for_nonredistributed_transitive_runtime_only"
NCCL_EXCEPTION_IDENTITY = "nvidia-nccl-cu12@2.31.2"
NCCL_EXCEPTION_REQUIRED_BY = "xgboost@3.2.0"
NCCL_EXCEPTION_LICENSE = "LicenseRef-NVIDIA-Proprietary"
NCCL_EXCEPTION_MARKER = (
    "sys_platform == 'linux' or (extra == 'extra-19-inclusive-shift-har-training-cpu' "
    "and extra == 'extra-19-inclusive-shift-har-training-cuda')"
)
PROHIBITED_LICENSE_TOKENS_V1 = (
    "affero general public license",
    "agpl",
    "commercial",
    "general public license",
    "gpl",
    "proprietary",
    "unknown",
)
NCCL_EXCEPTION_V1_POLICY_VALUES: dict[str, str | int] = {
    "license_expression": NCCL_EXCEPTION_LICENSE,
    "required_by": NCCL_EXCEPTION_REQUIRED_BY,
    "dependency_marker": NCCL_EXCEPTION_MARKER,
    "platform_system": "Linux",
    "platform_machine": "x86_64",
    "scope": LICENSE_EXCEPTION_SCOPE,
    "lock_path": "uv.lock",
    "review_path": "docs/release/NVIDIA_NCCL_CU12_2_31_2_REVIEW.json",
    "wheel_filename": "nvidia_nccl_cu12-2.31.2-py3-none-manylinux_2_18_x86_64.whl",
    "wheel_url": (
        "https://files.pythonhosted.org/packages/0f/36/"
        "104de52d6368f5b7f886e8fd252e0a438fe73a215e59b1b47f93a80ae2ea/"
        "nvidia_nccl_cu12-2.31.2-py3-none-manylinux_2_18_x86_64.whl"
    ),
    "wheel_size_bytes": 342105414,
    "wheel_sha256": "f9b1dc3c2a7e20176054144ebb3b32fea83b40402ee5d7ac7045cd11ecc956c0",
    "metadata_url": (
        "https://files.pythonhosted.org/packages/0f/36/"
        "104de52d6368f5b7f886e8fd252e0a438fe73a215e59b1b47f93a80ae2ea/"
        "nvidia_nccl_cu12-2.31.2-py3-none-manylinux_2_18_x86_64.whl.metadata"
    ),
    "metadata_size_bytes": 2097,
    "metadata_sha256": "ac64882e612ff2e1c4674386c2a662e0e0b6cae12e85ebe789b1608fa7c3a5fd",
    "embedded_license_member": "nvidia_nccl_cu12-2.31.2.dist-info/licenses/License.txt",
    "embedded_license_size_bytes": 1895,
    "embedded_license_sha256": ("0f0174a6b4e0b33ac26375bf729533075b32f4c51a5b6802a3d742d7dcdc9a76"),
    "terms_sla_url": (
        "https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/sla/index.html"
    ),
    "terms_sla_size_bytes": 95492,
    "terms_sla_sha256": "77e3db074f479b97051023bb8d9c06e1212a88375827f948eb869c73873e9052",
    "terms_bsd_url": (
        "https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/bsd/index.html"
    ),
    "terms_bsd_size_bytes": 23224,
    "terms_bsd_sha256": "d6e509a646b74aa552a17d1e864ca2d00d1cf34d975be6567a40431d4f0ec297",
}
NCCL_EXCEPTION_V1_LOCK_PACKAGE: dict[str, Any] = {
    "name": "nvidia-nccl-cu12",
    "version": "2.31.2",
    "source": {"registry": "https://pypi.org/simple"},
    "wheels": [
        {
            "url": (
                "https://files.pythonhosted.org/packages/37/85/"
                "b073e54c993cd9f79faa955d7c9bd7356da408935483aebc3cbb53a922ec/"
                "nvidia_nccl_cu12-2.31.2-py3-none-manylinux_2_18_aarch64.whl"
            ),
            "hash": "sha256:f208de397e431631eab0eca946444404a495d43a007535baa333d7de9e510ca2",
            "size": 342026203,
            "upload-time": "2026-08-11T23:23:40.509Z",
        },
        {
            "url": NCCL_EXCEPTION_V1_POLICY_VALUES["wheel_url"],
            "hash": f"sha256:{NCCL_EXCEPTION_V1_POLICY_VALUES['wheel_sha256']}",
            "size": NCCL_EXCEPTION_V1_POLICY_VALUES["wheel_size_bytes"],
            "upload-time": "2026-08-11T23:24:24.167Z",
        },
    ],
}
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_TAG_OBJECT_BYTES = 64 * 1024
TRACKED_GATE_NAMES = (
    "repository_scan",
    "secret_scan",
    "license_audit",
    "tests",
    "lint",
    "format",
    "types",
    "manifests",
    "splits",
    "configuration",
    "artifacts",
    "ci",
)
EXACT_GATE_NAMES = ("staged_index_scan", *TRACKED_GATE_NAMES[:-1], "ci_quality_proxy")
# Backwards-compatible public alias for the tracked precommit record contract.
GATE_NAMES = TRACKED_GATE_NAMES
TEXT_SUFFIXES = {
    "",
    ".bib",
    ".cff",
    ".csv",
    ".gitignore",
    ".gitleaksignore",
    ".in",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".svg",
    ".toml",
    ".txt",
    ".typed",
    ".xml",
    ".yaml",
    ".yml",
}
QUALITY_GATE_NAMES = frozenset(
    {
        "tests",
        "lint",
        "format",
        "types",
        "manifests",
        "splits",
        "configuration",
        "artifacts",
        "ci_quality_proxy",
    }
)
QUALITY_EVIDENCE_KIND = "ci_release_quality_gate_evidence"
QUALITY_EVIDENCE_SCOPE = "within_job_quality_proxy_not_completed_workflow"
SECRET_SCAN_SCOPE = "complete_commit_history_with_policy_pinned_tag_metadata"


class ReleaseGateError(RuntimeError):
    """Raised when release evidence is unsafe, incomplete, or inconsistent."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ReleaseGateError(f"{name} must be a string-keyed object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ReleaseGateError(
            f"{name} keys differ; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _exact_typed_structure(value: Any, expected: Any) -> bool:
    """Compare nested release-policy values without Python's bool/int/float coercions."""

    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(
            _exact_typed_structure(value[key], expected_value)
            for key, expected_value in expected.items()
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _exact_typed_structure(observed, pinned)
            for observed, pinned in zip(value, expected, strict=True)
        )
    return bool(value == expected)


def _array(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReleaseGateError(f"{name} must be an array")
    return value


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseGateError(f"{name} must be a non-empty string")
    return value


def _canonical_package_name(value: Any, *, name: str) -> str:
    """Return the PEP 503 comparison form for a validated package name."""

    return re.sub(r"[-_.]+", "-", _text(value, name=name)).casefold()


def _commit(value: Any, *, name: str = "commit") -> str:
    result = _text(value, name=name)
    if COMMIT_RE.fullmatch(result) is None:
        raise ReleaseGateError(f"{name} must be a full lowercase Git commit")
    return result


def _sha256(value: Any, *, name: str) -> str:
    result = _text(value, name=name)
    if SHA256_RE.fullmatch(result) is None:
        raise ReleaseGateError(f"{name} must be a lowercase SHA-256")
    return result


def license_exception_policy(policy: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return exact, candidate-verifiable dependency-licence exception records."""

    identities: list[str] = []
    for index, raw_identity in enumerate(
        _array(policy.get("license_exceptions"), name="license exceptions")
    ):
        identity = _text(raw_identity, name=f"license exception[{index}]")
        if identity != identity.casefold() or PACKAGE_IDENTITY_RE.fullmatch(identity) is None:
            raise ReleaseGateError(
                "license exceptions must be canonical lowercase package@version IDs"
            )
        identities.append(identity)
    if len(identities) != len(set(identities)):
        raise ReleaseGateError("license exceptions contain duplicates")
    if identities != [NCCL_EXCEPTION_IDENTITY]:
        raise ReleaseGateError("release policy v1 permits only the exact reviewed NCCL exception")

    raw_records = policy.get("license_exception_records")
    if raw_records is None:
        if identities:
            raise ReleaseGateError("every license exception requires a documented policy record")
        return {}
    records: dict[str, dict[str, Any]] = {}
    for raw_identity, raw_record in _mapping(raw_records, name="license exception records").items():
        identity = _text(raw_identity, name="license exception record identity")
        if identity != identity.casefold() or PACKAGE_IDENTITY_RE.fullmatch(identity) is None:
            raise ReleaseGateError(
                "license exception record keys must be canonical lowercase package@version IDs"
            )
        record = _mapping(raw_record, name=f"license exception record {identity}")
        _exact_keys(
            record,
            {
                "license_expression",
                "required_by",
                "dependency_marker",
                "platform_system",
                "platform_machine",
                "scope",
                "lock_path",
                "lock_sha256",
                "review_path",
                "review_sha256",
                "review_record_sha256",
                "wheel_filename",
                "wheel_url",
                "wheel_size_bytes",
                "wheel_sha256",
                "metadata_url",
                "metadata_size_bytes",
                "metadata_sha256",
                "embedded_license_member",
                "embedded_license_size_bytes",
                "embedded_license_sha256",
                "terms_sla_url",
                "terms_sla_size_bytes",
                "terms_sla_sha256",
                "terms_bsd_url",
                "terms_bsd_size_bytes",
                "terms_bsd_sha256",
            },
            name=f"license exception record {identity}",
        )
        required_by = _text(
            record.get("required_by"), name=f"license exception record {identity} required_by"
        )
        dependency_marker = _text(
            record.get("dependency_marker"),
            name=f"license exception record {identity} dependency_marker",
        )
        platform_system = _text(
            record.get("platform_system"),
            name=f"license exception record {identity} platform_system",
        )
        platform_machine = _text(
            record.get("platform_machine"),
            name=f"license exception record {identity} platform_machine",
        )
        scope = _text(record.get("scope"), name=f"license exception record {identity} scope")
        lock_path = _safe_relative(
            record.get("lock_path"), name=f"license exception record {identity} lock_path"
        )
        review_path = _safe_relative(
            record.get("review_path"), name=f"license exception record {identity} review_path"
        )
        if (
            required_by != required_by.casefold()
            or PACKAGE_IDENTITY_RE.fullmatch(required_by) is None
            or platform_system != "Linux"
            or platform_machine != "x86_64"
            or scope != LICENSE_EXCEPTION_SCOPE
            or lock_path != "uv.lock"
            or not review_path.startswith("docs/release/")
            or any(
                ord(character) < 32 or ord(character) == 127
                for value in (dependency_marker, scope)
                for character in value
            )
        ):
            raise ReleaseGateError(f"license exception record {identity} is not narrowly scoped")
        records[identity] = {
            "license_expression": _text(
                record.get("license_expression"),
                name=f"license exception record {identity} license_expression",
            ),
            "required_by": required_by,
            "dependency_marker": dependency_marker,
            "platform_system": platform_system,
            "platform_machine": platform_machine,
            "scope": scope,
            "lock_path": lock_path,
            "lock_sha256": _sha256(
                record.get("lock_sha256"),
                name=f"license exception record {identity} lock_sha256",
            ),
            "review_path": review_path,
            "review_sha256": _sha256(
                record.get("review_sha256"),
                name=f"license exception record {identity} review_sha256",
            ),
            "review_record_sha256": _sha256(
                record.get("review_record_sha256"),
                name=f"license exception record {identity} review_record_sha256",
            ),
            "wheel_filename": _text(
                record.get("wheel_filename"),
                name=f"license exception record {identity} wheel_filename",
            ),
            "wheel_url": _text(
                record.get("wheel_url"),
                name=f"license exception record {identity} wheel_url",
            ),
            "wheel_size_bytes": _exact_integer(
                record.get("wheel_size_bytes"),
                name=f"license exception record {identity} wheel_size_bytes",
                minimum=1,
            ),
            "wheel_sha256": _sha256(
                record.get("wheel_sha256"),
                name=f"license exception record {identity} wheel_sha256",
            ),
            "metadata_url": _text(
                record.get("metadata_url"),
                name=f"license exception record {identity} metadata_url",
            ),
            "metadata_size_bytes": _exact_integer(
                record.get("metadata_size_bytes"),
                name=f"license exception record {identity} metadata_size_bytes",
                minimum=1,
            ),
            "metadata_sha256": _sha256(
                record.get("metadata_sha256"),
                name=f"license exception record {identity} metadata_sha256",
            ),
            "embedded_license_member": _safe_relative(
                record.get("embedded_license_member"),
                name=f"license exception record {identity} embedded_license_member",
            ),
            "embedded_license_size_bytes": _exact_integer(
                record.get("embedded_license_size_bytes"),
                name=f"license exception record {identity} embedded_license_size_bytes",
                minimum=1,
            ),
            "embedded_license_sha256": _sha256(
                record.get("embedded_license_sha256"),
                name=f"license exception record {identity} embedded_license_sha256",
            ),
            "terms_sla_url": _text(
                record.get("terms_sla_url"),
                name=f"license exception record {identity} terms_sla_url",
            ),
            "terms_sla_size_bytes": _exact_integer(
                record.get("terms_sla_size_bytes"),
                name=f"license exception record {identity} terms_sla_size_bytes",
                minimum=1,
            ),
            "terms_sla_sha256": _sha256(
                record.get("terms_sla_sha256"),
                name=f"license exception record {identity} terms_sla_sha256",
            ),
            "terms_bsd_url": _text(
                record.get("terms_bsd_url"),
                name=f"license exception record {identity} terms_bsd_url",
            ),
            "terms_bsd_size_bytes": _exact_integer(
                record.get("terms_bsd_size_bytes"),
                name=f"license exception record {identity} terms_bsd_size_bytes",
                minimum=1,
            ),
            "terms_bsd_sha256": _sha256(
                record.get("terms_bsd_sha256"),
                name=f"license exception record {identity} terms_bsd_sha256",
            ),
        }
        bound = records[identity]
        if (
            identity != NCCL_EXCEPTION_IDENTITY
            or any(
                bound.get(key) != expected
                for key, expected in NCCL_EXCEPTION_V1_POLICY_VALUES.items()
            )
            or not str(bound["wheel_url"]).startswith("https://files.pythonhosted.org/packages/")
            or str(bound["wheel_url"]).rsplit("/", 1)[-1] != bound["wheel_filename"]
            or bound["metadata_url"] != f"{bound['wheel_url']}.metadata"
            or bound["terms_sla_url"]
            != "https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/sla/index.html"
            or bound["terms_bsd_url"]
            != "https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/bsd/index.html"
        ):
            raise ReleaseGateError(
                f"license exception record {identity} does not match the exact reviewed NCCL exception"
            )
    if set(records) != set(identities):
        raise ReleaseGateError("license exception IDs and documented records differ")
    return records


def _candidate_blob_with_sha256(
    root: Path, candidate: str, *, path: str, digest: str, name: str
) -> bytes:
    return _git_blob_payload(
        root,
        candidate,
        {"path": path, "size_bytes": 0, "file_sha256": digest},
        name=name,
    )


def _validated_license_review(
    value: Mapping[str, Any], *, expected_identity: str, exception: Mapping[str, Any]
) -> dict[str, Any]:
    _exact_keys(
        value,
        {
            "schema_version",
            "record_kind",
            "status",
            "review_date",
            "retrieval_date",
            "identity",
            "package",
            "dependency",
            "platform",
            "scope",
            "distribution",
            "official_terms",
            "review_findings",
            "controls",
            "verification_method",
            "record_sha256",
        },
        name="dependency licence exception review",
    )
    record_sha256 = _verify_self_hash(value, name="dependency licence exception review")
    package = _mapping(value.get("package"), name="licence review package")
    dependency = _mapping(value.get("dependency"), name="licence review dependency")
    reviewed_platform = _mapping(value.get("platform"), name="licence review platform")
    distribution = _mapping(value.get("distribution"), name="licence review distribution")
    wheel = _mapping(distribution.get("wheel"), name="licence review wheel")
    metadata = _mapping(distribution.get("core_metadata"), name="licence review metadata")
    embedded = _mapping(
        distribution.get("embedded_license"), name="licence review embedded licence"
    )
    terms = _mapping(value.get("official_terms"), name="licence review official terms")
    sla = _mapping(terms.get("nvidia_software_license_agreement"), name="licence review NVIDIA SLA")
    bsd = _mapping(terms.get("nvidia_bsd_license"), name="licence review NVIDIA BSD terms")
    controls = _mapping(value.get("controls"), name="licence review controls")
    method = _mapping(value.get("verification_method"), name="licence review method")
    for item, expected_keys, item_name in (
        (
            package,
            {"name", "version", "source_registry", "metadata_license_expression"},
            "licence review package",
        ),
        (dependency, {"required_by", "marker"}, "licence review dependency"),
        (reviewed_platform, {"system", "machine"}, "licence review platform"),
        (wheel, {"filename", "url", "size_bytes", "sha256"}, "licence review wheel"),
        (
            metadata,
            {"url", "size_bytes", "sha256", "license_expression"},
            "licence review metadata",
        ),
        (
            embedded,
            {"member", "size_bytes", "sha256", "classification"},
            "licence review embedded licence",
        ),
        (
            sla,
            {"url", "size_bytes", "sha256", "last_modified", "rendered_content_version"},
            "licence review NVIDIA SLA",
        ),
        (
            bsd,
            {"url", "size_bytes", "sha256", "last_modified", "rendered_content_version"},
            "licence review NVIDIA BSD terms",
        ),
    ):
        _exact_keys(item, expected_keys, name=item_name)
    _exact_keys(
        distribution,
        {"wheel", "core_metadata", "embedded_license", "wheel_member_count"},
        name="licence review distribution",
    )
    _exact_keys(
        terms,
        {"nvidia_software_license_agreement", "nvidia_bsd_license"},
        name="licence review official terms",
    )
    expected_controls = {
        "repository_redistributes_wheel": False,
        "release_assets_redistribute_wheel": False,
        "container_images_may_vendor_wheel": False,
        "third_party_terms_apply_to_local_installation": True,
        "exception_changes_repository_code_license": False,
    }
    _exact_keys(controls, set(expected_controls), name="licence review controls")
    _exact_keys(
        method,
        {"wheel", "core_metadata", "embedded_license", "official_terms"},
        name="licence review method",
    )
    identity = _text(value.get("identity"), name="licence review identity")
    package_name, package_version = identity.rsplit("@", 1)
    wheel_filename = _text(wheel.get("filename"), name="licence review wheel filename")
    wheel_url = _text(wheel.get("url"), name="licence review wheel URL")
    metadata_url = _text(metadata.get("url"), name="licence review metadata URL")
    sla_url = _text(sla.get("url"), name="licence review NVIDIA SLA URL")
    bsd_url = _text(bsd.get("url"), name="licence review NVIDIA BSD URL")
    findings = [
        _text(item, name=f"licence review finding[{index}]")
        for index, item in enumerate(
            _array(value.get("review_findings"), name="licence review findings")
        )
    ]
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("record_kind") != LICENSE_REVIEW_KIND
        or value.get("status") != LICENSE_REVIEW_STATUS
        or identity != expected_identity
        or identity != f"{package_name}@{package_version}".casefold()
        or package.get("name") != package_name
        or package.get("version") != package_version
        or package.get("source_registry") != "https://pypi.org/simple"
        or package.get("metadata_license_expression") != exception["license_expression"]
        or dependency.get("required_by") != exception["required_by"]
        or dependency.get("marker") != exception["dependency_marker"]
        or reviewed_platform.get("system") != exception["platform_system"]
        or reviewed_platform.get("machine") != exception["platform_machine"]
        or value.get("scope") != exception["scope"]
        or controls != expected_controls
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value.get("review_date")))
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value.get("retrieval_date")))
        or len(findings) != 4
        or any(not isinstance(item, str) or not item for item in method.values())
        or wheel_filename != exception["wheel_filename"]
        or not wheel_url.startswith("https://files.pythonhosted.org/packages/")
        or wheel_url.rsplit("/", 1)[-1] != wheel_filename
        or wheel_url != exception["wheel_url"]
        or wheel.get("size_bytes") != exception["wheel_size_bytes"]
        or wheel.get("sha256") != exception["wheel_sha256"]
        or metadata_url != f"{wheel_url}.metadata"
        or metadata_url != exception["metadata_url"]
        or metadata.get("size_bytes") != exception["metadata_size_bytes"]
        or metadata.get("sha256") != exception["metadata_sha256"]
        or metadata.get("license_expression") != exception["license_expression"]
        or embedded.get("member") != exception["embedded_license_member"]
        or embedded.get("size_bytes") != exception["embedded_license_size_bytes"]
        or embedded.get("sha256") != exception["embedded_license_sha256"]
        or embedded.get("classification") != "BSD-3-Clause-text"
        or sla_url != exception["terms_sla_url"]
        or sla.get("size_bytes") != exception["terms_sla_size_bytes"]
        or sla.get("sha256") != exception["terms_sla_sha256"]
        or bsd_url != exception["terms_bsd_url"]
        or bsd.get("size_bytes") != exception["terms_bsd_size_bytes"]
        or bsd.get("sha256") != exception["terms_bsd_sha256"]
        or sla.get("rendered_content_version") != "2.29.2"
        or bsd.get("rendered_content_version") != "2.29.2"
        or "/archives/nccl_2312/" not in sla_url
        or "/archives/nccl_2312/" not in bsd_url
        or any(controls.get(key) is not expected for key, expected in expected_controls.items())
    ):
        raise ReleaseGateError("dependency licence review scope or provenance differs")
    for item_name, item in (
        ("wheel", wheel),
        ("metadata", metadata),
        ("embedded licence", embedded),
        ("NVIDIA SLA", sla),
        ("NVIDIA BSD terms", bsd),
    ):
        _exact_integer(item.get("size_bytes"), name=f"licence review {item_name} size", minimum=1)
        _sha256(item.get("sha256"), name=f"licence review {item_name} SHA-256")
    _exact_integer(
        distribution.get("wheel_member_count"),
        name="licence review wheel member count",
        expected=75,
    )
    return {
        "review_record_sha256": record_sha256,
        "wheel_filename": wheel_filename,
        "wheel_url": wheel_url,
        "wheel_size_bytes": wheel["size_bytes"],
        "wheel_sha256": wheel["sha256"],
        "metadata_sha256": metadata["sha256"],
        "embedded_license_sha256": embedded["sha256"],
        "terms_sla_sha256": sla["sha256"],
        "terms_bsd_sha256": bsd["sha256"],
    }


def validate_license_exception_candidate_binding(
    root: Path, candidate: str, policy: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Bind every exception to exact candidate Git blobs and one uv resolution edge."""

    bindings: dict[str, dict[str, Any]] = {}
    for identity, exception in license_exception_policy(policy).items():
        lock_payload = _candidate_blob_with_sha256(
            root,
            candidate,
            path=exception["lock_path"],
            digest=exception["lock_sha256"],
            name=f"license exception {identity} lockfile",
        )
        try:
            lock = tomllib.loads(lock_payload.decode("utf-8"))
        except (UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise ReleaseGateError("candidate uv.lock is not canonical UTF-8 TOML") from exc
        raw_packages = _array(lock.get("package"), name="candidate uv.lock packages")
        package_name, package_version = identity.rsplit("@", 1)
        required_name, required_version = exception["required_by"].rsplit("@", 1)
        locked_package_matches: list[Mapping[str, Any]] = []
        parent_matches: list[Mapping[str, Any]] = []
        inbound_edges: list[dict[str, Any]] = []
        canonical_package_name = _canonical_package_name(
            package_name, name="license exception package name"
        )
        for index, raw_package in enumerate(raw_packages):
            locked_package = _mapping(raw_package, name=f"candidate uv.lock package[{index}]")
            owner_name = _text(
                locked_package.get("name"), name=f"candidate uv.lock package[{index}] name"
            )
            owner_version = _text(
                locked_package.get("version"),
                name=f"candidate uv.lock package[{index}] version",
            )
            if owner_name == package_name and owner_version == package_version:
                locked_package_matches.append(locked_package)
            if owner_name == required_name and owner_version == required_version:
                parent_matches.append(locked_package)
            for section in ("dependencies", "optional-dependencies", "dev-dependencies"):
                raw_container = locked_package.get(section)
                if raw_container is None:
                    continue
                groups: list[tuple[str | None, list[Any]]]
                if section == "dependencies":
                    groups = [
                        (
                            None,
                            _array(
                                raw_container,
                                name=f"candidate uv.lock package[{index}] {section}",
                            ),
                        )
                    ]
                else:
                    grouped = _mapping(
                        raw_container,
                        name=f"candidate uv.lock package[{index}] {section}",
                    )
                    groups = [
                        (
                            _text(group, name=f"candidate uv.lock package[{index}] group"),
                            _array(
                                values,
                                name=f"candidate uv.lock package[{index}] {section}.{group}",
                            ),
                        )
                        for group, values in grouped.items()
                    ]
                for group, raw_edges in groups:
                    for edge_index, raw_edge in enumerate(raw_edges):
                        edge = _mapping(
                            raw_edge,
                            name=(
                                f"candidate uv.lock package[{index}] {section} edge[{edge_index}]"
                            ),
                        )
                        edge_name = _canonical_package_name(
                            edge.get("name"),
                            name=(
                                f"candidate uv.lock package[{index}] {section}"
                                f" edge[{edge_index}] name"
                            ),
                        )
                        if edge_name == canonical_package_name:
                            inbound_edges.append(
                                {
                                    "owner_name": owner_name,
                                    "owner_version": owner_version,
                                    "section": section,
                                    "group": group,
                                    "edge": edge,
                                }
                            )
        if len(locked_package_matches) != 1 or len(parent_matches) != 1:
            raise ReleaseGateError(
                "license exception package or required parent is not uniquely locked"
            )
        locked_package = locked_package_matches[0]
        if not _exact_typed_structure(locked_package, NCCL_EXCEPTION_V1_LOCK_PACKAGE):
            raise ReleaseGateError(
                "candidate uv.lock exception package differs from the exact reviewed package stanza"
            )
        source = _mapping(locked_package.get("source"), name="exception package lock source")
        _exact_keys(source, {"registry"}, name="exception package lock source")
        review_payload = _candidate_blob_with_sha256(
            root,
            candidate,
            path=exception["review_path"],
            digest=exception["review_sha256"],
            name=f"license exception {identity} review",
        )
        review = _strict_json_bytes(review_payload, name=f"license exception {identity} review")
        reviewed = _validated_license_review(
            review,
            expected_identity=identity,
            exception=exception,
        )
        if reviewed["review_record_sha256"] != exception["review_record_sha256"]:
            raise ReleaseGateError("license exception review self-hash differs from policy")
        matching_wheels = []
        for raw_wheel in _array(locked_package.get("wheels"), name="exception package wheels"):
            wheel = _mapping(raw_wheel, name="exception package wheel")
            if wheel.get("url") == reviewed["wheel_url"]:
                matching_wheels.append(wheel)
        if len(matching_wheels) != 1:
            raise ReleaseGateError("reviewed exception wheel is not uniquely locked")
        locked_wheel = matching_wheels[0]
        parent = parent_matches[0]
        parent_source = _mapping(parent.get("source"), name="required parent lock source")
        _exact_keys(parent_source, {"registry"}, name="required parent lock source")
        expected_edge = {"name": package_name, "marker": exception["dependency_marker"]}
        if (
            source != {"registry": "https://pypi.org/simple"}
            or parent_source != {"registry": "https://pypi.org/simple"}
            or len(inbound_edges) != 1
            or inbound_edges[0]["owner_name"] != required_name
            or inbound_edges[0]["owner_version"] != required_version
            or inbound_edges[0]["section"] != "dependencies"
            or inbound_edges[0]["group"] is not None
            or not _exact_typed_structure(inbound_edges[0]["edge"], expected_edge)
            or locked_wheel.get("hash") != f"sha256:{reviewed['wheel_sha256']}"
            or locked_wheel.get("size") != reviewed["wheel_size_bytes"]
            or reviewed["wheel_filename"] != reviewed["wheel_url"].rsplit("/", 1)[-1]
        ):
            raise ReleaseGateError("candidate uv.lock differs from the reviewed exception edge")
        bindings[identity] = {**exception, **reviewed}
    return bindings


LICENSE_APPLICATION_KEYS = {
    "identity",
    "package",
    "version",
    "license",
    "required_by",
    "dependency_marker",
    "platform_system",
    "platform_machine",
    "scope",
    "lock_path",
    "lock_sha256",
    "review_path",
    "review_sha256",
    "review_record_sha256",
    "wheel_filename",
    "wheel_url",
    "wheel_size_bytes",
    "wheel_sha256",
    "metadata_url",
    "metadata_size_bytes",
    "metadata_sha256",
    "embedded_license_member",
    "embedded_license_size_bytes",
    "embedded_license_sha256",
    "terms_sla_url",
    "terms_sla_size_bytes",
    "terms_sla_sha256",
    "terms_bsd_url",
    "terms_bsd_size_bytes",
    "terms_bsd_sha256",
}


def _license_exception_application(
    package: Mapping[str, str], binding: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "identity": package["identity"],
        "package": package["package"],
        "version": package["version"],
        "license": package["license"],
        **{key: binding[key] for key in LICENSE_APPLICATION_KEYS - set(package)},
    }


def validate_license_exception_applications(
    value: Any,
    *,
    policy: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate full exception provenance; empty or fabricated applications fail closed."""

    records = license_exception_policy(policy)
    applications: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_application in enumerate(_array(value, name="license exceptions applied")):
        application = _mapping(raw_application, name=f"license exceptions applied[{index}]")
        _exact_keys(
            application,
            LICENSE_APPLICATION_KEYS,
            name=f"license exceptions applied[{index}]",
        )
        identity = _text(
            application.get("identity"), name=f"license exceptions applied[{index}].identity"
        )
        package = _text(
            application.get("package"), name=f"license exceptions applied[{index}].package"
        )
        version = _text(
            application.get("version"), name=f"license exceptions applied[{index}].version"
        )
        license_name = _text(
            application.get("license"), name=f"license exceptions applied[{index}].license"
        )
        record = records.get(identity)
        binding = bindings.get(identity)
        expected = (
            _license_exception_application(
                {
                    "identity": identity,
                    "package": package,
                    "version": version,
                    "license": license_name,
                },
                binding,
            )
            if binding is not None
            else None
        )
        if (
            identity != f"{package}@{version}".casefold()
            or identity in seen
            or record is None
            or binding is None
            or license_name != record["license_expression"]
            or dict(application) != expected
        ):
            raise ReleaseGateError("applied license exception differs from candidate provenance")
        seen.add(identity)
        applications.append(dict(application))
    if applications != sorted(applications, key=lambda item: str(item["identity"])):
        raise ReleaseGateError("applied license exceptions are not canonically ordered")
    return applications


def _normalized_license_packages(value: Any) -> list[dict[str, str]]:
    packages: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw_package in enumerate(_array(value, name="normalised licence packages")):
        package = _mapping(raw_package, name=f"normalised licence package[{index}]")
        _exact_keys(
            package,
            {"identity", "package", "version", "license"},
            name=f"normalised licence package[{index}]",
        )
        name = _text(package.get("package"), name=f"normalised package[{index}].package")
        version = _text(package.get("version"), name=f"normalised package[{index}].version")
        license_name = _text(package.get("license"), name=f"normalised package[{index}].license")
        identity = _text(package.get("identity"), name=f"normalised package[{index}].identity")
        expected_identity = f"{name}@{version}".casefold()
        if (
            identity != expected_identity
            or PACKAGE_IDENTITY_RE.fullmatch(identity) is None
            or identity in seen
            or any(
                character in "\t\r\n" or ord(character) < 32 or ord(character) == 127
                for field in (name, version, license_name)
                for character in field
            )
        ):
            raise ReleaseGateError("normalised licence package identity is invalid or duplicated")
        seen.add(identity)
        packages.append(
            {"identity": identity, "package": name, "version": version, "license": license_name}
        )
    if packages != sorted(packages, key=lambda item: item["identity"]):
        raise ReleaseGateError("normalised licence packages are not canonically ordered")
    return packages


def _license_inventory_digest(packages: Sequence[Mapping[str, str]]) -> str:
    rows = [f"{item['package']}\t{item['version']}\t{item['license']}" for item in packages]
    return hashlib.sha256(("\n".join(sorted(rows, key=str.casefold)) + "\n").encode()).hexdigest()


def _license_outcomes(
    packages: Sequence[Mapping[str, str]],
    *,
    environment: Mapping[str, str],
    policy: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    prohibited = tuple(
        _text(value, name="prohibited license token").casefold()
        for value in _array(policy.get("prohibited_license_tokens"), name="license tokens")
    )
    if prohibited != PROHIBITED_LICENSE_TOKENS_V1:
        raise ReleaseGateError(
            "release policy v1 prohibited licence tokens do not match the exact reviewed set"
        )
    by_identity = {item["identity"]: item for item in packages}
    for identity, binding in bindings.items():
        platform_matches = (
            environment["system"] == binding["platform_system"]
            and environment["machine"] == binding["platform_machine"]
        )
        if not platform_matches:
            continue
        package_present = identity in by_identity
        parent_present = binding["required_by"] in by_identity
        if not package_present or not parent_present:
            raise ReleaseGateError("exception package and reviewed parent must both be inventoried")
        if package_present and by_identity[identity]["license"] != binding["license_expression"]:
            raise ReleaseGateError("exception package licence differs from its reviewed metadata")

    applications: list[dict[str, Any]] = []
    violations: list[dict[str, str]] = []
    for package in packages:
        license_name = package["license"]
        if not any(token in license_name.casefold() for token in prohibited):
            continue
        candidate_binding = bindings.get(package["identity"])
        if (
            candidate_binding is not None
            and environment["system"] == candidate_binding["platform_system"]
            and environment["machine"] == candidate_binding["platform_machine"]
            and license_name == candidate_binding["license_expression"]
            and candidate_binding["required_by"] in by_identity
        ):
            applications.append(_license_exception_application(package, candidate_binding))
        else:
            violations.append(
                {
                    "package": package["package"],
                    "version": package["version"],
                    "license": license_name,
                }
            )
    return (
        sorted(applications, key=lambda item: str(item["identity"])),
        sorted(violations, key=lambda item: (item["package"].casefold(), item["version"])),
    )


def validate_license_audit_semantics(
    value: Mapping[str, Any],
    *,
    root: Path,
    candidate: str,
    policy: Mapping[str, Any],
) -> None:
    """Reconstruct a licence audit from its complete normalised package inventory."""

    environment = _mapping(value.get("audit_environment"), name="licence audit environment")
    _exact_keys(environment, {"system", "machine"}, name="licence audit environment")
    normalized_environment = {
        "system": _text(environment.get("system"), name="licence audit environment system"),
        "machine": _text(environment.get("machine"), name="licence audit environment machine"),
    }
    packages = _normalized_license_packages(value.get("packages"))
    bindings = validate_license_exception_candidate_binding(root, candidate, policy)
    expected_applications, expected_violations = _license_outcomes(
        packages,
        environment=normalized_environment,
        policy=policy,
        bindings=bindings,
    )
    observed_applications = validate_license_exception_applications(
        value.get("exceptions_applied"), policy=policy, bindings=bindings
    )
    observed_violations: list[dict[str, str]] = []
    for index, raw_violation in enumerate(
        _array(value.get("violations"), name="licence audit violations")
    ):
        violation = _mapping(raw_violation, name=f"licence audit violation[{index}]")
        _exact_keys(
            violation,
            {"package", "version", "license"},
            name=f"licence audit violation[{index}]",
        )
        observed_violations.append(
            {
                "package": _text(
                    violation.get("package"), name=f"licence audit violation[{index}].package"
                ),
                "version": _text(
                    violation.get("version"), name=f"licence audit violation[{index}].version"
                ),
                "license": _text(
                    violation.get("license"), name=f"licence audit violation[{index}].license"
                ),
            }
        )
    expected_status = "pass" if not expected_violations else "fail"
    if (
        value.get("package_count") != len(packages)
        or value.get("normalized_inventory_sha256") != _license_inventory_digest(packages)
        or observed_applications != expected_applications
        or observed_violations != expected_violations
        or value.get("status") != expected_status
    ):
        raise ReleaseGateError("licence audit cannot be reconstructed from its package inventory")


def _exact_integer(
    value: Any,
    *,
    name: str,
    expected: int | None = None,
    minimum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReleaseGateError(f"{name} must be an exact JSON integer")
    if expected is not None and value != expected:
        raise ReleaseGateError(f"{name} must equal {expected}")
    if minimum is not None and value < minimum:
        raise ReleaseGateError(f"{name} must be at least {minimum}")
    return value


def _timestamp(value: Any) -> str:
    result = _text(value, name="created_at_utc")
    if TIMESTAMP_RE.fullmatch(result) is None:
        raise ReleaseGateError("created_at_utc must use canonical ISO-8601 UTC Z form")
    try:
        parsed = datetime.fromisoformat(result[:-1] + "+00:00")
    except ValueError as exc:
        raise ReleaseGateError("created_at_utc must be valid ISO-8601") from exc
    if parsed.tzinfo != UTC:
        raise ReleaseGateError("created_at_utc must be UTC")
    return result


def _safe_relative(value: Any, *, name: str) -> str:
    result = _text(value, name=name)
    path = PurePosixPath(result)
    if (
        "\\" in result
        or ":" in result
        or any(ord(character) < 32 or ord(character) == 127 for character in result)
        or path.is_absolute()
        or path.as_posix() != result
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise ReleaseGateError(f"{name} must be a portable POSIX relative path")
    return result


def _git(
    root: Path, arguments: Sequence[str], *, binary: bool = False, check: bool = True
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", *arguments],
            cwd=root,
            check=check,
            capture_output=True,
            text=not binary,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseGateError(f"Git command failed: {' '.join(arguments)}") from exc


def _repository(path: str | Path) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise ReleaseGateError("repository_root may not be a symlink")
    root = source.resolve(strict=True)
    if not root.is_dir():
        raise ReleaseGateError("repository_root must be a directory")
    inside = _git(root, ("rev-parse", "--is-inside-work-tree"))
    assert isinstance(inside.stdout, str)
    if inside.stdout.strip() != "true":
        raise ReleaseGateError("repository_root is not a Git worktree")
    top_level = _git(root, ("rev-parse", "--show-toplevel"))
    assert isinstance(top_level.stdout, str)
    try:
        actual_root = Path(top_level.stdout.strip()).resolve(strict=True)
    except OSError as exc:
        raise ReleaseGateError("Git worktree root cannot be resolved") from exc
    if not root.samefile(actual_root):
        raise ReleaseGateError("repository_root must equal the Git worktree top-level")
    return root


def _require_candidate_head(root: Path, candidate: str, *, operation: str) -> None:
    head_result = _git(root, ("rev-parse", "HEAD"))
    assert isinstance(head_result.stdout, str)
    head = _commit(head_result.stdout.strip(), name="repository HEAD")
    if candidate != head:
        raise ReleaseGateError(f"{operation} candidate_commit differs from repository HEAD")


def _strict_json(path: str | Path, *, name: str) -> Mapping[str, Any]:
    try:
        return _mapping(load_json_strict(path), name=name)
    except (OSError, ValueError) as exc:
        raise ReleaseGateError(f"cannot load {name}: {exc}") from exc


def _strict_json_bytes(payload: bytes, *, name: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReleaseGateError(f"duplicate JSON key in {name}: {key!r}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> None:
        raise ReleaseGateError(f"non-finite JSON number in {name}: {token}")

    try:
        value = json.loads(
            payload.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_non_finite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseGateError(f"cannot load {name}: {exc}") from exc
    return _mapping(value, name=name)


def _policy(path: str | Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    source = Path(path)
    if source.is_symlink():
        raise ReleaseGateError("policy may not be a symlink")
    payload = source.resolve(strict=True).read_bytes()
    value = _strict_json_bytes(payload, name="release gate policy")
    if value.get("schema_version") != "1.0.0" or value.get("policy_kind") != (
        "final_release_gate_policy"
    ):
        raise ReleaseGateError("release gate policy version or kind differs")
    reference = {
        "path": "configs/release/release_gate_policy_v1.json",
        "size_bytes": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return value, reference


def _write_new(record: dict[str, Any], destination: str | Path, *, root: Path) -> None:
    raw = Path(destination)
    if any(part == ".." for part in raw.parts):
        raise ReleaseGateError("destination may not contain parent traversal")
    output = raw if raw.is_absolute() else root / raw
    try:
        relative = output.relative_to(root)
    except ValueError as exc:
        raise ReleaseGateError("destination escapes repository_root") from exc
    for component_count in range(1, len(relative.parts) + 1):
        component = root.joinpath(*relative.parts[:component_count])
        if component.is_symlink():
            raise ReleaseGateError("destination may not traverse a symlink")
    prospective = output.resolve(strict=False)
    try:
        prospective.relative_to(root)
    except ValueError as exc:
        raise ReleaseGateError("destination escapes repository_root") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    for component_count in range(1, len(relative.parts) + 1):
        component = root.joinpath(*relative.parts[:component_count])
        if component.is_symlink():
            raise ReleaseGateError("destination may not traverse a symlink")
    atomic_write_json_new(record, output, allowed_root=root)


def _self_hash(body: dict[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["record_sha256"] = canonical_json_sha256(result)
    return result


def _verify_self_hash(value: Mapping[str, Any], *, name: str) -> str:
    body = dict(value)
    observed = _sha256(body.pop("record_sha256", None), name=f"{name}.record_sha256")
    if canonical_json_sha256(body) != observed:
        raise ReleaseGateError(f"{name} self-hash does not reconstruct")
    return observed


def _tree_entries(root: Path, commit: str) -> list[tuple[str, str, str, int | None, str]]:
    result = _git(root, ("ls-tree", "-r", "-z", "-l", commit), binary=True)
    assert isinstance(result.stdout, bytes)
    entries: list[tuple[str, str, str, int | None, str]] = []
    for raw in (item for item in result.stdout.split(b"\x00") if item):
        try:
            metadata, raw_path = raw.split(b"\t", 1)
            mode, kind, object_id, size_text = metadata.decode("ascii").split(" ", 3)
            path = _safe_relative(raw_path.decode("utf-8"), name="Git tree path")
        except (UnicodeError, ValueError) as exc:
            raise ReleaseGateError("Git tree contains undecodable metadata or path") from exc
        size = None if size_text == "-" else int(size_text)
        entries.append((mode, kind, object_id, size, path))
    return entries


def _entry_index_row(
    *, mode: str, object_id: str, stage: int, kind: str, size: int | None, path: str
) -> str:
    return f"{mode} {object_id} {stage} {kind} {size} {path}"


def _candidate_tree_index(root: Path, commit: str) -> tuple[int, str]:
    rows = [
        _entry_index_row(
            mode=mode,
            object_id=object_id,
            stage=0,
            kind=kind,
            size=size,
            path=path,
        )
        for mode, kind, object_id, size, path in _tree_entries(root, commit)
    ]
    return len(rows), hashlib.sha256(("\n".join(sorted(rows)) + "\n").encode()).hexdigest()


def _content_scan_policy(
    policy: Mapping[str, Any],
) -> tuple[
    int,
    tuple[str, ...],
    set[str],
    dict[str, bytes],
    dict[str, set[str]],
    set[str],
]:
    maximum = policy.get("maximum_blob_size_bytes")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        raise ReleaseGateError("maximum_blob_size_bytes must be a positive integer")
    prefixes = tuple(
        _safe_relative(str(value).rstrip("/"), name="forbidden_path_prefix")
        for value in _array(policy.get("forbidden_path_prefixes"), name="forbidden prefixes")
    )
    suffixes = {
        _text(value, name="forbidden_suffix").casefold()
        for value in _array(policy.get("forbidden_suffixes"), name="forbidden suffixes")
    }
    signatures = {
        _text(name, name="signature name"): bytes.fromhex(_text(signature, name="signature"))
        for name, signature in _mapping(
            policy.get("disguised_binary_signatures"), name="binary signatures"
        ).items()
    }
    allowed_signatures = {
        str(name): {str(item).casefold() for item in _array(values, name="allowed suffixes")}
        for name, values in _mapping(
            policy.get("allowed_signature_suffixes"), name="allowed signatures"
        ).items()
    }
    allowed_binary_suffixes = {
        _text(value, name="allowed_binary_suffix").casefold()
        for value in _array(policy.get("allowed_binary_suffixes"), name="allowed binary suffixes")
    }
    if any(
        suffix
        not in {
            allowed_suffix
            for suffixes_for_signature in allowed_signatures.values()
            for allowed_suffix in suffixes_for_signature
        }
        for suffix in allowed_binary_suffixes
    ):
        raise ReleaseGateError("every allowed binary suffix must have an allowed magic signature")
    return maximum, prefixes, suffixes, signatures, allowed_signatures, allowed_binary_suffixes


def _is_forbidden_path(path: str, prefixes: Sequence[str]) -> bool:
    folded = path.casefold()
    return any(
        folded == prefix.casefold() or folded.startswith(f"{prefix.casefold()}/")
        for prefix in prefixes
    )


def _credential_path_policy(
    policy: Mapping[str, Any],
) -> tuple[set[str], tuple[str, ...], set[str]]:
    basenames = {
        _text(value, name="forbidden_basename").casefold()
        for value in _array(policy.get("forbidden_basenames"), name="forbidden basenames")
    }
    basename_prefixes = tuple(
        _text(value, name="forbidden_basename_prefix").casefold()
        for value in _array(
            policy.get("forbidden_basename_prefixes"), name="forbidden basename prefixes"
        )
    )
    allowed_paths = {
        _safe_relative(value, name="allowed_sensitive_path").casefold()
        for value in _array(policy.get("allowed_sensitive_paths"), name="allowed sensitive paths")
    }
    return basenames, basename_prefixes, allowed_paths


def _is_forbidden_credential_path(
    path: str,
    *,
    basenames: set[str],
    basename_prefixes: Sequence[str],
    allowed_paths: set[str],
) -> bool:
    folded = path.casefold()
    if folded in allowed_paths:
        return False
    basename = PurePosixPath(path).name.casefold()
    return basename in basenames or any(basename.startswith(prefix) for prefix in basename_prefixes)


def _tag_policy(
    policy: Mapping[str, Any],
) -> tuple[
    set[str],
    dict[str, tuple[str, str, str, str, str]],
    dict[str, tuple[str, str, str]],
]:
    value = _mapping(policy.get("git_refs"), name="git_refs policy")
    _exact_keys(
        value,
        {
            "non_tag_refs_must_target_commits",
            "tags_must_be_annotated_direct_to_commits",
            "required_for_release_tags",
            "pinned_annotated_tags",
            "permitted_candidate_tags",
            "historical_notes",
        },
        name="git_refs policy",
    )
    if (
        value.get("non_tag_refs_must_target_commits") is not True
        or value.get("tags_must_be_annotated_direct_to_commits") is not True
    ):
        raise ReleaseGateError("git_refs policy must fail closed on every ref and tag target")
    required = {
        _text(item, name="required release tag")
        for item in _array(value.get("required_for_release_tags"), name="required_for_release_tags")
    }
    pinned: dict[str, tuple[str, str, str, str, str]] = {}
    for raw_name, raw_item in _mapping(
        value.get("pinned_annotated_tags"), name="pinned_annotated_tags"
    ).items():
        name = _text(raw_name, name="pinned tag name")
        if TAG_NAME_RE.fullmatch(name) is None:
            raise ReleaseGateError("pinned tag name is not release-portable")
        item = _mapping(raw_item, name=f"pinned tag {name}")
        _exact_keys(
            item,
            {"object_id", "target_commit", "message", "tagger_name", "tagger_email"},
            name=f"pinned tag {name}",
        )
        pinned[name] = (
            _commit(item.get("object_id"), name=f"pinned tag {name} object_id"),
            _commit(item.get("target_commit"), name=f"pinned tag {name} target_commit"),
            _text(item.get("message"), name=f"pinned tag {name} message"),
            _text(item.get("tagger_name"), name=f"pinned tag {name} tagger_name"),
            _text(item.get("tagger_email"), name=f"pinned tag {name} tagger_email"),
        )
    permitted: dict[str, tuple[str, str, str]] = {}
    for raw_name, raw_item in _mapping(
        value.get("permitted_candidate_tags"), name="permitted_candidate_tags"
    ).items():
        name = _text(raw_name, name="permitted candidate tag name")
        if TAG_NAME_RE.fullmatch(name) is None:
            raise ReleaseGateError("permitted candidate tag name is not release-portable")
        item = _mapping(raw_item, name=f"permitted candidate tag {name}")
        _exact_keys(
            item,
            {"message", "tagger_name", "tagger_email"},
            name=f"permitted candidate tag {name}",
        )
        message = _text(item.get("message"), name=f"permitted candidate tag {name} message")
        tagger_name = _text(
            item.get("tagger_name"), name=f"permitted candidate tag {name} tagger_name"
        )
        tagger_email = _text(
            item.get("tagger_email"), name=f"permitted candidate tag {name} tagger_email"
        )
        if any(
            ord(character) < 32 or ord(character) == 127
            for text in (message, tagger_name, tagger_email)
            for character in text
        ) or any(character in tagger_email for character in "<> "):
            raise ReleaseGateError("permitted candidate tag metadata contains unsafe characters")
        permitted[name] = (message, tagger_name, tagger_email)
    notes = _mapping(value.get("historical_notes"), name="git_refs historical_notes")
    if any(name not in pinned for name in notes) or any(
        not isinstance(note, str) or not note for note in notes.values()
    ):
        raise ReleaseGateError("historical tag notes must describe pinned tags")
    if set(pinned) & set(permitted) or required != set(pinned) | set(permitted):
        raise ReleaseGateError("required release tags must exactly match pinned and permitted tags")
    return required, pinned, permitted


def inspect_git_ref_state(
    root: Path, policy: Mapping[str, Any], *, candidate: str
) -> tuple[dict[str, int | str], list[dict[str, str]]]:
    required, pinned, permitted = _tag_policy(policy)
    listing = _git(
        root,
        ("for-each-ref", "--format=%(refname)%09%(objectname)%09%(objecttype)"),
    )
    assert isinstance(listing.stdout, str)
    violations: list[dict[str, str]] = []
    ref_rows: list[str] = []
    local_main_state: tuple[str, str, str | None] | None = None
    remote_main_state: tuple[str, str, str | None] | None = None
    observed_tags: set[str] = set()
    tag_ref_count = 0
    validated_tag_count = 0
    invalid_ref_target_count = 0
    for line in listing.stdout.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ReleaseGateError("Git ref inventory returned malformed metadata")
        refname, raw_object_id, object_type = parts
        object_id = _commit(raw_object_id, name=f"{refname} object ID")
        peeled_result = _git(root, ("rev-parse", "--verify", f"{refname}^{{commit}}"), check=False)
        peeled: str | None = None
        if peeled_result.returncode == 0:
            assert isinstance(peeled_result.stdout, str)
            peeled = _commit(peeled_result.stdout.strip(), name=f"{refname} commit target")
        is_tag_ref = refname.startswith("refs/tags/")
        if not is_tag_ref:
            if refname == "refs/heads/main":
                local_main_state = (object_type, object_id, peeled)
            elif refname == "refs/remotes/origin/main":
                remote_main_state = (object_type, object_id, peeled)
            else:
                ref_rows.append(
                    f"ref-target\t{refname}\t{object_type}\t{object_id}\t{peeled or '-'}"
                )
            if object_type != "commit" or peeled != object_id:
                invalid_ref_target_count += 1
                violations.append(
                    {
                        "code": "non_commit_ref_target",
                        "commit": candidate,
                        "path": refname,
                        "detail": f"{object_type}:{object_id}",
                    }
                )
            if object_type == "tag":
                violations.append(
                    {
                        "code": "tag_object_outside_tag_namespace",
                        "commit": candidate,
                        "path": refname,
                        "detail": object_id,
                    }
                )
            continue

        tag_ref_count += 1
        tag_name = refname.removeprefix("refs/tags/")
        observed_tags.add(tag_name)
        ref_rows.append(f"tag\t{tag_name}\t{object_id}\t{object_type}\t{peeled or '-'}")
        if TAG_NAME_RE.fullmatch(tag_name) is None:
            violations.append(
                {
                    "code": "unsafe_tag_name",
                    "commit": candidate,
                    "path": refname,
                    "detail": tag_name,
                }
            )
            continue
        if object_type != "tag":
            invalid_ref_target_count += 1
            violations.append(
                {
                    "code": "lightweight_or_non_tag_ref",
                    "commit": candidate,
                    "path": refname,
                    "detail": f"{object_type}:{object_id}",
                }
            )
            continue
        tag_object = _git(root, ("cat-file", "tag", object_id), binary=True)
        assert isinstance(tag_object.stdout, bytes)
        payload = tag_object.stdout
        if len(payload) > MAX_TAG_OBJECT_BYTES:
            violations.append(
                {
                    "code": "oversized_tag_object",
                    "commit": candidate,
                    "path": refname,
                    "detail": str(len(payload)),
                }
            )
            continue
        try:
            header_bytes, message_bytes = payload.split(b"\n\n", 1)
            header_lines = header_bytes.decode("utf-8").splitlines()
            message = message_bytes.decode("utf-8")
        except (UnicodeError, ValueError) as exc:
            raise ReleaseGateError(f"tag object {tag_name} is not canonical UTF-8") from exc
        if len(header_lines) != 4:
            violations.append(
                {
                    "code": "noncanonical_tag_headers",
                    "commit": candidate,
                    "path": refname,
                    "detail": str(len(header_lines)),
                }
            )
            continue
        object_line, type_line, tag_line, tagger_line = header_lines
        direct_target = object_line.removeprefix("object ")
        tagger_match = re.fullmatch(r"tagger (.+) <([^<>]+)> ([0-9]+) ([+-][0-9]{4})", tagger_line)
        if (
            not object_line.startswith("object ")
            or COMMIT_RE.fullmatch(direct_target) is None
            or type_line != "type commit"
            or tag_line != f"tag {tag_name}"
            or tagger_match is None
            or peeled != direct_target
        ):
            invalid_ref_target_count += 1
            violations.append(
                {
                    "code": "noncanonical_tag_target",
                    "commit": candidate,
                    "path": refname,
                    "detail": object_id,
                }
            )
            continue
        if tag_name in pinned:
            (
                expected_object,
                expected_target,
                expected_message,
                expected_name,
                expected_email,
            ) = pinned[tag_name]
            assert tagger_match is not None
            if (
                object_id != expected_object
                or direct_target != expected_target
                or message != f"{expected_message}\n"
                or tagger_match.group(1) != expected_name
                or tagger_match.group(2) != expected_email
            ):
                violations.append(
                    {
                        "code": "pinned_tag_mismatch",
                        "commit": candidate,
                        "path": refname,
                        "detail": object_id,
                    }
                )
                continue
        elif tag_name in permitted:
            expected_message, expected_name, expected_email = permitted[tag_name]
            assert tagger_match is not None
            if (
                direct_target != candidate
                or message != f"{expected_message}\n"
                or tagger_match.group(1) != expected_name
                or tagger_match.group(2) != expected_email
            ):
                violations.append(
                    {
                        "code": "candidate_tag_metadata_mismatch",
                        "commit": candidate,
                        "path": refname,
                        "detail": object_id,
                    }
                )
                continue
        else:
            violations.append(
                {
                    "code": "unreviewed_annotated_tag",
                    "commit": candidate,
                    "path": refname,
                    "detail": object_id,
                }
            )
            continue
        validated_tag_count += 1
    for missing in sorted(required - observed_tags):
        violations.append(
            {
                "code": "required_release_tag_missing",
                "commit": candidate,
                "path": f"refs/tags/{missing}",
                "detail": "required by git_refs policy",
            }
        )
    # A local release commit necessarily precedes its push, so a cached
    # origin/main may truthfully lag. Prefer local main when it exists and use
    # the remote-tracking ref only in checkouts that do not create local main.
    release_main_ref = "refs/heads/main"
    release_main_state = local_main_state
    if release_main_state is None:
        release_main_ref = "refs/remotes/origin/main"
        release_main_state = remote_main_state
    if release_main_state is None:
        violations.append(
            {
                "code": "release_main_ref_missing",
                "commit": candidate,
                "path": "refs/heads/main|refs/remotes/origin/main",
                "detail": "no canonical main ref targets candidate_commit",
            }
        )
    elif release_main_state != ("commit", candidate, candidate):
        object_type, object_id, _peeled = release_main_state
        violations.append(
            {
                "code": "release_main_ref_mismatch",
                "commit": candidate,
                "path": release_main_ref,
                "detail": f"{object_type}:{object_id}",
            }
        )
    else:
        ref_rows.append(f"release-main\tcommit\t{candidate}\t{candidate}")
    metrics: dict[str, int | str] = {
        "ref_count": len(ref_rows),
        "tag_ref_count": tag_ref_count,
        "validated_annotated_tag_count": validated_tag_count,
        "invalid_ref_target_count": invalid_ref_target_count,
        "ref_metadata_index_sha256": hashlib.sha256(
            ("\n".join(sorted(ref_rows)) + "\n").encode()
        ).hexdigest(),
    }
    return metrics, violations


def _blob_payload(root: Path, object_id: str, *, expected_size: int) -> bytes:
    result = _git(root, ("cat-file", "blob", object_id), binary=True)
    assert isinstance(result.stdout, bytes)
    if len(result.stdout) != expected_size:
        raise ReleaseGateError("Git blob size differs from tree/index metadata")
    return result.stdout


def _blob_payloads(root: Path, expected_sizes: Mapping[str, int]) -> Iterator[tuple[str, bytes]]:
    """Read already-admitted Git objects in bounded batches, checking every object ID.

    Path, extension and size admission remain the callers' responsibility. No
    worktree/raw paths are opened. Batching avoids thousands of Windows process
    launches without altering payload inspection or alias/path-specific checks.
    """

    def read_batch(batch: list[tuple[str, int]]) -> Iterator[tuple[str, bytes]]:
        response = subprocess.run(
            ["git", "--no-replace-objects", "cat-file", "--batch"],
            cwd=root,
            input="".join(f"{object_id}\n" for object_id, _ in batch).encode("ascii"),
            capture_output=True,
            check=False,
        )
        if response.returncode:
            raise ReleaseGateError("batched Git object read failed")
        offset = 0
        for object_id, size in batch:
            end = response.stdout.find(b"\n", offset)
            expected = f"{object_id} blob {size}".encode("ascii")
            if end < 0 or response.stdout[offset:end] != expected:
                raise ReleaseGateError("batched Git object identity/type/size mismatch")
            payload = response.stdout[end + 1 : end + 1 + size]
            offset = end + 1 + size
            if len(payload) != size or response.stdout[offset : offset + 1] != b"\n":
                raise ReleaseGateError("batched Git object payload is truncated")
            digest = hashlib.sha1(
                f"blob {size}\0".encode("ascii") + payload, usedforsecurity=False
            ).hexdigest()
            if digest != object_id:
                raise ReleaseGateError("batched Git object payload hash mismatch")
            offset += 1
            yield object_id, payload
        if offset != len(response.stdout):
            raise ReleaseGateError("batched Git object response has trailing content")

    batch: list[tuple[str, int]] = []
    batch_bytes = 0
    for object_id, size in sorted(expected_sizes.items()):
        _commit(object_id, name="blob object ID")
        if size < 0:
            raise ReleaseGateError("negative Git blob size")
        if batch and (batch_bytes + size > 32 * 1024 * 1024 or len(batch) >= 128):
            yield from read_batch(batch)
            batch, batch_bytes = [], 0
        batch.append((object_id, size))
        batch_bytes += size
    if batch:
        yield from read_batch(batch)


def _blob_content_violations(
    payload: bytes,
    *,
    path: str,
    signatures: Mapping[str, bytes],
    allowed_signatures: Mapping[str, set[str]],
    allowed_binary_suffixes: set[str],
) -> list[tuple[str, str]]:
    suffix = PurePosixPath(path).suffix.casefold()
    violations: list[tuple[str, str]] = []
    if payload.startswith(b"version https://git-lfs.github.com/spec/v1"):
        violations.append(("git_lfs_pointer", "Git LFS pointers are forbidden"))
    if suffix in TEXT_SUFFIXES:
        if b"\x00" in payload:
            violations.append(("disguised_binary_nul", "NUL byte in text-declared blob"))
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError:
            violations.append(("invalid_text_encoding", "text-declared blob is not UTF-8"))
        else:
            if PurePosixPath(path).name.casefold() == ".gitattributes" and re.search(
                r"\bfilter\s*=\s*lfs\b", decoded, flags=re.IGNORECASE
            ):
                violations.append(
                    ("git_lfs_filter", "Git LFS filters are forbidden in .gitattributes")
                )
    elif suffix not in allowed_binary_suffixes:
        violations.append(("unsupported_binary_extension", suffix or "<none>"))
    elif not any(
        payload.startswith(signatures[name])
        for name, suffixes_for_signature in allowed_signatures.items()
        if suffix in suffixes_for_signature and name in signatures
    ):
        violations.append(
            ("invalid_allowed_binary_signature", f"{suffix} lacks its required magic")
        )
    for signature_name, signature in signatures.items():
        if payload.startswith(signature) and suffix not in allowed_signatures.get(
            signature_name, set()
        ):
            violations.append(("disguised_binary_signature", signature_name))
            break
    return violations


def _index_entries(root: Path) -> list[tuple[str, str, int, str, str, int | None]]:
    listing = _git(root, ("ls-files", "--stage", "-z"), binary=True)
    assert isinstance(listing.stdout, bytes)
    parsed: list[tuple[str, str, int, str]] = []
    for raw in (item for item in listing.stdout.split(b"\x00") if item):
        try:
            metadata, raw_path = raw.split(b"\t", 1)
            mode, object_id, stage_text = metadata.decode("ascii").split(" ", 2)
            path = _safe_relative(raw_path.decode("utf-8"), name="Git index path")
            stage = int(stage_text)
        except (UnicodeError, ValueError) as exc:
            raise ReleaseGateError("Git index contains undecodable metadata or path") from exc
        parsed.append((mode, object_id, stage, path))

    object_ids = sorted({object_id for _mode, object_id, _stage, _path in parsed})
    metadata_by_object: dict[str, tuple[str, int | None]] = {}
    if object_ids:
        request = ("\n".join(object_ids) + "\n").encode("ascii")
        try:
            checked = subprocess.run(
                [
                    "git",
                    "--no-replace-objects",
                    "cat-file",
                    "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                ],
                cwd=root,
                input=request,
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ReleaseGateError("Git index object inspection failed") from exc
        lines = checked.stdout.decode("ascii").splitlines()
        if len(lines) != len(object_ids):
            raise ReleaseGateError("Git index object inspection returned an incomplete response")
        for requested, line in zip(object_ids, lines, strict=True):
            parts = line.split(" ")
            if len(parts) == 2 and parts == [requested, "missing"]:
                metadata_by_object[requested] = ("missing", None)
                continue
            if len(parts) != 3 or parts[0] != requested:
                raise ReleaseGateError("Git index object inspection returned unexpected metadata")
            try:
                size = int(parts[2])
            except ValueError as exc:
                raise ReleaseGateError("Git index object size is invalid") from exc
            metadata_by_object[requested] = (parts[1], size)
    return [(*entry, *metadata_by_object.get(entry[1], ("missing", None))) for entry in parsed]


def scan_index(
    *, repository_root: str | Path, policy_path: str | Path, created_at_utc: str
) -> dict[str, Any]:
    """Inspect the exact Git index tree without reading worktree payloads or writing Git objects."""

    root = _repository(repository_root)
    policy, policy_reference = _policy(policy_path)
    staged_policy = _mapping(policy.get("staged_index_scan"), name="staged_index_scan")
    if staged_policy != {
        "required_before_release_candidate_commit": True,
        "scan_scope": "exact_git_index",
    }:
        raise ReleaseGateError("staged_index_scan policy is missing or incompatible")
    maximum, prefixes, suffixes, signatures, allowed_signatures, allowed_binary_suffixes = (
        _content_scan_policy(policy)
    )
    credential_basenames, credential_prefixes, allowed_sensitive_paths = _credential_path_policy(
        policy
    )
    head_result = _git(root, ("rev-parse", "HEAD"))
    assert isinstance(head_result.stdout, str)
    head = _commit(head_result.stdout.strip(), name="HEAD")
    changed_result = _git(root, ("diff", "--cached", "--name-only", "-z"), binary=True)
    assert isinstance(changed_result.stdout, bytes)
    changed_paths = [value for value in changed_result.stdout.split(b"\x00") if value]

    violations: list[dict[str, str]] = []
    replace_result = _git(root, ("for-each-ref", "--format=%(refname)", "refs/replace/"))
    assert isinstance(replace_result.stdout, str)
    replace_refs = sorted(value for value in replace_result.stdout.splitlines() if value)
    graft_result = _git(root, ("rev-parse", "--git-path", "info/grafts"))
    assert isinstance(graft_result.stdout, str)
    graft_path = Path(graft_result.stdout.strip())
    if not graft_path.is_absolute():
        graft_path = root / graft_path
    grafts_file_present = graft_path.is_symlink() or (
        graft_path.exists() and graft_path.stat().st_size > 0
    )
    for reference in replace_refs:
        violations.append(
            {
                "code": "git_replace_ref",
                "path": reference,
                "detail": "replace refs are forbidden for release evidence",
            }
        )
    if grafts_file_present:
        violations.append(
            {
                "code": "git_grafts_file",
                "path": ".git/info/grafts",
                "detail": "legacy grafts can hide reachable history",
            }
        )
    folded_paths: dict[str, str] = {}
    index_rows: list[str] = []
    inspection_targets: set[tuple[str, str]] = set()
    unique_blobs: set[str] = set()
    entries = _index_entries(root)
    for mode, object_id, stage, path, kind, size in entries:
        index_rows.append(
            _entry_index_row(
                mode=mode,
                object_id=object_id,
                stage=stage,
                kind=kind,
                size=size,
                path=path,
            )
        )
        folded = path.casefold()
        previous = folded_paths.get(folded)
        if previous is not None and previous != path:
            violations.append(
                {"code": "case_collision", "path": path, "detail": f"collides with {previous}"}
            )
        folded_paths[folded] = path
        if stage != 0:
            violations.append(
                {"code": "unmerged_index_entry", "path": path, "detail": f"stage {stage}"}
            )
            continue
        if mode == "120000":
            violations.append({"code": "symlink", "path": path, "detail": mode})
            continue
        if mode == "160000" or kind == "commit":
            violations.append({"code": "submodule", "path": path, "detail": mode})
            continue
        if mode not in {"100644", "100755"} or kind != "blob" or size is None:
            violations.append(
                {
                    "code": "unsupported_index_entry",
                    "path": path,
                    "detail": f"{mode} {kind}",
                }
            )
            continue
        unique_blobs.add(object_id)
        prohibited_path = _is_forbidden_path(path, prefixes) or _is_forbidden_credential_path(
            path,
            basenames=credential_basenames,
            basename_prefixes=credential_prefixes,
            allowed_paths=allowed_sensitive_paths,
        )
        suffix = PurePosixPath(path).suffix.casefold()
        prohibited_suffix = suffix in suffixes
        if prohibited_path:
            violations.append(
                {
                    "code": "forbidden_path",
                    "path": path,
                    "detail": "protected/raw/transient prefix",
                }
            )
        if prohibited_suffix:
            violations.append({"code": "forbidden_extension", "path": path, "detail": suffix})
        if size > maximum:
            violations.append({"code": "oversized_blob", "path": path, "detail": str(size)})
        if not prohibited_path and not prohibited_suffix and size <= maximum:
            inspection_targets.add((object_id, path))

    paths_by_object: dict[str, list[str]] = {}
    for object_id, path in sorted(inspection_targets):
        paths_by_object.setdefault(object_id, []).append(path)
    sizes = {
        object_id: size
        for _mode, object_id, _stage, _path, _kind, size in entries
        if object_id in paths_by_object and size is not None
    }
    for object_id, payload in _blob_payloads(root, sizes):
        for path in paths_by_object[object_id]:
            for code, detail in _blob_content_violations(
                payload,
                path=path,
                signatures=signatures,
                allowed_signatures=allowed_signatures,
                allowed_binary_suffixes=allowed_binary_suffixes,
            ):
                violations.append({"code": code, "path": path, "detail": detail})

    unique_violations = sorted(
        {(item["code"], item["path"], item["detail"]): item for item in violations}.values(),
        key=lambda item: (item["code"], item["path"], item["detail"]),
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": INDEX_SCAN_KIND,
        "status": "pass" if not unique_violations else "fail",
        "created_at_utc": _timestamp(created_at_utc),
        "head_commit": head,
        "scan_scope": "exact_git_index",
        "policy": policy_reference,
        "index_entry_count": len(entries),
        "staged_change_count": len(changed_paths),
        "unique_blob_count": len(unique_blobs),
        "replace_ref_count": len(replace_refs),
        "grafts_file_present": grafts_file_present,
        "index_entry_index_sha256": hashlib.sha256(
            ("\n".join(sorted(index_rows)) + "\n").encode()
        ).hexdigest(),
        "violations": unique_violations,
        "raw_worktree_paths_opened": False,
        "git_objects_written": False,
    }
    return _self_hash(body)


def scan_repository(
    *,
    repository_root: str | Path,
    candidate_commit: str,
    policy_path: str | Path,
    created_at_utc: str,
) -> dict[str, Any]:
    """Scan the candidate tree and every reachable historical Git tree."""

    root = _repository(repository_root)
    candidate = _commit(candidate_commit, name="candidate_commit")
    _require_candidate_head(root, candidate, operation="repository scan")
    exists = _git(root, ("cat-file", "-e", f"{candidate}^{{commit}}"), check=False)
    if exists.returncode != 0:
        raise ReleaseGateError("candidate_commit is not present in the repository")
    policy, policy_reference = _policy(policy_path)
    maximum, prefixes, suffixes, signatures, allowed_signatures, allowed_binary_suffixes = (
        _content_scan_policy(policy)
    )
    credential_basenames, credential_prefixes, allowed_sensitive_paths = _credential_path_policy(
        policy
    )
    revisions = _git(root, ("rev-list", "--all", candidate))
    assert isinstance(revisions.stdout, str)
    commits = sorted(set(revisions.stdout.splitlines()) | {candidate})
    ref_metrics, ref_violations = inspect_git_ref_state(root, policy, candidate=candidate)
    violations: list[dict[str, Any]] = list(ref_violations)
    objects: dict[str, tuple[int, str]] = {}
    path_observations = 0
    inspection_targets: dict[tuple[str, str], str] = {}
    shallow_result = _git(root, ("rev-parse", "--is-shallow-repository"))
    assert isinstance(shallow_result.stdout, str)
    shallow = shallow_result.stdout.strip() == "true"
    replace_result = _git(root, ("for-each-ref", "--format=%(refname)", "refs/replace/"))
    assert isinstance(replace_result.stdout, str)
    replace_refs = sorted(value for value in replace_result.stdout.splitlines() if value)
    graft_result = _git(root, ("rev-parse", "--git-path", "info/grafts"))
    assert isinstance(graft_result.stdout, str)
    graft_path = Path(graft_result.stdout.strip())
    if not graft_path.is_absolute():
        graft_path = root / graft_path
    grafts_file_present = graft_path.is_symlink() or (
        graft_path.exists() and graft_path.stat().st_size > 0
    )
    if shallow:
        violations.append(
            {
                "code": "shallow_repository",
                "commit": candidate,
                "path": ".git",
                "detail": "complete reachable history is unavailable",
            }
        )
    for reference in replace_refs:
        violations.append(
            {
                "code": "git_replace_ref",
                "commit": candidate,
                "path": reference,
                "detail": "replace refs are forbidden for release evidence",
            }
        )
    if grafts_file_present:
        violations.append(
            {
                "code": "git_grafts_file",
                "commit": candidate,
                "path": ".git/info/grafts",
                "detail": "legacy grafts can hide reachable history",
            }
        )
    for revision in commits:
        folded_paths: dict[str, str] = {}
        for mode, kind, object_id, size, path in _tree_entries(root, revision):
            path_observations += 1
            folded = path.casefold()
            previous = folded_paths.get(folded)
            if previous is not None and previous != path:
                violations.append(
                    {
                        "code": "case_collision",
                        "commit": revision,
                        "path": path,
                        "detail": f"collides with {previous}",
                    }
                )
            folded_paths[folded] = path
            if mode == "120000":
                violations.append(
                    {"code": "symlink", "commit": revision, "path": path, "detail": mode}
                )
                continue
            if mode == "160000" or kind == "commit":
                violations.append(
                    {"code": "submodule", "commit": revision, "path": path, "detail": mode}
                )
                continue
            if kind != "blob" or size is None:
                violations.append(
                    {
                        "code": "unsupported_tree_entry",
                        "commit": revision,
                        "path": path,
                        "detail": f"{mode} {kind}",
                    }
                )
                continue
            objects.setdefault(object_id, (size, path))
            prohibited_path = _is_forbidden_path(path, prefixes) or _is_forbidden_credential_path(
                path,
                basenames=credential_basenames,
                basename_prefixes=credential_prefixes,
                allowed_paths=allowed_sensitive_paths,
            )
            prohibited_suffix = PurePosixPath(path).suffix.casefold() in suffixes
            if prohibited_path:
                violations.append(
                    {
                        "code": "forbidden_path",
                        "commit": revision,
                        "path": path,
                        "detail": "protected/raw/transient prefix",
                    }
                )
            if prohibited_suffix:
                violations.append(
                    {
                        "code": "forbidden_extension",
                        "commit": revision,
                        "path": path,
                        "detail": PurePosixPath(path).suffix.casefold(),
                    }
                )
            if size > maximum:
                violations.append(
                    {
                        "code": "oversized_blob",
                        "commit": revision,
                        "path": path,
                        "detail": str(size),
                    }
                )
            if prohibited_path or prohibited_suffix or size > maximum:
                continue
            inspection_targets.setdefault((object_id, path), revision)
    paths_by_object: dict[str, list[tuple[str, str]]] = {}
    for (object_id, path), revision in sorted(inspection_targets.items()):
        paths_by_object.setdefault(object_id, []).append((path, revision))
    sizes = {object_id: objects[object_id][0] for object_id in paths_by_object}
    for object_id, payload in _blob_payloads(root, sizes):
        for path, revision in paths_by_object[object_id]:
            for code, detail in _blob_content_violations(
                payload,
                path=path,
                signatures=signatures,
                allowed_signatures=allowed_signatures,
                allowed_binary_suffixes=allowed_binary_suffixes,
            ):
                violations.append(
                    {"code": code, "commit": revision, "path": path, "detail": detail}
                )
    # Deduplicate violations caused by the same path/object recurring unchanged in history.
    unique_violations = sorted(
        {(item["code"], item["path"], item["detail"]): item for item in violations}.values(),
        key=lambda item: (str(item["code"]), str(item["path"]), str(item["commit"])),
    )
    object_digest_rows = [f"{oid} {size} {path}" for oid, (size, path) in sorted(objects.items())]
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": SCAN_KIND,
        "status": "pass" if not unique_violations else "fail",
        "created_at_utc": _timestamp(created_at_utc),
        "candidate_commit": candidate,
        "scan_scope": (
            "candidate_tree_and_complete_reachable_git_history"
            if not shallow and not grafts_file_present
            else "candidate_tree_and_observed_incomplete_history"
        ),
        "policy": policy_reference,
        "commit_count": len(commits),
        "unique_blob_count": len(objects),
        "tree_entry_observation_count": path_observations,
        **ref_metrics,
        "shallow_repository": shallow,
        "replace_ref_count": len(replace_refs),
        "grafts_file_present": grafts_file_present,
        "scanned_object_index_sha256": hashlib.sha256(
            ("\n".join(object_digest_rows) + "\n").encode()
        ).hexdigest(),
        "violations": unique_violations,
        "raw_worktree_paths_opened": False,
    }
    return _self_hash(body)


def validate_secret_scan(
    *,
    repository_root: str | Path,
    report_path: str | Path,
    candidate_commit: str,
    policy_path: str | Path,
    config_path: str | Path,
    ignore_path: str | Path,
    gitleaks_version: str,
    gitleaks_exit_code: int,
    created_at_utc: str,
) -> dict[str, Any]:
    """Validate a Gitleaks JSON report and bind it to an exact candidate."""

    root = _repository(repository_root)
    policy, policy_reference = _policy(policy_path)
    gitleaks_policy = _mapping(policy.get("gitleaks"), name="gitleaks policy")
    expected_version = _text(gitleaks_policy.get("version"), name="gitleaks.version")
    if gitleaks_version != expected_version:
        raise ReleaseGateError("Gitleaks version differs from the policy pin")
    if gitleaks_policy.get("required_scope") != SECRET_SCAN_SCOPE:
        raise ReleaseGateError("Gitleaks policy does not declare the required commit/tag scope")
    candidate = _commit(candidate_commit, name="candidate_commit")
    _require_candidate_head(root, candidate, operation="secret scan")
    ref_metrics, ref_violations = inspect_git_ref_state(root, policy, candidate=candidate)
    if ref_violations:
        raise ReleaseGateError("secret scan cannot attest unreviewed or unsafe Git ref metadata")
    expected_config_path = _safe_relative(
        gitleaks_policy.get("config_path"), name="gitleaks.config_path"
    )
    expected_config_sha256 = _sha256(
        gitleaks_policy.get("config_sha256"), name="gitleaks.config_sha256"
    )
    config_source = Path(config_path)
    if config_source.is_symlink():
        raise ReleaseGateError("Gitleaks configuration may not be a symlink")
    config_resolved = config_source.resolve(strict=True)
    try:
        config_relative = _safe_relative(
            config_resolved.relative_to(root).as_posix(), name="Gitleaks configuration"
        )
    except ValueError as exc:
        raise ReleaseGateError("Gitleaks configuration escapes repository_root") from exc
    if config_relative != expected_config_path or not config_resolved.is_file():
        raise ReleaseGateError("Gitleaks configuration differs from the policy path")
    config_payload = config_resolved.read_bytes()
    config_reference = {
        "path": config_relative,
        "size_bytes": len(config_payload),
        "file_sha256": hashlib.sha256(config_payload).hexdigest(),
    }
    if config_reference["file_sha256"] != expected_config_sha256:
        raise ReleaseGateError("Gitleaks configuration SHA-256 differs from the policy pin")
    if _git_blob_reference(root, candidate, config_reference, name="Gitleaks configuration") != (
        config_reference
    ):
        raise ReleaseGateError("Gitleaks configuration differs from the candidate blob")
    expected_ignore_path = _safe_relative(
        gitleaks_policy.get("ignore_path"), name="gitleaks.ignore_path"
    )
    ignore_source = Path(ignore_path)
    if ignore_source.is_symlink():
        raise ReleaseGateError("Gitleaks ignore file may not be a symlink")
    ignore_resolved = ignore_source.resolve(strict=True)
    try:
        ignore_relative = _safe_relative(
            ignore_resolved.relative_to(root).as_posix(), name="Gitleaks ignore file"
        )
    except ValueError as exc:
        raise ReleaseGateError("Gitleaks ignore file escapes repository_root") from exc
    if ignore_relative != expected_ignore_path or not ignore_resolved.is_file():
        raise ReleaseGateError("Gitleaks ignore file differs from the policy path")
    ignore_payload = ignore_resolved.read_bytes()
    try:
        ignore_text = ignore_payload.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ReleaseGateError("Gitleaks ignore file must be UTF-8") from exc
    _validate_gitleaks_ignore_entries(policy, ignore_text)
    ignore_reference = {
        "path": ignore_relative,
        "size_bytes": len(ignore_payload),
        "file_sha256": hashlib.sha256(ignore_payload).hexdigest(),
    }
    if _git_blob_reference(root, candidate, ignore_reference, name="Gitleaks ignore file") != (
        ignore_reference
    ):
        raise ReleaseGateError("Gitleaks ignore file differs from the candidate blob")
    source = Path(report_path)
    if source.is_symlink():
        raise ReleaseGateError("Gitleaks report may not be a symlink")
    report_resolved = source.resolve(strict=True)
    if not report_resolved.is_file() or report_resolved.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ReleaseGateError("Gitleaks report exceeds the evidence size cap")
    payload = report_resolved.read_bytes()
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise ReleaseGateError("Gitleaks report exceeds the evidence size cap")
    try:
        findings = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseGateError("Gitleaks report is not UTF-8 JSON") from exc
    if not isinstance(findings, list):
        raise ReleaseGateError("Gitleaks JSON report must be an array")
    if not isinstance(gitleaks_exit_code, int) or isinstance(gitleaks_exit_code, bool):
        raise ReleaseGateError("Gitleaks exit code must be an integer")
    execution_complete = (gitleaks_exit_code == 0 and not findings) or (
        gitleaks_exit_code == 1 and bool(findings)
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": SECRET_KIND,
        "status": "pass" if execution_complete and not findings else "fail",
        "created_at_utc": _timestamp(created_at_utc),
        "candidate_commit": candidate,
        "scan_scope": SECRET_SCAN_SCOPE,
        "ref_metadata_index_sha256": ref_metrics["ref_metadata_index_sha256"],
        "validated_annotated_tag_count": ref_metrics["validated_annotated_tag_count"],
        "gitleaks_version": gitleaks_version,
        "gitleaks_exit_code": gitleaks_exit_code,
        "execution_status": "completed" if execution_complete else "operational_error",
        "policy": policy_reference,
        "config": config_reference,
        "ignore": ignore_reference,
        "raw_report": {
            "basename": source.name,
            "size_bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        },
        "finding_count": len(findings),
    }
    return _self_hash(body)


def audit_licenses(
    *,
    repository_root: str | Path,
    inventory_path: str | Path,
    candidate_commit: str,
    policy_path: str | Path,
    pip_licenses_version: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Validate pip-licenses JSON against the versioned release policy."""

    root = _repository(repository_root)
    candidate = _commit(candidate_commit, name="candidate_commit")
    _require_candidate_head(root, candidate, operation="license audit")
    if pip_licenses_version != "5.5.5":
        raise ReleaseGateError("pip-licenses version must equal the locked 5.5.5 release tool")
    policy, policy_reference = _policy(policy_path)
    source = Path(inventory_path)
    if source.is_symlink():
        raise ReleaseGateError("license inventory may not be a symlink")
    inventory_resolved = source.resolve(strict=True)
    if not inventory_resolved.is_file() or inventory_resolved.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ReleaseGateError("license inventory exceeds the evidence size cap")
    payload = inventory_resolved.read_bytes()
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise ReleaseGateError("license inventory exceeds the evidence size cap")
    try:
        packages = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseGateError("license inventory is not UTF-8 JSON") from exc
    values = _array(packages, name="license inventory")
    normalized_packages: list[dict[str, str]] = []
    for index, raw_package in enumerate(values):
        package = _mapping(raw_package, name=f"license inventory[{index}]")
        name = _text(package.get("Name"), name=f"license inventory[{index}].Name")
        version = _text(package.get("Version"), name=f"license inventory[{index}].Version")
        license_name = _text(package.get("License"), name=f"license inventory[{index}].License")
        # Distribution metadata can contain a multiline license (e.g. aeon).
        # Retain the original inventory hash and all words; only normalize
        # whitespace for the single-line canonical package record.
        license_name = " ".join(license_name.split())
        identity = f"{name}@{version}".casefold()
        normalized_packages.append(
            {"identity": identity, "package": name, "version": version, "license": license_name}
        )
    normalized_packages = _normalized_license_packages(
        sorted(normalized_packages, key=lambda item: item["identity"])
    )
    audit_environment = {"system": platform.system(), "machine": platform.machine()}
    bindings = validate_license_exception_candidate_binding(root, candidate, policy)
    exceptions_applied, violations = _license_outcomes(
        normalized_packages,
        environment=audit_environment,
        policy=policy,
        bindings=bindings,
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": LICENSE_KIND,
        "status": "pass" if not violations else "fail",
        "created_at_utc": _timestamp(created_at_utc),
        "candidate_commit": candidate,
        "pip_licenses_version": pip_licenses_version,
        "policy": policy_reference,
        "inventory": {
            "basename": source.name,
            "size_bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        },
        "audit_environment": audit_environment,
        "packages": normalized_packages,
        "package_count": len(normalized_packages),
        "normalized_inventory_sha256": _license_inventory_digest(normalized_packages),
        "exceptions_applied": exceptions_applied,
        "violations": violations,
    }
    record = _self_hash(body)
    validate_license_audit_semantics(
        record,
        root=root,
        candidate=candidate,
        policy=policy,
    )
    return record


def _evidence_snapshot(
    root: Path, value: Mapping[str, Any], *, name: str
) -> tuple[dict[str, Any], bytes]:
    path = _safe_relative(value.get("path"), name=f"{name}.path")
    expected = _sha256(value.get("expected_sha256"), name=f"{name}.expected_sha256")
    source = root / Path(path)
    current = root
    for part in PurePosixPath(path).parts:
        current = current / part
        if current.is_symlink():
            raise ReleaseGateError(f"{name} may not traverse a symlink")
    resolved = source.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReleaseGateError(f"{name} escapes repository_root") from exc
    if not resolved.is_file() or resolved.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ReleaseGateError(f"{name} must be a regular file within the evidence size cap")
    payload = resolved.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ReleaseGateError(f"{name} SHA-256 differs")
    return {"path": path, "size_bytes": len(payload), "file_sha256": observed}, payload


def _evidence_reference(root: Path, value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    reference, _payload = _evidence_snapshot(root, value, name=name)
    return reference


def _file_reference_shape(value: Any, *, name: str) -> dict[str, Any]:
    reference = _mapping(value, name=name)
    if set(reference) != {"path", "size_bytes", "file_sha256"}:
        raise ReleaseGateError(f"{name} keys differ")
    path = _safe_relative(reference.get("path"), name=f"{name}.path")
    size = reference.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ReleaseGateError(f"{name}.size_bytes must be a non-negative integer")
    digest = _sha256(reference.get("file_sha256"), name=f"{name}.file_sha256")
    return {"path": path, "size_bytes": size, "file_sha256": digest}


def _git_blob_reference(
    root: Path, commit: str, value: Mapping[str, Any], *, name: str
) -> dict[str, Any]:
    path = _safe_relative(value.get("path"), name=f"{name}.path")
    expected = _sha256(value.get("file_sha256"), name=f"{name}.file_sha256")
    tree = _git(root, ("ls-tree", "-z", commit, "--", path), binary=True)
    assert isinstance(tree.stdout, bytes)
    entries = [entry for entry in tree.stdout.split(b"\x00") if entry]
    if len(entries) != 1:
        raise ReleaseGateError(f"{name} is not one tracked candidate file")
    try:
        metadata, observed_path = entries[0].split(b"\t", 1)
        mode, kind, _object_id = metadata.decode("ascii").split(" ", 2)
        decoded_path = observed_path.decode("utf-8")
    except (UnicodeError, ValueError) as exc:
        raise ReleaseGateError(f"{name} has invalid Git tree metadata") from exc
    if decoded_path != path or kind != "blob" or mode not in {"100644", "100755"}:
        raise ReleaseGateError(f"{name} must be a regular tracked file")
    blob = _git(root, ("cat-file", "blob", f"{commit}:{path}"), binary=True)
    assert isinstance(blob.stdout, bytes)
    observed = hashlib.sha256(blob.stdout).hexdigest()
    if observed != expected:
        raise ReleaseGateError(f"{name} SHA-256 differs from its bound Git blob")
    return {"path": path, "size_bytes": len(blob.stdout), "file_sha256": observed}


def _git_blob_payload(root: Path, commit: str, value: Mapping[str, Any], *, name: str) -> bytes:
    _git_blob_reference(root, commit, value, name=name)
    path = _safe_relative(value.get("path"), name=f"{name}.path")
    blob = _git(root, ("cat-file", "blob", f"{commit}:{path}"), binary=True)
    assert isinstance(blob.stdout, bytes)
    return blob.stdout


def _validate_gitleaks_ignore_entries(policy: Mapping[str, Any], ignore_text: str) -> None:
    gitleaks_policy = _mapping(policy.get("gitleaks"), name="gitleaks policy")
    reviewed = [
        _text(value, name="gitleaks.reviewed_ignore_entries entry")
        for value in _array(
            gitleaks_policy.get("reviewed_ignore_entries"),
            name="gitleaks.reviewed_ignore_entries",
        )
    ]
    observed = [
        line.strip()
        for line in ignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if observed != reviewed:
        raise ReleaseGateError("Gitleaks ignore entries differ from the reviewed policy")


def _tracked_report_policy_reference(
    root: Path,
    report_path: str | Path,
    parent_commit: str,
    policy_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve a pending report's policy without trusting mutable worktree bytes."""

    try:
        return _git_blob_reference(root, parent_commit, policy_reference, name="policy")
    except ReleaseGateError as parent_error:
        source = Path(report_path).resolve(strict=True)
        try:
            relative = _safe_relative(
                source.relative_to(root).as_posix(), name="tracked report path"
            )
        except ValueError as exc:
            raise ReleaseGateError(
                "tracked precommit report policy is not bound to repository history"
            ) from exc
        payload = source.read_bytes()
        revisions = _git(root, ("rev-list", "--all", "--", relative))
        assert isinstance(revisions.stdout, str)
        for candidate in revisions.stdout.splitlines():
            candidate = _commit(candidate, name="report-containing commit")
            observed_parent = _git(root, ("rev-parse", f"{candidate}^"), check=False)
            if observed_parent.returncode != 0:
                continue
            assert isinstance(observed_parent.stdout, str)
            if observed_parent.stdout.strip().casefold() != parent_commit:
                continue
            report_blob = _git(root, ("cat-file", "blob", f"{candidate}:{relative}"), binary=True)
            assert isinstance(report_blob.stdout, bytes)
            if report_blob.stdout != payload:
                continue
            try:
                return _git_blob_reference(root, candidate, policy_reference, name="policy")
            except ReleaseGateError:
                continue
        raise ReleaseGateError(
            "tracked precommit report policy is not bound to its parent or direct content commit"
        ) from parent_error


def validate_ci_gate_bundle(
    *,
    repository_root: str | Path,
    candidate_commit: str,
    policy_path: str | Path,
    evidence_root: str | Path,
    created_at_utc: str,
) -> dict[str, Any]:
    """Validate the candidate-bound CI gate bundle without claiming completed CI."""

    root = _repository(repository_root)
    candidate = _commit(candidate_commit, name="candidate_commit")
    head_result = _git(root, ("rev-parse", "HEAD"))
    assert isinstance(head_result.stdout, str)
    if head_result.stdout.strip().casefold() != candidate:
        raise ReleaseGateError("CI bundle candidate differs from repository HEAD")
    status_result = _git(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    assert isinstance(status_result.stdout, str)
    if status_result.stdout.strip():
        raise ReleaseGateError("CI bundle repository worktree is not clean")
    parent_result = _git(root, ("rev-parse", f"{candidate}^"), check=False)
    if parent_result.returncode != 0:
        raise ReleaseGateError("CI bundle candidate has no resolvable first parent")
    assert isinstance(parent_result.stdout, str)
    parent = _commit(parent_result.stdout.strip(), name="candidate first parent")
    policy_value, policy_reference = _policy(policy_path)
    bundle_root = Path(evidence_root)
    if bundle_root.is_symlink():
        raise ReleaseGateError("CI evidence root may not be a symlink")
    resolved_root = bundle_root.resolve(strict=True)
    try:
        resolved_root.relative_to(root)
    except ValueError as exc:
        raise ReleaseGateError("CI evidence root escapes repository_root") from exc
    if not resolved_root.is_dir():
        raise ReleaseGateError("CI evidence root must be a directory")

    paths = {
        "repository_scan": resolved_root / "repository_scan.json",
        "secret_scan": resolved_root / "secret_scan.json",
        "license_audit": resolved_root / "license_audit.json",
        **{name: resolved_root / "quality_gates.json" for name in QUALITY_GATE_NAMES},
    }
    references: dict[str, dict[str, Any]] = {}
    for gate_name, path in paths.items():
        if path.is_symlink():
            raise ReleaseGateError(f"CI gate {gate_name} evidence may not be a symlink")
        resolved_path = path.resolve(strict=True)
        if not resolved_path.is_file() or resolved_path.stat().st_size > MAX_EVIDENCE_BYTES:
            raise ReleaseGateError(f"CI gate {gate_name} evidence exceeds the evidence size cap")
        payload = resolved_path.read_bytes()
        relative = _safe_relative(
            path.resolve().relative_to(root).as_posix(), name=f"CI gate {gate_name} evidence"
        )
        reference = {
            "path": relative,
            "size_bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        }
        evidence = _strict_json_bytes(payload, name=f"CI gate {gate_name} evidence")
        _validate_exact_gate_evidence(
            root,
            gate_name=gate_name,
            evidence=evidence,
            candidate=candidate,
            parent=parent,
            policy=policy_value,
            policy_reference=policy_reference,
        )
        references[gate_name] = reference
    return _self_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "record_kind": "ci_release_gate_bundle_validation",
            "status": "pass",
            "created_at_utc": _timestamp(created_at_utc),
            "candidate_commit": candidate,
            "parent_commit": parent,
            "ci_evidence_scope": QUALITY_EVIDENCE_SCOPE,
            "external_completed_ci_required_for_release_inventory": True,
            "validated_gates": sorted(references),
            "evidence": references,
        }
    )


def _validate_exact_gate_evidence(
    root: Path,
    *,
    gate_name: str,
    evidence: Mapping[str, Any],
    candidate: str,
    parent: str,
    policy: Mapping[str, Any],
    policy_reference: Mapping[str, Any],
) -> None:
    """Fail closed on the semantics of one exact-candidate gate record."""

    _verify_self_hash(evidence, name=f"gates.{gate_name} evidence")
    if evidence.get("schema_version") != SCHEMA_VERSION or evidence.get("status") != "pass":
        raise ReleaseGateError(f"gates.{gate_name} evidence version or status differs")

    if gate_name == "staged_index_scan":
        _exact_keys(
            evidence,
            {
                "schema_version",
                "record_kind",
                "status",
                "created_at_utc",
                "head_commit",
                "scan_scope",
                "policy",
                "index_entry_count",
                "staged_change_count",
                "unique_blob_count",
                "replace_ref_count",
                "grafts_file_present",
                "index_entry_index_sha256",
                "violations",
                "raw_worktree_paths_opened",
                "git_objects_written",
                "record_sha256",
            },
            name="gates.staged_index_scan evidence",
        )
        _timestamp(evidence.get("created_at_utc"))
        _exact_integer(
            evidence.get("replace_ref_count"),
            name="gates.staged_index_scan evidence.replace_ref_count",
            expected=0,
        )
        if (
            evidence.get("record_kind") != INDEX_SCAN_KIND
            or evidence.get("head_commit") != parent
            or evidence.get("scan_scope") != "exact_git_index"
            or evidence.get("policy") != policy_reference
            or evidence.get("raw_worktree_paths_opened") is not False
            or evidence.get("git_objects_written") is not False
            or evidence.get("grafts_file_present") is not False
        ):
            raise ReleaseGateError("staged_index_scan evidence is incompatible or unsafe")
        violations = _array(
            evidence.get("violations"), name="gates.staged_index_scan evidence.violations"
        )
        staged_changes = evidence.get("staged_change_count")
        entry_count = evidence.get("index_entry_count")
        unique_blob_count = evidence.get("unique_blob_count")
        digest = evidence.get("index_entry_index_sha256")
        if (
            violations
            or not isinstance(staged_changes, int)
            or isinstance(staged_changes, bool)
            or staged_changes < 1
            or not isinstance(entry_count, int)
            or isinstance(entry_count, bool)
            or entry_count < 1
            or not isinstance(unique_blob_count, int)
            or isinstance(unique_blob_count, bool)
            or unique_blob_count < 1
            or unique_blob_count > entry_count
        ):
            raise ReleaseGateError("staged_index_scan evidence coverage is incomplete")
        _sha256(digest, name="gates.staged_index_scan evidence.index_entry_index_sha256")
        candidate_count, candidate_digest = _candidate_tree_index(root, candidate)
        if entry_count != candidate_count or digest != candidate_digest:
            raise ReleaseGateError("staged_index_scan does not reconstruct the candidate tree")
        return

    if evidence.get("candidate_commit") != candidate:
        raise ReleaseGateError(f"gates.{gate_name} evidence is bound to another commit")

    if gate_name in QUALITY_GATE_NAMES:
        _exact_keys(
            evidence,
            {
                "schema_version",
                "record_kind",
                "status",
                "candidate_commit",
                "created_at_utc",
                "gates",
                "gate_statuses",
                "scope",
                "ci_evidence_scope",
                "external_completed_ci_required_for_release_inventory",
                "gate_evidence",
                "tool_versions",
                "machine",
                "record_sha256",
            },
            name=f"gates.{gate_name} quality evidence",
        )
        if (
            evidence.get("record_kind") != QUALITY_EVIDENCE_KIND
            or evidence.get("ci_evidence_scope") != QUALITY_EVIDENCE_SCOPE
            or evidence.get("external_completed_ci_required_for_release_inventory") is not True
        ):
            raise ReleaseGateError(f"gates.{gate_name} quality evidence scope differs")
        declared_gates = [
            _text(value, name=f"gates.{gate_name} evidence gate")
            for value in _array(evidence.get("gates"), name=f"gates.{gate_name} evidence.gates")
        ]
        covered = set(declared_gates)
        expected_gates = set(QUALITY_GATE_NAMES)
        statuses = _mapping(
            evidence.get("gate_statuses"), name=f"gates.{gate_name} evidence.gate_statuses"
        )
        gate_evidence = _mapping(
            evidence.get("gate_evidence"), name=f"gates.{gate_name} evidence.gate_evidence"
        )
        executed_gates = expected_gates - {"ci_quality_proxy"}
        tools = _mapping(
            evidence.get("tool_versions"), name=f"gates.{gate_name} evidence.tool_versions"
        )
        machine = _mapping(evidence.get("machine"), name=f"gates.{gate_name} evidence.machine")
        if (
            len(covered) != len(declared_gates)
            or covered != expected_gates
            or set(statuses) != covered
            or any(value not in {"pass", "fail"} for value in statuses.values())
            or ("pass" if all(value == "pass" for value in statuses.values()) else "fail")
            != evidence.get("status")
            or gate_name not in covered
            or statuses.get(gate_name) != "pass"
            or set(gate_evidence) != executed_gates
            or set(tools) != {"python", "uv", "pytest", "ruff", "mypy"}
            or any(not isinstance(value, str) or not value for value in tools.values())
            or not str(tools.get("uv", "")).startswith("uv 0.11.29")
            or set(machine) != {"runner_os", "architecture", "platform"}
            or machine.get("runner_os") != "Linux"
            or any(not isinstance(machine.get(key), str) or not machine.get(key) for key in machine)
            or evidence.get("scope")
            != "upstream synthetic-validation matrix and preceding release-security steps"
        ):
            raise ReleaseGateError(f"gates.{gate_name} quality coverage is inconsistent")
        _timestamp(evidence.get("created_at_utc"))
        for executed_gate in sorted(executed_gates):
            item = _mapping(
                gate_evidence[executed_gate],
                name=f"gates.{gate_name} evidence.gate_evidence.{executed_gate}",
            )
            if set(item) != {
                "exit_code",
                "started_at_utc",
                "completed_at_utc",
                "log_path",
                "log_size_bytes",
                "log_sha256",
            }:
                raise ReleaseGateError(f"quality gate {executed_gate} provenance keys differ")
            exit_code = item.get("exit_code")
            started = _timestamp(item.get("started_at_utc"))
            completed = _timestamp(item.get("completed_at_utc"))
            if datetime.fromisoformat(completed[:-1] + "+00:00") < datetime.fromisoformat(
                started[:-1] + "+00:00"
            ):
                raise ReleaseGateError(f"quality gate {executed_gate} timestamps are reversed")
            if (
                not isinstance(exit_code, int)
                or isinstance(exit_code, bool)
                or (statuses[executed_gate] == "pass" and exit_code != 0)
                or (statuses[executed_gate] == "fail" and exit_code == 0)
            ):
                raise ReleaseGateError(f"quality gate {executed_gate} exit code is inconsistent")
            log_reference = _evidence_reference(
                root,
                {
                    "path": item.get("log_path"),
                    "expected_sha256": item.get("log_sha256"),
                },
                name=f"quality gate {executed_gate} log",
            )
            if item.get("log_size_bytes") != log_reference["size_bytes"]:
                raise ReleaseGateError(f"quality gate {executed_gate} log size differs")
        return

    if gate_name == "repository_scan":
        _exact_keys(
            evidence,
            {
                "schema_version",
                "record_kind",
                "status",
                "created_at_utc",
                "candidate_commit",
                "scan_scope",
                "policy",
                "commit_count",
                "unique_blob_count",
                "tree_entry_observation_count",
                "ref_count",
                "tag_ref_count",
                "validated_annotated_tag_count",
                "invalid_ref_target_count",
                "ref_metadata_index_sha256",
                "shallow_repository",
                "replace_ref_count",
                "grafts_file_present",
                "scanned_object_index_sha256",
                "violations",
                "raw_worktree_paths_opened",
                "record_sha256",
            },
            name="gates.repository_scan evidence",
        )
        _timestamp(evidence.get("created_at_utc"))
        commit_count = evidence.get("commit_count")
        unique_blob_count = evidence.get("unique_blob_count")
        observation_count = evidence.get("tree_entry_observation_count")
        candidate_policy_payload = _git_blob_payload(
            root, candidate, policy_reference, name="release gate policy"
        )
        candidate_policy = _strict_json_bytes(
            candidate_policy_payload, name="candidate release gate policy"
        )
        ref_metrics, ref_violations = inspect_git_ref_state(
            root, candidate_policy, candidate=candidate
        )
        if ref_violations:
            raise ReleaseGateError("repository_scan current Git ref state is unsafe")
        for name, value in {
            "commit_count": commit_count,
            "unique_blob_count": unique_blob_count,
            "tree_entry_observation_count": observation_count,
            "ref_count": evidence.get("ref_count"),
            "tag_ref_count": evidence.get("tag_ref_count"),
            "validated_annotated_tag_count": evidence.get("validated_annotated_tag_count"),
            "invalid_ref_target_count": evidence.get("invalid_ref_target_count"),
        }.items():
            minimum = (
                1
                if name
                in {
                    "commit_count",
                    "unique_blob_count",
                    "tree_entry_observation_count",
                    "ref_count",
                }
                else 0
            )
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ReleaseGateError(f"repository_scan {name} is incomplete")
        assert isinstance(unique_blob_count, int)
        assert isinstance(observation_count, int)
        if observation_count < unique_blob_count:
            raise ReleaseGateError("repository_scan observation coverage is incomplete")
        _sha256(
            evidence.get("scanned_object_index_sha256"),
            name="repository_scan.scanned_object_index_sha256",
        )
        _sha256(
            evidence.get("ref_metadata_index_sha256"),
            name="repository_scan.ref_metadata_index_sha256",
        )
        for metric in (
            "ref_count",
            "tag_ref_count",
            "validated_annotated_tag_count",
            "invalid_ref_target_count",
            "ref_metadata_index_sha256",
        ):
            if evidence.get(metric) != ref_metrics[metric]:
                raise ReleaseGateError(f"repository_scan {metric} differs from current Git refs")
        _exact_integer(
            evidence.get("replace_ref_count"),
            name="repository_scan.replace_ref_count",
            expected=0,
        )
        if (
            evidence.get("record_kind") != SCAN_KIND
            or evidence.get("policy") != policy_reference
            or evidence.get("scan_scope") != "candidate_tree_and_complete_reachable_git_history"
            or evidence.get("raw_worktree_paths_opened") is not False
            or evidence.get("shallow_repository") is not False
            or evidence.get("grafts_file_present") is not False
            or _array(evidence.get("violations"), name="repository_scan violations")
        ):
            raise ReleaseGateError("repository_scan evidence is incomplete or unsafe")
        return

    if gate_name == "secret_scan":
        _exact_keys(
            evidence,
            {
                "schema_version",
                "record_kind",
                "status",
                "created_at_utc",
                "candidate_commit",
                "scan_scope",
                "ref_metadata_index_sha256",
                "validated_annotated_tag_count",
                "gitleaks_version",
                "gitleaks_exit_code",
                "execution_status",
                "policy",
                "config",
                "ignore",
                "raw_report",
                "finding_count",
                "record_sha256",
            },
            name="gates.secret_scan evidence",
        )
        _timestamp(evidence.get("created_at_utc"))
        finding_count = evidence.get("finding_count")
        config = _mapping(evidence.get("config"), name="secret_scan config")
        ignore = _mapping(evidence.get("ignore"), name="secret_scan ignore")
        raw_report = _mapping(evidence.get("raw_report"), name="secret_scan raw_report")
        for name, reference in {"config": config, "ignore": ignore}.items():
            _exact_keys(
                reference,
                {"path", "size_bytes", "file_sha256"},
                name=f"secret_scan {name}",
            )
        _exact_keys(
            raw_report,
            {"basename", "size_bytes", "file_sha256"},
            name="secret_scan raw_report",
        )
        raw_basename = _text(raw_report.get("basename"), name="secret_scan raw report basename")
        raw_size = raw_report.get("size_bytes")
        if (
            PurePosixPath(raw_basename).name != raw_basename
            or "\\" in raw_basename
            or ":" in raw_basename
            or any(ord(character) < 32 or ord(character) == 127 for character in raw_basename)
            or not isinstance(raw_size, int)
            or isinstance(raw_size, bool)
            or raw_size < 0
        ):
            raise ReleaseGateError("secret_scan raw report reference is invalid")
        _sha256(raw_report.get("file_sha256"), name="secret_scan raw report file_sha256")
        observed_config = _git_blob_reference(
            root, candidate, config, name="Gitleaks configuration"
        )
        observed_ignore = _git_blob_reference(root, candidate, ignore, name="Gitleaks ignore file")
        policy_payload = _git_blob_payload(
            root, candidate, policy_reference, name="release gate policy"
        )
        candidate_policy = _strict_json_bytes(policy_payload, name="candidate release gate policy")
        ref_metrics, ref_violations = inspect_git_ref_state(
            root, candidate_policy, candidate=candidate
        )
        if ref_violations:
            raise ReleaseGateError("secret_scan current Git ref state is unsafe")
        gitleaks_policy = _mapping(
            candidate_policy.get("gitleaks"), name="candidate Gitleaks policy"
        )
        ignore_payload = _git_blob_payload(root, candidate, ignore, name="Gitleaks ignore file")
        try:
            ignore_text = ignore_payload.decode("utf-8-sig")
        except UnicodeError as exc:
            raise ReleaseGateError("Gitleaks ignore file must be UTF-8") from exc
        _validate_gitleaks_ignore_entries(candidate_policy, ignore_text)
        _exact_integer(
            evidence.get("gitleaks_exit_code"),
            name="secret_scan.gitleaks_exit_code",
            expected=0,
        )
        if (
            evidence.get("record_kind") != SECRET_KIND
            or evidence.get("policy") != policy_reference
            or evidence.get("scan_scope") != SECRET_SCAN_SCOPE
            or evidence.get("ref_metadata_index_sha256") != ref_metrics["ref_metadata_index_sha256"]
            or evidence.get("validated_annotated_tag_count")
            != ref_metrics["validated_annotated_tag_count"]
            or evidence.get("gitleaks_version") != gitleaks_policy.get("version")
            or gitleaks_policy.get("required_scope") != SECRET_SCAN_SCOPE
            or config.get("path") != gitleaks_policy.get("config_path")
            or config.get("file_sha256") != gitleaks_policy.get("config_sha256")
            or ignore.get("path") != gitleaks_policy.get("ignore_path")
            or evidence.get("execution_status") != "completed"
            or observed_config != config
            or observed_ignore != ignore
            or not isinstance(finding_count, int)
            or isinstance(finding_count, bool)
            or finding_count != 0
        ):
            raise ReleaseGateError("secret_scan evidence is incomplete or inconsistent")
        return

    if gate_name == "license_audit":
        _exact_keys(
            evidence,
            {
                "schema_version",
                "record_kind",
                "status",
                "created_at_utc",
                "candidate_commit",
                "pip_licenses_version",
                "policy",
                "inventory",
                "audit_environment",
                "packages",
                "package_count",
                "normalized_inventory_sha256",
                "exceptions_applied",
                "violations",
                "record_sha256",
            },
            name="gates.license_audit evidence",
        )
        _timestamp(evidence.get("created_at_utc"))
        package_count = evidence.get("package_count")
        inventory = _mapping(evidence.get("inventory"), name="license_audit inventory")
        _exact_keys(
            inventory,
            {"basename", "size_bytes", "file_sha256"},
            name="license_audit inventory",
        )
        inventory_basename = _text(
            inventory.get("basename"), name="license_audit inventory basename"
        )
        inventory_size = inventory.get("size_bytes")
        if (
            PurePosixPath(inventory_basename).name != inventory_basename
            or "\\" in inventory_basename
            or ":" in inventory_basename
            or any(ord(character) < 32 or ord(character) == 127 for character in inventory_basename)
            or not isinstance(inventory_size, int)
            or isinstance(inventory_size, bool)
            or inventory_size < 0
        ):
            raise ReleaseGateError("license_audit inventory reference is invalid")
        _sha256(inventory.get("file_sha256"), name="license_audit inventory file_sha256")
        _sha256(
            evidence.get("normalized_inventory_sha256"),
            name="license_audit normalized_inventory_sha256",
        )
        validate_license_audit_semantics(
            evidence,
            root=root,
            candidate=candidate,
            policy=policy,
        )
        if (
            evidence.get("record_kind") != LICENSE_KIND
            or evidence.get("policy") != policy_reference
            or evidence.get("pip_licenses_version") != "5.5.5"
            or evidence.get("audit_environment") != {"system": "Linux", "machine": "x86_64"}
            or not isinstance(package_count, int)
            or isinstance(package_count, bool)
            or package_count < 1
            or _array(evidence.get("violations"), name="license_audit violations")
        ):
            raise ReleaseGateError("license_audit evidence is incomplete or inconsistent")
        return

    raise ReleaseGateError(f"no semantic validator exists for gate {gate_name}")


def assemble_report(
    *,
    repository_root: str | Path,
    policy_path: str | Path,
    mode: str,
    created_at_utc: str,
    spec_path: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble a truthful tracked precommit report or exact-candidate attestation."""

    root = _repository(repository_root)
    _policy_value, policy_reference = _policy(policy_path)
    head_result = _git(root, ("rev-parse", "HEAD"))
    assert isinstance(head_result.stdout, str)
    head = _commit(head_result.stdout.strip(), name="HEAD")
    if mode == "tracked_precommit_report":
        if spec_path is not None:
            raise ReleaseGateError("tracked_precommit_report does not accept an attestation spec")
        return _self_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "record_kind": REPORT_KIND,
                "mode": mode,
                "status": "pending",
                "created_at_utc": _timestamp(created_at_utc),
                "repository": {
                    "parent_commit": head,
                    "content_commit": None,
                    "worktree_clean": False,
                },
                "policy": policy_reference,
                "gates": {
                    name: {"status": "not_run", "evidence": None} for name in TRACKED_GATE_NAMES
                },
                "external_candidate_attestation": {
                    "status": "pending",
                    "expected_root": ".audit/release-attestations/<candidate_commit>/",
                    "report_path": None,
                },
            }
        )
    if mode != "exact_candidate_attestation" or spec_path is None:
        raise ReleaseGateError("exact_candidate_attestation requires --spec")
    spec = _strict_json(spec_path, name="exact candidate attestation spec")
    expected_keys = {
        "schema_version",
        "spec_kind",
        "created_at_utc",
        "candidate_commit",
        "parent_commit",
        "worktree_clean",
        "gates",
    }
    if set(spec) != expected_keys:
        raise ReleaseGateError("exact candidate attestation spec keys differ")
    if spec.get("schema_version") != "1.0.0" or spec.get("spec_kind") != (
        "final_release_gate_attestation_spec"
    ):
        raise ReleaseGateError("exact candidate attestation spec version or kind differs")
    candidate = _commit(spec.get("candidate_commit"), name="candidate_commit")
    parent = _commit(spec.get("parent_commit"), name="parent_commit")
    if spec.get("worktree_clean") is not True:
        raise ReleaseGateError("exact candidate attestation must record a clean worktree")
    if candidate != head:
        raise ReleaseGateError("candidate_commit differs from repository HEAD")
    status_result = _git(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    assert isinstance(status_result.stdout, str)
    if status_result.stdout.strip():
        raise ReleaseGateError("exact candidate worktree is not clean")
    parent_result = _git(root, ("rev-parse", f"{candidate}^"), check=False)
    if parent_result.returncode != 0:
        raise ReleaseGateError("exact candidate must have a resolvable first parent")
    assert isinstance(parent_result.stdout, str)
    actual_parent = _commit(parent_result.stdout.strip(), name="candidate first parent")
    if parent != actual_parent:
        raise ReleaseGateError("declared parent_commit differs from candidate first parent")
    gates_spec = _mapping(spec.get("gates"), name="gates")
    if set(gates_spec) != set(EXACT_GATE_NAMES):
        raise ReleaseGateError("exact candidate attestation must include every release gate")
    gates: dict[str, Any] = {}
    for name in EXACT_GATE_NAMES:
        gate = _mapping(gates_spec[name], name=f"gates.{name}")
        if set(gate) != {"status", "path", "expected_sha256"} or gate.get("status") != "pass":
            raise ReleaseGateError(f"gates.{name} must be a pinned pass")
        reference, payload = _evidence_snapshot(root, gate, name=f"gates.{name}")
        evidence = _strict_json_bytes(payload, name=f"gates.{name} evidence")
        _validate_exact_gate_evidence(
            root,
            gate_name=name,
            evidence=evidence,
            candidate=candidate,
            parent=parent,
            policy=_policy_value,
            policy_reference=policy_reference,
        )
        gates[name] = {"status": "pass", "evidence": reference}
    report_path = f".audit/release-attestations/{candidate}/final_release_gate_report.json"
    return _self_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "record_kind": REPORT_KIND,
            "mode": mode,
            "status": "pass",
            "created_at_utc": _timestamp(spec.get("created_at_utc")),
            "repository": {
                "parent_commit": parent,
                "content_commit": candidate,
                "worktree_clean": True,
            },
            "policy": policy_reference,
            "gates": gates,
            "external_candidate_attestation": {
                "status": "complete",
                "expected_root": ".audit/release-attestations/<candidate_commit>/",
                "report_path": report_path,
            },
        }
    )


def validate_report(
    report_path: str | Path, *, repository_root: str | Path | None = None
) -> dict[str, Any]:
    """Validate self-hash, exact keys, and mode-specific release semantics."""

    try:
        value = _strict_json(report_path, name="final release gate report")
        _verify_self_hash(value, name="final release gate report")
        expected = {
            "schema_version",
            "record_kind",
            "mode",
            "status",
            "created_at_utc",
            "repository",
            "policy",
            "gates",
            "external_candidate_attestation",
            "record_sha256",
        }
        if set(value) != expected:
            raise ReleaseGateError("final release gate report keys differ")
        if value.get("schema_version") != SCHEMA_VERSION or value.get("record_kind") != REPORT_KIND:
            raise ReleaseGateError("final release gate report version or kind differs")
        _timestamp(value.get("created_at_utc"))
        repository = _mapping(value.get("repository"), name="repository")
        if set(repository) != {"parent_commit", "content_commit", "worktree_clean"}:
            raise ReleaseGateError("repository binding keys differ")
        _commit(repository.get("parent_commit"), name="repository.parent_commit")
        content = repository.get("content_commit")
        if content is not None:
            _commit(content, name="repository.content_commit")
        if not isinstance(repository.get("worktree_clean"), bool):
            raise ReleaseGateError("repository.worktree_clean must be boolean")
        policy_ref = _file_reference_shape(value.get("policy"), name="policy")
        gates = _mapping(value.get("gates"), name="gates")
        mode = value.get("mode")
        external = _mapping(
            value.get("external_candidate_attestation"), name="external_candidate_attestation"
        )
        if set(external) != {"status", "expected_root", "report_path"}:
            raise ReleaseGateError("external_candidate_attestation keys differ")
        if external.get("expected_root") != ".audit/release-attestations/<candidate_commit>/":
            raise ReleaseGateError("external candidate attestation root differs")
        if mode == "tracked_precommit_report":
            if set(gates) != set(TRACKED_GATE_NAMES):
                raise ReleaseGateError("tracked precommit release gate set differs")
            for name, raw_gate in gates.items():
                gate = _mapping(raw_gate, name=f"gates.{name}")
                if set(gate) != {"status", "evidence"}:
                    raise ReleaseGateError(f"gates.{name} keys differ")
                if gate.get("status") != "not_run" or gate.get("evidence") is not None:
                    raise ReleaseGateError(f"gates.{name} must be not_run with null evidence")
            if (
                value.get("status") != "pending"
                or content is not None
                or repository.get("worktree_clean") is not False
                or external.get("status") != "pending"
                or external.get("report_path") is not None
            ):
                raise ReleaseGateError("tracked precommit report overstates release completion")
        elif mode == "exact_candidate_attestation":
            if set(gates) != set(EXACT_GATE_NAMES):
                raise ReleaseGateError("exact candidate release gate set differs")
            candidate = _commit(content, name="repository.content_commit")
            for name, raw_gate in gates.items():
                gate = _mapping(raw_gate, name=f"gates.{name}")
                if set(gate) != {"status", "evidence"}:
                    raise ReleaseGateError(f"gates.{name} keys differ")
                if gate.get("status") != "pass":
                    raise ReleaseGateError(f"gates.{name} must pass")
                _file_reference_shape(gate.get("evidence"), name=f"gates.{name}.evidence")
            expected_path = (
                f".audit/release-attestations/{candidate}/final_release_gate_report.json"
            )
            if (
                value.get("status") != "pass"
                or repository.get("worktree_clean") is not True
                or external.get("status") != "complete"
                or external.get("report_path") != expected_path
            ):
                raise ReleaseGateError("exact candidate attestation is incomplete")
        else:
            raise ReleaseGateError("release gate mode is invalid")
        if mode == "exact_candidate_attestation" and repository_root is None:
            raise ReleaseGateError(
                "exact candidate attestation validation requires repository_root"
            )
        if repository_root is not None:
            root = _repository(repository_root)
            if mode == "tracked_precommit_report":
                observed_ref = _tracked_report_policy_reference(
                    root,
                    report_path,
                    _commit(
                        repository.get("parent_commit"),
                        name="repository.parent_commit",
                    ),
                    policy_ref,
                )
            else:
                observed_ref = _git_blob_reference(
                    root,
                    _commit(content, name="repository.content_commit"),
                    policy_ref,
                    name="policy",
                )
            if observed_ref != policy_ref:
                raise ReleaseGateError("policy reference size differs")
            if mode == "exact_candidate_attestation":
                assert content is not None
                candidate_policy = _strict_json_bytes(
                    _git_blob_payload(root, content, policy_ref, name="policy"),
                    name="candidate release gate policy",
                )
                parent_result = _git(root, ("rev-parse", f"{content}^"), check=False)
                if parent_result.returncode != 0:
                    raise ReleaseGateError("exact candidate first parent cannot be resolved")
                assert isinstance(parent_result.stdout, str)
                actual_parent = _commit(parent_result.stdout.strip(), name="candidate first parent")
                if repository.get("parent_commit") != actual_parent:
                    raise ReleaseGateError(
                        "exact candidate report parent differs from candidate first parent"
                    )
                for name, raw_gate in gates.items():
                    gate = _mapping(raw_gate, name=f"gates.{name}")
                    reference = _mapping(gate.get("evidence"), name=f"gates.{name}.evidence")
                    observed, payload = _evidence_snapshot(
                        root,
                        {
                            "path": reference.get("path"),
                            "expected_sha256": reference.get("file_sha256"),
                        },
                        name=f"gates.{name}",
                    )
                    if observed != reference:
                        raise ReleaseGateError(f"gates.{name} evidence size differs")
                    evidence = _strict_json_bytes(payload, name=f"gates.{name} evidence")
                    _validate_exact_gate_evidence(
                        root,
                        gate_name=name,
                        evidence=evidence,
                        candidate=content,
                        parent=actual_parent,
                        policy=candidate_policy,
                        policy_reference=policy_ref,
                    )
        return {
            "valid": True,
            "mode": mode,
            "status": value["status"],
            "record_sha256": value["record_sha256"],
            "errors": [],
        }
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)]}


def _write_and_status(record: dict[str, Any], args: argparse.Namespace) -> int:
    root = _repository(args.repository_root)
    _write_new(record, args.output, root=root)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] in {"pass", "pending"} else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_scan = subparsers.add_parser("scan-index")
    index_scan.add_argument("--repository-root", type=Path, required=True)
    index_scan.add_argument("--policy", type=Path, required=True)
    index_scan.add_argument("--created-at-utc", required=True)
    index_scan.add_argument("--output", type=Path, required=True)
    scan = subparsers.add_parser("scan-repository")
    scan.add_argument("--repository-root", type=Path, required=True)
    scan.add_argument("--candidate-commit", required=True)
    scan.add_argument("--policy", type=Path, required=True)
    scan.add_argument("--created-at-utc", required=True)
    scan.add_argument("--output", type=Path, required=True)
    secret = subparsers.add_parser("validate-secret-scan")
    secret.add_argument("--repository-root", type=Path, required=True)
    secret.add_argument("--report", type=Path, required=True)
    secret.add_argument("--candidate-commit", required=True)
    secret.add_argument("--policy", type=Path, required=True)
    secret.add_argument("--config", type=Path, required=True)
    secret.add_argument("--ignore", type=Path, required=True)
    secret.add_argument("--gitleaks-version", required=True)
    secret.add_argument("--gitleaks-exit-code", type=int, required=True)
    secret.add_argument("--created-at-utc", required=True)
    secret.add_argument("--output", type=Path, required=True)
    licenses = subparsers.add_parser("audit-licenses")
    licenses.add_argument("--repository-root", type=Path, required=True)
    licenses.add_argument("--inventory", type=Path, required=True)
    licenses.add_argument("--candidate-commit", required=True)
    licenses.add_argument("--policy", type=Path, required=True)
    licenses.add_argument("--pip-licenses-version", required=True)
    licenses.add_argument("--created-at-utc", required=True)
    licenses.add_argument("--output", type=Path, required=True)
    ci_bundle = subparsers.add_parser("validate-ci-bundle")
    ci_bundle.add_argument("--repository-root", type=Path, required=True)
    ci_bundle.add_argument("--candidate-commit", required=True)
    ci_bundle.add_argument("--policy", type=Path, required=True)
    ci_bundle.add_argument("--evidence-root", type=Path, required=True)
    ci_bundle.add_argument("--created-at-utc", required=True)
    ci_bundle.add_argument("--output", type=Path, required=True)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--repository-root", type=Path, required=True)
    assemble.add_argument("--policy", type=Path, required=True)
    assemble.add_argument(
        "--mode",
        choices=("tracked_precommit_report", "exact_candidate_attestation"),
        required=True,
    )
    assemble.add_argument("--created-at-utc", required=True)
    assemble.add_argument("--spec", type=Path)
    assemble.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--repository-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "scan-index":
            record = scan_index(
                repository_root=args.repository_root,
                policy_path=args.policy,
                created_at_utc=args.created_at_utc,
            )
            return _write_and_status(record, args)
        if args.command == "scan-repository":
            record = scan_repository(
                repository_root=args.repository_root,
                candidate_commit=args.candidate_commit,
                policy_path=args.policy,
                created_at_utc=args.created_at_utc,
            )
            return _write_and_status(record, args)
        if args.command == "validate-secret-scan":
            record = validate_secret_scan(
                repository_root=args.repository_root,
                report_path=args.report,
                candidate_commit=args.candidate_commit,
                policy_path=args.policy,
                config_path=args.config,
                ignore_path=args.ignore,
                gitleaks_version=args.gitleaks_version,
                gitleaks_exit_code=args.gitleaks_exit_code,
                created_at_utc=args.created_at_utc,
            )
            return _write_and_status(record, args)
        if args.command == "audit-licenses":
            record = audit_licenses(
                repository_root=args.repository_root,
                inventory_path=args.inventory,
                candidate_commit=args.candidate_commit,
                policy_path=args.policy,
                pip_licenses_version=args.pip_licenses_version,
                created_at_utc=args.created_at_utc,
            )
            return _write_and_status(record, args)
        if args.command == "validate-ci-bundle":
            record = validate_ci_gate_bundle(
                repository_root=args.repository_root,
                candidate_commit=args.candidate_commit,
                policy_path=args.policy,
                evidence_root=args.evidence_root,
                created_at_utc=args.created_at_utc,
            )
            return _write_and_status(record, args)
        if args.command == "assemble":
            record = assemble_report(
                repository_root=args.repository_root,
                policy_path=args.policy,
                mode=args.mode,
                created_at_utc=args.created_at_utc,
                spec_path=args.spec,
            )
            return _write_and_status(record, args)
        report = validate_report(args.report, repository_root=args.repository_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["valid"] is True else 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
