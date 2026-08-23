"""Locked label ontology loading and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from inclusive_shift_har.manifests.canonical import canonical_json_sha256

EXPECTED_RELEASED_LABELS = frozenset(
    {"Ramp ascent", "Ramp descent", "Sitting", "Standing", "Walking", "jogging"}
)


@dataclass(frozen=True)
class LockedOntology:
    ontology_id: str
    config_path: str
    config_sha256: str
    released_labels: dict[str, str]
    tracks: dict[str, Mapping[str, Any]]
    class_orders: dict[str, tuple[str, ...]]
    explicit_class_order: bool

    def labels_for_released_label(self, released_label: str) -> dict[str, str]:
        if released_label not in self.released_labels:
            raise ValueError(f"released label is absent from locked ontology: {released_label!r}")
        canonical = self.released_labels[released_label]
        labels: dict[str, str] = {}
        inclusive = self.tracks["inclusive_native"]["classes"]
        if canonical in inclusive:
            labels["inclusive_native"] = str(inclusive[canonical])
        functional = self.tracks["functional_core"]["classes"]
        if canonical in functional:
            labels["functional_core"] = str(functional[canonical])
        cross_source = self.tracks["cross_source_core"]["exact_classes"]
        if canonical in cross_source:
            labels["cross_source_core_exact"] = str(cross_source[canonical])
        return labels

    def runnable_track_schema(self, track_name: str) -> dict[str, Any]:
        if track_name not in self.class_orders:
            raise ValueError(f"track has no runnable numeric class schema: {track_name}")
        order = self.class_orders[track_name]
        schema: dict[str, Any] = {
            "class_count": len(order),
            "class_order": list(order),
            "index_by_class": {label: index for index, label in enumerate(order)},
            "track": track_name,
        }
        schema["class_schema_sha256"] = canonical_json_sha256(schema)
        return schema


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_locked_ontology(path: str | Path) -> LockedOntology:
    source = Path(path)
    parsed = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _require_mapping(parsed, name="ontology root")
    if root.get("schema_version") != "1.0.0":
        raise ValueError("ontology schema_version must equal '1.0.0'")
    if root.get("status") != "locked_conditional_released_block_protocol":
        raise ValueError("ontology status is not locked for the released-block protocol")
    if root.get("dataset_id") != "inclusivehar_v4":
        raise ValueError("ontology dataset_id must equal 'inclusivehar_v4'")
    if root.get("source_label_column") != "label":
        raise ValueError("ontology must use the observed lowercase label column")
    if root.get("source_label_is_model_feature") is not False:
        raise ValueError("source label must remain excluded from model features")

    released_value = _require_mapping(root.get("released_labels"), name="released_labels")
    released_labels = {str(key): str(value) for key, value in released_value.items()}
    if set(released_labels) != EXPECTED_RELEASED_LABELS:
        raise ValueError("released label set differs from the locked InclusiveHAR v4 labels")
    if len(set(released_labels.values())) != len(released_labels):
        raise ValueError("released label canonical names must be unique")

    tracks_value = _require_mapping(root.get("tracks"), name="tracks")
    required_tracks = {
        "ambulatory_walking_sensitivity",
        "cross_source_core",
        "functional_core",
        "inclusive_native",
    }
    if set(tracks_value) != required_tracks:
        raise ValueError("ontology tracks differ from the locked set")
    tracks = {
        str(name): _require_mapping(value, name=f"tracks.{name}")
        for name, value in tracks_value.items()
    }
    if tracks["functional_core"].get("status") != "primary_runnable_after_split_audit":
        raise ValueError("functional_core must remain the primary runnable ontology")
    functional_classes = _require_mapping(
        tracks["functional_core"].get("classes"), name="functional_core.classes"
    )
    if dict(functional_classes) != {
        "sitting": "sitting",
        "standing": "standing",
        "walking": "mobility",
    }:
        raise ValueError("functional_core mapping changed from the locked semantics")
    cross_source = tracks["cross_source_core"]
    if cross_source.get("status") != "blocked_not_classification_ready":
        raise ValueError("cross_source_core must remain blocked until its unblock condition")
    if dict(_require_mapping(cross_source.get("exact_classes"), name="exact_classes")) != {
        "sitting": "sitting"
    }:
        raise ValueError("cross_source exact labels changed")
    if root.get("class_count_policy") != (
        "Read only from this locked ontology; never infer from validation or target labels."
    ):
        raise ValueError("class-count policy is not locked")

    ontology_id = root.get("ontology_id")
    if ontology_id not in {
        "inclusivehar-v4-ontology-v1",
        "inclusivehar-v4-ontology-v1.1",
    }:
        raise ValueError("ontology_id is not a supported locked version")
    expected_orders = {
        "functional_core": ("mobility", "sitting", "standing"),
        "inclusive_native": (
            "jogging",
            "ramp_ascent",
            "ramp_descent",
            "sitting",
            "standing",
            "walking",
        ),
    }
    explicit_class_order = ontology_id == "inclusivehar-v4-ontology-v1.1"
    if explicit_class_order:
        for track_name, expected_order in expected_orders.items():
            order = tracks[track_name].get("class_order")
            if order != list(expected_order):
                raise ValueError(f"{track_name} numeric class order changed")
            classes = _require_mapping(
                tracks[track_name].get("classes"), name=f"{track_name}.classes"
            )
            if set(classes.values()) != set(expected_order):
                raise ValueError(f"{track_name} class order does not cover mapped classes")
    else:
        for track_name in expected_orders:
            if "class_order" in tracks[track_name]:
                raise ValueError("legacy ontology v1 may not silently gain a class order")
    return LockedOntology(
        ontology_id=str(ontology_id),
        config_path=source.as_posix(),
        config_sha256=canonical_json_sha256(root),
        released_labels=released_labels,
        tracks=tracks,
        class_orders=expected_orders,
        explicit_class_order=explicit_class_order,
    )
