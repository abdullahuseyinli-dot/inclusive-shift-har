"""Strong engineered-feature baselines with fixed class/probability alignment."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from sklearn.svm import SVC  # type: ignore[import-untyped]
from xgboost import XGBClassifier

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.preprocessing.features import extract_engineered_features


@dataclass(frozen=True)
class ClassicalConfig:
    model_name: str
    num_classes: int
    seed: int
    xgboost_device: str = "cpu"

    def __post_init__(self) -> None:
        choices = {"random_forest", "xgboost", "svm_rbf", "logistic_regression"}
        if self.model_name not in choices:
            raise ValueError(
                f"unknown classical baseline {self.model_name!r}; choices: {sorted(choices)}"
            )
        if self.num_classes < 2 or self.seed < 0:
            raise ValueError("classical baseline requires at least two classes and a valid seed")
        if self.xgboost_device not in {"cpu", "cuda"}:
            raise ValueError("xgboost_device must be cpu or cuda")


@dataclass
class FittedClassicalModel:
    estimator: Any
    config: ClassicalConfig
    feature_names: tuple[str, ...]
    channel_names: tuple[str, ...]
    training_participants: tuple[str, ...]
    lineage: dict[str, Any]


def _build_estimator(config: ClassicalConfig) -> Any:
    if config.model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=config.seed,
            n_jobs=1,
        )
    if config.model_name == "xgboost":
        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=2.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=config.num_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            device=config.xgboost_device,
            random_state=config.seed,
            n_jobs=1,
        )
    if config.model_name == "svm_rbf":
        return Pipeline(
            (
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=10.0,
                        kernel="rbf",
                        gamma="scale",
                        class_weight="balanced",
                        probability=False,
                        random_state=config.seed,
                    ),
                ),
            )
        )
    return Pipeline(
        (
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=config.seed,
                ),
            ),
        )
    )


def _estimator_classes(estimator: Any) -> NDArray[np.int64]:
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    if classes.ndim != 1:
        raise ValueError("classical estimator returned an invalid class index schema")
    return classes


def fit_classical_model(
    train_windows: NDArray[np.float32],
    train_labels: NDArray[np.int64],
    train_participant_ids: list[str],
    *,
    config: ClassicalConfig,
    channel_names: tuple[str, ...],
    lineage: dict[str, Any],
) -> FittedClassicalModel:
    """Fit an engineered-feature baseline on a source training partition only."""

    if train_windows.shape[0] != train_labels.size or train_labels.size != len(
        train_participant_ids
    ):
        raise ValueError("classical training arrays are not aligned")
    observed_classes = np.unique(train_labels)
    expected_classes = np.arange(config.num_classes, dtype=np.int64)
    if not np.array_equal(observed_classes, expected_classes):
        raise ValueError("training partition does not contain the full locked class schema")
    features = extract_engineered_features(train_windows, channel_names=channel_names)
    estimator = _build_estimator(config)
    estimator.fit(features.values, train_labels)
    if not np.array_equal(_estimator_classes(estimator), expected_classes):
        raise AssertionError("fitted estimator class order differs from the locked ontology")
    return FittedClassicalModel(
        estimator=estimator,
        config=config,
        feature_names=features.names,
        channel_names=channel_names,
        training_participants=tuple(sorted(set(train_participant_ids))),
        lineage=lineage,
    )


def _stable_softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return np.asarray(exponential / exponential.sum(axis=1, keepdims=True), dtype=np.float64)


def predict_classical_model(
    fitted: FittedClassicalModel,
    windows: NDArray[np.float32],
    labels: NDArray[np.int64],
    participant_ids: list[str],
    *,
    class_names: tuple[str, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Predict with explicit probability-column verification and participant metrics."""

    features = extract_engineered_features(windows, channel_names=fitted.channel_names)
    if features.names != fitted.feature_names:
        raise ValueError("engineered feature schema changed after model fitting")
    estimator = fitted.estimator
    if hasattr(estimator, "predict_proba") and fitted.config.model_name != "svm_rbf":
        probabilities = np.asarray(estimator.predict_proba(features.values), dtype=np.float64)
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    else:
        decision = np.asarray(estimator.decision_function(features.values), dtype=np.float64)
        if decision.ndim == 1:
            decision = np.stack((-decision, decision), axis=1)
        logits = decision
        probabilities = _stable_softmax(logits)
    expected_classes = np.arange(fitted.config.num_classes, dtype=np.int64)
    if not np.array_equal(_estimator_classes(estimator), expected_classes):
        raise ValueError("estimator probability columns no longer match the locked class order")
    if probabilities.shape != (labels.size, fitted.config.num_classes):
        raise ValueError("classical predictions are not aligned with labels/classes")
    report = classification_report(
        labels,
        probabilities,
        participant_ids,
        class_names=class_names,
    )
    return logits, probabilities, report


def save_classical_checkpoint_create_only(
    fitted: FittedClassicalModel,
    path: Path,
) -> dict[str, Any]:
    """Persist a trusted local pickle under an immutable create-only run path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "warning": "Load only trusted local artifacts; pickle is executable serialization.",
        "configuration": asdict(fitted.config),
        "configuration_sha256": canonical_json_sha256(asdict(fitted.config)),
        "feature_names": fitted.feature_names,
        "channel_names": fitted.channel_names,
        "training_participants": fitted.training_participants,
        "lineage": fitted.lineage,
        "estimator": fitted.estimator,
    }
    with path.open("xb") as stream:
        pickle.dump(payload, stream, protocol=5)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "configuration_sha256": payload["configuration_sha256"],
    }


def load_classical_checkpoint(path: Path) -> FittedClassicalModel:
    """Load a trusted local classical artifact and reconstruct its fixed schemas."""

    with path.open("rb") as stream:
        payload = pickle.load(stream)
    config = ClassicalConfig(**payload["configuration"])
    expected_hash = canonical_json_sha256(asdict(config))
    if expected_hash != payload["configuration_sha256"]:
        raise ValueError("classical checkpoint configuration hash mismatch")
    return FittedClassicalModel(
        estimator=payload["estimator"],
        config=config,
        feature_names=tuple(payload["feature_names"]),
        channel_names=tuple(payload["channel_names"]),
        training_participants=tuple(payload["training_participants"]),
        lineage=dict(payload["lineage"]),
    )
