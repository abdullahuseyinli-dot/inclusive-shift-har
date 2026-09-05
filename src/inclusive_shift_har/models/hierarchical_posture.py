"""Fixed-budget participant-balanced posture forests; no inference metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]

from inclusive_shift_har.models.classical import ClassicalConfig, _build_estimator

FloatArray = NDArray[np.float64]
VARIANTS = (
    "RandomForest-6ch",
    "PB-RF-6ch",
    "PB-RF-D9",
    "HPF-unweighted",
    "PB-HPF",
    "PB-HPF-shuffled-posture",
)


def participant_class_weights(
    labels: NDArray[np.int64], participants: NDArray[np.str_]
) -> FloatArray:
    """Give every participant equal mass, divided among their observed classes."""
    if labels.ndim != 1 or labels.size == 0 or participants.shape != labels.shape:
        raise ValueError("nonempty aligned participant/class arrays are required")
    weights = np.zeros(labels.size, dtype=np.float64)
    for person in np.unique(participants):
        selected = participants == person
        classes, counts = np.unique(labels[selected], return_counts=True)
        for label, count in zip(classes, counts, strict=True):
            weights[selected & (labels == label)] = 1.0 / (classes.size * count)
    weights /= weights.mean()
    return weights


@dataclass
class FittedPostureForest:
    variant: str
    root: Any
    specialist: Any | None
    feature_count: int
    training_participants: tuple[str, ...]
    shuffled_training_labels: int = 0

    def predict(self, features: FloatArray) -> FloatArray:
        """Accept signal features only: labels and participant IDs are not inputs."""
        if features.ndim != 2 or features.shape[1] != self.feature_count:
            raise ValueError("posture forest inference feature schema differs")
        if not np.isfinite(features).all():
            raise ValueError("posture forest inference features must be finite")
        root = np.asarray(self.root.predict_proba(features), dtype=np.float64)
        if self.specialist is None:
            probability = root
        else:
            posture = np.asarray(self.specialist.predict_proba(features), dtype=np.float64)
            probability = np.column_stack(
                (root[:, 0], root[:, 1] * posture[:, 0], root[:, 1] * posture[:, 1])
            )
        if probability.shape != (len(features), 3) or not np.allclose(
            probability.sum(axis=1), 1.0, atol=1e-12, rtol=0.0
        ):
            raise ValueError("posture forest returned invalid probability columns")
        return probability

    def mechanism(self, features: FloatArray) -> dict[str, Any]:
        if self.specialist is None:
            return {"hierarchy_used": False, "shuffled_training_labels": 0}
        root = np.asarray(self.root.predict_proba(features), dtype=np.float64)
        return {
            "hierarchy_used": True,
            "specialist_evaluated_window_count": len(features),
            "mean_stationary_probability_mass": float(root[:, 1].mean()),
            "root_stationary_argmax_fraction": float(np.mean(root.argmax(axis=1) == 1)),
            "shuffled_training_labels": self.shuffled_training_labels,
        }

    def size_summary(self) -> dict[str, int]:
        estimators = list(self.root.estimators_)
        if self.specialist is not None:
            estimators.extend(self.specialist.estimators_)
        return {
            "tree_count": len(estimators),
            "tree_node_count": sum(int(tree.tree_.node_count) for tree in estimators),
            "feature_count": self.feature_count,
        }


def fit_posture_forest(
    features: FloatArray,
    labels: NDArray[np.int64],
    participants: NDArray[np.str_],
    *,
    variant: str,
    seed: int,
) -> FittedPostureForest:
    """Fit one predeclared forest using only an outer training partition."""
    if variant not in VARIANTS or seed < 0:
        raise ValueError("unknown posture forest variant or invalid seed")
    if features.ndim != 2 or len(features) != labels.size or participants.shape != labels.shape:
        raise ValueError("posture forest training arrays do not align")
    if not np.isfinite(features).all() or not np.array_equal(np.unique(labels), [0, 1, 2]):
        raise ValueError("training requires finite features and the complete three-class ontology")
    hierarchy = variant.startswith("HPF-") or variant.startswith("PB-HPF")
    weighted = variant.startswith("PB-")

    def forest(trees: int, random_seed: int) -> Any:
        return RandomForestClassifier(
            n_estimators=trees,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight=None if weighted else "balanced_subsample",
            random_state=random_seed,
            n_jobs=1,
        )

    if not hierarchy:
        root = (
            _build_estimator(ClassicalConfig("random_forest", num_classes=3, seed=seed))
            if variant == "RandomForest-6ch"
            else forest(500, seed)
        )
        weights = participant_class_weights(labels, participants) if weighted else None
        root.fit(features, labels, sample_weight=weights)
        if not np.array_equal(root.classes_, [0, 1, 2]):
            raise ValueError("flat forest class order differs from the protocol")
        return FittedPostureForest(
            variant, root, None, features.shape[1], tuple(np.unique(participants).tolist())
        )

    binary = (labels != 0).astype(np.int64)
    root = forest(250, seed)
    root.fit(
        features,
        binary,
        sample_weight=participant_class_weights(binary, participants) if weighted else None,
    )
    stationary = labels != 0
    posture_labels = (labels[stationary] - 1).copy()
    posture_people = participants[stationary]
    shuffled = 0
    if variant == "PB-HPF-shuffled-posture":
        rng = np.random.default_rng(seed + 20000)
        original = posture_labels.copy()
        for person in np.unique(posture_people):
            selected = posture_people == person
            posture_labels[selected] = rng.permutation(posture_labels[selected])
        shuffled = int(np.count_nonzero(original != posture_labels))
    specialist = forest(250, seed + 10000)
    specialist.fit(
        features[stationary],
        posture_labels,
        sample_weight=participant_class_weights(posture_labels, posture_people)
        if weighted
        else None,
    )
    if not np.array_equal(root.classes_, [0, 1]) or not np.array_equal(specialist.classes_, [0, 1]):
        raise ValueError("hierarchical forest probability columns differ from the protocol")
    return FittedPostureForest(
        variant,
        root,
        specialist,
        features.shape[1],
        tuple(np.unique(participants).tolist()),
        shuffled,
    )
