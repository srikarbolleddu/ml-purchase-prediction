"""Load the trained pipeline and turn predictions into explanations a visitor can read."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import skops.io as sio
from sklearn.pipeline import Pipeline

from app.features import CATEGORICAL_FEATURES, FIELDS_BY_NAME, NUMERIC_FEATURES, records_to_frame

ARTIFACT_DIR = Path(__file__).parent / "model_artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.skops"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"
SAMPLES_PATH = ARTIFACT_DIR / "samples.json"

# skops refuses to load any type not explicitly trusted. These are the only non-default types a
# scikit-learn Pipeline(ColumnTransformer + OneHotEncoder + DecisionTree) serializes; anything
# else in the file means it is not the artifact we trained, so loading fails closed.
TRUSTED_TYPES = frozenset(
    {
        "builtins.object",
        "numpy.dtypes.Float64DType",
        "numpy.dtypes.Int64DType",
        "numpy.dtypes.ObjectDType",
        "sklearn.tree._tree.Tree",
    }
)


@dataclass(frozen=True)
class EncodedFeature:
    """One column of the encoded matrix, mapped back to the form field it came from."""

    field: str
    category: str | None  # set for one-hot columns


def encoded_features(pipeline: Pipeline) -> list[EncodedFeature]:
    """Column order of `pipeline[:-1].transform(...)`: one-hot categoricals, then numerics."""
    encoder = pipeline.named_steps["prep"].named_transformers_["cat"]
    features = [
        EncodedFeature(field=name, category=str(category))
        for name, categories in zip(CATEGORICAL_FEATURES, encoder.categories_, strict=True)
        for category in categories
    ]
    features.extend(EncodedFeature(field=name, category=None) for name in NUMERIC_FEATURES)
    return features


def save(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    sio.dump(pipeline, path)


def load(path: Path = MODEL_PATH) -> Pipeline:
    untrusted = set(sio.get_untrusted_types(file=path))
    unexpected = untrusted - TRUSTED_TYPES
    if unexpected:
        raise RuntimeError(f"refusing to load model with unexpected types: {sorted(unexpected)}")
    return sio.load(path, trusted=sorted(untrusted))


def _format_number(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.2f}".rstrip("0").rstrip(".")


def _rule_text(feature: EncodedFeature, threshold: float, went_left: bool) -> str:
    field = FIELDS_BY_NAME[feature.field]
    if feature.category is not None:
        options = field.to_schema()["options"]
        value_label = next(o["label"] for o in options if o["value"] == feature.category)
        # One-hot columns are 0/1, split at 0.5: left = "not this category".
        return f"{field.label.rstrip('?')}: {'not ' if went_left else ''}{value_label}"
    if feature.field == "days_since_last_contact" and threshold >= 30:
        return f"Customer was {'' if went_left else 'never '}contacted in a previous campaign"
    unit = f" {field.unit}" if field.unit else ""
    op = "≤" if went_left else ">"
    return f"{field.label} {op} {_format_number(threshold)}{unit}"


class PurchaseModel:
    def __init__(self, pipeline: Pipeline, manifest: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.manifest = manifest
        self._features = encoded_features(pipeline)
        tree = pipeline.named_steps["model"]
        self._tree = tree.tree_
        self._positive = list(tree.classes_).index(1)
        self._node_rates: list[float] = manifest["node_purchase_rate"]

    @property
    def version(self) -> str:
        return str(self.manifest["model_version"])

    @property
    def threshold(self) -> float:
        return float(self.manifest["threshold"])

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        frame = records_to_frame([record])
        probability = float(self.pipeline.predict_proba(frame)[0, self._positive])
        encoded = self.pipeline.named_steps["prep"].transform(frame)
        positive = probability >= self.threshold
        return {
            "task": "classification",
            "score": round(probability, 4),
            "score_label": "Purchase probability",
            "threshold": self.threshold,
            "positive": positive,
            "outcome": "Likely to buy" if positive else "Unlikely to buy",
            "explanation": {
                "kind": "path",
                "title": "How the tree decided",
                "value_label": "Purchase rate of training customers at this step",
                "steps": self._decision_path(np.asarray(encoded)[0]),
            },
            "model_version": self.version,
        }

    def _decision_path(self, row: np.ndarray) -> list[dict[str, Any]]:
        tree = self._tree
        node = 0
        steps: list[dict[str, Any]] = []
        while tree.children_left[node] != tree.children_right[node]:  # until a leaf
            feature = self._features[int(tree.feature[node])]
            threshold = float(tree.threshold[node])
            went_left = bool(row[int(tree.feature[node])] <= threshold)
            node = int(tree.children_left[node] if went_left else tree.children_right[node])
            # Real (unweighted) training purchase rate of customers reaching this node -- the
            # tree's own `value` is distorted by class_weight="balanced", so train.py records it.
            steps.append(
                {
                    "label": _rule_text(feature, threshold, went_left),
                    "field": feature.field,
                    "value": self._node_rates[node],
                    "count": int(tree.n_node_samples[node]),
                }
            )
        return steps


@lru_cache(maxsize=1)
def get_model() -> PurchaseModel:
    manifest = json.loads(MANIFEST_PATH.read_text())
    return PurchaseModel(load(MODEL_PATH), manifest)


@lru_cache(maxsize=1)
def get_samples() -> list[dict[str, Any]]:
    return json.loads(SAMPLES_PATH.read_text())
