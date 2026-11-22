"""Reproducible training: download the pinned dataset → clean → compare trees → write artifacts.

    uv run python training/train.py

Writes:
  app/model_artifacts/model.skops, manifest.json, samples.json   (shipped with the API)
  reports/metrics.json      held-out metrics -- the only source for numbers quoted anywhere
  reports/explore.json      precomputed data for the demo's interactive "Explore" charts
  reports/golden.json       fixed inputs + expected outputs, checked by tests/test_golden.py
  reports/figures/*.png     charts for the README / case study / blog

Method follows the decision-tree approach of the UT Austin PGP-AIML capstone (default tree →
pre-pruning → cost-complexity post-pruning), rebuilt on the openly licensed UCI dataset.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import model as model_io  # noqa: E402
from app.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    FIELDS,
    NEVER_CONTACTED,
    NUMERIC_FEATURES,
    TARGET,
    clean_raw,
    records_to_frame,
)

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DATA_SHA256 = "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4"
CACHE = ROOT / ".data" / "bank+marketing.zip"
SEED = 42
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def fetch_dataset() -> pd.DataFrame:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {DATA_URL}")
        # DATA_URL is a constant https URL and the payload is checksum-verified below.
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:  # nosec B310
            CACHE.write_bytes(response.read())
    payload = CACHE.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != DATA_SHA256:
        raise SystemExit(f"dataset checksum mismatch: {digest} != {DATA_SHA256}")
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        inner_bytes = outer.read("bank-additional.zip")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        with inner.open("bank-additional/bank-additional-full.csv") as handle:
            return pd.read_csv(handle, sep=";")


def build_pipeline(**tree_params: Any) -> Pipeline:
    prep = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    tree = DecisionTreeClassifier(class_weight="balanced", random_state=SEED, **tree_params)
    return Pipeline([("prep", prep), ("model", tree)])


def scores(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
    }


def tuned_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    """Cutoff maximizing validation F1 (0.5 ignores the 11% base rate)."""
    grid = np.round(np.linspace(0.05, 0.95, 91), 2)
    return float(max(grid, key=lambda t: f1_score(y_true, (proba >= t).astype(int))))


def tree_size(pipeline: Pipeline) -> dict[str, int]:
    tree = pipeline.named_steps["model"]
    return {"leaves": int(tree.get_n_leaves()), "depth": int(tree.get_depth())}


def proba(pipeline: Pipeline, x: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(x)[:, 1]


def main() -> None:
    df = clean_raw(fetch_dataset())
    x, y = df[FEATURE_ORDER], df[TARGET]
    # 60/20/20 stratified: train fits, validation chooses hyperparameters, test is touched once.
    x_train, x_rest, y_train, y_rest = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=SEED
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SEED
    )
    print(f"rows: train={len(x_train)} val={len(x_val)} test={len(x_test)}")

    # Hyperparameters are chosen by validation PR-AUC (average precision): it scores how well the
    # tree *ranks* likely buyers, independent of any one cutoff, and suits an 11%-positive target.
    # The cutoff itself is then tuned separately on validation (tuned_threshold).
    candidates: dict[str, Pipeline] = {}

    # 1. Default tree: grows until leaves are pure -- memorizes the training set.
    candidates["Default tree"] = build_pipeline().fit(x_train, y_train)

    # 2. Pre-pruning: limit growth up front.
    best_pre, best_pre_score = None, -1.0
    for depth in (3, 4, 5, 6, 8, 10):
        for leaf in (10, 25, 50, 100, 200):
            candidate = build_pipeline(max_depth=depth, min_samples_leaf=leaf).fit(x_train, y_train)
            score = scores(y_val, proba(candidate, x_val))["pr_auc"]
            if score > best_pre_score:
                best_pre, best_pre_score = candidate, score
    if best_pre is None:
        raise RuntimeError("pre-pruning search produced no model")
    candidates["Pre-pruned tree"] = best_pre

    # 3. Post-pruning: grow fully, then cut back with cost-complexity alpha.
    full = candidates["Default tree"]
    x_train_enc = full.named_steps["prep"].transform(x_train)
    path = full.named_steps["model"].cost_complexity_pruning_path(x_train_enc, y_train)
    # clip: the path can contain tiny negative alphas from floating-point error
    alphas = np.unique(np.clip(np.quantile(path.ccp_alphas[:-1], np.linspace(0, 1, 60)), 0, None))
    pruning_curve = []
    best_post, best_post_score = None, -1.0
    for alpha in alphas:
        candidate = build_pipeline(ccp_alpha=float(alpha)).fit(x_train, y_train)
        train_scores = scores(y_train, proba(candidate, x_train))
        val_scores = scores(y_val, proba(candidate, x_val))
        pruning_curve.append(
            {
                "alpha": float(alpha),
                **tree_size(candidate),
                "train_pr_auc": train_scores["pr_auc"],
                "val_pr_auc": val_scores["pr_auc"],
                "train_recall": train_scores["recall"],
                "val_recall": val_scores["recall"],
            }
        )
        if val_scores["pr_auc"] > best_post_score:
            best_post, best_post_score = candidate, val_scores["pr_auc"]
    if best_post is None:
        raise RuntimeError("post-pruning search produced no model")
    candidates["Post-pruned tree"] = best_post

    thresholds = {name: tuned_threshold(y_val, proba(p, x_val)) for name, p in candidates.items()}
    comparison: dict[str, dict[str, Any]] = {
        name: {
            **tree_size(p),
            "threshold": thresholds[name],
            "train": scores(y_train, proba(p, x_train), thresholds[name]),
            "validation": scores(y_val, proba(p, x_val), thresholds[name]),
            "test": scores(y_test, proba(p, x_test), thresholds[name]),
        }
        for name, p in candidates.items()
    }
    final_name = max(
        ("Pre-pruned tree", "Post-pruned tree"),
        key=lambda n: comparison[n]["validation"]["pr_auc"],
    )
    final = candidates[final_name]
    threshold = thresholds[final_name]
    test_pred = (proba(final, x_test) >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_test, test_pred).ravel())

    # Feature importance, summed back from one-hot columns to the form field they came from.
    encoded = model_io.encoded_features(final)
    importance: dict[str, float] = {}
    for feat, value in zip(encoded, final.named_steps["model"].feature_importances_, strict=True):
        importance[feat.field] = importance.get(feat.field, 0.0) + float(value)
    labels = {f.name: f.label for f in FIELDS}
    importance_rows = sorted(
        (
            {"field": k, "label": labels[k], "importance": round(v, 4)}
            for k, v in importance.items()
        ),
        key=lambda r: -float(r["importance"]),
    )

    # Real purchase rate of training customers reaching each node (for the decision-path UI).
    tree = final.named_steps["model"]
    node_matrix = tree.decision_path(final.named_steps["prep"].transform(x_train))
    hits = np.asarray(node_matrix.sum(axis=0)).ravel()
    buyers = np.asarray(node_matrix.T @ y_train.to_numpy()).ravel()
    node_rates = [round(float(b / h), 4) if h else 0.0 for b, h in zip(buyers, hits, strict=True)]

    params_hash = hashlib.sha256(str(tree.get_params()).encode()).hexdigest()[:8]
    model_version = f"{final_name.split()[0].lower()}-tree-{params_hash}"
    artifacts = model_io.ARTIFACT_DIR
    artifacts.mkdir(parents=True, exist_ok=True)
    model_io.save(final, model_io.MODEL_PATH)
    manifest = {
        "model_version": model_version,
        "model": final_name,
        "params": {k: v for k, v in tree.get_params().items() if v is not None},
        "threshold": threshold,
        "node_purchase_rate": node_rates,
        "dataset": {"url": DATA_URL, "sha256": DATA_SHA256, "license": "CC BY 4.0"},
    }
    (artifacts / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Curated demo samples: real held-out customers, both outcomes, chosen deterministically.
    test_proba = proba(final, x_test)
    test_frame = x_test.assign(actual=y_test.to_numpy(), p=test_proba)
    # Three buyers and three non-buyers, including one mistake each way, so the demo shows where
    # the model is right and where it isn't. `t` = held-out test customers with their score.
    t = test_frame
    buyers, non_buyers = t[t.actual == 1], t[t.actual == 0]
    picks = [
        (
            "Buyer: previous campaign was a success",
            buyers[buyers.poutcome == "success"].sort_values("p").iloc[-1:],
        ),
        (
            "Buyer: first-time contact on a mobile",
            buyers[(buyers.poutcome == "nonexistent") & (buyers.contact == "cellular")]
            .sort_values("p")
            .iloc[-1:],
        ),
        ("Buyer the model misses", buyers[buyers.p < threshold].sort_values("p").iloc[-1:]),
        (
            "Non-buyer: May telephone campaign",
            non_buyers[(non_buyers.month == "may") & (non_buyers.contact == "telephone")]
            .sort_values("p")
            .iloc[:1],
        ),
        (
            "Non-buyer: already has a housing loan",
            non_buyers[non_buyers.housing == "yes"]
            .sort_values("p")
            .iloc[len(non_buyers) // 4 : len(non_buyers) // 4 + 1],
        ),
        (
            "Non-buyer the model flags as a buyer",
            non_buyers[non_buyers.p >= threshold].sort_values("p").iloc[-1:],
        ),
    ]
    samples: list[dict[str, Any]] = []
    for title, row in picks:
        if row.empty:
            continue
        record = row.iloc[0]
        inputs: dict[str, Any] = {}
        for name in FEATURE_ORDER:
            value = record[name]
            if name in NUMERIC_FEATURES:
                value = float(value)
                value = int(value) if value.is_integer() else value
            inputs[name] = value
        if inputs["days_since_last_contact"] == NEVER_CONTACTED:
            inputs["days_since_last_contact"] = None
        samples.append(
            {
                "title": title,
                "inputs": inputs,
                "actual": "purchased" if record.actual else "did not purchase",
            }
        )
    (artifacts / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")

    REPORTS.mkdir(exist_ok=True)
    metrics = {
        "dataset": {
            "name": "UCI Bank Marketing (bank-additional-full)",
            "rows": len(df),
            "positive_rate": round(float(y.mean()), 4),
            "split": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
            "excluded_columns": ["duration (target leakage)"],
        },
        "final_model": final_name,
        "threshold": threshold,
        "test": comparison[final_name]["test"],
        "test_confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "comparison": comparison,
        "model_version": model_version,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Threshold sweep on the test set for the interactive confusion matrix.
    sweep = []
    for t in np.round(np.linspace(0.05, 0.95, 19), 2):
        pred = (test_proba >= t).astype(int)
        s_tn, s_fp, s_fn, s_tp = (int(v) for v in confusion_matrix(y_test, pred).ravel())
        sweep.append({"threshold": float(t), "tn": s_tn, "fp": s_fp, "fn": s_fn, "tp": s_tp})

    explore = {
        "pruning_curve": pruning_curve,
        "selected_alpha": float(final.named_steps["model"].ccp_alpha),
        "model_comparison": [
            {"model": name, **c["test"], **{k: c[k] for k in ("leaves", "depth")}}
            for name, c in comparison.items()
        ],
        "feature_importance": importance_rows[:10],
        "threshold_sweep": sweep,
        "purchase_rate_by": {
            col: (df.groupby(col)[TARGET].mean().round(4).sort_values(ascending=False).to_dict())
            for col in ("poutcome", "month", "job", "contact")
        },
    }
    (REPORTS / "explore.json").write_text(json.dumps(explore, indent=2) + "\n")

    # Golden outputs go through the same API-side conversion (records_to_frame) as live requests.
    golden = [
        {
            "inputs": s["inputs"],
            "probability": round(float(proba(final, records_to_frame([s["inputs"]]))[0]), 4),
        }
        for s in samples
    ]
    (REPORTS / "golden.json").write_text(json.dumps(golden, indent=2) + "\n")

    # The API serves these two (GET /explore), so ship copies alongside the model.
    for name in ("metrics.json", "explore.json"):
        (artifacts / name).write_text((REPORTS / name).read_text())

    write_figures(pruning_curve, importance_rows, comparison)
    print(
        json.dumps(
            {"final": final_name, "test": comparison[final_name]["test"], **tree_size(final)},
            indent=2,
        )
    )


def write_figures(
    curve: list[dict[str, Any]], importance: list[dict[str, Any]], comparison: dict[str, Any]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    leaves = [c["leaves"] for c in curve]
    ax.plot(leaves, [c["train_pr_auc"] for c in curve], label="train PR-AUC")
    ax.plot(leaves, [c["val_pr_auc"] for c in curve], label="validation PR-AUC")
    ax.set_xscale("log")
    ax.set_xlabel("tree size (leaves, log scale) — more pruning to the left")
    ax.set_ylabel("PR-AUC (buyers)")
    ax.set_title("Cost-complexity pruning: the gap between train and validation is overfitting")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "pruning_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    top = importance[:10][::-1]
    ax.barh([r["label"] for r in top], [r["importance"] for r in top])
    ax.set_xlabel("importance (share of impurity reduction)")
    ax.set_title("What drives a term-deposit purchase")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(comparison)
    width = 0.25
    for i, split in enumerate(("train", "validation", "test")):
        ax.bar(
            [n + i * width for n in range(len(names))],
            [comparison[name][split]["f1"] for name in names],
            width,
            label=split,
        )
    ax.set_xticks([n + width for n in range(len(names))], names)
    ax.set_ylabel("F1 (buyers)")
    ax.set_title("Default vs pre-pruned vs post-pruned")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
