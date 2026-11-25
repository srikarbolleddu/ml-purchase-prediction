"""Turn the training reports into the ml-demo-kit/v1 `/explore` payload: a list of generic chart
specs (line, bar, threshold, table) that the shared ExplorePanel renders without knowing anything
about this project."""

from __future__ import annotations

from typing import Any

_MONTHS = ("mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def _rate_bars(
    rates: dict[str, float], order: tuple[str, ...] | None = None
) -> list[dict[str, Any]]:
    keys = [k for k in order if k in rates] if order else sorted(rates, key=lambda k: -rates[k])
    return [{"label": k.replace("-", " ").capitalize(), "value": rates[k]} for k in keys]


def build_explore(explore: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics["test"]
    base_rate = metrics["dataset"]["positive_rate"]
    final = metrics["final_model"]
    final_shape = metrics["comparison"][final]
    rates = explore["purchase_rate_by"]
    # The pruning curve runs from the full tree (most leaves) down to a stump; plot it by size.
    curve = sorted(explore["pruning_curve"], key=lambda p: p["leaves"])

    return {
        "contract": "ml-demo-kit/v1",
        "headline": [
            {
                "label": "Precision",
                "value": test["precision"],
                "format": "percent",
                "hint": f"{test['precision'] / base_rate:.1f}x the {base_rate:.0%} base rate",
            },
            {
                "label": "Recall",
                "value": test["recall"],
                "format": "percent",
                "hint": "Share of all buyers reached",
            },
            {
                "label": "PR-AUC",
                "value": test["pr_auc"],
                "format": "number",
                "hint": "Held-out test set",
            },
            {
                "label": "Tree size",
                "value": final_shape["leaves"],
                "format": "integer",
                "hint": f"{final}, depth {final_shape['depth']}",
            },
        ],
        "charts": [
            {
                "id": "pruning",
                "kind": "line",
                "title": "Pruning: bigger trees memorize, smaller trees generalize",
                "caption": (
                    "Each point is a cost-complexity pruned tree. Drag the slider: the training "
                    "score keeps rising as the tree grows, but the validation score peaks early "
                    "and then falls. That gap is overfitting."
                ),
                "x": {"key": "leaves", "label": "Leaves in the tree", "scale": "log"},
                "y": {"label": "PR-AUC", "format": "number"},
                "series": [
                    {"key": "train_pr_auc", "label": "Training"},
                    {"key": "val_pr_auc", "label": "Validation"},
                ],
                "points": [
                    {
                        "leaves": p["leaves"],
                        "train_pr_auc": p["train_pr_auc"],
                        "val_pr_auc": p["val_pr_auc"],
                    }
                    for p in curve
                ],
            },
            {
                "id": "threshold",
                "kind": "threshold",
                "title": "Choose the call list: the decision threshold",
                "caption": (
                    "Customers scoring above the threshold get a call. Lower it to reach more "
                    "buyers (recall) at the cost of more wasted calls (precision)."
                ),
                "positive_label": "Buys",
                "negative_label": "Doesn't buy",
                "selected": metrics["threshold"],
                "sweep": explore["threshold_sweep"],
            },
            {
                "id": "importance",
                "kind": "bar",
                "title": "What the tree relies on",
                "caption": (
                    "Economic context at the time of the call dominates. When a campaign runs "
                    "matters more than who is called."
                ),
                "format": "percent",
                "items": [
                    {"label": f["label"], "value": f["importance"]}
                    for f in explore["feature_importance"]
                ],
            },
            {
                "id": "rate-month",
                "kind": "bar",
                "title": "Purchase rate by month of the call",
                "caption": f"Base rate across all calls: {base_rate:.1%}.",
                "format": "percent",
                "reference": {"value": base_rate, "label": "Base rate"},
                "items": _rate_bars(rates["month"], _MONTHS),
            },
            {
                "id": "rate-poutcome",
                "kind": "bar",
                "title": "Purchase rate by previous campaign outcome",
                "caption": "Customers who bought last time are by far the likeliest to buy again.",
                "format": "percent",
                "reference": {"value": base_rate, "label": "Base rate"},
                "items": _rate_bars(rates["poutcome"]),
            },
            {
                "id": "comparison",
                "kind": "table",
                "title": "Default vs pre-pruned vs post-pruned (held-out test set)",
                "caption": f"The {final.lower()} was chosen on validation PR-AUC.",
                "columns": [
                    {"key": "model", "label": "Model"},
                    {"key": "leaves", "label": "Leaves", "format": "integer"},
                    {"key": "precision", "label": "Precision", "format": "percent"},
                    {"key": "recall", "label": "Recall", "format": "percent"},
                    {"key": "pr_auc", "label": "PR-AUC", "format": "number"},
                ],
                "rows": explore["model_comparison"],
                "highlight": {"key": "model", "value": final},
            },
        ],
    }
