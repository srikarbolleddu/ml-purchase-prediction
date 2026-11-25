import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from app.api.explore import build_explore
from app.features import FIELDS
from app.model import get_model, get_samples
from app.schemas import PredictionInput, PredictionOutput, Sample, SchemaResponse

REPORTS_DIR = Path(__file__).resolve().parents[1] / "model_artifacts"

api_router = APIRouter()

GROUPS = [
    {"id": "customer", "label": "Customer"},
    {"id": "campaign", "label": "Campaign contact"},
    {"id": "economy", "label": "Economic context"},
]


@lru_cache(maxsize=1)
def _explore() -> dict[str, Any]:
    return json.loads((REPORTS_DIR / "explore.json").read_text())


@lru_cache(maxsize=1)
def _metrics() -> dict[str, Any]:
    return json.loads((REPORTS_DIR / "metrics.json").read_text())


@api_router.get("/schema", response_model=SchemaResponse, tags=["demo"])
def schema() -> dict[str, Any]:
    """Input fields for the demo form (types, ranges, options, help text)."""
    model = get_model()
    return {
        "contract": "ml-demo-kit/v1",
        "title": "Will this customer buy a term deposit?",
        "description": (
            "Change the customer's profile, the call, or the economy at the time, and see how a "
            "pruned decision tree's prediction and reasoning change."
        ),
        "task": "classification",
        "target": "purchased",
        "groups": GROUPS,
        "fields": [f.to_schema() for f in FIELDS],
        "model_version": model.version,
        "threshold": model.threshold,
    }


@api_router.get("/samples", response_model=list[Sample], tags=["demo"])
def samples() -> list[dict[str, Any]]:
    """Real held-out customers (both outcomes) for one-click demo inputs."""
    return get_samples()


@api_router.get("/explore", tags=["demo"])
def explore() -> dict[str, Any]:
    """Precomputed chart specs for the interactive Explore panel (ml-demo-kit/v1)."""
    return build_explore(_explore(), _metrics())


@api_router.post("/predict", response_model=PredictionOutput, tags=["demo"])
def predict(payload: PredictionInput) -> dict[str, Any]:
    """Purchase probability plus the decision-tree rules that produced it."""
    return get_model().predict(payload.model_dump())
