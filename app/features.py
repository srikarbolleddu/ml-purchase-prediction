"""Feature definitions and cleaning, shared by training (`training/train.py`) and serving (`app/`).

This module is the notebook's data-preparation steps turned into plain, tested functions. Because
both training and the API import it, the preprocessing a model was trained with can never drift
from the preprocessing applied to a live request.

Dataset: UCI Bank Marketing, `bank-additional-full.csv` (Moro, Cortez & Rita, 2014), CC BY 4.0.
https://archive.ics.uci.edu/dataset/222/bank+marketing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

TARGET = "purchased"

# The raw CSV's sentinel for "never contacted in a previous campaign".
NEVER_CONTACTED = 999


@dataclass(frozen=True)
class Field:
    """One model input. Drives training columns, request validation and the demo form."""

    name: str
    label: str
    kind: Literal["number", "category"]
    group: Literal["customer", "campaign", "economy"]
    help: str
    source_column: str
    options: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    unit: str | None = None
    optional: bool = False
    option_labels: dict[str, str] = field(default_factory=dict)

    def to_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "group": self.group,
            "help": self.help,
            "optional": self.optional,
        }
        if self.kind == "category":
            schema["options"] = [
                {"value": value, "label": self.option_labels.get(value, _humanize(value))}
                for value in self.options
            ]
        else:
            schema.update(
                {"min": self.minimum, "max": self.maximum, "step": self.step, "unit": self.unit}
            )
        return schema


def _humanize(value: str) -> str:
    return value.replace(".", " ").replace("-", " ").replace("_", " ").strip().capitalize()


_YES_NO_UNKNOWN = ("no", "yes", "unknown")

FIELDS: tuple[Field, ...] = (
    Field(
        "age",
        "Age",
        "number",
        "customer",
        "Customer's age in years.",
        "age",
        minimum=17,
        maximum=98,
        step=1,
        unit="years",
    ),
    Field(
        "job",
        "Job",
        "category",
        "customer",
        "Type of job.",
        "job",
        options=(
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown",
        ),
    ),
    Field(
        "marital",
        "Marital status",
        "category",
        "customer",
        "Marital status.",
        "marital",
        options=("married", "single", "divorced", "unknown"),
    ),
    Field(
        "education",
        "Education",
        "category",
        "customer",
        "Highest education level.",
        "education",
        options=(
            "basic.4y",
            "basic.6y",
            "basic.9y",
            "high.school",
            "professional.course",
            "university.degree",
            "illiterate",
            "unknown",
        ),
        option_labels={
            "basic.4y": "Basic (4 years)",
            "basic.6y": "Basic (6 years)",
            "basic.9y": "Basic (9 years)",
            "high.school": "High school",
            "professional.course": "Professional course",
            "university.degree": "University degree",
        },
    ),
    Field(
        "default",
        "Credit in default?",
        "category",
        "customer",
        "Whether the customer has credit in default.",
        "default",
        options=_YES_NO_UNKNOWN,
    ),
    Field(
        "housing",
        "Housing loan?",
        "category",
        "customer",
        "Whether the customer has a housing loan.",
        "housing",
        options=_YES_NO_UNKNOWN,
    ),
    Field(
        "loan",
        "Personal loan?",
        "category",
        "customer",
        "Whether the customer has a personal loan.",
        "loan",
        options=_YES_NO_UNKNOWN,
    ),
    Field(
        "contact",
        "Contact channel",
        "category",
        "campaign",
        "How the customer is contacted.",
        "contact",
        options=("cellular", "telephone"),
    ),
    Field(
        "month",
        "Contact month",
        "category",
        "campaign",
        "Month of the last contact.",
        "month",
        options=("mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
    ),
    Field(
        "day_of_week",
        "Contact weekday",
        "category",
        "campaign",
        "Weekday of the last contact.",
        "day_of_week",
        options=("mon", "tue", "wed", "thu", "fri"),
    ),
    Field(
        "campaign",
        "Contacts this campaign",
        "number",
        "campaign",
        "Number of contacts made during this campaign, including the last one.",
        "campaign",
        minimum=1,
        maximum=56,
        step=1,
    ),
    Field(
        "days_since_last_contact",
        "Days since previous-campaign contact",
        "number",
        "campaign",
        "Days since the customer was last contacted in a previous campaign. "
        "Leave empty if never contacted.",
        "pdays",
        minimum=0,
        maximum=30,
        step=1,
        unit="days",
        optional=True,
    ),
    Field(
        "previous",
        "Contacts before this campaign",
        "number",
        "campaign",
        "Number of contacts before this campaign.",
        "previous",
        minimum=0,
        maximum=7,
        step=1,
    ),
    Field(
        "poutcome",
        "Previous campaign outcome",
        "category",
        "campaign",
        "Outcome of the previous marketing campaign.",
        "poutcome",
        options=("nonexistent", "failure", "success"),
        option_labels={"nonexistent": "No previous campaign"},
    ),
    Field(
        "emp_var_rate",
        "Employment variation rate",
        "number",
        "economy",
        "Quarterly employment variation rate (macro indicator).",
        "emp.var.rate",
        minimum=-3.4,
        maximum=1.4,
        step=0.1,
    ),
    Field(
        "cons_price_idx",
        "Consumer price index",
        "number",
        "economy",
        "Monthly consumer price index.",
        "cons.price.idx",
        minimum=92.2,
        maximum=94.8,
        step=0.001,
    ),
    Field(
        "cons_conf_idx",
        "Consumer confidence index",
        "number",
        "economy",
        "Monthly consumer confidence index.",
        "cons.conf.idx",
        minimum=-50.8,
        maximum=-26.9,
        step=0.1,
    ),
    Field(
        "euribor3m",
        "Euribor 3-month rate",
        "number",
        "economy",
        "Daily Euribor 3-month interest rate.",
        "euribor3m",
        minimum=0.6,
        maximum=5.1,
        step=0.001,
        unit="%",
    ),
    Field(
        "nr_employed",
        "Number employed",
        "number",
        "economy",
        "Quarterly number of employees (thousands).",
        "nr.employed",
        minimum=4963,
        maximum=5229,
        step=0.1,
        unit="thousand",
    ),
)

FIELDS_BY_NAME = {f.name: f for f in FIELDS}
NUMERIC_FEATURES = [f.name for f in FIELDS if f.kind == "number"]
CATEGORICAL_FEATURES = [f.name for f in FIELDS if f.kind == "category"]
FEATURE_ORDER = [f.name for f in FIELDS]

# Excluded on purpose: call `duration` is only known *after* the call ends, and the dataset
# authors note it almost determines the outcome -- using it would leak the answer into the model.
LEAKY_COLUMNS = ("duration",)


def clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw UCI CSV → model-ready frame: renamed columns, leak removed, target encoded."""
    df = raw.drop(columns=list(LEAKY_COLUMNS), errors="ignore")
    df = df.rename(columns={f.source_column: f.name for f in FIELDS})
    df[TARGET] = (df.pop("y") == "yes").astype(int)
    return df[[*FEATURE_ORDER, TARGET]]


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """API inputs → the same columns `clean_raw` produces (minus the target).

    `days_since_last_contact` is optional in the API (empty = never contacted); the training data
    encodes that as the 999 sentinel, so map it back here.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {name: record.get(name) for name in FEATURE_ORDER}
        if row["days_since_last_contact"] is None:
            row["days_since_last_contact"] = NEVER_CONTACTED
        rows.append(row)
    frame = pd.DataFrame(rows, columns=pd.Index(FEATURE_ORDER))
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(float)
    return frame
