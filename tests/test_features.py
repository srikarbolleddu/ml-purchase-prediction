from typing import Any

import pandas as pd

from app.features import (
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    FIELDS,
    NEVER_CONTACTED,
    TARGET,
    clean_raw,
    records_to_frame,
)


def _raw_row(**overrides):
    row = {f.source_column: (f.options[0] if f.kind == "category" else 1.0) for f in FIELDS}
    row.update({"duration": 300, "y": "yes"})
    row.update(overrides)
    return row


def test_clean_raw_drops_leaky_duration_and_encodes_target():
    df = clean_raw(pd.DataFrame([_raw_row(), _raw_row(y="no")]))

    assert "duration" not in df.columns
    assert list(df.columns) == [*FEATURE_ORDER, TARGET]
    assert df[TARGET].tolist() == [1, 0]


def test_clean_raw_renames_dotted_source_columns():
    df = clean_raw(pd.DataFrame([_raw_row(**{"emp.var.rate": -1.8, "pdays": 999})]))

    assert df.loc[0, "emp_var_rate"] == -1.8
    assert df.loc[0, "days_since_last_contact"] == NEVER_CONTACTED


def test_records_to_frame_maps_never_contacted_back_to_sentinel():
    record: dict[str, Any] = dict.fromkeys(FEATURE_ORDER, 1.0)
    record.update(dict.fromkeys(CATEGORICAL_FEATURES, "x"))
    record["days_since_last_contact"] = None

    frame = records_to_frame([record])

    assert frame.loc[0, "days_since_last_contact"] == NEVER_CONTACTED
    assert list(frame.columns) == FEATURE_ORDER


def test_every_field_schema_is_renderable():
    for field in FIELDS:
        schema = field.to_schema()
        assert schema["label"] and schema["help"]
        if field.kind == "category":
            assert {o["value"] for o in schema["options"]} == set(field.options)
        else:
            assert schema["min"] < schema["max"]
