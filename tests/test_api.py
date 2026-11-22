from app.core.config import settings


def test_health(client):
    assert client.get("/api/v1/health").json() == {"status": "ok"}


def test_schema_lists_every_field_with_groups(client):
    body = client.get("/api/v1/schema").json()

    assert {g["id"] for g in body["groups"]} == {"customer", "campaign", "economy"}
    assert len(body["fields"]) == 19
    assert 0 < body["threshold"] < 1


def test_samples_cover_both_outcomes(client):
    samples = client.get("/api/v1/samples").json()

    assert {s["actual"] for s in samples} == {"purchased", "did not purchase"}


def test_explore_has_chart_data(client):
    body = client.get("/api/v1/explore").json()

    assert body["pruning_curve"] and body["threshold_sweep"] and body["feature_importance"]
    assert body["metrics"]["final_model"]


def test_predict_returns_probability_and_decision_path(client, sample_inputs):
    body = client.post("/api/v1/predict", json=sample_inputs).json()

    assert 0 <= body["probability"] <= 1
    assert body["will_purchase"] == (body["probability"] >= body["threshold"])
    assert body["decision_path"] and all(step["rule"] for step in body["decision_path"])


def test_predict_accepts_never_contacted(client, sample_inputs):
    sample_inputs["days_since_last_contact"] = None

    assert client.post("/api/v1/predict", json=sample_inputs).status_code == 200


def test_predict_rejects_unknown_category(client, sample_inputs):
    sample_inputs["job"] = "astronaut"

    assert client.post("/api/v1/predict", json=sample_inputs).status_code == 422


def test_predict_rejects_out_of_range_number(client, sample_inputs):
    sample_inputs["age"] = 500

    assert client.post("/api/v1/predict", json=sample_inputs).status_code == 422


def test_predict_rejects_extra_fields(client, sample_inputs):
    sample_inputs["duration"] = 300  # the leaky column must not sneak back in

    assert client.post("/api/v1/predict", json=sample_inputs).status_code == 422


def test_oversized_body_is_rejected_before_parsing(client):
    response = client.post(
        "/api/v1/predict",
        content=b"x" * (settings.MAX_BODY_BYTES + 1),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413


def test_rate_limit_returns_429(client, sample_inputs, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 2)

    codes = [client.post("/api/v1/predict", json=sample_inputs).status_code for _ in range(3)]

    assert codes == [200, 200, 429]


def test_cors_allows_portfolio_origin_only(client):
    allowed = client.get("/api/v1/health", headers={"Origin": "https://srikarbolleddu.com"})
    other = client.get("/api/v1/health", headers={"Origin": "https://evil.example"})

    assert allowed.headers["access-control-allow-origin"] == "https://srikarbolleddu.com"
    assert "access-control-allow-origin" not in other.headers
