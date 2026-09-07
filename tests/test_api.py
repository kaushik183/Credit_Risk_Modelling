"""API contract tests: schema validation, known-good and known-bad payloads,
health, and the training/serving parity check."""
from __future__ import annotations

import copy

import pytest


# ---------------------------------------------------------------------------
# Ops endpoints
# ---------------------------------------------------------------------------
def test_health_reports_the_model_as_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]
    assert body["uptime_seconds"] >= 0


def test_model_info_exposes_version_features_and_metrics(client, metadata):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == metadata["model_version"]
    assert body["n_features"] == len(metadata["feature_columns"])
    assert body["metrics"]["xgboost_calibrated_test"]["roc_auc"] > 0.5
    assert "auc_inflation" in body["leakage_ablation"]


def test_model_info_never_advertises_a_post_disbursal_feature(client):
    """A regression guard on the leakage decision, at the API boundary."""
    body = client.get("/model/info").json()
    forbidden = {"principal_outstanding", "net_disbursement", "gst",
                 "processing_fee", "installment_start_dt"}
    assert forbidden.isdisjoint(body["feature_columns"])
    assert forbidden.isdisjoint(body["raw_input_columns"])


def test_openapi_docs_are_served(client):
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


# ---------------------------------------------------------------------------
# Known-good payload
# ---------------------------------------------------------------------------
def test_known_good_payload_scores_successfully(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert 300 <= body["credit_score"] <= 900
    assert body["rating"] in {"Poor", "Average", "Good", "Excellent"}
    assert body["decision"] in {"approve", "decline"}


def test_prediction_matches_what_train_py_recorded(client, valid_payload, metadata):
    """The training/serving parity test.

    ``train.py`` stored one test row and the probability the freshly fitted
    pipeline gave it. If the API - after a joblib round-trip, through Pydantic
    coercion and a fresh process - returns a different number, then feature
    engineering has drifted between training and serving. This is the single
    most important test in the suite.
    """
    expected = metadata["prediction_fixture"]["expected_probability"]
    body = client.post("/predict", json=valid_payload).json()
    assert body["probability_of_default"] == pytest.approx(expected, abs=1e-5)
    assert body["credit_score"] == metadata["prediction_fixture"]["expected_credit_score"]


def test_scoring_is_idempotent(client, valid_payload):
    first = client.post("/predict", json=valid_payload).json()
    second = client.post("/predict", json=valid_payload).json()
    assert first["probability_of_default"] == second["probability_of_default"]


def test_riskier_applicant_scores_worse(client, valid_payload):
    """A directional sanity check: worse credit behaviour must not score better."""
    safe = copy.deepcopy(valid_payload)
    safe.update(credit_utilization_ratio=5, delinquent_months=0, total_dpd=0)
    risky = copy.deepcopy(valid_payload)
    risky.update(credit_utilization_ratio=98, delinquent_months=30,
                 total_dpd=900, total_loan_months=36)

    safe_pd = client.post("/predict", json=safe).json()["probability_of_default"]
    risky_pd = client.post("/predict", json=risky).json()["probability_of_default"]
    assert risky_pd > safe_pd


# ---------------------------------------------------------------------------
# Known-bad payloads: every one must be a 422, never a 500
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field,bad_value,reason",
    [
        ("age", 15, "under 18 cannot contract"),
        ("age", 130, "implausible age"),
        ("income", 0, "income must be positive"),
        ("income", -50_000, "negative income"),
        ("credit_utilization_ratio", 150, "percent cannot exceed 100"),
        ("credit_utilization_ratio", -1, "percent cannot be negative"),
        ("loan_tenure_months", 0, "a loan must last at least a month"),
        ("employment_status", "Freelance", "not an allowed category"),
        ("loan_purpose", "Wedding", "not an allowed category"),
        ("residence_type", "owned", "wrong case, categories are exact"),
        ("number_of_dependants", -1, "cannot be negative"),
        ("loan_amount", 0, "must be positive"),
    ],
)
def test_out_of_range_values_return_422(client, valid_payload, field, bad_value, reason):
    payload = copy.deepcopy(valid_payload)
    payload[field] = bad_value
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, f"{field}={bad_value} ({reason}) should be rejected"
    assert "detail" in response.json()


def test_missing_required_field_returns_422(client, valid_payload):
    payload = copy.deepcopy(valid_payload)
    del payload["income"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert any("income" in str(err["loc"]) for err in response.json()["detail"])


def test_empty_body_returns_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_wrong_type_returns_422(client, valid_payload):
    payload = copy.deepcopy(valid_payload)
    payload["income"] = "a lot"
    assert client.post("/predict", json=payload).status_code == 422


# --- cross-field rules -----------------------------------------------------
def test_loan_amount_above_sanction_is_rejected(client, valid_payload):
    payload = copy.deepcopy(valid_payload)
    payload["sanction_amount"] = 100_000
    payload["loan_amount"] = 500_000
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "sanction" in response.text


def test_delinquent_months_above_history_is_rejected(client, valid_payload):
    payload = copy.deepcopy(valid_payload)
    payload["total_loan_months"] = 12
    payload["delinquent_months"] = 40
    assert client.post("/predict", json=payload).status_code == 422


def test_dpd_without_delinquency_is_rejected(client, valid_payload):
    payload = copy.deepcopy(valid_payload)
    payload["delinquent_months"] = 0
    payload["total_dpd"] = 200
    assert client.post("/predict", json=payload).status_code == 422


def test_applicant_with_no_credit_accounts_is_rejected(client, valid_payload):
    payload = copy.deepcopy(valid_payload)
    payload["number_of_open_accounts"] = 0
    payload["number_of_closed_accounts"] = 0
    assert client.post("/predict", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# Batch endpoint
# ---------------------------------------------------------------------------
def test_batch_returns_one_prediction_per_applicant(client, valid_payload):
    response = client.post("/predict/batch", json={"applicants": [valid_payload] * 5})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 5
    assert len(body["predictions"]) == 5


def test_batch_agrees_with_single_prediction(client, valid_payload):
    """Batching must be a performance optimisation, never a semantic change."""
    single = client.post("/predict", json=valid_payload).json()
    batched = client.post("/predict/batch", json={"applicants": [valid_payload]}).json()
    assert batched["predictions"][0]["probability_of_default"] == single["probability_of_default"]


def test_batch_rejects_an_empty_list(client):
    assert client.post("/predict/batch", json={"applicants": []}).status_code == 422


def test_batch_rejects_an_oversized_list(client, valid_payload):
    response = client.post("/predict/batch", json={"applicants": [valid_payload] * 1001})
    assert response.status_code == 422


def test_one_bad_applicant_rejects_the_whole_batch(client, valid_payload):
    bad = copy.deepcopy(valid_payload)
    bad["age"] = 5
    response = client.post("/predict/batch", json={"applicants": [valid_payload, bad]})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
def test_responses_carry_a_request_id_and_latency_header(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Response-Time-ms"]) >= 0
