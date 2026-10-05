import os

import pytest
from fastapi.testclient import TestClient

import app as application

client = TestClient(application.app, raise_server_exceptions=False)

VALID_APPLICANT = {
    "checking_balance": "0 <= Rs. < 2000",
    "credit_duration": 12,
    "credit_amount": 25000,
    "credit_history": "all settled till now",
    "credit_purpose": "electronics",
    "savings_balance": "Rs. < 1000",
    "employment_duration": "1 to 4 years",
    "installment_rate": 3,
    "marital_status_gender": "single male",
    "other_parties": "no one",
    "residence_duration": "2 to 3 years",
    "property_type": "real estate",
    "age": 35,
    "installment_plans": "none",
    "housing_type": "own",
    "existing_credits": 1,
    "job_type": "employee with official position",
    "num_dependents": 1,
    "foreign_worker": "yes",
}


def model_is_available() -> bool:
    try:
        application.get_classifier()
        return True
    except Exception:
        return False


needs_model = pytest.mark.skipif(not model_is_available(), reason="no trained model in any registry")


def test_health_is_independent_of_the_model():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_reflects_the_model_availability():
    response = client.get("/ready")
    assert response.status_code == (200 if model_is_available() else 503)


def test_index_renders_the_form():
    response = client.get("/")
    assert response.status_code == 200
    assert "checking_balance" in response.text


def test_every_form_field_is_covered_by_the_validator():
    assert set(application.VALIDATOR.field_names) == set(application.FIELD_NAMES)


def test_an_unknown_category_is_rejected_before_it_reaches_the_encoder():
    payload = {**VALID_APPLICANT, "housing_type": "a mansion on the moon"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "housing_type" in " ".join(response.json()["errors"])


def test_a_number_outside_its_range_is_rejected():
    payload = {**VALID_APPLICANT, "age": 7}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "age" in " ".join(response.json()["errors"])


def test_a_non_numeric_value_is_rejected():
    payload = {**VALID_APPLICANT, "credit_amount": "plenty"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_missing_fields_are_reported():
    payload = {key: value for key, value in VALID_APPLICANT.items() if key != "age"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "age" in " ".join(response.json()["errors"])


def test_training_is_refused_without_a_configured_token(monkeypatch):
    monkeypatch.delenv("TRAIN_API_TOKEN", raising=False)
    assert client.post("/train").status_code == 403


def test_training_is_refused_with_a_wrong_token(monkeypatch):
    monkeypatch.setenv("TRAIN_API_TOKEN", "the-real-token")
    response = client.post("/train", headers={"X-Train-Token": "guessed"})
    assert response.status_code == 401


def test_training_is_not_reachable_over_get():
    assert client.get("/train").status_code == 405


@needs_model
def test_a_valid_applicant_is_scored():
    response = client.post("/predict", json=VALID_APPLICANT)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] is True
    assert body["credit_score"] in {"good", "bad"}
    assert 0.0 <= body["risk_probability"] <= 100.0


@needs_model
def test_the_classifier_is_loaded_once_and_reused():
    application.reset_classifier()
    first = application.get_classifier()
    second = application.get_classifier()

    assert first is second


@needs_model
def test_the_html_form_returns_a_verdict():
    response = client.post("/", data={key: str(value) for key, value in VALID_APPLICANT.items()})

    assert response.status_code == 200
    assert "Creditworthy" in response.text


@needs_model
def test_the_html_form_reports_an_invalid_value_and_keeps_the_input():
    payload = {**VALID_APPLICANT, "foreign_worker": "maybe"}
    response = client.post("/", data={key: str(value) for key, value in payload.items()})

    assert response.status_code == 422
    assert "foreign_worker" in response.text
