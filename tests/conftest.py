"""Shared pytest fixtures.

The API tests need a trained model.  Rather than training inside the test
suite (slow and non-deterministic across machines), we skip the whole API
module when the artifact is absent and tell the developer what to run.
"""
from __future__ import annotations

import json

import pytest

from src import config


@pytest.fixture(scope="session")
def metadata() -> dict:
    if not config.METADATA_PATH.exists():
        pytest.skip("models/metadata.json missing - run `python -m src.train` first")
    return json.loads(config.METADATA_PATH.read_text())


@pytest.fixture(scope="session")
def client():
    """A TestClient with the app's lifespan actually run, so the model loads."""
    if not config.MODEL_PATH.exists():
        pytest.skip("models/model.joblib missing - run `python -m src.train` first")

    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def valid_payload(metadata: dict) -> dict:
    """A known-good applicant: the exact row train.py saved as a fixture."""
    return dict(metadata["prediction_fixture"]["payload"])
