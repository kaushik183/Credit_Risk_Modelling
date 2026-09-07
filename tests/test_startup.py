"""The app must refuse to start without a model, rather than boot and 500.

A container that cannot serve a single valid request should not report itself
healthy - otherwise an orchestrator keeps routing traffic to it.
"""
from __future__ import annotations

import pytest

from src import config


def test_app_fails_loudly_when_the_model_file_is_missing(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import api.main as api_main

    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "does_not_exist.joblib")
    monkeypatch.setattr(api_main.config, "MODEL_PATH", tmp_path / "does_not_exist.joblib")

    with pytest.raises(RuntimeError, match="Model artifact not found"), TestClient(api_main.app):
        pass


def test_app_fails_loudly_when_metadata_is_missing(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import api.main as api_main

    if not config.MODEL_PATH.exists():
        pytest.skip("models/model.joblib missing - run `python -m src.train` first")

    monkeypatch.setattr(api_main.config, "METADATA_PATH", tmp_path / "no_metadata.json")

    with pytest.raises(RuntimeError, match="metadata not found"), TestClient(api_main.app):
        pass
