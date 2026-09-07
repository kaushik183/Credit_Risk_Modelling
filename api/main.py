"""FastAPI service for credit default risk scoring.

Run locally::

    uvicorn api.main:app --reload

Design notes worth defending in review:

* The model is loaded **once**, in the lifespan handler, not per request.
  Loading a 300 KB joblib file on every call would dominate the latency budget.
* If the model file is missing the app **fails loudly at startup** rather than
  booting and returning 500s. A container that will never serve a valid
  request should not report itself healthy.
* Feature engineering is not reimplemented here. It travels inside the
  serialized pipeline (``src.features.build_features``), so the API physically
  cannot compute a feature differently from training.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src import config
from src.evaluate import clip_pd, pd_to_credit_score, score_to_rating
from src.features import RAW_INPUTS

# ---------------------------------------------------------------------------
# Structured logging: one JSON line per request, so a log aggregator can chart
# latency and error rate without parsing prose.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("api")

# Populated at startup by the lifespan handler.
ARTIFACTS: dict[str, Any] = {"model": None, "metadata": {}, "threshold": 0.5, "started_at": time.time()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, at startup, and fail loudly if it is not there."""
    if not config.MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {config.MODEL_PATH}. "
            "Run `python -m src.train` before starting the API."
        )

    started = time.time()
    ARTIFACTS["model"] = joblib.load(config.MODEL_PATH)

    if config.METADATA_PATH.exists():
        ARTIFACTS["metadata"] = json.loads(config.METADATA_PATH.read_text())
    else:
        raise RuntimeError(f"Model metadata not found at {config.METADATA_PATH}.")

    ARTIFACTS["threshold"] = float(ARTIFACTS["metadata"].get("threshold", {}).get("value", 0.5))
    ARTIFACTS["started_at"] = time.time()

    logger.info(json.dumps({
        "event": "startup",
        "model_version": ARTIFACTS["metadata"].get("model_version"),
        "threshold": ARTIFACTS["threshold"],
        "load_seconds": round(time.time() - started, 3),
    }))
    yield
    logger.info(json.dumps({"event": "shutdown"}))


app = FastAPI(
    title="Credit Default Risk API",
    description=(
        "Scores a loan application at **application time** and returns a calibrated "
        "probability of default, a 300-900 credit score, a rating band and an "
        "approve/decline recommendation.\n\n"
        "The model sees only information a lender has *before* disbursing the loan - "
        "no post-disbursal columns - so the score is reproducible in a real "
        "underwriting flow."
    ),
    version=config.MODEL_VERSION,
    lifespan=lifespan,
)

# Import after `app` exists to keep the module import graph obvious.
from api.schemas import (  # noqa: E402
    ApplicantRequest,
    BatchPredictionResponse,
    BatchRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Attach a request id and log method, path, status and latency."""
    request_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(json.dumps({
            "event": "request_failed", "request_id": request_id,
            "path": request.url.path, "method": request.method,
        }))
        raise
    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(json.dumps({
        "event": "request",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": round(latency_ms, 2),
    }))
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-ms"] = f"{latency_ms:.2f}"
    return response


def _require_model():
    """Return the loaded pipeline or raise a 503."""
    model = ARTIFACTS.get("model")
    if model is None:  # pragma: no cover - lifespan raises before this is reachable
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded; the service is not ready to score.",
        )
    return model


def _score(applicants: list[ApplicantRequest]) -> list[PredictionResponse]:
    """Score a list of validated applications in one vectorised batch."""
    model = _require_model()
    threshold = ARTIFACTS["threshold"]
    version = ARTIFACTS["metadata"].get("model_version", config.MODEL_VERSION)

    # One frame for the whole batch: a single predict_proba call is far
    # cheaper than one per applicant.
    frame = pd.DataFrame([a.model_dump() for a in applicants])
    # Column order is enforced here so the pipeline never has to guess.
    frame = frame[RAW_INPUTS]

    # Clamped away from exactly 0 and 1: see config.PD_FLOOR for why.
    probabilities = clip_pd(model.predict_proba(frame)[:, 1])
    scores = pd_to_credit_score(probabilities)
    ratings = score_to_rating(scores)

    return [
        PredictionResponse(
            probability_of_default=round(float(p), 6),
            credit_score=int(s),
            rating=str(r),
            decision="decline" if p >= threshold else "approve",
            threshold=threshold,
            model_version=version,
        )
        for p, s, r in zip(probabilities, scores, ratings, strict=True)
    ]


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "service": "Credit Default Risk API",
        "version": config.MODEL_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness plus whether the model actually loaded."""
    loaded = ARTIFACTS.get("model") is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=ARTIFACTS["metadata"].get("model_version") if loaded else None,
        uptime_seconds=round(time.time() - ARTIFACTS["started_at"], 1),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
async def model_info() -> ModelInfoResponse:
    """Version, training date, feature list and headline metrics."""
    meta = ARTIFACTS["metadata"]
    if not meta:
        raise HTTPException(status_code=503, detail="Model metadata is not loaded.")
    return ModelInfoResponse(
        model_version=meta.get("model_version", config.MODEL_VERSION),
        trained_at=meta.get("trained_at", "unknown"),
        algorithm=meta.get("algorithm", "unknown"),
        feature_columns=meta.get("feature_columns", []),
        raw_input_columns=meta.get("raw_input_columns", []),
        n_features=len(meta.get("feature_columns", [])),
        metrics=meta.get("metrics", {}),
        threshold=meta.get("threshold", {}),
        dataset=meta.get("dataset", {}),
        leakage_ablation=meta.get("leakage_ablation", {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
async def predict(applicant: ApplicantRequest) -> PredictionResponse:
    """Score a single loan application."""
    return _score([applicant])[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["scoring"])
async def predict_batch(payload: BatchRequest) -> BatchPredictionResponse:
    """Score up to 1,000 applications in one request."""
    predictions = _score(payload.applicants)
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Turn a stray ValueError into a 400 rather than an opaque 500."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})
