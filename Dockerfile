# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Credit Default Risk API
#
# Two stages so the wheel-building toolchain never ships to production.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# build-essential provides the C toolchain some wheels need to compile. It is
# needed only to *install* packages, which is exactly why this is a separate
# stage - none of it reaches the final image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt


# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# libgomp1 is XGBoost's OpenMP runtime. Without it `import xgboost` fails at
# load time with a dlopen error, which is a confusing way to discover a missing
# system package - so it is installed explicitly rather than hoped for.
# curl is here only so HEALTHCHECK below has something to call with.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user. If the process is ever compromised it should not
# own the filesystem.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=builder /install /usr/local

# Only what the service actually needs to serve. src/ comes along because the
# serialized pipeline imports src.features at unpickle time - that is the
# training/serving contract, not an accident.
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser models/ ./models/

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    WEB_CONCURRENCY=2

EXPOSE 8000

# The model loads at startup and the app refuses to boot without it, so a
# passing healthcheck genuinely means "ready to score".
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Shell form so ${PORT} is expanded - Render and Fly assign the port at runtime.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY}
