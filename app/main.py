"""
FastAPI inference server for TESA-CF.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Try it:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "The game keeps crashing every time I open it!"}'

Design notes:
    * The model is loaded ONCE in the lifespan hook and stored on app.state.
    * Endpoints are plain `def` (not `async def`): model inference is CPU-bound
      and blocking, so FastAPI runs them in its thread pool instead of freezing
      the event loop (which would block /health and every other request).
    * /health = liveness (process is up), /ready = readiness (model loaded).
      Load balancers / ECS / Kubernetes use them differently.
    * Tests inject a fake pipeline via app.state, so CI never downloads RoBERTa.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Protocol

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints

from configs.config import MAX_BATCH_SIZE, MAX_TEXT_CHARS, MODEL_VERSION
from src.schemas import SentimentResult

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("tesa_cf.api")


class Predictor(Protocol):
    """Anything with these methods can serve requests (real pipeline or a test fake)."""
    def predict_batch(self, texts: list[str]) -> list[SentimentResult]: ...


# ── Lifespan: load model once ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "pipeline", None) is None and os.getenv("SKIP_MODEL_LOAD") != "1":
        from src.inference import InferencePipeline   # heavy import, done lazily
        start = time.perf_counter()
        logger.info("Loading inference pipeline...")
        app.state.pipeline = InferencePipeline()
        logger.info("Pipeline ready in %.1fs", time.perf_counter() - start)
    yield
    app.state.pipeline = None
    logger.info("Pipeline released.")


app = FastAPI(
    title="TESA-CF Sentiment Analysis API",
    description="Topic-aware multi-domain sentiment analysis "
                "(frozen Twitter-RoBERTa + topic embedding + skip-connected head).",
    version="1.1.0",
    lifespan=lifespan,
)
app.state.pipeline = None


# ── Middleware: request id + latency logging ─────────────────────────────────

@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info("rid=%s %s %s -> %d (%.1f ms)",
                request_id, request.method, request.url.path, response.status_code, elapsed_ms)
    return response


# ── Schemas ───────────────────────────────────────────────────────────────────

Text = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS),
]


class PredictRequest(BaseModel):
    text: Text = Field(..., examples=["The game keeps crashing every time I open it!"])


class BatchPredictRequest(BaseModel):
    texts: list[Text] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE,
        examples=[["Great hotel, friendly staff", "Worst patch ever, servers are down"]],
    )


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    predicted_topic: str
    cleaned_text: str
    warnings: list[str] = []


class PredictResponse(Prediction):
    latency_ms: float
    model_version: str


class BatchPredictResponse(BaseModel):
    predictions: list[Prediction]
    count: int
    latency_ms: float
    model_version: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_pipeline(request: Request) -> Predictor:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return pipeline


def _to_prediction(r: SentimentResult) -> Prediction:
    return Prediction(
        label=r.label,
        confidence=r.confidence,
        probabilities=r.probabilities,
        predicted_topic=r.predicted_topic,
        cleaned_text=r.cleaned_text,
        warnings=list(r.warnings),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness probe: the process is up (does not care about the model)."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready(request: Request) -> dict:
    """Readiness probe: only 200 when the model can serve traffic."""
    if getattr(request.app.state, "pipeline", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.get("/model/info", tags=["ops"])
def model_info(request: Request) -> dict:
    pipeline = _get_pipeline(request)
    return {
        "model_version": MODEL_VERSION,
        "classes": getattr(pipeline, "class_names", None),
        "topics": getattr(pipeline, "topic_names", None),
        "device": getattr(pipeline, "device", None),
        "max_batch_size": MAX_BATCH_SIZE,
        "max_text_chars": MAX_TEXT_CHARS,
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    pipeline = _get_pipeline(request)
    start = time.perf_counter()
    result = pipeline.predict_batch([body.text])[0]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return PredictResponse(**_to_prediction(result).model_dump(),
                           latency_ms=latency_ms, model_version=MODEL_VERSION)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(body: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    pipeline = _get_pipeline(request)
    start = time.perf_counter()
    results = pipeline.predict_batch(body.texts)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return BatchPredictResponse(
        predictions=[_to_prediction(r) for r in results],
        count=len(results),
        latency_ms=latency_ms,
        model_version=MODEL_VERSION,
    )
