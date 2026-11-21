"""
FastAPI inference server for TESA-CF.

Start server:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Test:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "The game keeps crashing every time I open it!"}'
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import time

from src.inference import InferencePipeline, SentimentResult


# ── Global pipeline instance ──────────────────────────────────────────────────
# Dùng một instance duy nhất — không load lại model mỗi request
# Đây là điểm quan trọng nhất: load model 1 lần khi server khởi động
pipeline: InferencePipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model khi server start, giải phóng khi server tắt.
    Lifespan context manager là cách FastAPI khuyên dùng từ v0.95+
    thay cho @app.on_event("startup") cũ.
    """
    global pipeline
    print("[server] Loading inference pipeline...")
    pipeline = InferencePipeline()
    print("[server] Pipeline ready. Server accepting requests.")
    yield
    # Cleanup khi server tắt
    del pipeline
    print("[server] Pipeline released.")


app = FastAPI(
    title="TESA-CF Sentiment Analysis API",
    description="Topic-Aware Multi-Domain Sentiment Analysis with Explainable Counterfactual Interventions",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Raw text to classify",
        example="The game keeps crashing every time I open it!",
    )

    @field_validator("text", mode="before")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    predicted_topic: str
    cleaned_text: str
    latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Dùng để kiểm tra server còn sống không — Docker và load balancer cần cái này.
    """
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict sentiment for a single text input.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    start = time.perf_counter()

    result: SentimentResult = pipeline.predict_sentiment(request.text)

    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        label=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        predicted_topic=result.predicted_topic,
        cleaned_text=result.cleaned_text,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/batch")
async def predict_batch(texts: list[str]):
    """
    Predict sentiment for multiple texts at once.
    Giới hạn 32 texts mỗi lần để tránh timeout.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if len(texts) > 32:
        raise HTTPException(
            status_code=400,
            detail="Maximum 32 texts per batch request.",
        )

    results = pipeline.predict_batch(texts)

    return [
        {
            "label": r.label,
            "confidence": r.confidence,
            "predicted_topic": r.predicted_topic,
        }
        for r in results
    ]