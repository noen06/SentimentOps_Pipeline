"""
Shared fixtures. The API is tested with a FAKE pipeline so tests run in
milliseconds and CI never downloads RoBERTa / TensorFlow weights.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SKIP_MODEL_LOAD", "1")

from src.schemas import SentimentResult  # noqa: E402


class FakePipeline:
    class_names = ["Negative", "Neutral", "Positive"]
    topic_names = ["gaming", "hotel", "product"]
    device = "cpu"

    def __init__(self):
        self.calls: list[list[str]] = []

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:
        self.calls.append(list(texts))
        out = []
        for t in texts:
            label = "Negative" if "bad" in t.lower() else "Positive"
            probs = {"Negative": 0.9 if label == "Negative" else 0.05,
                     "Neutral": 0.05,
                     "Positive": 0.05 if label == "Negative" else 0.9}
            out.append(SentimentResult(label=label, confidence=0.9, probabilities=probs,
                                       predicted_topic="product", cleaned_text=t.lower()))
        return out


@pytest.fixture
def fake_pipeline():
    return FakePipeline()


@pytest.fixture
def client(fake_pipeline):
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:          # runs lifespan (skipped load via env)
        app.state.pipeline = fake_pipeline
        yield c
    app.state.pipeline = None
