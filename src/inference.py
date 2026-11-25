"""
Production inference pipeline for TESA-CF.

InferencePipeline loads every artifact ONCE (at server startup) and exposes:
    - predict_batch(texts)      -> list[SentimentResult]   (the real work)
    - predict_sentiment(text)   -> SentimentResult         (wrapper over batch)

Design notes (why it looks like this):
    * True batching: one tokenizer call + one RoBERTa forward pass + one
      vectorised TF-IDF predict + one Keras call for the whole batch,
      instead of looping N times.
    * Dynamic padding (padding=True): pad to the longest text in the batch,
      not to MAX_LEN=128. Because we use attention-mask mean pooling, pad
      tokens never affect the embedding -> same result, much less compute.
    * Direct Keras call `model(x, training=False)` instead of `model.predict()`
      — predict() builds a tf.data pipeline every call (slow for small inputs).
    * TensorFlow head is pinned to CPU with tf.config.set_visible_devices,
      so TF does not grab the GPU memory PyTorch needs.
      (Setting CUDA_VISIBLE_DEVICES after TF is imported has no effect.)
    * A lock serialises access: HuggingFace fast tokenizers are not safe to
      call from several threads at once ("Already borrowed" error), and FastAPI
      runs sync endpoints in a thread pool.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
import torch
from transformers import AutoModel, AutoTokenizer

from configs.config import (
    ARTIFACT_DIR,
    INFER_BATCH_SIZE,
    MAX_LEN,
    ROBERTA_CHECKPOINT,
    TORCH_NUM_THREADS,
)
from src.data_preprocess import clean_text
from src.model import build_sentiment_head
from src.schemas import SentimentResult

logger = logging.getLogger(__name__)

EMPTY_TEXT_WARNING = "empty_after_cleaning: text had no usable words, returned a neutral fallback"


def _pin_tensorflow_to_cpu() -> None:
    """Hide GPUs from TensorFlow. Must run before TF initialises its devices."""
    try:
        tf.config.set_visible_devices([], "GPU")
    except (RuntimeError, ValueError) as exc:  # devices already initialised
        logger.warning("Could not pin TensorFlow to CPU: %s", exc)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling: average only over real tokens, ignore padding."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)   # (B, L, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                     # (B, 768)
    counts = mask.sum(dim=1).clamp(min=1e-9)                           # (B, 1)
    return summed / counts


class InferencePipeline:
    """
    Usage:
        pipeline = InferencePipeline()
        pipeline.predict_sentiment("The game keeps crashing!").label
        pipeline.predict_batch(["great hotel", "awful service"])
    """

    def __init__(self, artifact_dir: Path = ARTIFACT_DIR, force_cpu_keras: bool = True) -> None:
        if force_cpu_keras:
            _pin_tensorflow_to_cpu()
        if TORCH_NUM_THREADS > 0:
            torch.set_num_threads(TORCH_NUM_THREADS)

        artifact_dir = Path(artifact_dir)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()

        logger.info("Loading sklearn artifacts from %s", artifact_dir)
        self._topic_pipeline = joblib.load(artifact_dir / "topic_pipeline.joblib")
        label_encoder = joblib.load(artifact_dir / "label_encoder.joblib")
        topic_encoder = joblib.load(artifact_dir / "topic_encoder.joblib")

        # Pre-compute names once (no inverse_transform per request)
        self.class_names: list[str] = [str(c) for c in label_encoder.classes_]
        self.topic_names: list[str] = [str(t) for t in topic_encoder.classes_]
        self.num_topics = len(self.topic_names)

        logger.info("Loading RoBERTa (%s) on %s", ROBERTA_CHECKPOINT, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(ROBERTA_CHECKPOINT)
        self._roberta = AutoModel.from_pretrained(ROBERTA_CHECKPOINT).to(self._device).eval()
        for p in self._roberta.parameters():
            p.requires_grad = False

        logger.info("Loading Keras sentiment head")
        self._head = build_sentiment_head(num_topics=self.num_topics)
        self._head.load_weights(str(artifact_dir / "sentiment_head.weights.h5"))

        # Warm-up: first call pays graph / kernel initialisation cost,
        # do it now instead of on the first user request.
        self.predict_batch(["warm up the model"])
        logger.info("InferencePipeline ready (topics=%d, device=%s)", self.num_topics, self._device)

    # ── Info ──────────────────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        return str(self._device)

    # ── Core steps ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _embed(self, texts: list[str]) -> np.ndarray:
        """Cleaned texts -> (N, 768) float32 embeddings, chunked by INFER_BATCH_SIZE."""
        chunks = []
        for start in range(0, len(texts), INFER_BATCH_SIZE):
            batch = texts[start:start + INFER_BATCH_SIZE]
            enc = self._tokenizer(
                batch,
                max_length=MAX_LEN,
                truncation=True,
                padding=True,              # dynamic padding
                return_tensors="pt",
            ).to(self._device)
            out = self._roberta(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            chunks.append(mean_pool(out.last_hidden_state, enc["attention_mask"]).cpu().numpy())
        return np.concatenate(chunks).astype(np.float32)

    def _classify(self, cleaned: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (probs (N,3), topic_ids (N,)) for non-empty cleaned texts."""
        embeddings = self._embed(cleaned)
        topic_ids = np.asarray(self._topic_pipeline.predict(cleaned), dtype=np.int32)
        probs = self._head(
            {
                "text_embedding": tf.constant(embeddings),
                "topic_id": tf.constant(topic_ids.reshape(-1, 1)),
            },
            training=False,   # BatchNorm uses moving stats, Dropout off
        ).numpy()
        return probs, topic_ids

    def _to_result(self, probs: np.ndarray, topic_id: int, cleaned: str) -> SentimentResult:
        idx = int(np.argmax(probs))
        topic = self.topic_names[topic_id] if 0 <= topic_id < self.num_topics else "unknown"
        return SentimentResult(
            label=self.class_names[idx],
            confidence=round(float(probs[idx]), 4),
            probabilities={n: round(float(p), 4) for n, p in zip(self.class_names, probs)},
            predicted_topic=topic,
            cleaned_text=cleaned,
        )

    def _empty_result(self, cleaned: str) -> SentimentResult:
        uniform = round(1.0 / len(self.class_names), 4)
        return SentimentResult(
            label="Neutral",
            confidence=uniform,
            probabilities={n: uniform for n in self.class_names},
            predicted_topic="unknown",
            cleaned_text=cleaned,
            warnings=[EMPTY_TEXT_WARNING],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Batch inference. Order of results == order of inputs."""
        if not texts:
            return []

        cleaned = [clean_text(t) for t in texts]
        valid_idx = [i for i, c in enumerate(cleaned) if c]
        results: list[SentimentResult | None] = [None] * len(texts)

        if valid_idx:
            valid_texts = [cleaned[i] for i in valid_idx]
            with self._lock:
                probs, topic_ids = self._classify(valid_texts)
            for j, i in enumerate(valid_idx):
                results[i] = self._to_result(probs[j], int(topic_ids[j]), cleaned[i])

        for i, c in enumerate(cleaned):
            if results[i] is None:
                results[i] = self._empty_result(c)

        return results  # type: ignore[return-value]

    def predict_sentiment(self, text: str) -> SentimentResult:
        """Single-text inference (same code path as batch -> no train/serve drift)."""
        return self.predict_batch([text])[0]
