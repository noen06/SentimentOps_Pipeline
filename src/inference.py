"""
Production inference pipeline.
InferencePipeline loads all artifacts once at startup and
exposes a single predict_sentiment() method for serving.

Memory management strategy:
- RoBERTa runs on GPU if available, releases after extraction.
- TF Keras head runs on CPU by default to avoid CUDA context conflict.
  Set TF_FORCE_GPU=1 env var to override if you have enough VRAM.
"""

import os
import gc
import numpy as np
import torch
import joblib
import tensorflow as tf
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from configs.config import (
    ROBERTA_CHECKPOINT, MAX_LEN, ROBERTA_DIM,
    TOPIC_PIPELINE_PATH, LABEL_ENCODER_PATH, KERAS_WEIGHTS_PATH,
    ARTIFACT_DIR, ID_TO_LABEL, NUM_CLASSES, TOPIC_EMBEDDING_DIM,
    DROPOUT_RATE,
)
from src.data_preprocess import clean_text
from src.model import build_sentiment_head


@dataclass
class SentimentResult:
    label: str          # "Negative" | "Neutral" | "Positive"
    confidence: float   # probability of predicted class
    probabilities: dict[str, float]  # full distribution
    predicted_topic: str             # topic predicted by auxiliary classifier
    cleaned_text: str               # text after preprocessing


class InferencePipeline:
    """
    Loads all artifacts once and provides fast single-text inference.

    Usage:
        pipeline = InferencePipeline()
        result = pipeline.predict_sentiment("The game keeps crashing!")
        print(result.label, result.confidence)
    """

    def __init__(
        self,
        artifact_dir: Path = ARTIFACT_DIR,
        force_cpu_keras: bool = True,
    ) -> None:
        print("[InferencePipeline] Initializing...")

        self._artifact_dir = artifact_dir
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Step 1: Load sklearn artifacts (fast, CPU only)
        self._topic_pipeline  = self._load_topic_pipeline()
        self._label_encoder   = self._load_label_encoder()
        self._topic_encoder   = joblib.load(artifact_dir / "topic_encoder.joblib")
        self._num_topics      = len(self._topic_encoder.classes_)

        # Step 2: Load RoBERTa tokenizer + model (PyTorch)
        self._tokenizer, self._roberta = self._load_roberta()

        # Step 3: Load Keras classification head
        # Force TF to use CPU to avoid CUDA conflict with PyTorch
        if force_cpu_keras:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self._keras_model = self._load_keras_model()

        print("[InferencePipeline] All artifacts loaded. Ready.")

    # ── Private loaders ───────────────────────────────────────────────────────

    def _load_topic_pipeline(self):
        print("[InferencePipeline] Loading topic classifier...")
        return joblib.load(TOPIC_PIPELINE_PATH)

    def _load_label_encoder(self):
        print("[InferencePipeline] Loading label encoder...")
        return joblib.load(LABEL_ENCODER_PATH)

    def _load_roberta(self) -> tuple:
        print(f"[InferencePipeline] Loading RoBERTa on {self._device}...")
        tokenizer = AutoTokenizer.from_pretrained(ROBERTA_CHECKPOINT)
        model = AutoModel.from_pretrained(ROBERTA_CHECKPOINT)
        model.eval()
        model.to(self._device)
        for param in model.parameters():
            param.requires_grad = False
        return tokenizer, model

    def _load_keras_model(self) -> tf.keras.Model:
        print("[InferencePipeline] Loading Keras classification head...")
        model = build_sentiment_head(num_topics=self._num_topics)
        model.load_weights(str(KERAS_WEIGHTS_PATH))
        return model

    # ── Core inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_embedding(self, text: str) -> np.ndarray:
        """Extract 768-d mean-pooled embedding for a single cleaned text."""
        encoding = self._tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)

        outputs = self._roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling (same as training)
        mask_expanded = attention_mask.unsqueeze(-1).expand(
            outputs.last_hidden_state.size()
        ).float()
        sum_emb  = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embedding = (sum_emb / sum_mask).cpu().numpy()

        return embedding.astype(np.float32)  # shape: (1, 768)

    def predict_sentiment(self, text: str) -> SentimentResult:
        """
        Full inference pipeline for a single raw text string.

        Steps:
        1. Clean text
        2. Extract RoBERTa embedding (PyTorch)
        3. Predict topic (Sklearn TF-IDF + LR)
        4. Run Keras classification head
        5. Return structured result
        """
        # ── 1. Preprocess ─────────────────────────────────────────────────────
        cleaned = clean_text(text)
        if not cleaned:
            return SentimentResult(
                label="Neutral",
                confidence=0.0,
                probabilities={"Negative": 0.0, "Neutral": 1.0, "Positive": 0.0},
                predicted_topic="unknown",
                cleaned_text=cleaned,
            )

        # ── 2. Embed ──────────────────────────────────────────────────────────
        embedding = self._extract_embedding(cleaned)  # (1, 768)

        # ── 3. Predict topic ──────────────────────────────────────────────────
        topic_id_arr  = self._topic_pipeline.predict([cleaned])  # (1,)
        topic_id      = int(topic_id_arr[0])

        # Decode topic name for logging/output
        try:
            topic_name = self._topic_encoder.inverse_transform([topic_id])[0]
        except Exception:
            topic_name = "unknown"

        topic_id_input = np.array([[topic_id]], dtype=np.int32)  # (1, 1)

        # ── 4. Keras classification head ──────────────────────────────────────
        probs = self._keras_model.predict(
            {"text_embedding": embedding, "topic_id": topic_id_input},
            verbose=0,
        )[0]  # shape: (3,)

        pred_idx    = int(np.argmax(probs))
        pred_label  = self._label_encoder.inverse_transform([pred_idx])[0]
        confidence  = float(probs[pred_idx])

        prob_dict = {
            self._label_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(probs)
        }

        return SentimentResult(
            label=pred_label,
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in prob_dict.items()},
            predicted_topic=topic_name,
            cleaned_text=cleaned,
        )

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Convenience wrapper for batch inference (calls predict_sentiment in loop)."""
        return [self.predict_sentiment(t) for t in texts]