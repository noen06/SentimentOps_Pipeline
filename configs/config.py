"""
Centralized configuration for the TESA-CF pipeline.
All hyperparameters and paths live here — never hardcode these elsewhere.

Serving settings can be overridden with environment variables
(useful for Docker / ECS / SageMaker without rebuilding the image).
"""

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT_DIR / "artifacts"))
DATA_DIR = ROOT_DIR / "dataset"

TOPIC_PIPELINE_PATH = ARTIFACT_DIR / "topic_pipeline.joblib"
LABEL_ENCODER_PATH  = ARTIFACT_DIR / "label_encoder.joblib"
TOPIC_ENCODER_PATH  = ARTIFACT_DIR / "topic_encoder.joblib"
KERAS_WEIGHTS_PATH  = ARTIFACT_DIR / "sentiment_head.weights.h5"
EMBEDDINGS_CACHE    = ARTIFACT_DIR / "embeddings_cache.npz"


# ── Reproducibility ───────────────────────────────────────────────────────────
SEED: int = 42


# ── RoBERTa / Embedding ───────────────────────────────────────────────────────
ROBERTA_CHECKPOINT: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LEN: int = 128
ROBERTA_DIM: int = 768
EMBED_BATCH_SIZE: int = 64   # batch size for offline embedding extraction


# ── Topic Classifier (Sklearn) ────────────────────────────────────────────────
TFIDF_MAX_FEATURES: int = 20_000
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)
LR_C: float = 10.0
CV_FOLDS: int = 5


# ── Sentiment Head (Keras) ────────────────────────────────────────────────────
TOPIC_EMBEDDING_DIM: int = 32
DROPOUT_RATE: float = 0.4
DENSE_UNITS: tuple[int, ...] = (512, 256, 256, 64)   # documentation only
NUM_CLASSES: int = 3

LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 0.01
MAX_EPOCHS: int = 30
EARLY_STOPPING_PATIENCE: int = 5
REDUCE_LR_PATIENCE: int = 3
REDUCE_LR_FACTOR: float = 0.5
TRAIN_BATCH_SIZE: int = 16
EVAL_BATCH_SIZE: int = 32


# ── Serving ───────────────────────────────────────────────────────────────────
MAX_TEXT_CHARS: int = _env_int("MAX_TEXT_CHARS", 512)    # per-text request limit
MAX_BATCH_SIZE: int = _env_int("MAX_BATCH_SIZE", 32)     # texts per /predict/batch
INFER_BATCH_SIZE: int = _env_int("INFER_BATCH_SIZE", 32) # RoBERTa forward chunk
TORCH_NUM_THREADS: int = _env_int("TORCH_NUM_THREADS", 0)  # 0 = library default
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "tesa-cf-1.0.0")


# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL_MAP: dict[str, int] = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
}
ID_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
