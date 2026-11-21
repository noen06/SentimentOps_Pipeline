"""
End-to-end training script.
Run: python -m src.train
"""

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

from configs.config import (
    SEED, DATA_DIR, ARTIFACT_DIR,
    TOPIC_PIPELINE_PATH, LABEL_ENCODER_PATH, KERAS_WEIGHTS_PATH,
    EMBEDDINGS_CACHE, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    MAX_EPOCHS, EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, LR_C, CV_FOLDS,
)
from src.data_preprocess import clean_dataframe, encode_labels, save_encoder
from src.features import extract_and_cache
from src.model import build_sentiment_head, compile_model


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """Load and concatenate your three source datasets here."""
    # Adapt this to your actual data loading logic
    df = pd.read_csv(data_path)
    return df


def make_tf_dataset(
    embeddings: np.ndarray,
    topic_ids: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from numpy arrays.
    Using tf.data is significantly faster than fitting on raw numpy.
    """
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"text_embedding": embeddings, "topic_id": topic_ids}, labels)
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels), seed=SEED)
    dataset = (
        dataset
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def train_topic_classifier(
    train_texts: list[str],
    train_topics: np.ndarray,
) -> Pipeline:
    """
    Train TF-IDF + Logistic Regression topic classifier with cross-validation.
    Saves the fitted pipeline to disk for inference.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
        )),
        ("clf", LogisticRegression(
            C=LR_C,
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
        )),
    ])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(pipeline, train_texts, train_topics, cv=cv, scoring="accuracy")
    print(f"[train] Topic classifier CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipeline.fit(train_texts, train_topics)
    joblib.dump(pipeline, TOPIC_PIPELINE_PATH)
    print(f"[train] Topic pipeline saved to: {TOPIC_PIPELINE_PATH}")

    return pipeline


def main() -> None:
    set_seeds()

    # ── 1. Load & clean data ──────────────────────────────────────────────────
    raw_df = load_raw_data(DATA_DIR / "merged_twitter_sentiment.csv")
    df = clean_dataframe(raw_df, text_col="text", label_col="sentiment")

    # ── 2. Encode sentiment labels ─────────────────────────────────────────────
    labels, label_encoder = encode_labels(df["sentiment"])
    save_encoder(label_encoder, LABEL_ENCODER_PATH)
    num_classes = len(label_encoder.classes_)

    # ── 3. Encode topic labels (integer IDs) ───────────────────────────────────
    topics, topic_encoder = encode_labels(df["topic"])
    num_topics = len(topic_encoder.classes_)
    joblib.dump(topic_encoder, ARTIFACT_DIR / "topic_encoder.joblib")

    # ── 4. Stratified split ───────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split

    X_trainval, X_test, y_trainval, y_test, t_trainval, t_test = train_test_split(
        df["text"].tolist(), labels, topics,
        test_size=0.15, stratify=labels, random_state=SEED,
    )
    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X_trainval, y_trainval, t_trainval,
        test_size=0.15 / 0.85, stratify=y_trainval, random_state=SEED,
    )
    print(f"[train] Split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

    # ── 5. Extract RoBERTa embeddings (cached) ────────────────────────────────
    train_emb, val_emb, test_emb = extract_and_cache(
        X_train, X_val, X_test,
        cache_path=EMBEDDINGS_CACHE,
        force_recompute=False,
    )

    # ── 6. Train topic classifier ─────────────────────────────────────────────
    topic_pipeline = train_topic_classifier(X_train, t_train)

    # Predict topic IDs for all splits using the trained classifier
    train_topic_ids = topic_pipeline.predict(X_train).astype(np.int32)
    val_topic_ids   = topic_pipeline.predict(X_val).astype(np.int32)

    # ── 7. Build tf.data datasets ─────────────────────────────────────────────
    train_ds = make_tf_dataset(train_emb, train_topic_ids, y_train, TRAIN_BATCH_SIZE, shuffle=True)
    val_ds   = make_tf_dataset(val_emb, val_topic_ids, y_val, EVAL_BATCH_SIZE)

    # ── 8. Compute class weights ──────────────────────────────────────────────
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"[train] Class weights: {class_weight_dict}")

    # ── 9. Build & compile model ──────────────────────────────────────────────
    model = build_sentiment_head(num_topics=num_topics)
    model = compile_model(model)
    model.summary()

    # ── 10. Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(KERAS_WEIGHTS_PATH),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    # ── 11. Train ─────────────────────────────────────────────────────────────
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    print(f"[train] Best val_loss: {min(history.history['val_loss']):.4f}")
    print(f"[train] Weights saved to: {KERAS_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()