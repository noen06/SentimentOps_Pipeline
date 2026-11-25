"""
TensorFlow/Keras classification head.
Dual-input: (roberta_embedding, topic_id) → 3-class softmax.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

from configs.config import (
    ROBERTA_DIM,
    TOPIC_EMBEDDING_DIM,
    DROPOUT_RATE,
    NUM_CLASSES,
    LEARNING_RATE,
    WEIGHT_DECAY,
)


def build_sentiment_head(num_topics: int) -> Model:
    """
    Build the topic-aware skip-connected classification head.

    Architecture (follows Algorithm 1 in the paper):
        Input: etext (768-d) + topic_id (int)
        ẽtext = Dropout(BN(etext))
        etopic = Embedding(topic_id) → 32-d
        f = Concat([ẽtext, etopic]) → 800-d
        z1 = Dropout(BN(ReLU(Dense(512)(f))))
        z2 = ReLU(Dense(256)(z1))
        z_skip = Concat([z2, f]) → 1056-d   ← skip connection
        z3 = Dropout(BN(ReLU(Dense(256)(z_skip))))
        z4 = ReLU(Dense(64)(z3))
        ŷ = Softmax(Dense(3)(z4))
    """
    # ── Inputs ────────────────────────────────────────────────────────────────
    text_input  = layers.Input(shape=(ROBERTA_DIM,), name="text_embedding")
    topic_input = layers.Input(shape=(1,), dtype=tf.int32, name="topic_id")

    # ── Topic embedding branch ────────────────────────────────────────────────
    topic_emb = layers.Embedding(
        input_dim=num_topics + 1,   # +1 for unknown topic at inference
        output_dim=TOPIC_EMBEDDING_DIM,
        name="topic_embedding",
    )(topic_input)
    topic_emb = layers.Flatten(name="topic_flatten")(topic_emb)

    # ── Pre-fusion BN + Dropout on text embedding ─────────────────────────────
    x_text = layers.BatchNormalization(name="bn_input")(text_input)
    x_text = layers.Dropout(DROPOUT_RATE, name="dropout_input")(x_text)

    # ── Fusion vector f (800-d) ───────────────────────────────────────────────
    fused = layers.Concatenate(name="fusion")([x_text, topic_emb])

    # ── Dense block 1: 512 units ──────────────────────────────────────────────
    z1 = layers.Dense(512, activation="relu", name="dense_512")(fused)
    z1 = layers.BatchNormalization(name="bn_512")(z1)
    z1 = layers.Dropout(DROPOUT_RATE, name="dropout_512")(z1)

    # ── Dense block 2: 256 units ──────────────────────────────────────────────
    z2 = layers.Dense(256, activation="relu", name="dense_256a")(z1)

    # ── Skip connection: concat z2 with original fusion vector ───────────────
    # z_skip shape: (batch, 256 + 800) = (batch, 1056)
    z_skip = layers.Concatenate(name="skip_connection")([z2, fused])

    # ── Dense block 3: 256 units (post-skip) ─────────────────────────────────
    z3 = layers.Dense(256, activation="relu", name="dense_256b")(z_skip)
    z3 = layers.BatchNormalization(name="bn_256b")(z3)
    z3 = layers.Dropout(DROPOUT_RATE, name="dropout_256b")(z3)

    # ── Dense block 4: 64 units ───────────────────────────────────────────────
    z4 = layers.Dense(64, activation="relu", name="dense_64")(z3)

    # ── Output: 3-class softmax ───────────────────────────────────────────────
    output = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(z4)

    model = Model(
        inputs=[text_input, topic_input],
        outputs=output,
        name="TESA_CF_SentimentHead",
    )
    return model


def compile_model(model: Model) -> Model:
    """
    Compile with AdamW (decoupled weight decay).
    tf.keras.optimizers.AdamW is built into TF >= 2.11, so the old
    tensorflow_addons dependency (deprecated, unmaintained) is no longer needed.
    """
    optimizer = optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model