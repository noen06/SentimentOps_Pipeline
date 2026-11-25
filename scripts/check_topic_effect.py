"""
check_topic_effect.py - how much does the topic id change the prediction?
Feeds ONE sentence through the head with every possible topic id.
Run: python -m scripts.check_topic_effect
"""
import numpy as np
import tensorflow as tf

from src.data_preprocess import clean_text
from src.inference import InferencePipeline

p = InferencePipeline()
text = "My PS5 controller stopped working after the update"
emb = p._embed([clean_text(text)])                      # (1, 768)

for tid, name in enumerate(p.topic_names):
    probs = p._head(
        {"text_embedding": tf.constant(emb), "topic_id": tf.constant([[tid]], dtype=tf.int32)},
        training=False,
    ).numpy()[0]
    print(f"{name:22s} {p.class_names[int(np.argmax(probs))]:9s} {probs.max():.3f}")
