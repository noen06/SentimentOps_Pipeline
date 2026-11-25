"""
check_topic_bias.py - does a topic id push predictions toward Positive/Negative
regardless of the text? Averages P(Positive) for clearly negative vs clearly
positive sentences under every topic id.
Run: python -m scripts.check_topic_bias
"""
import numpy as np
import tensorflow as tf

from src.data_preprocess import clean_text
from src.inference import InferencePipeline

p = InferencePipeline()

neg = [
    "This is terrible and I am so disappointed",
    "Worst experience ever, completely useless",
    "It stopped working and nobody will help me",
    "I hate how slow and broken this is",
    "Awful quality, a total waste of money",
    "The update ruined everything, so frustrating",
    "I am angry that this keeps failing again",
    "Really bad, I regret buying it",
]
pos = [
    "This is amazing and I am so happy with it",
    "Best experience ever, absolutely wonderful",
    "It works perfectly and the support was great",
    "I love how fast and smooth this is",
    "Excellent quality, worth every penny",
    "The update improved everything, so impressive",
    "I am delighted that this keeps working well",
    "Really good, I am glad I bought it",
]

pos_idx = p.class_names.index("Positive")


def mean_p_positive(sentences: list[str]) -> np.ndarray:
    emb = p._embed([clean_text(s) for s in sentences])  # one batched RoBERTa pass
    out = np.zeros(p.num_topics)
    for tid in range(p.num_topics):
        ids = tf.constant(np.full((len(sentences), 1), tid, dtype=np.int32))
        probs = p._head({"text_embedding": tf.constant(emb), "topic_id": ids}, training=False).numpy()
        out[tid] = probs[:, pos_idx].mean()
    return out


neg_mean, pos_mean = mean_p_positive(neg), mean_p_positive(pos)
print(f"{'topic':22s} {'P(Pos) neg-sent':>16s} {'P(Pos) pos-sent':>16s}")
for i, t in enumerate(p.topic_names):
    print(f"{t:22s} {neg_mean[i]:16.3f} {pos_mean[i]:16.3f}")
