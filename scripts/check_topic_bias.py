# check_topic_bias.py
import numpy as np
from src.inference import InferencePipeline
from src.data_preprocess import clean_text

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

topics = list(p._topic_encoder.classes_)
pos_idx = list(p._label_encoder.classes_).index("Positive")

def mean_p_positive(sentences):
    out = np.zeros(len(topics))
    for s in sentences:
        emb = p._extract_embedding(clean_text(s))
        for tid in range(len(topics)):
            probs = p._keras_model(
                {"text_embedding": emb, "topic_id": np.array([[tid]], dtype=np.int32)},
                training=False,
            ).numpy()[0]
            out[tid] += probs[pos_idx]
    return out / len(sentences)

neg_mean = mean_p_positive(neg)
pos_mean = mean_p_positive(pos)

print(f"{'topic':22s} {'P(Pos) neg-sent':>16s} {'P(Pos) pos-sent':>16s}")
for i, t in enumerate(topics):
    print(f"{t:22s} {neg_mean[i]:16.3f} {pos_mean[i]:16.3f}")