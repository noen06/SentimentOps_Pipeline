# check_topic_effect.py
import numpy as np
from src.inference import InferencePipeline
from src.data_preprocess import clean_text

p = InferencePipeline()
text = "My PS5 controller stopped working after the update"
emb = p._extract_embedding(clean_text(text))

for tid, name in enumerate(p._topic_encoder.classes_):
    probs = p._keras_model(
        {"text_embedding": emb, "topic_id": np.array([[tid]], dtype=np.int32)},
        training=False,
    ).numpy()[0]
    label = p._label_encoder.inverse_transform([int(np.argmax(probs))])[0]
    print(f"{name:22s} {label:9s} {probs.max():.3f}")