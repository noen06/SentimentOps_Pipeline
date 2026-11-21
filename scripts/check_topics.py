# check_topics.py
from src.inference import InferencePipeline

p = InferencePipeline()
tests = [
    "My PS5 controller stopped working after the update",
    "Xbox Game Pass has so many great games",
    "Amazon delivered my package two days late",
    "The hotel room was clean and the staff were friendly",
    "My iPhone battery drains so fast",
    "Another patch, another broken champion ruining ranked",
]
for t in tests:
    r = p.predict_sentiment(t)
    print(f"{r.predicted_topic:18s} {r.label:9s} {r.confidence:.3f} | {t}")