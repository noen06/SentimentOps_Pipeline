"""Architecture regression test: param count must match the paper (835,267)."""
import numpy as np
import pytest

pytest.importorskip("tensorflow")
from src.model import build_sentiment_head  # noqa: E402


def test_param_count_matches_paper():
    model = build_sentiment_head(num_topics=13)
    assert model.count_params() == 835_267


def test_output_is_probability_distribution():
    model = build_sentiment_head(num_topics=13)
    emb = np.random.rand(4, 768).astype("float32")
    topics = np.array([[0], [3], [12], [13]], dtype="int32")   # 13 = unknown slot
    probs = model([emb, topics], training=False).numpy()
    assert probs.shape == (4, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
