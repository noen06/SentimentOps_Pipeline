"""
Smoke test against a RUNNING server (real model).
    uvicorn app.main:app --port 8000
    python test_api.py
Unit tests (no model, no server) live in tests/ -> run `pytest`.
"""

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    r = requests.get(f"{BASE_URL}/ready")
    assert r.status_code == 200, "model not loaded yet"
    print(f"OK health/ready: {r.json()}")


def test_single_predict():
    r = requests.post(f"{BASE_URL}/predict",
                      json={"text": "The game keeps crashing every time I open it!"})
    assert r.status_code == 200
    d = r.json()
    print(f"OK predict: {d['label']} ({d['confidence']}) topic={d['predicted_topic']} "
          f"latency={d['latency_ms']}ms rid={r.headers.get('X-Request-ID')}")


def test_batch_predict():
    texts = [
        "I absolutely love this product, works perfectly!",
        "Terrible service, never buying again.",
        "It arrived on time I guess.",
    ]
    r = requests.post(f"{BASE_URL}/predict/batch", json={"texts": texts})
    assert r.status_code == 200
    d = r.json()
    assert d["count"] == len(texts)
    print(f"OK batch ({d['count']} in {d['latency_ms']}ms):")
    for text, p in zip(texts, d["predictions"]):
        print(f"  [{p['label']:8s}] {text[:50]}")


def test_validation():
    assert requests.post(f"{BASE_URL}/predict", json={"text": "   "}).status_code == 422
    assert requests.post(f"{BASE_URL}/predict/batch", json={"texts": []}).status_code == 422
    print("OK invalid input rejected with 422")


if __name__ == "__main__":
    test_health()
    test_single_predict()
    test_batch_predict()
    test_validation()
    print("\nAll smoke tests passed!")
