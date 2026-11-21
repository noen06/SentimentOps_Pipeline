"""
Quick smoke test for the deployed API.
Run: python test_api.py
"""

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    print(f"✓ Health check: {data}")


def test_single_predict():
    payload = {"text": "The game keeps crashing every time I open it!"}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Single predict:")
    print(f"  Label:     {data['label']}")
    print(f"  Confidence:{data['confidence']}")
    print(f"  Topic:     {data['predicted_topic']}")
    print(f"  Latency:   {data['latency_ms']} ms")


def test_batch_predict():
    texts = [
        "I absolutely love this product, works perfectly!",
        "Terrible service, never buying again.",
        "It arrived on time I guess.",
    ]
    response = requests.post(f"{BASE_URL}/predict/batch", json=texts)
    assert response.status_code == 200
    results = response.json()
    print(f"✓ Batch predict ({len(results)} results):")
    for text, result in zip(texts, results):
        print(f"  [{result['label']:8s}] {text[:50]}")


def test_empty_text():
    payload = {"text": "   "}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    # Pydantic min_length=1 sẽ reject cái này với 422
    assert response.status_code == 422
    print("✓ Empty text correctly rejected with 422")


if __name__ == "__main__":
    print("Running API smoke tests...\n")
    test_health()
    test_single_predict()
    test_batch_predict()
    test_empty_text()
    print("\nAll tests passed!")