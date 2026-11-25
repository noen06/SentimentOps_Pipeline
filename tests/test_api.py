from configs.config import MAX_BATCH_SIZE, MAX_TEXT_CHARS, MODEL_VERSION


def test_health_is_always_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_when_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["model_version"] == MODEL_VERSION


def test_ready_503_when_not_loaded(client):
    from app.main import app
    app.state.pipeline = None
    assert client.get("/ready").status_code == 503
    assert client.post("/predict", json={"text": "hi"}).status_code == 503


def test_predict_single(client, fake_pipeline):
    r = client.post("/predict", json={"text": "  This is bad  "})
    assert r.status_code == 200
    d = r.json()
    assert d["label"] == "Negative"
    assert set(d["probabilities"]) == {"Negative", "Neutral", "Positive"}
    assert d["model_version"] == MODEL_VERSION
    assert "latency_ms" in d
    assert fake_pipeline.calls == [["This is bad"]]          # whitespace stripped


def test_predict_batch_single_call_and_order(client, fake_pipeline):
    texts = ["good one", "bad one", "great"]
    r = client.post("/predict/batch", json={"texts": texts})
    assert r.status_code == 200
    d = r.json()
    assert d["count"] == 3
    assert [p["label"] for p in d["predictions"]] == ["Positive", "Negative", "Positive"]
    assert len(fake_pipeline.calls) == 1                     # true batching: one call


def test_validation_errors(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422
    assert client.post("/predict", json={"text": "a" * (MAX_TEXT_CHARS + 1)}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422
    too_many = ["ok"] * (MAX_BATCH_SIZE + 1)
    assert client.post("/predict/batch", json={"texts": too_many}).status_code == 422
    assert client.post("/predict/batch", json=["old", "format"]).status_code == 422


def test_request_id_and_timing_headers(client):
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    assert float(r.headers["X-Process-Time-Ms"]) >= 0


def test_model_info(client):
    d = client.get("/model/info").json()
    assert d["classes"] == ["Negative", "Neutral", "Positive"]
    assert d["max_batch_size"] == MAX_BATCH_SIZE
