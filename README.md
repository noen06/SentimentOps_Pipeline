# SentimentOps: Sentiment Model Serving Pipeline

![CI](https://github.com/noen06/SentimentOps_Pipeline/actions/workflows/ci.yml/badge.svg)

Production-style serving pipeline for **TESA-CF** (Topic-Aware Explainable Sentiment Analysis), a multi-domain sentiment model from the paper *"Topic-Aware Multi-Domain Sentiment Analysis with Explainable Counterfactual Interventions"* (EXPLAINS 2026).

The research notebook is refactored into a modular, config-driven Python package and served as a REST API:

- **FastAPI** with true batch inference
- **Pydantic** input validation
- **Liveness and readiness probes**
- **Request tracing**
- **Unit tests** with a fake model
- **Docker** image
- **GitHub Actions** CI

## How the model works

```
text ──► clean_text ──┬──► TF-IDF + Logistic Regression ──► topic_id ──► topic embedding (32-d) ─┐
                      │                                                                          ├──► dense head (skip + BN) ──► Negative / Neutral / Positive
                      └──► Twitter-RoBERTa (frozen, masked mean-pooled 768-d) ───────────────────┘
```

1. A **topic classifier** (TF-IDF + Logistic Regression, scikit-learn) predicts the domain of the text. There are 13 domains.
2. **Frozen Twitter-RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) produces a 768-d sentence embedding.
3. A **Keras dual-input head** fuses the embedding with a learned topic embedding, using a skip connection and BatchNorm. It outputs 3 sentiment classes and has 835,267 trainable parameters.

## Results (paper)

On 44,396 texts across 13 domains (stratified 70/15/15 split):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| **TESA-CF (seed 42)** | **0.8793** | **0.8682** |
| TESA-CF (5-seed mean ± std) | 0.8737 ± 0.0023 | 0.8612 ± 0.0028 |
| Best TF-IDF baseline | – | 0.850 |
| Zero-shot Twitter-RoBERTa | – | 0.663 |

Ablation (macro-F1): no skip connection 0.8574 · no topic embedding 0.8491 · no BatchNorm 0.8356.

**Reproducing with this repo:** the artifacts in `artifacts/` were retrained with the packaged preprocessing. That cleaning keeps 44,441 rows, so the test split has 6,667 texts. `python -m src.evaluate` on these saved weights gives **accuracy 0.8806, macro-F1 0.8698**, within the seed-to-seed variance reported above. Evaluating saved weights is deterministic. Retraining from scratch varies slightly because of random initialization, shuffling and non-deterministic GPU ops.

## Project structure

```
SentimentOps_Pipeline/
├── app/main.py              # FastAPI app: lifespan loading, middleware, endpoints
├── configs/config.py        # all hyper-parameters + env-overridable serving limits
├── src/
│   ├── data_preprocess.py   # clean_text (shared by training AND serving)
│   ├── features.py          # RoBERTa embedding extraction + masked mean pooling
│   ├── model.py             # Keras dual-input sentiment head
│   ├── inference.py         # InferencePipeline: batched, thread-safe, warmed up
│   ├── schemas.py           # SentimentResult dataclass
│   ├── train.py / evaluate.py
├── artifacts/               # topic_pipeline, label/topic encoders, head weights
├── dataset/                 # merged_twitter_sentiment.csv (13 domains; 44,441 rows after cleaning)
├── tests/                   # pytest: API (fake model), preprocessing, model shape
├── scripts/                 # diagnostics: topic bias / topic effect / topic predictions
├── test_api.py              # smoke test against a running server (real model)
├── Dockerfile, .dockerignore
└── .github/workflows/ci.yml
```

## Setup

With `venv`:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

Or with conda:

```bash
conda create -p ./.conda python=3.11 -y && conda activate ./.conda
pip install -r requirements-dev.txt
```

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness: the process is up (always 200) |
| GET | `/ready` | Readiness: 200 only when the model is loaded, otherwise 503 |
| GET | `/model/info` | Model version, classes, topics, device, limits |
| POST | `/predict` | `{"text": "..."}` → one prediction |
| POST | `/predict/batch` | `{"texts": ["...", "..."]}` → predictions in the same order (one batched forward pass) |

Example:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"text": "The game keeps crashing every time I open it!"}'
```

```json
{
  "label": "Negative", "confidence": 0.97,
  "probabilities": {"Negative": 0.97, "Neutral": 0.02, "Positive": 0.01},
  "predicted_topic": "Apple", "cleaned_text": "the game keeps crashing every time i open it",
  "warnings": [], "latency_ms": 41.3, "model_version": "tesa-cf-1.0.0"
}
```

**Limits and validation:**

- Whitespace is stripped. Each text must be 1–512 characters, and a batch can hold 1–32 texts. Anything else returns **422**.
- A text that becomes empty after cleaning (e.g. `"!!! 123"`) returns `Neutral` with a warning instead of crashing.
- Every response carries `X-Request-ID` (pass your own or one is generated) and `X-Process-Time-Ms` headers.

**Configuration** (environment variables): `MAX_TEXT_CHARS`, `MAX_BATCH_SIZE`, `INFER_BATCH_SIZE`, `TORCH_NUM_THREADS`, `MODEL_VERSION`, `ARTIFACT_DIR`, `LOG_LEVEL`, `SKIP_MODEL_LOAD`.

## Tests

```bash
pytest                    # 19 unit tests, ~5 s, no model download (API uses a fake pipeline)
python test_api.py        # smoke test against a running server with the real model
```

## Docker

```bash
docker build -t sentimentops .
docker run -p 8000:8000 sentimentops
```

The image is built for serving:

- CPU-only PyTorch, so the image stays small.
- RoBERTa weights are baked into the image, so containers don't download at startup.
- It runs as a non-root user.
- A `HEALTHCHECK` probes `/ready`.
- It runs one uvicorn worker. The model is about 500 MB in RAM, so scale out with more containers rather than more workers.

## CI (GitHub Actions)

On every push and pull request:

1. Install the light dependencies, run the API and preprocessing tests with `SKIP_MODEL_LOAD=1`.
2. Build the Docker image.

## Serving design & optimizations

| Before (notebook-style) | After |
|---|---|
| Batch endpoint looped over texts one by one | One tokenizer + RoBERTa + head call per batch, with dynamic padding: **≈2.4× faster** on 5 texts (172 ms → 73 ms, CPU) |
| `model.predict()` per request (slow, builds a tf.data pipeline) | `model(x, training=False)`: direct call, BatchNorm in inference mode |
| `async def` endpoints running blocking inference | Plain `def` endpoints, so FastAPI runs them in a threadpool and `/health` never freezes |
| Model loaded at import time | Loaded once in the lifespan hook, with a warm-up call before serving traffic |
| No input limits | Pydantic `StringConstraints` + batch size limit (protects memory and latency) |
| Single `/health` | `/health` (liveness) vs `/ready` (readiness) for load balancers and orchestrators |
| Shared tokenizer across threads | Lock around inference (HF fast tokenizer is not thread-safe) |
| TensorFlow grabbing the GPU next to PyTorch | Keras head pinned to CPU (it is tiny); RoBERTa uses CUDA if available |

**Correctness:**

- Single-text and batch predictions are **bit-identical** (max probability difference 0.0).
- `src.evaluate` checks that `InferencePipeline` agrees with the offline batch path.
- Training and serving use the **same** `clean_text`, which avoids train/serve skew.

**Known limitation:** the topic embedding carries a domain prior. `python -m scripts.check_topic_effect` shows a negative sentence can flip to Positive under topics whose training data is mostly positive (e.g. PS5). This is what the paper's counterfactual analysis studies.

## Deploying on AWS (options)

| Option | When |
|---|---|
| **SageMaker real-time endpoint** (custom container from ECR) | Steady traffic, low latency, managed autoscaling and monitoring |
| **SageMaker Serverless Inference** | Spiky or low traffic. Pays per request, but has cold starts |
| **SageMaker Batch Transform** | Offline scoring of large datasets |
| **ECR + ECS Fargate + ALB** | Run this exact container. The ALB health check targets `/ready`; autoscale on CPU or request count |
| **Lambda (container image)** | Possible (≤10 GB image), but cold starts with a 500 MB model hurt latency |

Logs go to **CloudWatch**. In production, also track latency p50/p95, error rate, and the predicted-label distribution, which helps detect drift.

## Train and evaluate

```bash
python -m src.train       # trains topic classifier + sentiment head, writes artifacts/
python -m src.evaluate    # test-split metrics + per-topic metrics + serving parity check
```

`artifacts/embeddings_cache.npz` (RoBERTa embeddings, ~127 MB) is not committed. It is rebuilt automatically by `src.train`.
