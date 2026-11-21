# SentimentOps: Sentiment Model Serving Pipeline

Production-style serving pipeline for **TESA-CF** (Topic-Aware Explainable Sentiment Analysis), a multi-domain sentiment model from the paper *"Topic-Aware Multi-Domain Sentiment Analysis with Explainable Counterfactual Interventions"* (EXPLAINS 2026).

The original research notebook is refactored into a modular, config-driven Python package and served as a REST API with FastAPI.

> **Status:** in progress. Next steps: Docker image, cloud deployment, GitHub Actions CI.

## How the model works

```
text ──► clean_text ──┬──► TF-IDF + Logistic Regression ──► topic_id ──► topic embedding ─┐
                      │                                                                   ├──► dense head ──► Negative / Neutral / Positive
                      └──► Twitter-RoBERTa (frozen, mean-pooled 768-d) ──────────────────┘
```

1. A **topic classifier** (TF-IDF + Logistic Regression, scikit-learn) predicts the domain of the text.
2. **Frozen Twitter-RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) produces a 768-d sentence embedding.
3. A **Keras dual-input head** fuses the embedding with a learned topic embedding and outputs 3 sentiment classes.

## Project structure

```
SentimentOps_Pipeline/
├── app/
│   └── main.py              # FastAPI server (/health, /predict, /predict/batch)
├── artifacts/               # Trained model files loaded at inference time
├── configs/
│   └── config.py            # All paths and hyperparameters in one place
├── dataset/
│   └── merged_twitter_sentiment.csv
├── scripts/                 # Ad-hoc checks on topic bias / topic effect
├── src/
│   ├── data_preprocess.py   # Text cleaning, label encoding
│   ├── features.py          # RoBERTa embedding extraction + caching
│   ├── model.py             # Keras sentiment head
│   ├── train.py             # End-to-end training
│   ├── evaluate.py          # Test-set evaluation + inference parity check
│   └── inference.py         # InferencePipeline used by the API
├── test_api.py              # API smoke tests
└── requirements.txt
```

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
```

The RoBERTa checkpoint is downloaded from Hugging Face on first run.

## Run the API

Trained artifacts are included in `artifacts/`, so the API runs without retraining.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

### Endpoints

| Method | Path             | Body                          | Returns                                   |
|--------|------------------|-------------------------------|-------------------------------------------|
| GET    | `/health`        | –                             | Server and model status                   |
| POST   | `/predict`       | `{"text": "..."}` (1–512 chars) | Label, confidence, class probabilities, predicted topic, cleaned text, latency (ms) |
| POST   | `/predict/batch` | `["text 1", "text 2", ...]`   | List of predictions                       |

Example:

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "The game keeps crashing every time I open it!"}'
```

### Smoke tests

With the server running:

```bash
python test_api.py
```

Covers health, single predict, batch predict, and rejection of empty input (HTTP 422).

## Train and evaluate

```bash
python -m src.train       # trains topic classifier + sentiment head, writes artifacts/
python -m src.evaluate    # evaluates on the held-out test split
```

`artifacts/embeddings_cache.npz` (RoBERTa embeddings, ~127 MB) is not committed. It is rebuilt automatically by `src.train`.

Debug scripts are run from the project root, e.g. `python -m scripts.check_topics`.

## Results (paper)

On 44,396 texts across 13 domains: **87.93% accuracy, 0.868 macro-F1**, ahead of the best TF-IDF baseline (0.850) and zero-shot RoBERTa (0.663).
