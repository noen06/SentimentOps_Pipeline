# ---- TESA-CF inference image (CPU) ------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# 1) Dependencies first -> cached layer unless requirements change.
#    CPU-only torch wheel keeps the image ~1.5GB smaller than the CUDA build.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

# 2) Bake RoBERTa weights into the image so cold start never hits the internet.
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
m='cardiffnlp/twitter-roberta-base-sentiment-latest'; \
AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"

# 3) App code + trained artifacts (head weights, topic pipeline, encoders).
COPY configs ./configs
COPY src ./src
COPY app ./app
COPY artifacts ./artifacts

# 4) Run as non-root.
RUN useradd --create-home appuser && chown -R appuser /app /opt/hf
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready').status==200 else 1)"

# 1 worker per container: the model is ~500MB in RAM; scale out with more containers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
