"""
Đánh giá artifact đã train trên tập test được dựng lại từ cùng seed.
Chạy: python -m src.evaluate            (kèm kiểm tra khớp InferencePipeline 200 câu)
      python -m src.evaluate --parity 0 (bỏ qua kiểm tra khớp)
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from configs.config import (
    SEED, DATA_DIR, ARTIFACT_DIR,
    TOPIC_PIPELINE_PATH, LABEL_ENCODER_PATH, KERAS_WEIGHTS_PATH, EMBEDDINGS_CACHE,
)
from src.data_preprocess import clean_dataframe
from src.model import build_sentiment_head


def rebuild_test_split():
    """Dựng lại tập test đúng như train.py: cùng cleaning, cùng seed, cùng stratify."""
    raw_df = pd.read_csv(DATA_DIR / "merged_twitter_sentiment.csv")
    df = clean_dataframe(raw_df, text_col="text", label_col="sentiment")
    print(f"[evaluate] Corpus sau khi làm sạch: {len(df)} dòng")

    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    labels = label_encoder.transform(df["sentiment"]).astype(np.int32)

    # Chỉ cần lần chia đầu tiên (tách test 15%). Truyền thêm cột topic dạng chuỗi
    # không làm đổi chỉ số được chọn vì chỉ số chỉ phụ thuộc seed, độ dài và stratify.
    _, X_test, _, y_test, _, topic_test = train_test_split(
        df["text"].tolist(), labels, df["topic"].tolist(),
        test_size=0.15, stratify=labels, random_state=SEED,
    )
    return X_test, y_test, np.array(topic_test), label_encoder


def main(parity_n: int) -> None:
    import tensorflow as tf  # import muộn để lỗi cài đặt hiện rõ ở đây

    X_test, y_test, topic_names, label_encoder = rebuild_test_split()
    print(f"[evaluate] Kích thước tập test: {len(X_test)}")

    # Embedding của test đã được cache lúc train, thứ tự khớp với X_test
    test_emb = np.load(EMBEDDINGS_CACHE)["test"]
    if len(test_emb) != len(X_test):
        raise RuntimeError(
            f"Cache có {len(test_emb)} embedding nhưng tập test dựng lại có {len(X_test)} dòng. "
            "Cleaning/split hiện tại không khớp với lần train, kết quả sẽ không đáng tin."
        )

    topic_pipeline = joblib.load(TOPIC_PIPELINE_PATH)
    topic_encoder = joblib.load(ARTIFACT_DIR / "topic_encoder.joblib")
    test_topic_ids = topic_pipeline.predict(X_test).astype(np.int32)

    model = build_sentiment_head(num_topics=len(topic_encoder.classes_))
    model.load_weights(str(KERAS_WEIGHTS_PATH))
    probs = model.predict(
        {"text_embedding": test_emb, "topic_id": test_topic_ids.reshape(-1, 1)},
        batch_size=256, verbose=0,
    )
    y_pred = np.argmax(probs, axis=1)

    # ── Toàn bộ tập test ─────────────────────────────────────────────────────
    print("\n=== Toàn bộ tập test ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro F1 : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=4))
    print("Confusion matrix (hàng = thật, cột = dự đoán):")
    print(confusion_matrix(y_test, y_pred))

    # ── Tách theo miền (dùng topic thật để chia miền) ────────────────────────
    print("\n=== Theo miền ===")
    is_hotel = topic_names == "hotel"
    for name, mask in [("Social-media", ~is_hotel), ("Hotel-review", is_hotel)]:
        acc = accuracy_score(y_test[mask], y_pred[mask])
        f1 = f1_score(y_test[mask], y_pred[mask], average="macro")
        print(f"{name:13s} n={mask.sum():5d}  acc={acc:.4f}  macro-F1={f1:.4f}")

    # ── Kiểm tra InferencePipeline (đường đi từng câu) khớp với đường batch ──
    if parity_n > 0:
        from src.inference import InferencePipeline

        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_test), size=min(parity_n, len(X_test)), replace=False)
        pipeline = InferencePipeline()
        agree = 0
        for i in idx:
            result = pipeline.predict_sentiment(X_test[i])
            agree += int(result.label == label_encoder.classes_[y_pred[i]])
        print(f"\n=== Kiểm tra khớp: InferencePipeline vs đường batch ===")
        print(f"Cùng nhãn: {agree}/{len(idx)} ({agree / len(idx):.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity", type=int, default=200)
    main(parser.parse_args().parity)