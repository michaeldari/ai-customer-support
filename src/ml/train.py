import pandas as pd
import joblib
import logging
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    recall_score,
)
from src.utils.config import settings
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(y_true, y_pred, name):
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(f"reports/{name}_confusion_matrix.png")
    plt.close()


def train_triage_model():
    logging.info("Loading data...")
    df = pd.read_csv(os.path.join(settings.DATA_DIR, "tickets_train.csv"))

    # Combine subject and body for better context
    X = df["subject"] + " " + df["body"]
    y_cat = df["category_label"]
    y_pri = df["priority_label"]

    logging.info("Training Category Model...")
    cat_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )
    cat_pipeline.fit(X, y_cat)

    logging.info("Training Priority Model...")
    pri_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )
    pri_pipeline.fit(X, y_pri)

    cat_pred = cat_pipeline.predict(X)
    pri_pred = pri_pipeline.predict(X)

    save_confusion_matrix(y_cat, cat_pred, "category")

    metrics = {
        "category_macro_f1": f1_score(y_cat, cat_pred, average="macro"),
        "priority_f1_weighted": f1_score(y_pri, pri_pred, average="weighted"),
        "priority_recall_weighted": recall_score(y_pri, pri_pred, average="weighted"),
    }

    logging.info(f"Saving artifacts to {settings.ARTIFACTS_DIR}...")
    joblib.dump(
        cat_pipeline, os.path.join(settings.ARTIFACTS_DIR, "category_model.joblib")
    )
    joblib.dump(
        pri_pipeline, os.path.join(settings.ARTIFACTS_DIR, "priority_model.joblib")
    )

    with open("reports/ml_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info("Training complete.")


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    train_triage_model()
