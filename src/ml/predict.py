import pandas as pd
import joblib
import logging
import os
import sys


def run_predictions():
    cat_model_path = "artifacts/category_model.joblib"
    pri_model_path = "artifacts/priority_model.joblib"

    if not os.path.exists(cat_model_path) or not os.path.exists(pri_model_path):
        logging.error("Model artifacts not found in artifacts/ folder!")
        sys.exit(1)

    logging.info("Loading pipelines (Vectorizer + Model)...")
    cat_pipeline = joblib.load(cat_model_path)
    pri_pipeline = joblib.load(pri_model_path)

    logging.info("Reading tickets_test.csv...")
    test_df = pd.read_csv("data/tickets_test.csv")

    X_test = test_df["subject"].fillna("") + " " + test_df["body"].fillna("")

    logging.info("Generating predictions...")
    test_df["predicted_category"] = cat_pipeline.predict(X_test)
    test_df["predicted_priority"] = pri_pipeline.predict(X_test)

    output_path = "artifacts/predictions.csv"
    test_df.to_csv(output_path, index=False)
    logging.info(f"Success! Batch predictions saved to {output_path}")


if __name__ == "__main__":
    run_predictions()
