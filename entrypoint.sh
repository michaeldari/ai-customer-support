#!/bin/bash
set -e

echo "--- Step 1: Training ML Models ---"
python src/ml/train.py

echo "--- Step 2: Run batch predictions on the test file ---"
python src/ml/predict.py

echo "--- Step 3: Ingesting Knowledge Base ---"
python src/rag/ingest.py

echo "--- Step 4: Generate RAG Evaluation Report ---"
python src/rag/evaluate.py

echo "--- Step 5: Starting FastAPI ---"
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000