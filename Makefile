.PHONY: setup train ingest api test clean

# Default command: setup and run everything
all: setup train ingest api

# Install local dependencies (for local dev)
setup:
	pip install -r requirements.txt

# Run the ML training script
train:
	export PYTHONPATH=$${PYTHONPATH}:. && python3 src/ml/train.py

# Ingest KB docs into ChromaDB
ingest:
	export PYTHONPATH=$${PYTHONPATH}:. && python3 src/rag/ingest.py

# Launch the API via Docker
api:
	docker-compose up --build

# Run the adversarial and unit tests
test:
	docker-compose exec copilot-app python -m pytest tests/test_adversarial.py -v

# Run the RAG evaluation
eval:
	docker-compose exec copilot-app python src/rag/evaluate.py
