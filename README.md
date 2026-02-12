# AcmeDesk Support Copilot

An AI-powered support assistant that triages incoming tickets using Scikit-Learn and generates grounded responses using RAG (ChromaDB + GPT-3.5).

## 🚀 Quick Start (Docker)

1. **Configure Environment**: Create a `.env` file in the root:
   ```env
   OPENAI_API_KEY=your_sk_key_here

2. **Run Everything**: 
docker-compose up --build

## 🛠 Commands

We use a `Makefile` to simplify common tasks:

- **Build & Run API**: `make api`
- **Run Triage Training**: `make train`
- **Run Knowledge Ingestion**: `make ingest`
- **Run Adversarial Tests**: `make test`
- **Run RAG Offline Eval**: `make eval`

3. **Explore the API:**: 
Visit http://localhost:8000/docs to test the endpoints.

3. **Project Structure:**: 
Visit http://localhost:8000/docs to test the endpoints.

src/ml: Triage classification logic (TF-IDF + Logistic Regression).

src/rag: Vector search and LLM generation logic.

reports/: Contains ml_metrics.json and rag_eval.md.

artifacts/: Saved .joblib model files.