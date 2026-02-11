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


### 2. ARCHITECTURE.md
This demonstrates your system design thinking.



```markdown
# System Architecture

## Overview
The system follows a "Triage-then-Generate" pattern. By classifying the ticket first, we can dynamically adjust the RAG behavior (e.g., prioritizing certain docs or changing the LLM's persona for P0 issues).

### Component Breakdown
1. **Triage Engine**: 
   - **Model**: Dual Logistic Regression pipelines with TF-IDF vectorization. 
   - **Rationale**: Fast inference (<10ms), low memory footprint, and high interpretability.
2. **RAG Engine**:
   - **Vector Store**: ChromaDB (persistent local storage).
   - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` for high-quality local vectorization.
   - **LLM**: GPT-3.5-Turbo with JSON-mode output for structured responses.

## Productionization Plan
- **Scaling**: Move from local ChromaDB to a managed service like Pinecone or Weaviate.
- **Monitoring**: Implement Prometheus metrics for `triage_confidence` and `llm_latency`.
- **Security**: Add a PII-scrubbing layer (e.g., Microsoft Presidio) before sending data to the OpenAI API.
- **Cost Control**: Implement semantic caching; if a similar question was asked recently, return the cached RAG response.

## Evaluation Notes
- **ML Triage**: Achieved 1.0 F1 on categories due to high keyword distinctness in the synthetic set.
- **RAG Evaluation**: The 0.76 Groundedness score in `reports/rag_eval_raw.md` reflects the model's behavior on "Out-of-Distribution" questions (e.g., questions about Dark Mode which are not in the provided KB).
- **Adversarial**: See `tests/test_adversarial.py` for successful mitigation of prompt injection and out-of-scope refusals.

## 🛡️ Reliability & Guardrails

To ensure production-grade safety, the following logic is implemented in `engine.py`:

* **Human-in-the-Loop (HITL)**: If the Triage model's confidence is < 60%, the response is flagged for human review, and a notice is prepended to the internal steps.
* **Mandatory Grounding**: The LLM is constrained to answer ONLY using the provided context. If the answer is missing, it triggers a mandatory refusal phrase: *"I don't have enough info to answer this."*
* **Adversarial Defense**: 
    * **Prompt Injection**: Specific rules block "ignore instructions" or "system prompt leakage" attempts.
    * **Out-of-Scope Blocking**: Requests for jokes, code, or recipes are treated as "missing info" to prevent the bot from being misused.