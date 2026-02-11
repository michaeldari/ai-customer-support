Based on your current system logic and the feedback provided, here is the refined `ARCHITECTURE.md`. This version is structured to demonstrate professional-grade "System Design" thinking, focusing on the specific pipelines, tradeoffs, and production roadmap you have built.

---

# System Architecture: AcmeDesk Support Copilot

## 1. System Overview

The AcmeDesk Support Copilot utilizes a **"Triage-First RAG"** architecture. Unlike a standard chatbot, this system first classifies the intent and priority of a ticket to dynamically adjust the retrieval strategy and the LLM’s persona.

### High-Level Component Diagram

---

## 2. Technical Pipelines

### A. Ingestion Flow (Offline)

1. **Markdown Processing**: Support documents are parsed and split into semantic chunks.
2. **Vectorization**: Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2`.
3. **Storage**: Vectors are persisted in **ChromaDB** with metadata (category, source, chunk_id).

### B. Inference Flow (Real-time)

1. **FastAPI Entry**: The user query and metadata enter the system.
2. **ML Triage**: A Scikit-Learn pipeline (TF-IDF + Logistic Regression) predicts `category` and `priority`.
3. **Hybrid Retrieval**:
* **Tier 1**: Attempts a filtered search within the predicted category.
* **Tier 2 (Fallback)**: If Tier 1 returns no results, it executes a global search across all categories.


4. **Prompt Construction**: System instructions, Triage metadata, and retrieved Context are combined into a structured prompt.
5. **LLM Generation**: OpenAI (GPT-3.5) generates a JSON response containing the `draft_reply` and `internal_next_steps`.

---

## 3. Design Tradeoffs

| Feature | Choice | Tradeoff |
| --- | --- | --- |
| **Triage Model** | TF-IDF + LogReg | **Pro:** Ultra-fast (<10ms), low memory, highly interpretable. **Con:** Struggles with deep semantic nuances compared to BERT. |
| **Vector DB** | ChromaDB | **Pro:** Ease of local persistence, zero-latency network overhead for this prototype. **Con:** Harder to scale horizontally compared to Pinecone. |
| **LLM** | GPT-3.5-Turbo | **Pro:** High cost-efficiency and speed. **Con:** Smaller context window and lower reasoning capability than GPT-4. |

---

## 4. Productionization & Scaling Plan

### 🚀 Scaling & Performance

* **Retrieval Optimization**: To improve the 0.76 Groundedness score, I would implement **Cross-Encoder Re-ranking** (e.g., using Cohere Rerank). This ensures the final context provided to the LLM is the most semantically relevant.
* **Semantic Caching**: Using Redis to store previous query/response pairs. If a new query has >95% similarity to a cached one, we serve the result in <50ms, bypassing the LLM and saving costs.

### 🛡️ Security & Reliability

* **PII Scrubbing**: Implement a layer (e.g., **Microsoft Presidio**) before the LLM call to ensure customer emails, phone numbers, or passwords never leave the internal infrastructure.
* **Human-in-the-Loop (HITL)**: If Triage confidence is `< 0.60`, the system automatically flags the response in the `internal_next_steps` for a human agent to review before sending.

### 📊 Observability

* **Traceability**: Integrate **LangSmith** or **Arize Phoenix** to monitor "Traceability"—allowing us to see exactly which retrieved chunk led to a specific response in real-time.
* **Drift Detection**: Monitor the triage model for "Concept Drift" to ensure classification accuracy remains high as new support topics emerge.

---

## 5. Data Lifecycle

1. **Ingestion**: Files are chunked and embedded using `all-MiniLM-L6-v2`.
2. **Storage**: Vectors and metadata are persisted in `ChromaDB`.
3. **Retrieval**: At runtime, the query is embedded and matched using **Cosine Similarity**.
