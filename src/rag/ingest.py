import os
import chromadb
import logging
from chromadb.utils import embedding_functions
from src.utils.config import settings
from chromadb.config import Settings


def ingest_docs():
    client = chromadb.PersistentClient(
        path=settings.CHROMA_DB_DIR, settings=Settings(anonymized_telemetry=False)
    )

    # Use local embeddings to save cost and increase speed
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(name="kb_docs", embedding_function=ef)

    docs_dir = os.path.join(settings.DATA_DIR, "kb_docs")
    documents = []
    metadatas = []
    ids = []

    if not os.path.exists(docs_dir):
        logging.info(f"No kb_docs found at {docs_dir}. Skipping ingestion.")
        return

    logging.info(f"Ingesting documents from {docs_dir}...")

    for filename in os.listdir(docs_dir):
        if filename.endswith(".md"):
            category = filename.split("_")[0].capitalize()
            with open(os.path.join(docs_dir, filename), "r") as f:
                text = f.read()
                chunks = text.split("\n\n")
                for idx, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 10:
                        documents.append(chunk)
                        metadatas.append({"source": filename, "chunk_id": idx, "category": category})
                        ids.append(f"{filename}_{idx}")

    if documents:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        logging.info(f"Ingested {len(documents)} chunks.")


if __name__ == "__main__":
    ingest_docs()
