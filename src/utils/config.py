import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths
    DATA_DIR: str = "data"
    ARTIFACTS_DIR: str = "artifacts"
    CHROMA_DB_DIR: str = "chroma_db"

    # Model Params
    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Free local embeddings

    class Config:
        env_file = ".env"


settings = Settings()
os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)
