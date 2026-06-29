"""
Application configuration using environment variables.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama (local, no API key needed)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    # Model settings
    llm_model: str = "llama3"
    embedding_model: str = "all-MiniLM-L6-v2"

    # RAG settings
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.3

    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = base_dir / "data" / "documents"
    vector_store_dir: Path = base_dir / "data" / "vector_store"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
