"""
Configuración centralizada del Chatbot LSE-FIUBA.
Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración del sistema cargada desde .env"""

    # ── Rutas ──────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "raw"
    PROCESSED_DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "processed"
    INDEX_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "indexes"
    GRAPH_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "graphs"
    EVALUATION_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "evaluation"

    # ── LLM ────────────────────────────────────────────────
    LLM_BACKEND: str = "ollama"
    LLM_MODEL: str = "llama3"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024

    # ── Embeddings ─────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DEVICE: str = "cpu"

    # ── Chunking ───────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 128

    # ── Retrieval ──────────────────────────────────────────
    RAG_TOP_K: int = 5
    USE_MMR: bool = True
    USE_RERANKER: bool = True
    CONFIDENCE_THRESHOLD: float = 0.5
    ABSTENTION_THRESHOLD: float = 0.3

    # ── API ────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ── UI ─────────────────────────────────────────────────
    STREAMLIT_PORT: int = 8501

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
