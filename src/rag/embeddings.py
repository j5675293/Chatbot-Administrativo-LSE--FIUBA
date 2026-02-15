"""
Modelo de embeddings multilingüe para búsqueda semántica.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper para sentence-transformers multilingüe.

    Modelo por defecto: paraphrase-multilingual-MiniLM-L12-v2
    - 384 dimensiones
    - Soporta 50+ idiomas incluyendo español
    - ~120MB de tamaño
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.embedding_dim = 384
        self._model = None

    def _load_model(self):
        """Carga lazy del modelo."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Cargando modelo de embeddings: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Modelo cargado. Dimensiones: {self.embedding_dim}")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Codifica una lista de textos en vectores densos."""
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # Para cosine similarity con inner product
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Codifica una query individual."""
        self._load_model()
        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32)[0]
