"""
Retriever RAG con re-ranking por cross-encoder.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import logging
from typing import Optional

from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import FAISSVectorStore, SearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-ranking con cross-encoder para mejorar precisión."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Cargando cross-encoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name)

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int = 5
    ) -> list[SearchResult]:
        """Re-ordena resultados por relevancia con cross-encoder."""
        if not results:
            return []

        self._load_model()

        # Crear pares (query, text) para scoring
        pairs = [(query, r.text) for r in results]
        scores = self._model.predict(pairs)

        # Asignar scores y re-ordenar
        for result, score in zip(results, scores):
            result.score = float(score)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


class RAGRetriever:
    """Pipeline de retrieval con embedding, búsqueda y re-ranking."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: FAISSVectorStore,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_mmr: bool = True,
        program_filter: Optional[str] = None,
        rerank: bool = True,
    ) -> list[SearchResult]:
        """Pipeline completo de retrieval."""
        # 1. Embed query
        query_embedding = self.embedding_model.embed_query(query)

        # 2. Búsqueda FAISS
        fetch_k = top_k * 4 if rerank else top_k
        if program_filter:
            filter_meta = {"program_codes": [program_filter]}
            results = self.vector_store.search_with_filter(
                query_embedding, top_k=fetch_k, filter_metadata=filter_meta
            )
        elif use_mmr:
            results = self.vector_store.search_mmr(
                query_embedding, top_k=fetch_k, fetch_k=fetch_k * 2
            )
        else:
            results = self.vector_store.search(
                query_embedding, top_k=fetch_k
            )

        if not results:
            logger.warning(f"Sin resultados para: {query[:80]}")
            return []

        # 3. Re-ranking
        if rerank and self.reranker and len(results) > top_k:
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        logger.info(
            f"Retrieval: {len(results)} resultados, "
            f"scores: {[f'{r.score:.3f}' for r in results]}"
        )
        return results
