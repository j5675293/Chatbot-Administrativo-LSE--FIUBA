"""
Vector store basado en FAISS con búsqueda por similitud y MMR.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import pickle
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    document_name: str = ""
    page_numbers: list[int] = field(default_factory=list)
    section_title: str = ""


class FAISSVectorStore:
    """Vector store FAISS con IndexFlatIP para cosine similarity."""

    def __init__(self, embedding_dim: int = 384, index_path: Optional[Path] = None):
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks_metadata: list[dict] = []

        if index_path and Path(index_path).exists():
            self.load(index_path)

    def build_index(self, chunks: list, embeddings: np.ndarray) -> None:
        """Construye el índice FAISS desde chunks y embeddings."""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
            )

        # Normalizar embeddings para cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        # Crear índice
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

        # Almacenar metadata de chunks
        self.chunks_metadata = []
        for chunk in chunks:
            meta = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "document_name": chunk.document_name,
                "document_type": chunk.document_type,
                "page_numbers": chunk.page_numbers,
                "section_title": chunk.section_title,
                "metadata": chunk.metadata,
            }
            self.chunks_metadata.append(meta)

        logger.info(f"Índice FAISS construido: {self.index.ntotal} vectores")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[SearchResult]:
        """Búsqueda por similitud coseno."""
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < score_threshold:
                continue
            meta = self.chunks_metadata[idx]
            results.append(SearchResult(
                chunk_id=meta["chunk_id"],
                text=meta["text"],
                score=float(score),
                metadata=meta.get("metadata", {}),
                document_name=meta["document_name"],
                page_numbers=meta.get("page_numbers", []),
                section_title=meta.get("section_title", ""),
            ))

        return results

    def search_mmr(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[SearchResult]:
        """Maximal Marginal Relevance para diversidad en resultados."""
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        k = min(fetch_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        # Obtener embeddings de candidatos
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            # Reconstruir vector del índice
            vec = np.zeros(self.embedding_dim, dtype=np.float32)
            self.index.reconstruct(int(idx), vec)
            candidates.append({
                "idx": int(idx),
                "score": float(score),
                "embedding": vec,
            })

        if not candidates:
            return []

        # MMR selection
        selected = []
        remaining = list(range(len(candidates)))
        query_flat = query_vec.flatten()

        while len(selected) < min(top_k, len(candidates)) and remaining:
            best_score = -float("inf")
            best_idx = -1

            for i in remaining:
                cand = candidates[i]
                relevance = float(np.dot(query_flat, cand["embedding"]))

                # Máxima similitud con ya seleccionados
                max_sim = 0.0
                for j in selected:
                    sim = float(np.dot(cand["embedding"], candidates[j]["embedding"]))
                    max_sim = max(max_sim, sim)

                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

        # Construir resultados
        results = []
        for sel_idx in selected:
            cand = candidates[sel_idx]
            meta = self.chunks_metadata[cand["idx"]]
            results.append(SearchResult(
                chunk_id=meta["chunk_id"],
                text=meta["text"],
                score=cand["score"],
                metadata=meta.get("metadata", {}),
                document_name=meta["document_name"],
                page_numbers=meta.get("page_numbers", []),
                section_title=meta.get("section_title", ""),
            ))

        return results

    def search_with_filter(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Búsqueda con filtro por metadata (post-filtering)."""
        # Buscar más candidatos para compensar filtrado
        fetch_k = top_k * 5
        results = self.search(query_embedding, top_k=fetch_k, score_threshold=0.1)

        if not filter_metadata:
            return results[:top_k]

        filtered = []
        for result in results:
            match = True
            for key, value in filter_metadata.items():
                meta_value = result.metadata.get(key)
                if meta_value is None:
                    match = False
                    break
                if isinstance(value, list):
                    if isinstance(meta_value, list):
                        if not any(v in meta_value for v in value):
                            match = False
                    elif meta_value not in value:
                        match = False
                elif meta_value != value:
                    match = False

            if match:
                filtered.append(result)
                if len(filtered) >= top_k:
                    break

        return filtered

    def add_chunks(self, new_chunks: list, new_embeddings: np.ndarray) -> None:
        """Agrega chunks incrementalmente al índice."""
        faiss.normalize_L2(new_embeddings)

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        self.index.add(new_embeddings)

        for chunk in new_chunks:
            self.chunks_metadata.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "document_name": chunk.document_name,
                "document_type": chunk.document_type,
                "page_numbers": chunk.page_numbers,
                "section_title": chunk.section_title,
                "metadata": chunk.metadata,
            })

        logger.info(f"Agregados {len(new_chunks)} chunks. Total: {self.index.ntotal}")

    def save(self, path: Path) -> None:
        """Guarda índice FAISS y metadata a disco."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(path / "faiss_index.bin"))

        with open(path / "chunks_metadata.pkl", "wb") as f:
            pickle.dump(self.chunks_metadata, f)

        logger.info(f"Índice guardado en {path}")

    def load(self, path: Path) -> None:
        """Carga índice y metadata desde disco."""
        path = Path(path)

        index_file = path / "faiss_index.bin"
        meta_file = path / "chunks_metadata.pkl"

        if index_file.exists():
            self.index = faiss.read_index(str(index_file))
            logger.info(f"Índice FAISS cargado: {self.index.ntotal} vectores")

        if meta_file.exists():
            with open(meta_file, "rb") as f:
                self.chunks_metadata = pickle.load(f)
            logger.info(f"Metadata cargada: {len(self.chunks_metadata)} chunks")
