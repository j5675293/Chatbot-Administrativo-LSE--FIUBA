"""
Tests del sistema RAG.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.vector_store import FAISSVectorStore
from src.data_pipeline.chunker import Chunk


def _make_chunks(texts, doc_name="test.pdf"):
    """Helper para crear chunks de prueba."""
    return [
        Chunk(
            chunk_id=f"chunk_{i}",
            text=t,
            document_name=doc_name,
            document_type="resolucion",
        )
        for i, t in enumerate(texts)
    ]


class TestFAISSVectorStore:
    """Tests del vector store FAISS."""

    def setup_method(self):
        self.store = FAISSVectorStore(embedding_dim=4)

    def test_build_index_and_count(self):
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)

        chunks = _make_chunks(["texto uno", "texto dos", "texto tres"])
        self.store.build_index(chunks, embeddings)
        assert self.store.index.ntotal == 3

    def test_search_returns_results(self):
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)

        chunks = _make_chunks(["gato", "perro similar", "auto"])
        self.store.build_index(chunks, embeddings)

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = self.store.search(query, top_k=2)

        assert len(results) == 2
        # El primer resultado debería ser el más similar
        assert results[0].text == "gato"

    def test_empty_store_search(self):
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = self.store.search(query, top_k=5)
        assert len(results) == 0


class TestEmbeddingModel:
    """Tests del modelo de embeddings (requiere descarga del modelo)."""

    @pytest.mark.slow
    def test_embed_text(self):
        from src.rag.embeddings import EmbeddingModel

        model = EmbeddingModel(device="cpu")
        embedding = model.embed_query("Hola mundo")
        assert embedding.shape == (384,)

    @pytest.mark.slow
    def test_embed_batch(self):
        from src.rag.embeddings import EmbeddingModel

        model = EmbeddingModel(device="cpu")
        embeddings = model.embed_texts(["Hola", "Mundo"])
        assert embeddings.shape == (2, 384)

    @pytest.mark.slow
    def test_similarity(self):
        from src.rag.embeddings import EmbeddingModel

        model = EmbeddingModel(device="cpu")
        e1 = model.embed_query("requisitos de inscripción")
        e2 = model.embed_query("requerimientos para inscribirse")
        e3 = model.embed_query("clima en Buenos Aires")

        sim_related = float(np.dot(e1, e2))
        sim_unrelated = float(np.dot(e1, e3))

        # Los textos relacionados deberían tener mayor similitud
        assert sim_related > sim_unrelated
