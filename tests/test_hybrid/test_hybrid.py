"""
Tests del sistema híbrido.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.hybrid.hybrid_retriever import HybridRetriever, RetrievalMode
from src.hybrid.citation_manager import CitationManager


class TestHybridRetrieverWeights:
    """Tests del ajuste de pesos del retriever híbrido."""

    def test_structural_query_weights(self):
        """Queries estructurales deberían priorizar graph."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.rag_weight = 0.6
        retriever.graph_weight = 0.4

        rag_w, graph_w = retriever._adjust_weights(
            "¿Cuáles son los requisitos para la MIA?"
        )
        assert graph_w > rag_w

    def test_descriptive_query_weights(self):
        """Queries descriptivas deberían priorizar RAG."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.rag_weight = 0.6
        retriever.graph_weight = 0.4

        rag_w, graph_w = retriever._adjust_weights(
            "¿Qué es la CEIA y cuáles son sus objetivos?"
        )
        assert rag_w > graph_w

    def test_default_weights(self):
        """Queries genéricas deberían usar pesos default."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.rag_weight = 0.6
        retriever.graph_weight = 0.4

        rag_w, graph_w = retriever._adjust_weights(
            "¿Cuántas materias tiene la carrera?"
        )
        assert rag_w == 0.6
        assert graph_w == 0.4


class TestCitationManager:
    """Tests del gestor de citaciones."""

    def setup_method(self):
        self.manager = CitationManager()

    def test_create_citations(self):
        sources = [
            {
                "document_name": "Reglamento.pdf",
                "section_title": "Art. 2",
                "page_numbers": [3],
                "score": 0.85,
            },
            {
                "document_name": "CEIA.pdf",
                "section_title": "Plan de estudios",
                "page_numbers": [5, 6],
                "score": 0.72,
            },
        ]

        citations = self.manager.create_citations(sources)
        assert len(citations) == 2
        assert citations[0].citation_id == "[1]"
        assert citations[1].citation_id == "[2]"

    def test_format_answer_with_citations(self):
        sources = [
            {
                "document_name": "Reglamento.pdf",
                "section_title": "Art. 2",
                "page_numbers": [3],
                "score": 0.85,
            },
        ]

        citations = self.manager.create_citations(sources)
        formatted = self.manager.format_answer_with_citations(
            "La asistencia mínima es 75%.", citations
        )

        assert "La asistencia mínima es 75%." in formatted
        # Debería incluir las fuentes al final
        assert "Reglamento" in formatted
