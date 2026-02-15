"""
Tests de la API FastAPI.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


class TestSchemas:
    """Tests de los schemas de la API."""

    def test_chat_request_valid(self):
        from src.api.schemas import ChatRequest

        req = ChatRequest(question="¿Cuál es la asistencia mínima?")
        assert req.question == "¿Cuál es la asistencia mínima?"
        assert req.mode.value == "hybrid"

    def test_chat_request_with_mode(self):
        from src.api.schemas import ChatRequest, RetrievalModeEnum

        req = ChatRequest(
            question="test",
            mode=RetrievalModeEnum.rag,
        )
        assert req.mode == RetrievalModeEnum.rag

    def test_chat_request_with_filter(self):
        from src.api.schemas import ChatRequest

        req = ChatRequest(
            question="test",
            program_filter="CEIA",
        )
        assert req.program_filter == "CEIA"

    def test_chat_response_model(self):
        from src.api.schemas import ChatResponse, SourceCitation

        resp = ChatResponse(
            answer="La asistencia mínima es 75%",
            confidence=0.85,
            method="hybrid",
            sources=[
                SourceCitation(
                    document_name="Reglamento.pdf",
                    score=0.9,
                )
            ],
        )
        assert resp.confidence == 0.85
        assert len(resp.sources) == 1


class TestHealthEndpoint:
    """Tests del endpoint de salud (sin dependencias pesadas)."""

    def test_health_schema(self):
        from src.api.schemas import HealthResponse

        health = HealthResponse(
            status="ok",
            llm_available=True,
            documents_loaded=13,
            index_size=500,
            graph_nodes=45,
        )
        assert health.status == "ok"
        assert health.documents_loaded == 13
