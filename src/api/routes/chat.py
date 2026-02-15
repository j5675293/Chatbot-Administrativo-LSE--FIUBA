"""
Endpoints de chat para la API.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import time
import logging

from fastapi import APIRouter, Depends

from src.api.schemas import (
    ChatRequest, ChatResponse, SourceCitation,
    ComparisonRequest, ComparisonResponse,
    RetrievalModeEnum,
)
from src.api.dependencies import AppDependencies, get_dependencies
from src.hybrid.hybrid_retriever import RetrievalMode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _mode_to_retrieval(mode: RetrievalModeEnum) -> RetrievalMode:
    mapping = {
        RetrievalModeEnum.rag: RetrievalMode.RAG_ONLY,
        RetrievalModeEnum.graph: RetrievalMode.GRAPH_ONLY,
        RetrievalModeEnum.hybrid: RetrievalMode.HYBRID,
    }
    return mapping[mode]


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    deps: AppDependencies = Depends(get_dependencies),
) -> ChatResponse:
    """Endpoint principal de chat."""
    start = time.time()

    mode = _mode_to_retrieval(request.mode)

    # Retrieve
    hybrid_result = deps.hybrid_retriever.retrieve(
        query=request.question,
        mode=mode,
        top_k=5,
        program_filter=request.program_filter,
    )

    # Synthesize
    final = deps.answer_synthesizer.synthesize(
        query=request.question,
        hybrid_result=hybrid_result,
    )

    elapsed_ms = (time.time() - start) * 1000

    sources = [
        SourceCitation(
            document_name=s.get("document_name", ""),
            page_numbers=s.get("page_numbers", []),
            section_title=s.get("section_title", ""),
            text_snippet=s.get("text_snippet", ""),
            score=s.get("score", 0.0),
            source_type=s.get("source_type", "rag"),
        )
        for s in final.sources
    ]

    return ChatResponse(
        answer=final.answer,
        formatted_answer=final.formatted_answer,
        sources=sources,
        confidence=final.confidence,
        method=final.method,
        warnings=final.warnings,
        fallback_contacts=final.fallback_contacts,
        processing_time_ms=elapsed_ms,
    )


@router.post("/chat/compare", response_model=ComparisonResponse)
async def compare(
    request: ComparisonRequest,
    deps: AppDependencies = Depends(get_dependencies),
) -> ComparisonResponse:
    """Compara RAG vs GraphRAG vs Hybrid para la misma pregunta."""
    results = {}

    for mode_name, mode_enum in [
        ("rag", RetrievalMode.RAG_ONLY),
        ("graph", RetrievalMode.GRAPH_ONLY),
        ("hybrid", RetrievalMode.HYBRID),
    ]:
        start = time.time()

        hybrid_result = deps.hybrid_retriever.retrieve(
            query=request.question,
            mode=mode_enum,
            top_k=5,
            program_filter=request.program_filter,
        )

        final = deps.answer_synthesizer.synthesize(
            query=request.question,
            hybrid_result=hybrid_result,
        )

        elapsed_ms = (time.time() - start) * 1000

        sources = [
            SourceCitation(
                document_name=s.get("document_name", ""),
                page_numbers=s.get("page_numbers", []),
                section_title=s.get("section_title", ""),
                text_snippet=s.get("text_snippet", ""),
                score=s.get("score", 0.0),
            )
            for s in final.sources
        ]

        results[mode_name] = ChatResponse(
            answer=final.answer,
            formatted_answer=final.formatted_answer,
            sources=sources,
            confidence=final.confidence,
            method=mode_name,
            warnings=final.warnings,
            fallback_contacts=final.fallback_contacts,
            processing_time_ms=elapsed_ms,
        )

    return ComparisonResponse(
        rag_answer=results["rag"],
        graph_answer=results["graph"],
        hybrid_answer=results["hybrid"],
    )
