"""
Endpoints de chat para la API.
Incluye memoria conversacional, query expansion y feedback.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import time
import logging
import uuid

from fastapi import APIRouter, Depends

from src.api.schemas import (
    ChatRequest, ChatResponse, SourceCitation,
    ComparisonRequest, ComparisonResponse,
    FeedbackRequest, FeedbackResponse, FeedbackStatsResponse,
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
    """Endpoint principal de chat con memoria conversacional y query expansion."""
    start = time.time()

    session_id = request.session_id or str(uuid.uuid4())
    mode = _mode_to_retrieval(request.mode)
    query = request.question

    # 1. Contextualizar query con memoria conversacional
    if deps.conversation_memory and request.session_id:
        query = deps.conversation_memory.contextualize_query(session_id, query)

    # 2. Query expansion (solo para RAG y Hybrid)
    expanded_query = query
    if deps.query_expander and mode != RetrievalMode.GRAPH_ONLY:
        expansions = deps.query_expander.expand(query)
        if len(expansions) > 1:
            logger.info(f"Query expandida: {expansions}")

    # 3. Retrieve
    hybrid_result = deps.hybrid_retriever.retrieve(
        query=query,
        mode=mode,
        top_k=5,
        program_filter=request.program_filter,
    )

    # 4. Obtener historial conversacional
    chat_history = None
    if deps.conversation_memory and request.session_id:
        chat_history = deps.conversation_memory.get_chat_history(session_id)

    # 5. Synthesize
    final = deps.answer_synthesizer.synthesize(
        query=request.question,
        hybrid_result=hybrid_result,
        chat_history=chat_history,
    )

    elapsed_ms = (time.time() - start) * 1000

    # 6. Registrar en memoria conversacional
    if deps.conversation_memory:
        deps.conversation_memory.add_turn(session_id, "user", request.question)
        deps.conversation_memory.add_turn(
            session_id, "assistant", final.answer,
            metadata={"confidence": final.confidence, "method": final.method},
        )

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


# ── Feedback Endpoints ─────────────────────────────────────────


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    deps: AppDependencies = Depends(get_dependencies),
) -> FeedbackResponse:
    """Registra feedback del usuario sobre una respuesta."""
    entry = deps.feedback_collector.submit_feedback(
        session_id=request.session_id,
        question=request.question,
        answer=request.answer,
        rating=request.rating,
        method=request.method,
        confidence=request.confidence,
        is_correct=request.is_correct,
        is_complete=request.is_complete,
        user_comment=request.user_comment,
        expected_answer=request.expected_answer,
    )
    return FeedbackResponse(feedback_id=entry.feedback_id)


@router.get("/feedback/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(
    deps: AppDependencies = Depends(get_dependencies),
) -> FeedbackStatsResponse:
    """Obtiene estadísticas agregadas del feedback."""
    stats = deps.feedback_collector.get_stats()
    return FeedbackStatsResponse(
        total_entries=stats.total_entries,
        avg_rating=stats.avg_rating,
        correct_rate=stats.correct_rate,
        complete_rate=stats.complete_rate,
        by_method=stats.by_method,
        by_rating=stats.by_rating,
        low_rated_questions=stats.low_rated_questions,
        improvement_suggestions=stats.improvement_suggestions,
    )
