"""
Cadena RAG para respuesta a preguntas con citaciones.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.rag.retriever import RAGRetriever, SearchResult
from src.llm.llm_provider import LLMProvider
from src.llm.prompts import SYSTEM_PROMPT_ES, RAG_QA_PROMPT_ES

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    retrieval_scores: list[float] = field(default_factory=list)
    method: str = "rag"


class RAGChain:
    """Cadena RAG completa: retrieval + generación + citaciones."""

    def __init__(
        self,
        retriever: RAGRetriever,
        llm_provider: LLMProvider,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.llm = llm_provider
        self.top_k = top_k

    def answer(
        self,
        question: str,
        chat_history: Optional[list[dict]] = None,
        program_filter: Optional[str] = None,
    ) -> RAGResponse:
        """Pipeline completo: retrieve -> build context -> generate -> cite."""
        # 1. Retrieval
        results = self.retriever.retrieve(
            query=question,
            top_k=self.top_k,
            use_mmr=True,
            program_filter=program_filter,
        )

        if not results:
            return RAGResponse(
                answer=(
                    "No encontré información relevante para tu pregunta en los "
                    "documentos disponibles. Te recomiendo contactar a "
                    "gestion.academica.lse@fi.uba.ar para más información."
                ),
                confidence=0.0,
            )

        # 2. Construir contexto
        context = self._build_context(results)

        # 3. Generar respuesta
        prompt = RAG_QA_PROMPT_ES.format(context=context, question=question)

        if chat_history:
            messages = list(chat_history) + [{"role": "user", "content": prompt}]
            answer_text = self.llm.generate_with_history(
                messages, system_prompt=SYSTEM_PROMPT_ES
            )
        else:
            answer_text = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT_ES)

        # 4. Construir fuentes
        sources = self._extract_sources(results)

        # 5. Calcular confianza
        retrieval_scores = [r.score for r in results]
        confidence = self._compute_confidence(retrieval_scores, answer_text)

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            confidence=confidence,
            retrieval_scores=retrieval_scores,
            method="rag",
        )

    def _build_context(self, results: list[SearchResult]) -> str:
        """Construye string de contexto con marcadores de fuente."""
        context_parts = []
        for i, result in enumerate(results, 1):
            source_label = f"[Fuente {i}: {result.document_name}"
            if result.section_title:
                source_label += f", {result.section_title}"
            source_label += "]"

            context_parts.append(f"{source_label}\n{result.text}")

        return "\n\n---\n\n".join(context_parts)

    def _extract_sources(self, results: list[SearchResult]) -> list[dict]:
        """Extrae información de fuentes de los resultados."""
        sources = []
        for result in results:
            sources.append({
                "document_name": result.document_name,
                "page_numbers": result.page_numbers,
                "section_title": result.section_title,
                "text_snippet": result.text[:150] + "..." if len(result.text) > 150 else result.text,
                "score": result.score,
            })
        return sources

    def _compute_confidence(
        self, retrieval_scores: list[float], answer: str
    ) -> float:
        """Calcula confianza basada en scores de retrieval."""
        if not retrieval_scores:
            return 0.0

        avg_score = sum(retrieval_scores) / len(retrieval_scores)
        max_score = max(retrieval_scores)

        # Penalizar respuestas que dicen "no tengo información"
        no_info_phrases = [
            "no tengo información",
            "no encontré",
            "no puedo responder",
            "no dispongo",
        ]
        has_no_info = any(phrase in answer.lower() for phrase in no_info_phrases)
        if has_no_info:
            return min(avg_score * 0.3, 0.2)

        # Confianza: combinación de score promedio y máximo
        confidence = 0.6 * avg_score + 0.4 * max_score

        # Factor por cantidad de fuentes
        source_factor = min(len(retrieval_scores) / 3.0, 1.0)
        confidence *= (0.7 + 0.3 * source_factor)

        return min(max(confidence, 0.0), 1.0)
