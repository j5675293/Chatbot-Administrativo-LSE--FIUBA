"""
Sintetizador de respuestas del sistema híbrido.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.llm.llm_provider import LLMProvider
from src.llm.prompts import SYSTEM_PROMPT_ES, ANSWER_SYNTHESIS_PROMPT_ES, RAG_QA_PROMPT_ES
from src.hybrid.anti_hallucination import AntiHallucinationEngine
from src.hybrid.citation_manager import CitationManager, Citation
from src.hybrid.hybrid_retriever import HybridResult, RetrievalMode

logger = logging.getLogger(__name__)


@dataclass
class FinalAnswer:
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    method: str = "hybrid"
    rag_contribution: str = ""
    graph_contribution: str = ""
    warnings: list[str] = field(default_factory=list)
    fallback_contacts: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    formatted_answer: str = ""


class AnswerSynthesizer:
    """Genera respuestas finales integrando RAG + GraphRAG + anti-alucinación."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        anti_hallucination: Optional[AntiHallucinationEngine] = None,
        citation_manager: Optional[CitationManager] = None,
    ):
        self.llm = llm_provider
        self.anti_hallucination = anti_hallucination or AntiHallucinationEngine()
        self.citation_manager = citation_manager or CitationManager()

    def synthesize(
        self,
        query: str,
        hybrid_result: HybridResult,
        chat_history: Optional[list[dict]] = None,
    ) -> FinalAnswer:
        """Genera respuesta final desde resultados híbridos."""
        final = FinalAnswer(method=hybrid_result.retrieval_mode.value)

        # 1. Verificar si debe abstenerse
        max_confidence = max(
            hybrid_result.rag_confidence,
            hybrid_result.graph_confidence,
            0.01,
        )

        should_abstain, reason = self.anti_hallucination.should_abstain(
            max_confidence, query
        )

        if should_abstain and not hybrid_result.merged_context:
            contact = self.anti_hallucination.get_fallback_contact(query)
            final.answer = (
                f"{reason} Te recomiendo contactar a {contact} "
                f"para obtener información precisa."
            )
            final.confidence = 0.0
            final.fallback_contacts = [contact]
            final.formatted_answer = final.answer
            return final

        # 2. Generar respuesta con LLM
        if hybrid_result.retrieval_mode == RetrievalMode.HYBRID:
            rag_context = "\n".join(
                r.text for r in hybrid_result.rag_results
            ) if hybrid_result.rag_results else "Sin información RAG disponible."

            graph_context = "\n".join(
                r.subgraph_text for r in hybrid_result.graph_results if r.subgraph_text
            ) if hybrid_result.graph_results else "Sin información del grafo disponible."

            prompt = ANSWER_SYNTHESIS_PROMPT_ES.format(
                rag_context=rag_context[:2000],
                graph_context=graph_context[:2000],
                question=query,
            )
        else:
            prompt = RAG_QA_PROMPT_ES.format(
                context=hybrid_result.merged_context[:4000],
                question=query,
            )

        if chat_history:
            messages = list(chat_history) + [{"role": "user", "content": prompt}]
            answer_text = self.llm.generate_with_history(
                messages, system_prompt=SYSTEM_PROMPT_ES
            )
        else:
            answer_text = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT_ES)

        final.answer = answer_text

        # 3. Extraer fuentes
        sources = []
        for r in hybrid_result.rag_results:
            sources.append({
                "document_name": r.document_name,
                "page_numbers": r.page_numbers,
                "section_title": r.section_title,
                "text_snippet": r.text[:150],
                "score": r.score,
                "source_type": "rag",
            })
        for r in hybrid_result.graph_results:
            for entity in r.entities[:3]:
                sources.append({
                    "document_name": entity.get("properties", {}).get(
                        "source_document", "Grafo de conocimiento"
                    ),
                    "section_title": entity.get("name", ""),
                    "text_snippet": r.subgraph_text[:150] if r.subgraph_text else "",
                    "score": r.confidence,
                    "source_type": "graph",
                })
        final.sources = sources

        # 4. Verificar fidelidad
        faithfulness = self.anti_hallucination.check_faithfulness(
            answer_text, hybrid_result.merged_context
        )

        # 5. Cross-reference
        rag_ctx = " ".join(r.text for r in hybrid_result.rag_results)
        graph_ctx = " ".join(
            r.subgraph_text for r in hybrid_result.graph_results if r.subgraph_text
        )
        cross_ref_score = self.anti_hallucination.cross_reference_check(
            rag_ctx, graph_ctx, answer_text
        )

        # 6. Calcular confianza final
        retrieval_scores = [r.score for r in hybrid_result.rag_results]
        final.confidence = self.anti_hallucination.compute_confidence(
            retrieval_scores=retrieval_scores,
            faithfulness_score=faithfulness.score,
            source_count=len(sources),
            cross_reference_score=cross_ref_score,
        )

        # 7. Agregar warnings
        if faithfulness.unsupported_claims:
            final.warnings.append(
                "Algunas afirmaciones podrían no estar completamente "
                "respaldadas por los documentos oficiales."
            )

        if final.confidence < self.anti_hallucination.confidence_threshold:
            contact = self.anti_hallucination.get_fallback_contact(query)
            final.warnings.append(
                f"Confianza baja ({final.confidence:.0%}). "
                f"Verificar con {contact}."
            )
            final.fallback_contacts.append(contact)

        # 8. Formatear con citaciones
        citations = self.citation_manager.create_citations(sources)
        final.citations = citations
        final.formatted_answer = self.citation_manager.format_answer_with_citations(
            final.answer, citations
        )

        # 9. Contribuciones
        final.rag_contribution = f"{len(hybrid_result.rag_results)} chunks recuperados"
        final.graph_contribution = (
            f"{len(hybrid_result.graph_results)} subgrafos encontrados"
        )

        return final
