"""
Retriever híbrido que combina RAG vectorial y GraphRAG.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.rag.retriever import RAGRetriever, SearchResult
from src.graph_rag.graph_retriever import GraphRetriever, GraphSearchResult

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    RAG_ONLY = "rag_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"


@dataclass
class HybridResult:
    rag_results: list[SearchResult] = field(default_factory=list)
    graph_results: list[GraphSearchResult] = field(default_factory=list)
    merged_context: str = ""
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    rag_confidence: float = 0.0
    graph_confidence: float = 0.0


class HybridRetriever:
    """Combina RAG vectorial y GraphRAG con routing inteligente."""

    def __init__(
        self,
        rag_retriever: RAGRetriever,
        graph_retriever: GraphRetriever,
        rag_weight: float = 0.6,
        graph_weight: float = 0.4,
    ):
        self.rag_retriever = rag_retriever
        self.graph_retriever = graph_retriever
        self.rag_weight = rag_weight
        self.graph_weight = graph_weight

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 5,
        program_filter: Optional[str] = None,
    ) -> HybridResult:
        """Recupera información de ambos sistemas."""
        result = HybridResult(retrieval_mode=mode)

        # Clasificar query para ajustar pesos
        adjusted_rag_weight, adjusted_graph_weight = self._adjust_weights(query)

        if mode in (RetrievalMode.RAG_ONLY, RetrievalMode.HYBRID):
            try:
                result.rag_results = self.rag_retriever.retrieve(
                    query=query,
                    top_k=top_k,
                    use_mmr=True,
                    program_filter=program_filter,
                )
                if result.rag_results:
                    result.rag_confidence = sum(
                        r.score for r in result.rag_results
                    ) / len(result.rag_results)
            except Exception as e:
                logger.error(f"Error en RAG retrieval: {e}")

        if mode in (RetrievalMode.GRAPH_ONLY, RetrievalMode.HYBRID):
            try:
                result.graph_results = self.graph_retriever.retrieve(
                    query=query, top_k=top_k
                )
                if result.graph_results:
                    result.graph_confidence = sum(
                        r.confidence for r in result.graph_results
                    ) / len(result.graph_results)
            except Exception as e:
                logger.error(f"Error en Graph retrieval: {e}")

        # Merge contexts
        result.merged_context = self._merge_contexts(
            result.rag_results,
            result.graph_results,
            adjusted_rag_weight,
            adjusted_graph_weight,
        )

        return result

    def _adjust_weights(self, query: str) -> tuple[float, float]:
        """Ajusta pesos RAG/Graph según el tipo de query."""
        query_lower = query.lower()

        # Queries estructurales -> más Graph
        structural_keywords = [
            "requisito", "necesito para", "correlativa", "prerrequisito",
            "camino", "desde", "hasta", "pasos para",
            "antes de", "después de", "primero",
        ]
        if any(kw in query_lower for kw in structural_keywords):
            return 0.3, 0.7

        # Queries descriptivas -> más RAG
        descriptive_keywords = [
            "qué es", "cómo funciona", "explicar", "describir",
            "fundamentación", "objetivos", "perfil",
        ]
        if any(kw in query_lower for kw in descriptive_keywords):
            return 0.8, 0.2

        # Queries de path/multi-hop -> Graph only
        path_keywords = [
            "camino de", "desde .+ hasta", "cómo llego",
            "pasos desde", "trayecto",
        ]
        for kw in path_keywords:
            if re.search(kw, query_lower):
                return 0.1, 0.9

        # Default
        return self.rag_weight, self.graph_weight

    def _merge_contexts(
        self,
        rag_results: list[SearchResult],
        graph_results: list[GraphSearchResult],
        rag_weight: float,
        graph_weight: float,
    ) -> str:
        """Combina contextos de ambas fuentes."""
        parts = []

        # Contexto RAG
        if rag_results:
            rag_texts = []
            for i, r in enumerate(rag_results, 1):
                source = f"[RAG-{i}: {r.document_name}"
                if r.section_title:
                    source += f", {r.section_title}"
                source += f" (score: {r.score:.2f})]"
                rag_texts.append(f"{source}\n{r.text}")

            if rag_texts:
                parts.append(
                    "=== Información de documentos (RAG) ===\n"
                    + "\n\n".join(rag_texts)
                )

        # Contexto Graph
        if graph_results:
            graph_texts = []
            for i, r in enumerate(graph_results, 1):
                if r.subgraph_text:
                    graph_texts.append(
                        f"[Graph-{i} (confianza: {r.confidence:.2f})]\n{r.subgraph_text}"
                    )
                if r.path_description:
                    graph_texts.append(f"Camino: {r.path_description}")

            if graph_texts:
                parts.append(
                    "=== Información del grafo de conocimiento ===\n"
                    + "\n\n".join(graph_texts)
                )

        return "\n\n".join(parts)
