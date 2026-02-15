"""
Motor anti-alucinación multi-capa.
Verificación de fidelidad, scoring de confianza y cross-referencing.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaithfulnessCheck:
    is_faithful: bool = True
    score: float = 1.0
    unsupported_claims: list[str] = field(default_factory=list)
    supported_claims: list[str] = field(default_factory=list)


class AntiHallucinationEngine:
    """Motor anti-alucinación con múltiples capas de verificación."""

    # Contactos de fallback por tema
    FALLBACK_CONTACTS = {
        "inscripcion": "inscripcion.lse@fi.uba.ar",
        "gestion_proyectos": "direccion.posgrado.lse@fi.uba.ar",
        "trabajo_final": "direccion.posgrado.lse@fi.uba.ar",
        "default": "gestion.academica.lse@fi.uba.ar",
    }

    def __init__(
        self,
        llm_provider=None,
        embedding_model=None,
        confidence_threshold: float = 0.5,
        abstention_threshold: float = 0.3,
    ):
        self.llm = llm_provider
        self.embedding_model = embedding_model
        self.confidence_threshold = confidence_threshold
        self.abstention_threshold = abstention_threshold

    def check_faithfulness(self, answer: str, context: str) -> FaithfulnessCheck:
        """Verifica que cada claim en la respuesta esté respaldado por el contexto."""
        if not answer or not context:
            return FaithfulnessCheck(is_faithful=False, score=0.0)

        # Método 1: Verificación por embeddings (sin LLM)
        if self.embedding_model:
            return self._check_faithfulness_embeddings(answer, context)

        # Método 2: Verificación por LLM
        if self.llm:
            return self._check_faithfulness_llm(answer, context)

        # Método 3: Verificación heurística (sin modelos)
        return self._check_faithfulness_heuristic(answer, context)

    def _check_faithfulness_embeddings(
        self, answer: str, context: str
    ) -> FaithfulnessCheck:
        """Verificación por similitud semántica de claims contra contexto."""
        # Descomponer respuesta en oraciones (claims)
        claims = self._split_into_claims(answer)
        context_sentences = self._split_into_claims(context)

        if not claims or not context_sentences:
            return FaithfulnessCheck(score=0.5)

        # Embeddings
        claim_embeddings = self.embedding_model.embed_texts(claims)
        context_embeddings = self.embedding_model.embed_texts(context_sentences)

        supported = []
        unsupported = []

        for i, claim in enumerate(claims):
            # Similitud máxima con cualquier oración del contexto
            similarities = np.dot(claim_embeddings[i], context_embeddings.T)
            max_sim = float(np.max(similarities))

            if max_sim > 0.65:  # Threshold de soporte
                supported.append(claim)
            else:
                unsupported.append(claim)

        total = len(claims)
        score = len(supported) / total if total > 0 else 0.0

        return FaithfulnessCheck(
            is_faithful=score >= 0.7,
            score=score,
            supported_claims=supported,
            unsupported_claims=unsupported,
        )

    def _check_faithfulness_llm(self, answer: str, context: str) -> FaithfulnessCheck:
        """Verificación de fidelidad usando LLM."""
        from src.llm.prompts import FAITHFULNESS_CHECK_PROMPT_ES

        prompt = FAITHFULNESS_CHECK_PROMPT_ES.format(
            context=context[:3000],
            answer=answer,
        )

        response = self.llm.generate(prompt)

        # Intentar parsear JSON del response
        try:
            # Buscar JSON en la respuesta
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                score = float(data.get("overall_faithfulness", 0.5))
                claims = data.get("claims", [])
                supported = [c["claim"] for c in claims if c.get("supported")]
                unsupported = [c["claim"] for c in claims if not c.get("supported")]

                return FaithfulnessCheck(
                    is_faithful=score >= 0.7,
                    score=score,
                    supported_claims=supported,
                    unsupported_claims=unsupported,
                )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        return FaithfulnessCheck(score=0.5)

    def _check_faithfulness_heuristic(
        self, answer: str, context: str
    ) -> FaithfulnessCheck:
        """Verificación heurística sin modelos."""
        answer_lower = answer.lower()
        context_lower = context.lower()

        # Buscar datos específicos en la respuesta que deberían estar en el contexto
        # (números, porcentajes, plazos, nombres de programas)
        data_patterns = [
            r"\d+\s*(?:bimestres?|meses?|años?|%|por\s*ciento)",
            r"\b(?:CEIA|CESE|CEIoT|MIA|MIAE|MIoT|MCB)\b",
            r"\b(?:Art\.\s*\d+)\b",
        ]

        total_checks = 0
        passed_checks = 0

        for pattern in data_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            for match in matches:
                total_checks += 1
                if match.lower() in context_lower:
                    passed_checks += 1

        score = passed_checks / total_checks if total_checks > 0 else 0.7

        return FaithfulnessCheck(
            is_faithful=score >= 0.6,
            score=score,
        )

    def compute_confidence(
        self,
        retrieval_scores: list[float],
        faithfulness_score: float,
        source_count: int,
        cross_reference_score: float = 0.5,
    ) -> float:
        """Calcula confianza agregada de múltiples señales."""
        if not retrieval_scores:
            return 0.0

        avg_retrieval = sum(retrieval_scores) / len(retrieval_scores)
        source_factor = min(source_count / 3.0, 1.0)

        confidence = (
            0.30 * avg_retrieval
            + 0.30 * faithfulness_score
            + 0.15 * source_factor
            + 0.25 * cross_reference_score
        )

        return min(max(confidence, 0.0), 1.0)

    def cross_reference_check(
        self, rag_context: str, graph_context: str, answer: str
    ) -> float:
        """Verifica consistencia entre RAG y GraphRAG."""
        if not rag_context or not graph_context:
            return 0.5  # Sin información para comparar

        if self.embedding_model:
            rag_emb = self.embedding_model.embed_query(rag_context[:1000])
            graph_emb = self.embedding_model.embed_query(graph_context[:1000])
            similarity = float(np.dot(rag_emb, graph_emb))
            return max(0.0, min(similarity, 1.0))

        # Heurística: overlap de palabras clave
        rag_words = set(rag_context.lower().split())
        graph_words = set(graph_context.lower().split())

        if not rag_words or not graph_words:
            return 0.5

        overlap = len(rag_words & graph_words)
        total = len(rag_words | graph_words)

        return overlap / total if total > 0 else 0.5

    def should_abstain(self, confidence: float, query: str) -> tuple[bool, str]:
        """Decide si el sistema debe abstenerse de responder."""
        query_lower = query.lower()

        # Detectar preguntas fuera del dominio
        out_of_scope_indicators = [
            "precio", "costo", "cuánto sale", "cuánto cuesta",
            "opinión", "opinás", "pensás",
            "mejor", "peor", "recomendás",
            "otro universidad", "otra facultad",
        ]

        for indicator in out_of_scope_indicators:
            if indicator in query_lower:
                return True, (
                    "Esta pregunta está fuera del alcance de la información "
                    "disponible en los documentos del LSE."
                )

        if confidence < self.abstention_threshold:
            return True, (
                "No tengo suficiente información para responder con certeza."
            )

        return False, ""

    def get_fallback_contact(self, query: str) -> str:
        """Determina el email de contacto relevante para la query."""
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["inscripci", "inscribi", "matricul"]):
            return self.FALLBACK_CONTACTS["inscripcion"]
        if any(kw in query_lower for kw in ["proyecto", "gdp", "gti"]):
            return self.FALLBACK_CONTACTS["gestion_proyectos"]
        if any(kw in query_lower for kw in ["trabajo final", "tesis", "ttf", "defensa"]):
            return self.FALLBACK_CONTACTS["trabajo_final"]

        return self.FALLBACK_CONTACTS["default"]

    def _split_into_claims(self, text: str) -> list[str]:
        """Divide texto en oraciones/claims."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
