"""
Métricas RAGAS (Retrieval-Augmented Generation Assessment) para evaluación.
Implementa faithfulness, answer relevance y context precision sin dependencias externas.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import logging
import re
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RAGASResult:
    """Resultado de evaluación RAGAS para una pregunta."""
    question: str = ""
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    overall_score: float = 0.0
    details: dict = field(default_factory=dict)


class RAGASEvaluator:
    """Evaluador basado en métricas RAGAS usando embeddings y heurísticas."""

    def __init__(self, embedding_model=None, llm_provider=None):
        self.embedding_model = embedding_model
        self.llm = llm_provider

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
        expected_keywords: list[str] = None,
    ) -> RAGASResult:
        """Evalúa una respuesta con métricas RAGAS."""
        result = RAGASResult(question=question)

        result.faithfulness = self._compute_faithfulness(answer, contexts)
        result.answer_relevance = self._compute_answer_relevance(
            question, answer
        )
        result.context_precision = self._compute_context_precision(
            question, contexts
        )
        result.context_recall = self._compute_context_recall(
            answer, contexts, ground_truth, expected_keywords or []
        )

        # Score global ponderado
        result.overall_score = (
            0.30 * result.faithfulness
            + 0.25 * result.answer_relevance
            + 0.25 * result.context_precision
            + 0.20 * result.context_recall
        )

        return result

    def evaluate_batch(
        self, qa_pairs: list[dict], results_data: list[dict]
    ) -> dict:
        """Evalúa un lote de pares QA con RAGAS.

        Args:
            qa_pairs: Lista de dicts con 'question', 'expected_answer', 'expected_keywords'
            results_data: Lista de dicts con 'answer', 'contexts' (textos de los chunks)
        """
        all_results = []
        for qa, res in zip(qa_pairs, results_data):
            ragas = self.evaluate(
                question=qa["question"],
                answer=res.get("answer", ""),
                contexts=res.get("contexts", []),
                ground_truth=qa.get("expected_answer", ""),
                expected_keywords=qa.get("expected_keywords", []),
            )
            all_results.append(ragas)

        # Agregar métricas
        n = len(all_results) or 1
        summary = {
            "avg_faithfulness": sum(r.faithfulness for r in all_results) / n,
            "avg_answer_relevance": sum(r.answer_relevance for r in all_results) / n,
            "avg_context_precision": sum(r.context_precision for r in all_results) / n,
            "avg_context_recall": sum(r.context_recall for r in all_results) / n,
            "avg_overall": sum(r.overall_score for r in all_results) / n,
            "results": all_results,
        }
        return summary

    # ── Faithfulness ─────────────────────────────────────────────

    def _compute_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Mide qué porcentaje de claims en la respuesta están respaldados por el contexto.

        Faithfulness = |claims respaldados| / |total claims|
        """
        if not answer or not contexts:
            return 0.0

        claims = self._extract_claims(answer)
        if not claims:
            return 0.5

        context_full = " ".join(contexts).lower()

        if self.embedding_model:
            return self._faithfulness_embeddings(claims, contexts)

        # Heurística: verificar presencia de datos clave en contexto
        supported = 0
        for claim in claims:
            claim_lower = claim.lower()

            # Extraer datos específicos del claim (números, nombres propios, emails)
            data_tokens = set()
            data_tokens.update(re.findall(r"\d+", claim))
            data_tokens.update(
                re.findall(r"\b[A-Z]{2,}\b", claim)
            )  # Siglas
            data_tokens.update(
                re.findall(r"[\w.]+@[\w.]+", claim)
            )  # Emails

            if data_tokens:
                matched = sum(1 for t in data_tokens if t.lower() in context_full)
                if matched / len(data_tokens) >= 0.5:
                    supported += 1
            else:
                # Sin datos específicos: overlap de palabras
                claim_words = set(claim_lower.split()) - _STOPWORDS_ES
                context_words = set(context_full.split())
                if claim_words:
                    overlap = len(claim_words & context_words) / len(claim_words)
                    if overlap >= 0.4:
                        supported += 1

        return supported / len(claims)

    def _faithfulness_embeddings(
        self, claims: list[str], contexts: list[str]
    ) -> float:
        """Faithfulness usando similitud de embeddings."""
        claim_embs = self.embedding_model.embed_texts(claims)
        ctx_embs = self.embedding_model.embed_texts(contexts)

        supported = 0
        for i in range(len(claims)):
            similarities = np.dot(claim_embs[i], ctx_embs.T)
            if float(np.max(similarities)) > 0.60:
                supported += 1

        return supported / len(claims)

    # ── Answer Relevance ─────────────────────────────────────────

    def _compute_answer_relevance(self, question: str, answer: str) -> float:
        """Mide qué tan relevante es la respuesta para la pregunta.

        Usa similitud semántica entre pregunta y respuesta.
        """
        if not question or not answer:
            return 0.0

        # Penalizar respuestas que se abstienen
        abstention_phrases = [
            "no tengo información", "no encontré", "fuera del alcance",
            "no puedo responder", "contactar a",
        ]
        if any(p in answer.lower() for p in abstention_phrases):
            return 0.3  # Score bajo pero no 0 (la abstención puede ser correcta)

        if self.embedding_model:
            q_emb = self.embedding_model.embed_query(question)
            a_emb = self.embedding_model.embed_query(answer[:500])
            similarity = float(np.dot(q_emb, a_emb))
            return max(0.0, min(similarity, 1.0))

        # Heurística: overlap de palabras clave
        q_words = set(question.lower().split()) - _STOPWORDS_ES
        a_words = set(answer.lower().split()) - _STOPWORDS_ES
        if not q_words:
            return 0.5
        overlap = len(q_words & a_words) / len(q_words)
        return min(overlap * 1.5, 1.0)  # Amplificar un poco

    # ── Context Precision ────────────────────────────────────────

    def _compute_context_precision(
        self, question: str, contexts: list[str]
    ) -> float:
        """Mide qué porcentaje de los contextos recuperados son relevantes.

        Context Precision = |contextos relevantes| / |total contextos|
        """
        if not contexts:
            return 0.0

        if self.embedding_model:
            q_emb = self.embedding_model.embed_query(question)
            ctx_embs = self.embedding_model.embed_texts(
                [c[:500] for c in contexts]
            )
            similarities = np.dot(ctx_embs, q_emb)
            relevant = sum(1 for s in similarities if s > 0.35)
            return relevant / len(contexts)

        # Heurística
        q_words = set(question.lower().split()) - _STOPWORDS_ES
        relevant = 0
        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            if q_words:
                overlap = len(q_words & ctx_words) / len(q_words)
                if overlap >= 0.2:
                    relevant += 1
        return relevant / len(contexts)

    # ── Context Recall ───────────────────────────────────────────

    def _compute_context_recall(
        self,
        answer: str,
        contexts: list[str],
        ground_truth: str,
        expected_keywords: list[str],
    ) -> float:
        """Mide si el contexto contiene la información necesaria para la respuesta correcta.

        Context Recall = |keywords esperados en contexto| / |total keywords|
        """
        context_full = " ".join(contexts).lower()

        if expected_keywords:
            found = sum(
                1 for kw in expected_keywords
                if kw.lower() in context_full
            )
            return found / len(expected_keywords)

        if ground_truth and ground_truth != "ABSTAIN":
            gt_words = set(ground_truth.lower().split()) - _STOPWORDS_ES
            if gt_words:
                found = sum(1 for w in gt_words if w in context_full)
                return found / len(gt_words)

        return 0.5  # Sin ground truth, score neutro

    # ── Utilidades ───────────────────────────────────────────────

    def _extract_claims(self, text: str) -> list[str]:
        """Extrae claims (oraciones con contenido informativo) de un texto."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = []
        for s in sentences:
            s = s.strip()
            if len(s) > 15 and not s.startswith("["):
                claims.append(s)
        return claims


# Stopwords básicas en español para heurísticas
_STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
    "al", "a", "en", "con", "por", "para", "se", "su", "sus", "que",
    "es", "son", "fue", "ser", "no", "sí", "como", "más", "pero",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "muy", "también", "ya", "o", "u", "y", "e", "lo", "le", "les",
    "me", "te", "nos", "mi", "tu", "qué", "cuál", "cuáles", "cómo",
    "dónde", "cuándo", "cuánto", "cuántos", "hay", "tiene", "tienen",
}
