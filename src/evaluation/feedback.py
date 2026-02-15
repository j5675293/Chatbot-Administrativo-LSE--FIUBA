"""
Sistema de feedback Human-in-the-Loop.
Recolecta y almacena feedback de usuarios para mejora continua.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """Entrada de feedback individual."""
    feedback_id: str = ""
    session_id: str = ""
    question: str = ""
    answer: str = ""
    method: str = ""
    confidence: float = 0.0
    # Feedback del usuario
    rating: int = 0  # 1-5 estrellas
    is_correct: Optional[bool] = None  # Respuesta correcta?
    is_complete: Optional[bool] = None  # Respuesta completa?
    user_comment: str = ""
    expected_answer: str = ""  # Lo que el usuario esperaba
    # Metadata
    timestamp: float = 0.0
    sources_count: int = 0
    processing_time_ms: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.feedback_id:
            self.feedback_id = f"FB-{int(self.timestamp * 1000)}"


@dataclass
class FeedbackStats:
    """Estadísticas agregadas de feedback."""
    total_entries: int = 0
    avg_rating: float = 0.0
    correct_rate: float = 0.0
    complete_rate: float = 0.0
    avg_confidence: float = 0.0
    by_method: dict = field(default_factory=dict)
    by_rating: dict = field(default_factory=dict)
    low_rated_questions: list = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)


class FeedbackCollector:
    """Recolector y analizador de feedback de usuarios."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/evaluation/feedback.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[FeedbackEntry] = []
        self._load()

    def submit_feedback(
        self,
        session_id: str,
        question: str,
        answer: str,
        rating: int,
        method: str = "hybrid",
        confidence: float = 0.0,
        is_correct: Optional[bool] = None,
        is_complete: Optional[bool] = None,
        user_comment: str = "",
        expected_answer: str = "",
        processing_time_ms: float = 0.0,
        sources_count: int = 0,
    ) -> FeedbackEntry:
        """Registra una entrada de feedback."""
        entry = FeedbackEntry(
            session_id=session_id,
            question=question,
            answer=answer[:500],
            method=method,
            confidence=confidence,
            rating=max(1, min(5, rating)),
            is_correct=is_correct,
            is_complete=is_complete,
            user_comment=user_comment,
            expected_answer=expected_answer,
            processing_time_ms=processing_time_ms,
            sources_count=sources_count,
        )
        self._entries.append(entry)
        self._save()

        logger.info(
            f"Feedback registrado: rating={entry.rating}, "
            f"correct={entry.is_correct}, method={entry.method}"
        )
        return entry

    def get_stats(self) -> FeedbackStats:
        """Calcula estadísticas agregadas del feedback."""
        stats = FeedbackStats(total_entries=len(self._entries))

        if not self._entries:
            return stats

        entries = self._entries
        n = len(entries)

        # Promedios
        stats.avg_rating = sum(e.rating for e in entries) / n
        stats.avg_confidence = sum(e.confidence for e in entries) / n

        correct_entries = [e for e in entries if e.is_correct is not None]
        if correct_entries:
            stats.correct_rate = (
                sum(1 for e in correct_entries if e.is_correct)
                / len(correct_entries)
            )

        complete_entries = [e for e in entries if e.is_complete is not None]
        if complete_entries:
            stats.complete_rate = (
                sum(1 for e in complete_entries if e.is_complete)
                / len(complete_entries)
            )

        # Por método
        methods = set(e.method for e in entries)
        for method in methods:
            method_entries = [e for e in entries if e.method == method]
            m_n = len(method_entries)
            stats.by_method[method] = {
                "count": m_n,
                "avg_rating": sum(e.rating for e in method_entries) / m_n,
                "avg_confidence": sum(e.confidence for e in method_entries) / m_n,
            }

        # Por rating
        for r in range(1, 6):
            count = sum(1 for e in entries if e.rating == r)
            stats.by_rating[str(r)] = count

        # Preguntas con peor rating
        low_rated = sorted(
            [e for e in entries if e.rating <= 2],
            key=lambda e: e.rating,
        )
        stats.low_rated_questions = [
            {
                "question": e.question,
                "rating": e.rating,
                "method": e.method,
                "confidence": e.confidence,
                "comment": e.user_comment,
            }
            for e in low_rated[:10]
        ]

        # Sugerencias de mejora
        stats.improvement_suggestions = self._generate_suggestions(stats)

        return stats

    def get_entries(
        self,
        session_id: Optional[str] = None,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
    ) -> list[FeedbackEntry]:
        """Obtiene entradas filtradas."""
        entries = self._entries

        if session_id:
            entries = [e for e in entries if e.session_id == session_id]
        if min_rating is not None:
            entries = [e for e in entries if e.rating >= min_rating]
        if max_rating is not None:
            entries = [e for e in entries if e.rating <= max_rating]

        return entries

    def export_for_training(self) -> list[dict]:
        """Exporta feedback positivo como datos de entrenamiento potencial."""
        good_entries = [
            e for e in self._entries
            if e.rating >= 4 and e.is_correct is True
        ]
        return [
            {
                "question": e.question,
                "good_answer": e.answer,
                "method": e.method,
                "confidence": e.confidence,
            }
            for e in good_entries
        ]

    def _generate_suggestions(self, stats: FeedbackStats) -> list[str]:
        """Genera sugerencias de mejora basadas en el feedback."""
        suggestions = []

        if stats.avg_rating < 3.0:
            suggestions.append(
                "Rating promedio bajo ({:.1f}/5). Revisar calidad de respuestas.".format(
                    stats.avg_rating
                )
            )

        if stats.correct_rate < 0.7 and stats.total_entries >= 5:
            suggestions.append(
                "Tasa de respuestas correctas baja ({:.0%}). "
                "Considerar mejorar el retrieval o aumentar el corpus.".format(
                    stats.correct_rate
                )
            )

        if stats.complete_rate < 0.6 and stats.total_entries >= 5:
            suggestions.append(
                "Tasa de respuestas completas baja ({:.0%}). "
                "Considerar aumentar top_k o mejorar la síntesis.".format(
                    stats.complete_rate
                )
            )

        # Método con peor desempeño
        if stats.by_method:
            worst_method = min(
                stats.by_method.items(), key=lambda x: x[1]["avg_rating"]
            )
            if worst_method[1]["avg_rating"] < 3.0:
                suggestions.append(
                    f"Método '{worst_method[0]}' tiene rating bajo "
                    f"({worst_method[1]['avg_rating']:.1f}). Revisar configuración."
                )

        # Patrones en preguntas mal valoradas
        if stats.low_rated_questions:
            suggestions.append(
                f"{len(stats.low_rated_questions)} preguntas con rating <= 2. "
                f"Revisar para mejorar cobertura del conocimiento."
            )

        return suggestions

    def _save(self) -> None:
        """Guarda feedback en archivo JSON."""
        data = [asdict(e) for e in self._entries]
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        """Carga feedback desde archivo JSON."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._entries = [FeedbackEntry(**d) for d in data]
                logger.info(f"Cargadas {len(self._entries)} entradas de feedback")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Error cargando feedback: {e}")
                self._entries = []
