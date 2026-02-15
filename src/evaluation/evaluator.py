"""
Evaluador del sistema de chatbot. Mide retrieval quality, answer quality,
faithfulness, y compara RAG vs GraphRAG vs Hybrid.
Incluye métricas RAGAS (faithfulness, answer_relevance, context_precision).

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.evaluation.ragas_metrics import RAGASEvaluator, RAGASResult

logger = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    """Resultado de evaluación de una pregunta individual."""
    question_id: str
    question: str
    category: str
    difficulty: str
    # RAG
    rag_answer: str = ""
    rag_confidence: float = 0.0
    rag_time_ms: float = 0.0
    rag_sources: list = field(default_factory=list)
    # Graph
    graph_answer: str = ""
    graph_confidence: float = 0.0
    graph_time_ms: float = 0.0
    graph_sources: list = field(default_factory=list)
    # Hybrid
    hybrid_answer: str = ""
    hybrid_confidence: float = 0.0
    hybrid_time_ms: float = 0.0
    hybrid_sources: list = field(default_factory=list)
    # Métricas
    rag_keyword_hit_rate: float = 0.0
    graph_keyword_hit_rate: float = 0.0
    hybrid_keyword_hit_rate: float = 0.0
    rag_source_match: bool = False
    graph_source_match: bool = False
    hybrid_source_match: bool = False
    best_method: str = ""
    # RAGAS metrics
    rag_ragas: Optional[RAGASResult] = None
    graph_ragas: Optional[RAGASResult] = None
    hybrid_ragas: Optional[RAGASResult] = None


@dataclass
class EvaluationReport:
    """Reporte de evaluación completo."""
    total_questions: int = 0
    # Métricas por método
    rag_avg_keyword_hit: float = 0.0
    graph_avg_keyword_hit: float = 0.0
    hybrid_avg_keyword_hit: float = 0.0
    rag_avg_confidence: float = 0.0
    graph_avg_confidence: float = 0.0
    hybrid_avg_confidence: float = 0.0
    rag_avg_time_ms: float = 0.0
    graph_avg_time_ms: float = 0.0
    hybrid_avg_time_ms: float = 0.0
    rag_source_accuracy: float = 0.0
    graph_source_accuracy: float = 0.0
    hybrid_source_accuracy: float = 0.0
    # Conteos de mejor método
    rag_wins: int = 0
    graph_wins: int = 0
    hybrid_wins: int = 0
    # Por categoría
    results_by_category: dict = field(default_factory=dict)
    # Por dificultad
    results_by_difficulty: dict = field(default_factory=dict)
    # Detalle
    question_results: list[QuestionResult] = field(default_factory=list)
    # Abstención
    correct_abstentions: int = 0
    total_abstention_questions: int = 0
    # RAGAS agregadas
    ragas_summary: dict = field(default_factory=dict)


class Evaluator:
    """Evaluador principal del sistema de chatbot."""

    def __init__(
        self,
        hybrid_retriever,
        answer_synthesizer,
        embedding_model=None,
        llm_provider=None,
        output_dir: Optional[Path] = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.answer_synthesizer = answer_synthesizer
        self.output_dir = output_dir or Path("data/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ragas = RAGASEvaluator(
            embedding_model=embedding_model, llm_provider=llm_provider
        )

    def evaluate(
        self, qa_pairs: list[dict], verbose: bool = True
    ) -> EvaluationReport:
        """Ejecuta evaluación completa sobre conjunto de preguntas."""
        from src.hybrid.hybrid_retriever import RetrievalMode

        report = EvaluationReport(total_questions=len(qa_pairs))
        results = []

        for i, qa in enumerate(qa_pairs):
            if verbose:
                logger.info(
                    f"[{i+1}/{len(qa_pairs)}] Evaluando: {qa['question'][:60]}..."
                )

            qr = QuestionResult(
                question_id=qa["id"],
                question=qa["question"],
                category=qa.get("category", "unknown"),
                difficulty=qa.get("difficulty", "unknown"),
            )

            # Evaluar con cada método
            for mode_name, mode_enum in [
                ("rag", RetrievalMode.RAG_ONLY),
                ("graph", RetrievalMode.GRAPH_ONLY),
                ("hybrid", RetrievalMode.HYBRID),
            ]:
                start = time.time()
                try:
                    hybrid_result = self.hybrid_retriever.retrieve(
                        query=qa["question"], mode=mode_enum, top_k=5
                    )
                    final = self.answer_synthesizer.synthesize(
                        query=qa["question"], hybrid_result=hybrid_result
                    )
                    elapsed = (time.time() - start) * 1000

                    setattr(qr, f"{mode_name}_answer", final.answer)
                    setattr(qr, f"{mode_name}_confidence", final.confidence)
                    setattr(qr, f"{mode_name}_time_ms", elapsed)
                    setattr(qr, f"{mode_name}_sources", final.sources)

                except Exception as e:
                    logger.error(f"Error evaluando {mode_name}: {e}")
                    setattr(qr, f"{mode_name}_time_ms",
                            (time.time() - start) * 1000)

            # Calcular métricas clásicas
            self._compute_metrics(qr, qa)

            # Calcular métricas RAGAS por método
            for method in ["rag", "graph", "hybrid"]:
                answer_text = getattr(qr, f"{method}_answer", "")
                sources = getattr(qr, f"{method}_sources", [])
                contexts = [
                    s.get("text_snippet", "") for s in sources if s.get("text_snippet")
                ]
                ragas_result = self.ragas.evaluate(
                    question=qa["question"],
                    answer=answer_text,
                    contexts=contexts,
                    ground_truth=qa.get("expected_answer", ""),
                    expected_keywords=qa.get("expected_keywords", []),
                )
                setattr(qr, f"{method}_ragas", ragas_result)

            results.append(qr)

        report.question_results = results
        self._aggregate_report(report)

        # Guardar reporte
        self._save_report(report)

        return report

    def _compute_metrics(self, qr: QuestionResult, qa: dict) -> None:
        """Calcula métricas para una pregunta."""
        expected_keywords = qa.get("expected_keywords", [])
        expected_source = qa.get("expected_source")
        is_ood = qa.get("category") == "out_of_domain"

        for method in ["rag", "graph", "hybrid"]:
            answer = getattr(qr, f"{method}_answer", "").lower()

            # Keyword hit rate
            if expected_keywords and not is_ood:
                hits = sum(
                    1 for kw in expected_keywords
                    if kw.lower() in answer
                )
                hit_rate = hits / len(expected_keywords)
                setattr(qr, f"{method}_keyword_hit_rate", hit_rate)

            # Source match
            if expected_source and not is_ood:
                sources = getattr(qr, f"{method}_sources", [])
                source_match = any(
                    expected_source.lower() in s.get("document_name", "").lower()
                    for s in sources
                )
                setattr(qr, f"{method}_source_match", source_match)

        # Determinar mejor método
        scores = {
            "rag": qr.rag_keyword_hit_rate * 0.5 + qr.rag_confidence * 0.3
                   + (0.2 if qr.rag_source_match else 0.0),
            "graph": qr.graph_keyword_hit_rate * 0.5 + qr.graph_confidence * 0.3
                     + (0.2 if qr.graph_source_match else 0.0),
            "hybrid": qr.hybrid_keyword_hit_rate * 0.5 + qr.hybrid_confidence * 0.3
                      + (0.2 if qr.hybrid_source_match else 0.0),
        }
        qr.best_method = max(scores, key=scores.get)

    def _aggregate_report(self, report: EvaluationReport) -> None:
        """Agrega métricas del reporte."""
        results = report.question_results
        if not results:
            return

        n = len(results)

        # Promedios generales
        for method in ["rag", "graph", "hybrid"]:
            kw_hits = [getattr(r, f"{method}_keyword_hit_rate") for r in results]
            confs = [getattr(r, f"{method}_confidence") for r in results]
            times = [getattr(r, f"{method}_time_ms") for r in results]
            sources = [getattr(r, f"{method}_source_match") for r in results]

            setattr(report, f"{method}_avg_keyword_hit",
                    sum(kw_hits) / n if n else 0)
            setattr(report, f"{method}_avg_confidence",
                    sum(confs) / n if n else 0)
            setattr(report, f"{method}_avg_time_ms",
                    sum(times) / n if n else 0)
            setattr(report, f"{method}_source_accuracy",
                    sum(sources) / n if n else 0)

        # Conteo de wins
        report.rag_wins = sum(1 for r in results if r.best_method == "rag")
        report.graph_wins = sum(1 for r in results if r.best_method == "graph")
        report.hybrid_wins = sum(1 for r in results if r.best_method == "hybrid")

        # Por categoría
        categories = set(r.category for r in results)
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            report.results_by_category[cat] = {
                "count": len(cat_results),
                "rag_avg_keyword_hit": np.mean(
                    [r.rag_keyword_hit_rate for r in cat_results]
                ),
                "graph_avg_keyword_hit": np.mean(
                    [r.graph_keyword_hit_rate for r in cat_results]
                ),
                "hybrid_avg_keyword_hit": np.mean(
                    [r.hybrid_keyword_hit_rate for r in cat_results]
                ),
            }

        # Por dificultad
        difficulties = set(r.difficulty for r in results)
        for diff in difficulties:
            diff_results = [r for r in results if r.difficulty == diff]
            report.results_by_difficulty[diff] = {
                "count": len(diff_results),
                "rag_avg_keyword_hit": np.mean(
                    [r.rag_keyword_hit_rate for r in diff_results]
                ),
                "hybrid_avg_keyword_hit": np.mean(
                    [r.hybrid_keyword_hit_rate for r in diff_results]
                ),
            }

        # RAGAS summary
        for method in ["rag", "graph", "hybrid"]:
            ragas_list = [
                getattr(r, f"{method}_ragas")
                for r in results
                if getattr(r, f"{method}_ragas") is not None
            ]
            if ragas_list:
                rn = len(ragas_list)
                report.ragas_summary[method] = {
                    "avg_faithfulness": sum(r.faithfulness for r in ragas_list) / rn,
                    "avg_answer_relevance": sum(r.answer_relevance for r in ragas_list) / rn,
                    "avg_context_precision": sum(r.context_precision for r in ragas_list) / rn,
                    "avg_context_recall": sum(r.context_recall for r in ragas_list) / rn,
                    "avg_overall": sum(r.overall_score for r in ragas_list) / rn,
                }

        # Abstención
        ood_results = [r for r in results if r.category == "out_of_domain"]
        report.total_abstention_questions = len(ood_results)
        abstain_phrases = ["fuera del alcance", "no tengo", "contactar"]
        report.correct_abstentions = sum(
            1 for r in ood_results
            if any(p in r.hybrid_answer.lower() for p in abstain_phrases)
        )

    def _save_report(self, report: EvaluationReport) -> None:
        """Guarda el reporte en formato JSON."""
        output = {
            "summary": {
                "total_questions": report.total_questions,
                "rag": {
                    "avg_keyword_hit": round(report.rag_avg_keyword_hit, 3),
                    "avg_confidence": round(report.rag_avg_confidence, 3),
                    "avg_time_ms": round(report.rag_avg_time_ms, 1),
                    "source_accuracy": round(report.rag_source_accuracy, 3),
                    "wins": report.rag_wins,
                },
                "graph": {
                    "avg_keyword_hit": round(report.graph_avg_keyword_hit, 3),
                    "avg_confidence": round(report.graph_avg_confidence, 3),
                    "avg_time_ms": round(report.graph_avg_time_ms, 1),
                    "source_accuracy": round(report.graph_source_accuracy, 3),
                    "wins": report.graph_wins,
                },
                "hybrid": {
                    "avg_keyword_hit": round(report.hybrid_avg_keyword_hit, 3),
                    "avg_confidence": round(report.hybrid_avg_confidence, 3),
                    "avg_time_ms": round(report.hybrid_avg_time_ms, 1),
                    "source_accuracy": round(report.hybrid_source_accuracy, 3),
                    "wins": report.hybrid_wins,
                },
                "abstention": {
                    "total": report.total_abstention_questions,
                    "correct": report.correct_abstentions,
                    "accuracy": round(
                        report.correct_abstentions / report.total_abstention_questions
                        if report.total_abstention_questions > 0 else 0, 3
                    ),
                },
            },
            "ragas": {
                method: {k: round(v, 3) for k, v in data.items()}
                for method, data in report.ragas_summary.items()
            },
            "by_category": report.results_by_category,
            "by_difficulty": report.results_by_difficulty,
            "details": [
                {
                    "id": r.question_id,
                    "question": r.question,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "best_method": r.best_method,
                    "rag": {
                        "answer": r.rag_answer[:300],
                        "confidence": round(r.rag_confidence, 3),
                        "keyword_hit_rate": round(r.rag_keyword_hit_rate, 3),
                        "source_match": r.rag_source_match,
                        "time_ms": round(r.rag_time_ms, 1),
                    },
                    "graph": {
                        "answer": r.graph_answer[:300],
                        "confidence": round(r.graph_confidence, 3),
                        "keyword_hit_rate": round(r.graph_keyword_hit_rate, 3),
                        "source_match": r.graph_source_match,
                        "time_ms": round(r.graph_time_ms, 1),
                    },
                    "hybrid": {
                        "answer": r.hybrid_answer[:300],
                        "confidence": round(r.hybrid_confidence, 3),
                        "keyword_hit_rate": round(r.hybrid_keyword_hit_rate, 3),
                        "source_match": r.hybrid_source_match,
                        "time_ms": round(r.hybrid_time_ms, 1),
                    },
                }
                for r in report.question_results
            ],
        }

        # Convertir numpy types para serialización
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=convert)

        logger.info(f"Reporte guardado en: {report_path}")

    def print_summary(self, report: EvaluationReport) -> str:
        """Genera resumen legible del reporte."""
        lines = [
            "=" * 70,
            "REPORTE DE EVALUACIÓN - CHATBOT LSE-FIUBA",
            "Autor: Juan Ruiz Otondo - CEIA FIUBA",
            "=" * 70,
            f"\nTotal de preguntas evaluadas: {report.total_questions}\n",
            "-" * 70,
            f"{'Métrica':<35} {'RAG':>10} {'GraphRAG':>10} {'Hybrid':>10}",
            "-" * 70,
            f"{'Keyword Hit Rate':<35} {report.rag_avg_keyword_hit:>10.1%} "
            f"{report.graph_avg_keyword_hit:>10.1%} {report.hybrid_avg_keyword_hit:>10.1%}",
            f"{'Confianza Promedio':<35} {report.rag_avg_confidence:>10.1%} "
            f"{report.graph_avg_confidence:>10.1%} {report.hybrid_avg_confidence:>10.1%}",
            f"{'Tiempo Promedio (ms)':<35} {report.rag_avg_time_ms:>10.1f} "
            f"{report.graph_avg_time_ms:>10.1f} {report.hybrid_avg_time_ms:>10.1f}",
            f"{'Source Accuracy':<35} {report.rag_source_accuracy:>10.1%} "
            f"{report.graph_source_accuracy:>10.1%} {report.hybrid_source_accuracy:>10.1%}",
            f"{'Victorias':<35} {report.rag_wins:>10} "
            f"{report.graph_wins:>10} {report.hybrid_wins:>10}",
            "-" * 70,
        ]

        # RAGAS metrics
        if report.ragas_summary:
            lines.append("\n--- Métricas RAGAS ---")
            lines.append(f"{'Métrica':<35} {'RAG':>10} {'GraphRAG':>10} {'Hybrid':>10}")
            lines.append("-" * 70)
            for metric in ["avg_faithfulness", "avg_answer_relevance",
                           "avg_context_precision", "avg_context_recall", "avg_overall"]:
                label = metric.replace("avg_", "").replace("_", " ").title()
                vals = []
                for method in ["rag", "graph", "hybrid"]:
                    val = report.ragas_summary.get(method, {}).get(metric, 0)
                    vals.append(f"{val:>10.1%}")
                lines.append(f"{label:<35} {''.join(vals)}")

        # Abstención
        if report.total_abstention_questions > 0:
            acc = report.correct_abstentions / report.total_abstention_questions
            lines.append(
                f"\nAbstención correcta: {report.correct_abstentions}/"
                f"{report.total_abstention_questions} ({acc:.0%})"
            )

        # Por categoría
        lines.append("\n--- Resultados por categoría ---")
        for cat, data in report.results_by_category.items():
            lines.append(
                f"  {cat}: {data['count']} preguntas | "
                f"RAG: {data['rag_avg_keyword_hit']:.1%} | "
                f"Graph: {data['graph_avg_keyword_hit']:.1%} | "
                f"Hybrid: {data['hybrid_avg_keyword_hit']:.1%}"
            )

        lines.append("\n" + "=" * 70)

        summary = "\n".join(lines)
        logger.info("\n" + summary)
        return summary
