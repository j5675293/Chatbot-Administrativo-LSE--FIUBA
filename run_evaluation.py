"""
Script para ejecutar la evaluación comparativa del sistema.
Compara RAG vs GraphRAG vs Hybrid sobre el conjunto de test.

Autor: Juan Ruiz Otondo - CEIA FIUBA
Carrera de Especialización en Inteligencia Artificial
Laboratorio de Sistemas Embebidos - FIUBA - UBA

Uso:
    python run_evaluation.py                    # Evaluación completa
    python run_evaluation.py --quick            # Subset rápido (5 preguntas)
    python run_evaluation.py --category factual # Solo preguntas factuales
"""

import sys
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "evaluation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("evaluation")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluación del Chatbot LSE-FIUBA"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Ejecutar evaluación rápida (subset de 5 preguntas)",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Evaluar solo una categoría (factual, procedural, comparative, etc.)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("EVALUACIÓN COMPARATIVA - CHATBOT LSE-FIUBA")
    logger.info("Autor: Juan Ruiz Otondo - CEIA FIUBA")
    logger.info("=" * 70)

    from config.settings import Settings
    from src.api.dependencies import AppDependencies
    from src.evaluation.evaluator import Evaluator
    from src.evaluation.test_sets import EVALUATION_QA_PAIRS

    # Inicializar sistema
    settings = Settings()
    logger.info("Inicializando componentes del sistema...")

    deps = AppDependencies()
    deps.initialize()

    # Filtrar preguntas
    qa_pairs = EVALUATION_QA_PAIRS

    if args.category:
        qa_pairs = [q for q in qa_pairs if q["category"] == args.category]
        logger.info(f"Filtrado por categoría '{args.category}': {len(qa_pairs)} preguntas")

    if args.quick:
        qa_pairs = qa_pairs[:5]
        logger.info(f"Modo rápido: {len(qa_pairs)} preguntas")

    logger.info(f"\nEvaluando {len(qa_pairs)} preguntas...\n")

    # Ejecutar evaluación
    evaluator = Evaluator(
        hybrid_retriever=deps.hybrid_retriever,
        answer_synthesizer=deps.answer_synthesizer,
        output_dir=settings.EVALUATION_DIR,
    )

    report = evaluator.evaluate(qa_pairs, verbose=True)

    # Mostrar resumen
    summary = evaluator.print_summary(report)
    print(summary)

    logger.info(f"\nReporte detallado guardado en: {settings.EVALUATION_DIR}")


if __name__ == "__main__":
    main()
