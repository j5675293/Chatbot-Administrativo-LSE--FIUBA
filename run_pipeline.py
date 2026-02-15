"""
Script principal para ejecutar el pipeline de procesamiento de documentos.
Extrae, limpia, chunkea e indexa todos los PDFs del directorio de datos.

Autor: Juan Ruiz Otondo - CEIA FIUBA
Carrera de Especialización en Inteligencia Artificial
Laboratorio de Sistemas Embebidos - FIUBA - UBA

Uso:
    python run_pipeline.py                      # Procesar todos los documentos
    python run_pipeline.py --force              # Forzar reprocesamiento
    python run_pipeline.py --doc CEIA.pdf       # Procesar un documento
"""

import sys
import argparse
import logging
import shutil
from pathlib import Path

# Configurar path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import Settings
from src.data_pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import FAISSVectorStore
from src.graph_rag.entity_extractor import AcademicEntityExtractor
from src.graph_rag.relationship_mapper import RelationshipMapper
from src.graph_rag.graph_builder import KnowledgeGraphBuilder
from src.graph_rag.community_detector import CommunityDetector

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de procesamiento de documentos LSE-FIUBA"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Forzar reprocesamiento de todos los documentos",
    )
    parser.add_argument(
        "--doc", type=str, default=None,
        help="Procesar solo un documento específico (nombre del archivo)",
    )
    parser.add_argument(
        "--skip-graph", action="store_true",
        help="Omitir construcción del grafo de conocimiento",
    )
    parser.add_argument(
        "--pdf-dir", type=str, default=None,
        help="Directorio con los PDFs (default: data/raw)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    settings = Settings()

    logger.info("=" * 70)
    logger.info("PIPELINE DE PROCESAMIENTO - CHATBOT LSE-FIUBA")
    logger.info("Autor: Juan Ruiz Otondo - CEIA FIUBA")
    logger.info("=" * 70)

    # ── 1. Configurar directorios ────────────────────────────────
    raw_dir = Path(args.pdf_dir) if args.pdf_dir else settings.RAW_DATA_DIR
    processed_dir = settings.PROCESSED_DATA_DIR
    index_dir = settings.INDEX_DIR
    graph_dir = settings.GRAPH_DIR

    for d in [raw_dir, processed_dir, index_dir, graph_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 2. Copiar PDFs al directorio raw si no existen ───────────
    # Buscar PDFs en el directorio padre (donde están los originales)
    source_pdf_dir = ROOT.parent  # Directorio DOCUMENTOS CHATBOT
    pdf_files = list(source_pdf_dir.glob("*.pdf"))

    if pdf_files and not list(raw_dir.glob("*.pdf")):
        logger.info(f"Copiando {len(pdf_files)} PDFs desde {source_pdf_dir}")
        for pdf in pdf_files:
            dest = raw_dir / pdf.name
            if not dest.exists():
                shutil.copy2(pdf, dest)
                logger.info(f"  Copiado: {pdf.name}")

    # ── 3. Pipeline de extracción y procesamiento ────────────────
    logger.info("\n--- Etapa 1: Extracción y procesamiento de PDFs ---")

    pipeline = PipelineOrchestrator(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    if args.doc:
        doc_path = raw_dir / args.doc
        if not doc_path.exists():
            logger.error(f"Documento no encontrado: {doc_path}")
            sys.exit(1)
        # Procesar un solo documento
        chunks = pipeline.process_single(doc_path)
        all_chunks = pipeline.get_all_chunks()
        logger.info(f"Documento procesado: {len(chunks)} chunks nuevos")
    else:
        pdfs = list(raw_dir.glob("*.pdf"))
        if not pdfs:
            logger.error(f"No se encontraron PDFs en {raw_dir}")
            logger.info(f"Colocá los documentos PDF en: {raw_dir}")
            sys.exit(1)

        logger.info(f"Documentos a procesar: {len(pdfs)}")
        result = pipeline.process_all(force=args.force)
        all_chunks = pipeline.get_all_chunks()
        logger.info(
            f"Pipeline: {len(result['processed'])} procesados, "
            f"{len(result['skipped'])} sin cambios, "
            f"{len(result['errors'])} errores"
        )

    if not all_chunks:
        logger.error("No se generaron chunks. Verificar los PDFs en data/raw/")
        sys.exit(1)

    logger.info(f"Total de chunks disponibles: {len(all_chunks)}")

    # ── 4. Indexación vectorial (FAISS) ──────────────────────────
    logger.info("\n--- Etapa 2: Indexación vectorial (FAISS) ---")

    embedding_model = EmbeddingModel(
        model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE,
    )

    vector_store = FAISSVectorStore(embedding_dim=384)

    # Generar embeddings
    texts = [chunk.text for chunk in all_chunks]
    logger.info(f"Generando embeddings para {len(texts)} chunks...")
    embeddings = embedding_model.embed_texts(texts)

    # Construir índice
    logger.info("Construyendo índice FAISS...")
    vector_store.build_index(all_chunks, embeddings)

    logger.info(f"Guardando índice en {index_dir}...")
    vector_store.save(index_dir)

    logger.info(f"Índice FAISS creado con {vector_store.index.ntotal} vectores")

    # ── 5. Grafo de conocimiento ─────────────────────────────────
    if not args.skip_graph:
        logger.info("\n--- Etapa 3: Construcción del grafo de conocimiento ---")

        entity_extractor = AcademicEntityExtractor()
        relationship_mapper = RelationshipMapper()
        graph_builder = KnowledgeGraphBuilder()

        # Extraer entidades de cada chunk
        all_entities = []
        for chunk in all_chunks:
            entities = entity_extractor.extract_entities(
                chunk.text, chunk.document_name
            )
            all_entities.extend(entities)

        logger.info(f"Entidades extraídas: {len(all_entities)}")

        # Mapear relaciones
        full_text = "\n\n".join(c.text for c in all_chunks)
        all_relationships = relationship_mapper.extract_relationships(
            text=full_text,
            entities=all_entities,
        )
        logger.info(f"Relaciones mapeadas: {len(all_relationships)}")

        # Construir grafo
        graph_builder.build_graph(all_entities, all_relationships)
        logger.info(
            f"Grafo construido: {graph_builder.graph.number_of_nodes()} nodos, "
            f"{graph_builder.graph.number_of_edges()} aristas"
        )

        # Detección de comunidades
        try:
            detector = CommunityDetector(graph_builder.graph)
            communities = detector.detect_communities()
            logger.info(f"Comunidades detectadas: {len(communities)}")
        except Exception as e:
            logger.warning(f"No se pudieron detectar comunidades: {e}")

        # Guardar
        graph_builder.save(graph_dir)
        logger.info(f"Grafo guardado en {graph_dir}")

    # ── Resumen ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info(f"  Chunks disponibles: {len(all_chunks)}")
    logger.info(f"  Vectores indexados: {vector_store.index.ntotal}")
    if not args.skip_graph:
        logger.info(f"  Nodos del grafo: {graph_builder.graph.number_of_nodes()}")
        logger.info(f"  Aristas del grafo: {graph_builder.graph.number_of_edges()}")
    logger.info("=" * 70)
    logger.info("\nSiguiente paso: ejecutar la API con 'python run_api.py'")


if __name__ == "__main__":
    main()
