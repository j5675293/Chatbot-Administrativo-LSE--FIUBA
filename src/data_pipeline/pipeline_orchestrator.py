"""
Orquestador del pipeline de procesamiento de documentos.
Gestiona extracción, limpieza, chunking y metadata end-to-end.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from src.data_pipeline.pdf_extractor import PDFExtractor
from src.data_pipeline.text_cleaner import SpanishTextCleaner
from src.data_pipeline.chunker import DocumentChunker, Chunk
from src.data_pipeline.metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Pipeline end-to-end con detección de cambios y procesamiento incremental."""

    STATE_FILE = ".pipeline_state.json"

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        extractor: Optional[PDFExtractor] = None,
        cleaner: Optional[SpanishTextCleaner] = None,
        chunker: Optional[DocumentChunker] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.extractor = extractor or PDFExtractor()
        self.cleaner = cleaner or SpanishTextCleaner()
        self.chunker = chunker or DocumentChunker()
        self.metadata_extractor = metadata_extractor or MetadataExtractor()

        # Crear directorios
        for subdir in ["extracted", "cleaned", "chunks", "metadata"]:
            (self.processed_dir / subdir).mkdir(parents=True, exist_ok=True)

        self._state = self._load_state()

    def process_all(self, force: bool = False) -> dict:
        """Procesa todos los PDFs en raw_dir."""
        pdf_files = list(self.raw_dir.glob("*.pdf"))
        logger.info(f"Encontrados {len(pdf_files)} archivos PDF")

        results = {
            "processed": [],
            "skipped": [],
            "errors": [],
            "total_chunks": 0,
        }

        for pdf_path in pdf_files:
            if not force and not self._has_changed(pdf_path):
                results["skipped"].append(pdf_path.name)
                logger.info(f"  Sin cambios: {pdf_path.name}")
                continue

            try:
                chunks = self.process_single(pdf_path)
                results["processed"].append(pdf_path.name)
                results["total_chunks"] += len(chunks)
                logger.info(f"  Procesado: {pdf_path.name} -> {len(chunks)} chunks")
            except Exception as e:
                results["errors"].append({"file": pdf_path.name, "error": str(e)})
                logger.error(f"  Error en {pdf_path.name}: {e}")

        self._save_state()

        logger.info(
            f"Pipeline completado: {len(results['processed'])} procesados, "
            f"{len(results['skipped'])} sin cambios, "
            f"{len(results['errors'])} errores, "
            f"{results['total_chunks']} chunks totales"
        )
        return results

    def process_single(self, filepath: Path) -> list[Chunk]:
        """Pipeline completo para un documento."""
        filepath = Path(filepath)
        stem = filepath.stem

        # 1. Extracción
        logger.info(f"  [1/4] Extrayendo texto: {filepath.name}")
        doc = self.extractor.extract(filepath)

        # Guardar texto extraído
        extracted_path = self.processed_dir / "extracted" / f"{stem}.txt"
        extracted_path.write_text(doc.raw_text, encoding="utf-8")

        # 2. Limpieza
        logger.info(f"  [2/4] Limpiando texto: {filepath.name}")
        cleaning_result = self.cleaner.clean(doc.raw_text, doc.document_type.value)
        cleaned_text = cleaning_result.cleaned_text

        # Guardar texto limpio
        cleaned_path = self.processed_dir / "cleaned" / f"{stem}.txt"
        cleaned_path.write_text(cleaned_text, encoding="utf-8")

        # 3. Metadata del documento
        logger.info(f"  [3/4] Extrayendo metadata: {filepath.name}")
        doc_metadata = self.metadata_extractor.extract_document_metadata(
            filepath.name, cleaned_text
        )

        # Guardar metadata
        meta_path = self.processed_dir / "metadata" / f"{stem}.json"
        meta_path.write_text(
            json.dumps(doc_metadata.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 4. Chunking
        logger.info(f"  [4/4] Generando chunks: {filepath.name}")
        pages_text = [(p.page_number, p.text) for p in doc.pages]
        chunks = self.chunker.chunk_document(
            text=cleaned_text,
            document_name=filepath.name,
            document_type=doc_metadata.document_type,
            pages_text=pages_text,
        )

        # Enriquecer chunks con metadata
        for chunk in chunks:
            chunk_meta = self.metadata_extractor.extract_chunk_metadata(
                chunk.text, doc_metadata
            )
            chunk.metadata.update(chunk_meta)

        # Guardar chunks
        chunks_path = self.processed_dir / "chunks" / f"{stem}.json"
        chunks_data = [c.to_dict() for c in chunks]
        chunks_path.write_text(
            json.dumps(chunks_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Actualizar estado
        self._state[filepath.name] = self._compute_file_hash(filepath)

        return chunks

    def get_all_chunks(self) -> list[Chunk]:
        """Carga todos los chunks procesados desde disco."""
        chunks_dir = self.processed_dir / "chunks"
        all_chunks = []

        for chunk_file in sorted(chunks_dir.glob("*.json")):
            try:
                data = json.loads(chunk_file.read_text(encoding="utf-8"))
                for item in data:
                    all_chunks.append(Chunk.from_dict(item))
            except Exception as e:
                logger.error(f"Error cargando chunks de {chunk_file.name}: {e}")

        logger.info(f"Cargados {len(all_chunks)} chunks desde disco")
        return all_chunks

    def get_all_metadata(self) -> list[dict]:
        """Carga todos los metadatos de documentos."""
        meta_dir = self.processed_dir / "metadata"
        all_meta = []

        for meta_file in sorted(meta_dir.glob("*.json")):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                all_meta.append(data)
            except Exception as e:
                logger.error(f"Error cargando metadata de {meta_file.name}: {e}")

        return all_meta

    def _compute_file_hash(self, filepath: Path) -> str:
        """SHA-256 del archivo para detección de cambios."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def _has_changed(self, filepath: Path) -> bool:
        """Verifica si el archivo cambió desde el último procesamiento."""
        current_hash = self._compute_file_hash(filepath)
        stored_hash = self._state.get(filepath.name)
        return current_hash != stored_hash

    def _load_state(self) -> dict:
        """Carga estado del pipeline desde disco."""
        state_path = self.processed_dir / self.STATE_FILE
        if state_path.exists():
            try:
                return json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_state(self) -> None:
        """Persiste estado del pipeline."""
        state_path = self.processed_dir / self.STATE_FILE
        state_path.write_text(
            json.dumps(self._state, indent=2),
            encoding="utf-8",
        )
