"""
Tests del pipeline de procesamiento de datos.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data_pipeline.pdf_extractor import PDFExtractor, DocumentType
from src.data_pipeline.text_cleaner import SpanishTextCleaner
from src.data_pipeline.chunker import DocumentChunker
from src.data_pipeline.metadata_extractor import MetadataExtractor


class TestPDFExtractor:
    """Tests de extracción de PDFs."""

    def setup_method(self):
        self.extractor = PDFExtractor()

    def test_detect_document_type_faq(self):
        doc_type = self.extractor._detect_document_type("FAQ - MIA.pdf", "")
        assert doc_type == DocumentType.FAQ

    def test_detect_document_type_reglamento(self):
        doc_type = self.extractor._detect_document_type(
            "Reglamento de Cursada.pdf", ""
        )
        assert doc_type == DocumentType.REGULATION

    def test_detect_document_type_vinculacion(self):
        doc_type = self.extractor._detect_document_type(
            "PROGRAMA DE VINCULACIÓN.pdf", ""
        )
        assert doc_type == DocumentType.VINCULACION

    def test_detect_document_type_resolution_by_content(self):
        doc_type = self.extractor._detect_document_type(
            "CEIA.pdf", "VISTO el expediente..."
        )
        assert doc_type == DocumentType.RESOLUTION

    def test_header_removal(self):
        text = "EX-2020-02051595-UBA-SG#REC\nContenido real"
        cleaned = self.extractor._remove_headers_footers(text, DocumentType.RESOLUTION)
        assert "EX-2020" not in cleaned
        assert "Contenido real" in cleaned

    def test_extract_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            self.extractor.extract(Path("/nonexistent/file.pdf"))


class TestTextCleaner:
    """Tests de limpieza de texto."""

    def setup_method(self):
        self.cleaner = SpanishTextCleaner()

    def test_normalize_whitespace(self):
        result = self.cleaner.clean("Hola    mundo   test")
        assert "   " not in result.cleaned_text

    def test_clean_empty(self):
        result = self.cleaner.clean("")
        assert result.cleaned_text == ""

    def test_preserve_content(self):
        text = "Artículo 1°: Los estudiantes deben asistir."
        result = self.cleaner.clean(text)
        assert "estudiantes" in result.cleaned_text
        assert "asistir" in result.cleaned_text


class TestDocumentChunker:
    """Tests del chunking."""

    def setup_method(self):
        self.chunker = DocumentChunker(
            fixed_chunk_size=100, fixed_overlap=20
        )

    def test_chunk_short_text(self):
        chunks = self.chunker.chunk_document(
            text="Texto corto. Este es un texto de prueba suficientemente largo para no ser filtrado por min tokens.",
            document_name="test.pdf",
            document_type="resolucion",
        )
        assert len(chunks) >= 1

    def test_chunk_long_text(self):
        text = "Esta es una oración de prueba. " * 100
        chunks = self.chunker.chunk_document(
            text=text,
            document_name="test.pdf",
            document_type="resolucion",
        )
        assert len(chunks) > 1

    def test_chunk_metadata(self):
        chunks = self.chunker.chunk_document(
            text="Contenido de prueba largo para que no sea filtrado por tokens mínimos requeridos en el chunker.",
            document_name="CEIA.pdf",
            document_type="resolucion",
        )
        assert chunks[0].document_name == "CEIA.pdf"
        assert chunks[0].document_type == "resolucion"
        assert chunks[0].chunk_id is not None


class TestMetadataExtractor:
    """Tests de extracción de metadata."""

    def setup_method(self):
        self.extractor = MetadataExtractor()

    def test_extract_email(self):
        text = "Contactar a inscripcion.lse@fi.uba.ar para más info."
        meta = self.extractor.extract_document_metadata("test.pdf", text)
        assert "inscripcion.lse@fi.uba.ar" in meta.contact_emails

    def test_extract_from_known_document(self):
        meta = self.extractor.extract_document_metadata("CEIA.pdf", "contenido")
        assert meta is not None
        assert "CEIA" in meta.program_codes


class TestIntegration:
    """Tests de integración del pipeline."""

    def test_full_pipeline_with_text(self):
        """Prueba el pipeline completo con texto simulado."""
        cleaner = SpanishTextCleaner()
        chunker = DocumentChunker(fixed_chunk_size=100, fixed_overlap=20)

        raw_text = """
        Artículo 1°: La asistencia es obligatoria.
        El porcentaje mínimo es del 75%.

        Artículo 2°: La nota mínima para aprobar es 4 (cuatro).
        Los docentes informarán la modalidad de evaluación.
        """

        cleaned = cleaner.clean(raw_text)
        chunks = chunker.chunk_document(
            text=cleaned.cleaned_text,
            document_name="Reglamento.pdf",
            document_type="reglamento",
        )

        assert len(chunks) >= 1
        assert any("75" in c.text for c in chunks)
