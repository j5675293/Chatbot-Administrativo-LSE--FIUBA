"""
Extracción de texto y tablas desde PDFs.
Usa PyMuPDF (fitz) para texto y pdfplumber para tablas.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import re
import logging

import fitz  # PyMuPDF
import pdfplumber

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    RESOLUTION = "resolucion"
    FAQ = "faq"
    REGULATION = "reglamento"
    PROGRAM = "programa"
    VINCULACION = "vinculacion"


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    has_header: bool = False
    has_footer: bool = False


@dataclass
class ExtractedDocument:
    filename: str
    filepath: str
    document_type: DocumentType
    total_pages: int
    pages: list[ExtractedPage] = field(default_factory=list)
    raw_text: str = ""
    tables: list[list[list[str]]] = field(default_factory=list)
    extraction_metadata: dict = field(default_factory=dict)


class PDFExtractor:
    """Extracción dual de PDFs: PyMuPDF para texto, pdfplumber para tablas."""

    # Patrones de headers/footers institucionales
    HEADER_PATTERNS = [
        r"EX-\d{4}-\d+-.*?-UBA-.*?#.*?(?:_FI)?",
        r"ACS-\d{4}-\d+-E-UBA-SG#REC?",
        r"RESCS-\d{4}-\d+-E-UBA-REC",
        r"IF-\d{4}-\d+-.*?-UBA-.*",
        r"P[aá]gina\s+\d+\s+de\s+\d+",
        r"N[uú]mero:\s*\w+-\d+-\d+",
        r"CIUDAD\s+(?:AUT[OÓ]NOMA\s+)?DE\s+BUENOS\s+AIRES",
    ]

    FOOTER_PATTERNS = [
        r"Digitally signed by.*",
        r"Date:.*\d{4}\.\d{2}\.\d{2}.*",
        r"^.{0,5}UBA\s*fiuba.{0,5}$",
    ]

    def __init__(self, fallback_on_error: bool = True):
        self.fallback_on_error = fallback_on_error
        self._header_re = [re.compile(p, re.IGNORECASE) for p in self.HEADER_PATTERNS]
        self._footer_re = [re.compile(p, re.IGNORECASE) for p in self.FOOTER_PATTERNS]

    def extract(self, filepath: Path) -> ExtractedDocument:
        """Extrae texto y tablas de un archivo PDF."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"PDF no encontrado: {filepath}")

        logger.info(f"Extrayendo: {filepath.name}")

        # Extraer texto con PyMuPDF
        pages = self._extract_with_pymupdf(filepath)

        # Extraer tablas con pdfplumber
        all_tables = self._extract_tables_with_pdfplumber(filepath)

        # Combinar tablas en las páginas correspondientes
        for table_info in all_tables:
            page_num = table_info["page"]
            if page_num < len(pages):
                pages[page_num].tables.append(table_info["data"])

        # Texto completo
        raw_text = "\n\n".join(p.text for p in pages if p.text.strip())

        # Detectar tipo de documento
        doc_type = self._detect_document_type(filepath.name, raw_text)

        # Limpiar headers/footers de cada página
        for page in pages:
            page.text = self._remove_headers_footers(page.text, doc_type)

        # Recopilar todas las tablas
        doc_tables = []
        for page in pages:
            doc_tables.extend(page.tables)

        return ExtractedDocument(
            filename=filepath.name,
            filepath=str(filepath),
            document_type=doc_type,
            total_pages=len(pages),
            pages=pages,
            raw_text=raw_text,
            tables=doc_tables,
            extraction_metadata={
                "extractor": "pymupdf+pdfplumber",
                "tables_found": len(doc_tables),
                "pages_with_text": sum(1 for p in pages if p.text.strip()),
            },
        )

    def _extract_with_pymupdf(self, filepath: Path) -> list[ExtractedPage]:
        """Extracción primaria con PyMuPDF."""
        pages = []
        try:
            doc = fitz.open(str(filepath))
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")

                has_header = any(p.search(text[:200]) for p in self._header_re)
                has_footer = any(p.search(text[-200:]) for p in self._footer_re)

                pages.append(ExtractedPage(
                    page_number=page_num + 1,
                    text=text,
                    has_header=has_header,
                    has_footer=has_footer,
                ))
            doc.close()
        except Exception as e:
            logger.error(f"Error PyMuPDF en {filepath.name}: {e}")
            if not self.fallback_on_error:
                raise

        return pages

    def _extract_tables_with_pdfplumber(self, filepath: Path) -> list[dict]:
        """Extrae tablas usando pdfplumber (plan de estudios, etc.)."""
        tables = []
        try:
            with pdfplumber.open(str(filepath)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table in page_tables:
                            # Limpiar celdas None
                            cleaned = []
                            for row in table:
                                cleaned_row = [
                                    (cell.strip() if cell else "")
                                    for cell in row
                                ]
                                cleaned.append(cleaned_row)
                            tables.append({
                                "page": page_num,
                                "data": cleaned,
                            })
        except Exception as e:
            logger.warning(f"Error pdfplumber en {filepath.name}: {e}")

        return tables

    def _detect_document_type(self, filename: str, text: str) -> DocumentType:
        """Clasifica el tipo de documento por nombre y contenido."""
        filename_lower = filename.lower()

        if "faq" in filename_lower:
            return DocumentType.FAQ

        if "reglamento" in filename_lower:
            return DocumentType.REGULATION

        if "vinculaci" in filename_lower or "vinculacion" in filename_lower:
            return DocumentType.VINCULACION

        if "programa" in filename_lower and "vinculaci" not in filename_lower:
            return DocumentType.PROGRAM

        # Detectar resoluciones por contenido
        text_upper = text[:2000].upper()
        if "RESOLUCI" in text_upper or "RESCS-" in text_upper or "VISTO" in text_upper:
            return DocumentType.RESOLUTION

        # Detectar reglamento por estructura de artículos
        if re.search(r"Art[\.\s]*\d+", text[:3000]):
            return DocumentType.REGULATION

        # Detectar programa por contenido
        if "plan de estudios" in text[:3000].lower() or "programa" in filename_lower:
            return DocumentType.PROGRAM

        return DocumentType.RESOLUTION  # Default

    def _remove_headers_footers(self, text: str, doc_type: DocumentType) -> str:
        """Elimina headers y footers institucionales recurrentes."""
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append(line)
                continue

            is_header_footer = False
            for pattern in self._header_re + self._footer_re:
                if pattern.search(stripped):
                    is_header_footer = True
                    break

            if not is_header_footer:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)
