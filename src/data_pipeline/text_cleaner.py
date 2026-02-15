"""
Limpieza y normalización de texto en español para documentos académicos.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import re
import unicodedata
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CleaningResult:
    original_text: str
    cleaned_text: str
    changes_log: list[str] = field(default_factory=list)


class SpanishTextCleaner:
    """Normalización de texto para documentos académicos/administrativos en español."""

    # Headers institucionales
    INSTITUTIONAL_HEADERS = [
        r"UNIVERSIDAD\s+DE\s+BUENOS\s+AIRES",
        r"FACULTAD\s+DE\s+INGENIER[IÍ]A",
        r"Laboratorio\s+de\s+Sistemas\s+Embebidos",
        r"Secretar[ií]a\s+de\s+Posgrado",
        r"POSGRADOS?\s+DEL\s+LABORATORIO",
    ]

    # IDs de documentos
    DOCUMENT_ID_PATTERNS = [
        r"EX-\d{4}-\d+-\S+-UBA-\S+",
        r"ACS-\d{4}-\d+-E-UBA-SG#REC?",
        r"RESCS-\d{4}-\d+-E-UBA-REC",
        r"IF-\d{4}-\d+-\S+-UBA-\S+",
        r"NO-\d{4}-\d+-\S+-UBA-\S+",
    ]

    # Números de página
    PAGE_PATTERNS = [
        r"P[aá]gina\s+\d+\s+de\s+\d+",
        r"[-–—]\s*\d+\s*[-–—]",
        r"^\s*\d+\s*$",
    ]

    def __init__(self):
        self._institutional_re = [
            re.compile(p, re.IGNORECASE) for p in self.INSTITUTIONAL_HEADERS
        ]
        self._docid_re = [re.compile(p) for p in self.DOCUMENT_ID_PATTERNS]
        self._page_re = [re.compile(p, re.MULTILINE) for p in self.PAGE_PATTERNS]

    def clean(self, text: str, document_type: str = None) -> CleaningResult:
        """Pipeline completo de limpieza."""
        changes = []
        original = text

        # 1. Normalizar Unicode
        text = self._normalize_unicode(text)
        if text != original:
            changes.append("unicode_normalized")

        # 2. Corregir artefactos de encoding
        prev = text
        text = self._fix_encoding_artifacts(text)
        if text != prev:
            changes.append("encoding_fixed")

        # 3. Eliminar headers institucionales
        prev = text
        text = self._remove_institutional_headers(text)
        if text != prev:
            changes.append("headers_removed")

        # 4. Eliminar IDs de documentos
        prev = text
        text = self._remove_document_ids(text)
        if text != prev:
            changes.append("doc_ids_removed")

        # 5. Eliminar números de página
        prev = text
        text = self._remove_page_numbers(text)
        if text != prev:
            changes.append("page_numbers_removed")

        # 6. Corregir hyphenation de fin de línea
        prev = text
        text = self._fix_hyphenation(text)
        if text != prev:
            changes.append("hyphenation_fixed")

        # 7. Normalizar bullets
        prev = text
        text = self._normalize_bullets(text)
        if text != prev:
            changes.append("bullets_normalized")

        # 8. Preservar marcadores de estructura
        text = self._preserve_structure_markers(text)

        # 9. Normalizar whitespace (último paso)
        text = self._normalize_whitespace(text)

        return CleaningResult(
            original_text=original,
            cleaned_text=text.strip(),
            changes_log=changes,
        )

    def _normalize_unicode(self, text: str) -> str:
        """Normaliza caracteres Unicode preservando acentos españoles."""
        # NFC: composed form (é en vez de e + combining accent)
        text = unicodedata.normalize("NFC", text)
        return text

    def _fix_encoding_artifacts(self, text: str) -> str:
        """Corrige artefactos comunes de encoding PDF."""
        replacements = {
            "\u00c3\u00a1": "á",
            "\u00c3\u00a9": "é",
            "\u00c3\u00ad": "í",
            "\u00c3\u00b3": "ó",
            "\u00c3\u00ba": "ú",
            "\u00c3\u00b1": "ñ",
            "\u00c3\u00bc": "ü",
            "\u00c3\u0081": "Á",
            "\u00c3\u0089": "É",
            "\u00c3\u008d": "Í",
            "\u00c3\u0093": "Ó",
            "\u00c3\u009a": "Ú",
            "\u00c3\u0091": "Ñ",
            "\uf0b7": "•",  # Bullet point
            "\uf0a7": "•",
            "\u2022": "•",
            "\u2013": "–",  # En dash
            "\u2014": "—",  # Em dash
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\xa0": " ",    # Non-breaking space
            "\xad": "",     # Soft hyphen
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _remove_institutional_headers(self, text: str) -> str:
        """Elimina headers institucionales repetitivos."""
        for pattern in self._institutional_re:
            text = pattern.sub("", text)
        return text

    def _remove_document_ids(self, text: str) -> str:
        """Elimina IDs de referencia de documentos oficiales."""
        for pattern in self._docid_re:
            text = pattern.sub("", text)
        return text

    def _remove_page_numbers(self, text: str) -> str:
        """Elimina indicadores de número de página."""
        for pattern in self._page_re:
            text = pattern.sub("", text)
        return text

    def _fix_hyphenation(self, text: str) -> str:
        """Reúne palabras cortadas por guión de fin de línea."""
        # Patrón: palabra- \n continuación (sin mayúscula = no es inicio de oración)
        text = re.sub(
            r"(\w+)-\s*\n\s*([a-záéíóúñü])",
            r"\1\2",
            text,
        )
        return text

    def _normalize_bullets(self, text: str) -> str:
        """Estandariza caracteres de viñetas."""
        # Diferentes tipos de bullets a formato estándar
        text = re.sub(r"^[\s]*[►▪▸‣⁃◦●○■□–—]\s*", "• ", text, flags=re.MULTILINE)
        return text

    def _preserve_structure_markers(self, text: str) -> str:
        """Asegura que marcadores de estructura sean consistentes."""
        # Normalizar "Art." / "Artículo" / "ARTICULO"
        text = re.sub(
            r"(?:ART[IÍ]CULO|ARTICULO|Art\.?)\s*(\d+)",
            r"\nArt. \1",
            text,
            flags=re.IGNORECASE,
        )
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Colapsa espacios múltiples preservando saltos de párrafo."""
        # Reemplazar tabs por espacios
        text = text.replace("\t", " ")
        # Colapsar espacios horizontales múltiples
        text = re.sub(r"[^\S\n]+", " ", text)
        # Colapsar más de 2 newlines a 2 (párrafo)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Eliminar espacios al inicio/fin de cada línea
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        return text
