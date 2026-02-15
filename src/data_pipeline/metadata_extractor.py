"""
Extracción de metadatos para documentos del LSE-FIUBA.
Registry de los 13 documentos + extracción automática de entidades.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import re
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    filename: str
    document_type: str
    program_codes: list[str] = field(default_factory=list)
    program_full_names: list[str] = field(default_factory=list)
    degree_level: str = ""
    topics: list[str] = field(default_factory=list)
    resolution_number: Optional[str] = None
    version_date: Optional[str] = None
    contact_emails: list[str] = field(default_factory=list)
    key_entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "document_type": self.document_type,
            "program_codes": self.program_codes,
            "program_full_names": self.program_full_names,
            "degree_level": self.degree_level,
            "topics": self.topics,
            "resolution_number": self.resolution_number,
            "version_date": self.version_date,
            "contact_emails": self.contact_emails,
            "key_entities": self.key_entities,
        }


# ── Registry de los 13 documentos del LSE ────────────────────────
DOCUMENT_REGISTRY: dict[str, dict] = {
    "CEIA.pdf": {
        "program_codes": ["CEIA"],
        "program_full_names": ["Carrera de Especialización en Inteligencia Artificial"],
        "degree_level": "especializacion",
        "document_type": "resolucion",
        "topics": ["plan_de_estudios", "requisitos", "titulo", "objetivos"],
    },
    "CEIoT.pdf": {
        "program_codes": ["CEIoT"],
        "program_full_names": ["Carrera de Especialización en Internet de las Cosas"],
        "degree_level": "especializacion",
        "document_type": "resolucion",
        "topics": ["plan_de_estudios", "requisitos", "titulo", "objetivos"],
    },
    "CESE.pdf": {
        "program_codes": ["CESE"],
        "program_full_names": ["Carrera de Especialización en Sistemas Embebidos"],
        "degree_level": "especializacion",
        "document_type": "resolucion",
        "topics": ["plan_de_estudios", "requisitos", "titulo", "objetivos"],
    },
    "MCB.pdf": {
        "program_codes": ["MCB"],
        "program_full_names": ["Maestría en Ciencia de Datos y Bioestadística"],
        "degree_level": "maestria",
        "document_type": "resolucion",
        "topics": ["plan_de_estudios", "requisitos", "titulo", "objetivos", "tesis"],
    },
    "MIAE.pdf": {
        "program_codes": ["MIAE"],
        "program_full_names": ["Maestría en Inteligencia Artificial Embebida"],
        "degree_level": "maestria",
        "document_type": "resolucion",
        "topics": ["plan_de_estudios", "requisitos", "titulo", "objetivos", "tesis"],
    },
    "MIoT.pdf": {
        "program_codes": ["MIoT"],
        "program_full_names": ["Maestría en Internet de las Cosas"],
        "degree_level": "maestria",
        "document_type": "resolucion",
        "topics": ["plan_de_estudios", "requisitos", "titulo", "objetivos", "tesis"],
    },
    "FAQ - MIA.pdf": {
        "program_codes": ["MIA", "CEIA"],
        "program_full_names": [
            "Maestría en Inteligencia Artificial",
            "Carrera de Especialización en Inteligencia Artificial",
        ],
        "degree_level": "maestria",
        "document_type": "faq",
        "topics": [
            "inscripcion", "requisitos", "materias_optativas",
            "trabajo_final", "plazos", "preguntas_generales",
        ],
    },
    "FAQ - GdP, GTI, TTFA, TTFB.pdf": {
        "program_codes": ["ALL"],
        "program_full_names": ["Todas las carreras del LSE"],
        "degree_level": "all",
        "document_type": "faq",
        "topics": [
            "gestion_proyectos", "trabajo_final", "inscripcion",
            "talleres", "gdp", "gti", "ttfa", "ttfb",
        ],
    },
    "FAQ - Materias optativas.pdf": {
        "program_codes": ["ALL"],
        "program_full_names": ["Todas las carreras del LSE"],
        "degree_level": "all",
        "document_type": "faq",
        "topics": ["materias_optativas"],
    },
    "Reglamento de Cursada y Asistencia de los posgrados del LSE - 2025B3 (1) (1).pdf": {
        "program_codes": ["ALL"],
        "program_full_names": ["Todas las carreras del LSE"],
        "degree_level": "all",
        "document_type": "reglamento",
        "topics": [
            "asistencia", "calificacion", "plazos", "trabajo_final",
            "readmision", "baja", "aplazos", "normas_convivencia",
            "evaluacion", "examenes", "prorroga",
        ],
        "version": "junio/2025",
    },
    "LSE-FIUBA-Trabajo-Final.pdf": {
        "program_codes": ["ALL"],
        "program_full_names": ["Todas las carreras del LSE"],
        "degree_level": "all",
        "document_type": "reglamento",
        "topics": ["trabajo_final", "director", "jurado", "defensa"],
    },
    "PROGRAMA DE VINCULACIÓN.pdf": {
        "program_codes": ["ALL"],
        "program_full_names": ["Todas las carreras del LSE"],
        "degree_level": "all",
        "document_type": "programa",
        "topics": ["vinculacion", "empresas"],
    },
    "MIA-AE1-Programa.pdf": {
        "program_codes": ["MIA"],
        "program_full_names": ["Maestría en Inteligencia Artificial"],
        "degree_level": "maestria",
        "document_type": "programa",
        "topics": ["programa_materia", "contenidos", "bibliografia"],
    },
}

# Variantes de nombres de archivos para matching flexible
FILENAME_ALIASES: dict[str, str] = {
    "reglamento": "Reglamento de Cursada y Asistencia de los posgrados del LSE - 2025B3 (1) (1).pdf",
    "programa de vinculacion": "PROGRAMA DE VINCULACIÓN.pdf",
    "vinculacion": "PROGRAMA DE VINCULACIÓN.pdf",
}


class MetadataExtractor:
    """Extrae y enriquece metadatos para documentos y chunks."""

    # Patrones de programas
    PROGRAM_PATTERNS = {
        "CEIA": re.compile(
            r"\b(?:CEIA|Carrera\s+de\s+Especializaci[oó]n\s+en\s+Inteligencia\s+Artificial)\b",
            re.IGNORECASE,
        ),
        "CESE": re.compile(
            r"\b(?:CESE|Carrera\s+de\s+Especializaci[oó]n\s+en\s+Sistemas\s+Embebidos)\b",
            re.IGNORECASE,
        ),
        "CEIoT": re.compile(
            r"\b(?:CEIoT|Carrera\s+de\s+Especializaci[oó]n\s+en\s+Internet\s+de\s+las\s+Cosas)\b",
            re.IGNORECASE,
        ),
        "MIA": re.compile(
            r"\b(?:MIA|Maestr[ií]a\s+en\s+Inteligencia\s+Artificial)\b",
            re.IGNORECASE,
        ),
        "MIAE": re.compile(
            r"\b(?:MIAE|Maestr[ií]a\s+en\s+Inteligencia\s+Artificial\s+Embebida)\b",
            re.IGNORECASE,
        ),
        "MIoT": re.compile(
            r"\b(?:MIoT|Maestr[ií]a\s+en\s+Internet\s+de\s+las\s+Cosas)\b",
            re.IGNORECASE,
        ),
        "MCB": re.compile(
            r"\b(?:MCB|Maestr[ií]a\s+en\s+Ciencia\s+de\s+Datos)\b",
            re.IGNORECASE,
        ),
    }

    # Patrones de temas
    TOPIC_KEYWORDS = {
        "inscripcion": ["inscripci", "inscribi", "admisi", "postula"],
        "requisitos": ["requisit", "necesit", "condici", "requiere"],
        "plazos": ["plazo", "vencimient", "fecha l[ií]mite", "pr[oó]rroga"],
        "trabajo_final": ["trabajo final", "tesis", "defensa", "director", "jurado"],
        "materias_optativas": ["optativa", "electiva"],
        "asistencia": ["asistencia", "inasistencia", "ausenci"],
        "calificacion": ["calificaci", "nota", "aprobaci", "desaprobaci", "aplazo"],
        "baja": ["baja", "desistimiento"],
        "readmision": ["readmisi", "reincorpor"],
        "plan_de_estudios": ["plan de estudio", "plan de la carrera", "estructura curricular"],
        "correlatividades": ["correlativa", "prerrequisit"],
        "gestion_proyectos": ["gesti[oó]n de proyectos", "GdP"],
        "vinculacion": ["vinculaci", "empresa", "industria"],
    }

    EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[\w]+")
    RESOLUTION_PATTERN = re.compile(r"RESCS-\d{4}-\d+-E-UBA-REC")

    def __init__(self):
        self.registry = DOCUMENT_REGISTRY

    def extract_document_metadata(self, filename: str, text: str) -> DocumentMetadata:
        """Combina datos del registry con extracción del contenido."""
        # Buscar en registry (matching flexible)
        registry_data = self._find_in_registry(filename)

        metadata = DocumentMetadata(
            filename=filename,
            document_type=registry_data.get("document_type", "desconocido"),
            program_codes=registry_data.get("program_codes", []),
            program_full_names=registry_data.get("program_full_names", []),
            degree_level=registry_data.get("degree_level", ""),
            topics=registry_data.get("topics", []),
            version_date=registry_data.get("version", None),
        )

        # Enriquecer con extracción del contenido
        metadata.contact_emails = self._extract_emails(text)
        metadata.resolution_number = self._extract_resolution(text)

        # Agregar topics detectados en texto
        detected_topics = self._extract_topics(text)
        for topic in detected_topics:
            if topic not in metadata.topics:
                metadata.topics.append(topic)

        # Agregar programas mencionados en texto
        detected_programs = self._extract_program_references(text)
        for prog in detected_programs:
            if prog not in metadata.program_codes:
                metadata.program_codes.append(prog)

        return metadata

    def extract_chunk_metadata(
        self, chunk_text: str, doc_metadata: DocumentMetadata
    ) -> dict:
        """Enriquece metadata individual de un chunk."""
        chunk_meta = {
            "document_type": doc_metadata.document_type,
            "program_codes": list(doc_metadata.program_codes),
            "degree_level": doc_metadata.degree_level,
        }

        # Topics específicos del chunk
        chunk_topics = self._extract_topics(chunk_text)
        chunk_meta["topics"] = chunk_topics

        # Programas mencionados en el chunk
        chunk_programs = self._extract_program_references(chunk_text)
        if chunk_programs:
            chunk_meta["mentioned_programs"] = chunk_programs

        # Emails en el chunk
        emails = self._extract_emails(chunk_text)
        if emails:
            chunk_meta["contact_emails"] = emails

        return chunk_meta

    def _find_in_registry(self, filename: str) -> dict:
        """Busca el archivo en el registry con matching flexible."""
        # Búsqueda exacta
        if filename in self.registry:
            return self.registry[filename]

        # Búsqueda por nombre parcial
        filename_lower = filename.lower()
        for reg_name, data in self.registry.items():
            if reg_name.lower() in filename_lower or filename_lower in reg_name.lower():
                return data

        # Búsqueda por alias
        for alias, canonical in FILENAME_ALIASES.items():
            if alias in filename_lower:
                return self.registry.get(canonical, {})

        logger.warning(f"Documento no encontrado en registry: {filename}")
        return {}

    def _extract_emails(self, text: str) -> list[str]:
        """Extrae direcciones de email."""
        return list(set(self.EMAIL_PATTERN.findall(text)))

    def _extract_resolution(self, text: str) -> Optional[str]:
        """Extrae número de resolución."""
        match = self.RESOLUTION_PATTERN.search(text)
        return match.group(0) if match else None

    def _extract_topics(self, text: str) -> list[str]:
        """Detecta topics por keywords en el texto."""
        text_lower = text.lower()
        detected = []
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            for kw in keywords:
                if re.search(kw, text_lower):
                    detected.append(topic)
                    break
        return detected

    def _extract_program_references(self, text: str) -> list[str]:
        """Detecta códigos de programas mencionados."""
        found = []
        for code, pattern in self.PROGRAM_PATTERNS.items():
            if pattern.search(text):
                found.append(code)
        return found
