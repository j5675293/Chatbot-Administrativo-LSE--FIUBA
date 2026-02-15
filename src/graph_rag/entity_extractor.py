"""
Extracción de entidades del dominio académico LSE-FIUBA.
Combinación de reglas (regex) y extracción asistida por LLM.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import re
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class EntityType(Enum):
    PROGRAMA = "programa"
    MATERIA = "materia"
    TITULO = "titulo"
    REQUISITO = "requisito"
    PLAZO = "plazo"
    ARTICULO = "articulo"
    CONTACTO = "contacto"
    INSTITUCION = "institucion"
    RESOLUCION = "resolucion"
    MODALIDAD = "modalidad"
    PROCESO = "proceso"


@dataclass
class Entity:
    entity_id: str
    name: str
    entity_type: EntityType
    aliases: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    source_document: str = ""
    source_page: int = 0

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "aliases": self.aliases,
            "properties": self.properties,
            "source_document": self.source_document,
            "source_page": self.source_page,
        }


# ── Definiciones de programas del LSE ─────────────────────────
PROGRAM_DEFINITIONS = {
    "CEIA": {
        "full_name": "Carrera de Especialización en Inteligencia Artificial",
        "aliases": ["CEIA", "Especialización en IA", "Especialización en Inteligencia Artificial"],
        "degree_level": "especializacion",
        "title": "Especialista en Inteligencia Artificial",
    },
    "CESE": {
        "full_name": "Carrera de Especialización en Sistemas Embebidos",
        "aliases": ["CESE", "Especialización en Sistemas Embebidos"],
        "degree_level": "especializacion",
        "title": "Especialista en Sistemas Embebidos",
    },
    "CEIoT": {
        "full_name": "Carrera de Especialización en Internet de las Cosas",
        "aliases": ["CEIoT", "Especialización en IoT", "Especialización en Internet de las Cosas"],
        "degree_level": "especializacion",
        "title": "Especialista en Internet de las Cosas",
    },
    "MIA": {
        "full_name": "Maestría en Inteligencia Artificial",
        "aliases": ["MIA", "Maestría en IA"],
        "degree_level": "maestria",
        "title": "Magíster en Inteligencia Artificial",
    },
    "MIAE": {
        "full_name": "Maestría en Inteligencia Artificial Embebida",
        "aliases": ["MIAE", "Maestría en IA Embebida"],
        "degree_level": "maestria",
        "title": "Magíster en Inteligencia Artificial Embebida",
    },
    "MIoT": {
        "full_name": "Maestría en Internet de las Cosas",
        "aliases": ["MIoT", "Maestría en IoT"],
        "degree_level": "maestria",
        "title": "Magíster en Internet de las Cosas",
    },
    "MCB": {
        "full_name": "Maestría en Ciencia de Datos y Bioestadística",
        "aliases": ["MCB", "Maestría en Ciencia de Datos"],
        "degree_level": "maestria",
        "title": "Magíster en Ciencia de Datos y Bioestadística",
    },
}

# ── Materias conocidas ─────────────────────────────────────────
KNOWN_SUBJECTS = {
    "GdP": {
        "full_name": "Gestión de Proyectos",
        "aliases": ["GdP", "Gestión de Proyectos"],
    },
    "GTI": {
        "full_name": "Gestión de la Tecnología y la Innovación",
        "aliases": ["GTI", "Gestión de la Tecnología"],
    },
    "TTFA": {
        "full_name": "Taller de Trabajo Final A",
        "aliases": ["TTFA", "Taller de Trabajo Final A", "Taller TF A"],
    },
    "TTFB": {
        "full_name": "Taller de Trabajo Final B",
        "aliases": ["TTFB", "Taller de Trabajo Final B", "Taller TF B"],
    },
}


class AcademicEntityExtractor:
    """Extracción de entidades del dominio académico del LSE."""

    PROGRAM_PATTERNS = {
        code: re.compile(
            r"\b(?:" + re.escape(code) + r"|"
            + re.escape(info["full_name"]).replace(r"\ ", r"\s+")
            + r")\b",
            re.IGNORECASE,
        )
        for code, info in PROGRAM_DEFINITIONS.items()
    }

    SUBJECT_PATTERNS = {
        code: re.compile(
            r"\b(?:" + re.escape(code) + r"|"
            + re.escape(info["full_name"]).replace(r"\ ", r"\s+")
            + r")\b",
            re.IGNORECASE,
        )
        for code, info in KNOWN_SUBJECTS.items()
    }

    DEADLINE_PATTERN = re.compile(
        r"(\d+)\s*(?:bimestres?|meses?|a[nñ]os?)\s*(?:corridos?)?",
        re.IGNORECASE,
    )

    EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[\w]+")

    ARTICLE_PATTERN = re.compile(r"Art\.\s*(\d+)", re.IGNORECASE)

    PROCESS_KEYWORDS = {
        "inscripcion": ["inscripción", "inscripcion", "inscribir", "matricula"],
        "baja": ["baja", "desistimiento"],
        "readmision": ["readmisión", "readmision", "reincorporación"],
        "prorroga": ["prórroga", "prorroga", "extensión de plazo"],
        "defensa": ["defensa", "exposición del trabajo"],
        "evaluacion": ["evaluación", "examen", "parcial", "final"],
    }

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        self._seen_entities: dict[str, Entity] = {}

    def extract_entities(self, text: str, document_name: str = "") -> list[Entity]:
        """Extrae todas las entidades del texto."""
        entities = []

        entities.extend(self._extract_programs(text, document_name))
        entities.extend(self._extract_subjects(text, document_name))
        entities.extend(self._extract_deadlines(text, document_name))
        entities.extend(self._extract_contacts(text, document_name))
        entities.extend(self._extract_articles(text, document_name))
        entities.extend(self._extract_processes(text, document_name))
        entities.extend(self._extract_institutions(text, document_name))

        # Deduplicar
        entities = self._deduplicate(entities)

        logger.info(f"Extraídas {len(entities)} entidades de {document_name}")
        return entities

    def _extract_programs(self, text: str, doc_name: str) -> list[Entity]:
        """Extrae programas académicos."""
        entities = []
        for code, pattern in self.PROGRAM_PATTERNS.items():
            if pattern.search(text):
                info = PROGRAM_DEFINITIONS[code]
                entities.append(Entity(
                    entity_id=f"prog_{code}",
                    name=code,
                    entity_type=EntityType.PROGRAMA,
                    aliases=info["aliases"],
                    properties={
                        "full_name": info["full_name"],
                        "degree_level": info["degree_level"],
                        "title": info["title"],
                    },
                    source_document=doc_name,
                ))
        return entities

    def _extract_subjects(self, text: str, doc_name: str) -> list[Entity]:
        """Extrae materias/asignaturas."""
        entities = []
        for code, pattern in self.SUBJECT_PATTERNS.items():
            if pattern.search(text):
                info = KNOWN_SUBJECTS[code]
                entities.append(Entity(
                    entity_id=f"mat_{code}",
                    name=code,
                    entity_type=EntityType.MATERIA,
                    aliases=info["aliases"],
                    properties={"full_name": info["full_name"]},
                    source_document=doc_name,
                ))
        return entities

    def _extract_deadlines(self, text: str, doc_name: str) -> list[Entity]:
        """Extrae plazos y deadlines."""
        entities = []
        for match in self.DEADLINE_PATTERN.finditer(text):
            full_text = match.group(0)
            value = match.group(1)
            entity_id = f"plazo_{value}_{full_text.replace(' ', '_')[:20]}"

            # Evitar duplicados
            if entity_id not in self._seen_entities:
                entity = Entity(
                    entity_id=entity_id,
                    name=full_text.strip(),
                    entity_type=EntityType.PLAZO,
                    properties={
                        "value": int(value),
                        "unit": full_text.replace(value, "").strip(),
                        "context": text[max(0, match.start() - 50):match.end() + 50],
                    },
                    source_document=doc_name,
                )
                entities.append(entity)
                self._seen_entities[entity_id] = entity

        return entities

    def _extract_contacts(self, text: str, doc_name: str) -> list[Entity]:
        """Extrae emails de contacto."""
        entities = []
        for match in self.EMAIL_PATTERN.finditer(text):
            email = match.group(0)
            entity_id = f"contacto_{email.replace('@', '_at_').replace('.', '_')}"
            entities.append(Entity(
                entity_id=entity_id,
                name=email,
                entity_type=EntityType.CONTACTO,
                properties={"email": email},
                source_document=doc_name,
            ))
        return entities

    def _extract_articles(self, text: str, doc_name: str) -> list[Entity]:
        """Extrae artículos del reglamento."""
        entities = []
        for match in self.ARTICLE_PATTERN.finditer(text):
            art_num = match.group(1)
            entity_id = f"art_{art_num}_{doc_name.replace(' ', '_')[:15]}"

            # Extraer contenido del artículo (hasta el siguiente Art. o fin)
            start = match.start()
            next_art = self.ARTICLE_PATTERN.search(text, match.end())
            end = next_art.start() if next_art else min(start + 500, len(text))
            content = text[start:end].strip()

            entities.append(Entity(
                entity_id=entity_id,
                name=f"Art. {art_num}",
                entity_type=EntityType.ARTICULO,
                properties={
                    "number": int(art_num),
                    "content_preview": content[:200],
                    "full_content": content,
                },
                source_document=doc_name,
            ))
        return entities

    def _extract_processes(self, text: str, doc_name: str) -> list[Entity]:
        """Extrae procesos administrativos."""
        entities = []
        text_lower = text.lower()

        for process_name, keywords in self.PROCESS_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    entity_id = f"proc_{process_name}"
                    entities.append(Entity(
                        entity_id=entity_id,
                        name=process_name,
                        entity_type=EntityType.PROCESO,
                        aliases=keywords,
                        source_document=doc_name,
                    ))
                    break

        return entities

    def _extract_institutions(self, text: str, doc_name: str) -> list[Entity]:
        """Extrae instituciones mencionadas."""
        entities = []
        institutions = {
            "UBA": ["Universidad de Buenos Aires", "UBA"],
            "FIUBA": ["Facultad de Ingeniería", "FIUBA"],
            "LSE": ["Laboratorio de Sistemas Embebidos", "LSE"],
        }

        for code, aliases in institutions.items():
            for alias in aliases:
                if alias.lower() in text.lower():
                    entities.append(Entity(
                        entity_id=f"inst_{code}",
                        name=code,
                        entity_type=EntityType.INSTITUCION,
                        aliases=aliases,
                        source_document=doc_name,
                    ))
                    break

        return entities

    def _deduplicate(self, entities: list[Entity]) -> list[Entity]:
        """Elimina entidades duplicadas por entity_id."""
        seen = {}
        for entity in entities:
            if entity.entity_id not in seen:
                seen[entity.entity_id] = entity
        return list(seen.values())
