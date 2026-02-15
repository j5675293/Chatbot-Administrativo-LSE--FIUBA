"""
Mapeo de relaciones entre entidades del dominio académico LSE.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.graph_rag.entity_extractor import Entity, EntityType

logger = logging.getLogger(__name__)


class RelationType(Enum):
    REQUIERE_EGRESO_DE = "requiere_egreso_de"
    COMBINA_CON = "combina_con"
    PERTENECE_A = "pertenece_a"
    OTORGA_TITULO = "otorga_titulo"
    ES_CORRELATIVA_DE = "es_correlativa_de"
    ES_REQUISITO_PARA = "es_requisito_para"
    SE_DICTA_EN = "se_dicta_en"
    REGULA = "regula"
    TIENE_PLAZO = "tiene_plazo"
    APLICA_A = "aplica_a"
    CONTACTAR_PARA = "contactar_para"
    DOCUMENTADO_EN = "documentado_en"


@dataclass
class Relationship:
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    properties: dict = field(default_factory=dict)
    source_text: str = ""

    def to_dict(self) -> dict:
        return {
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relation_type": self.relation_type.value,
            "properties": self.properties,
            "source_text": self.source_text[:200],
        }


class RelationshipMapper:
    """Mapea relaciones entre entidades extraídas.

    Combina relaciones conocidas del dominio con extracción basada en reglas.
    """

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider

    def extract_relationships(
        self,
        text: str,
        entities: list[Entity],
        document_name: str = "",
    ) -> list[Relationship]:
        """Extrae relaciones del texto y agrega relaciones conocidas."""
        relationships = []

        # 1. Relaciones conocidas (hardcoded del dominio)
        relationships.extend(self._get_known_relationships(entities))

        # 2. Relaciones extraídas del texto por reglas
        relationships.extend(self._rule_based_extraction(text, entities))

        # Deduplicar
        relationships = self._deduplicate(relationships)

        logger.info(f"Extraídas {len(relationships)} relaciones de {document_name}")
        return relationships

    def _get_known_relationships(self, entities: list[Entity]) -> list[Relationship]:
        """Relaciones conocidas del dominio LSE-FIUBA."""
        rels = []
        entity_ids = {e.entity_id for e in entities}

        # ── Prerequisitos de maestrías ─────────────────────────
        # MIA requiere egreso de CEIA
        if "prog_MIA" in entity_ids and "prog_CEIA" in entity_ids:
            rels.append(Relationship(
                source_entity_id="prog_MIA",
                target_entity_id="prog_CEIA",
                relation_type=RelationType.REQUIERE_EGRESO_DE,
                properties={"weight": 1.0},
                source_text="La MIA requiere haber egresado de la CEIA",
            ))

        # MIAE combina CEIA + CESE
        if "prog_MIAE" in entity_ids:
            if "prog_CEIA" in entity_ids:
                rels.append(Relationship(
                    source_entity_id="prog_MIAE",
                    target_entity_id="prog_CEIA",
                    relation_type=RelationType.COMBINA_CON,
                    properties={"weight": 1.0},
                    source_text="La MIAE combina la CEIA y la CESE",
                ))
            if "prog_CESE" in entity_ids:
                rels.append(Relationship(
                    source_entity_id="prog_MIAE",
                    target_entity_id="prog_CESE",
                    relation_type=RelationType.COMBINA_CON,
                    properties={"weight": 1.0},
                    source_text="La MIAE combina la CEIA y la CESE",
                ))

        # MIoT requiere egreso de CEIoT
        if "prog_MIoT" in entity_ids and "prog_CEIoT" in entity_ids:
            rels.append(Relationship(
                source_entity_id="prog_MIoT",
                target_entity_id="prog_CEIoT",
                relation_type=RelationType.REQUIERE_EGRESO_DE,
                properties={"weight": 1.0},
                source_text="La MIoT requiere haber egresado de la CEIoT",
            ))

        # ── Cadena GdP -> TTFA -> TTFB ────────────────────────
        if "mat_GdP" in entity_ids and "mat_TTFA" in entity_ids:
            rels.append(Relationship(
                source_entity_id="mat_TTFA",
                target_entity_id="mat_GdP",
                relation_type=RelationType.ES_REQUISITO_PARA,
                properties={"weight": 1.0},
                source_text="Para inscribirse en TTFA es necesario haber aprobado GdP",
            ))

        if "mat_TTFA" in entity_ids and "mat_TTFB" in entity_ids:
            rels.append(Relationship(
                source_entity_id="mat_TTFB",
                target_entity_id="mat_TTFA",
                relation_type=RelationType.ES_REQUISITO_PARA,
                properties={"weight": 1.0},
                source_text="Para inscribirse en TTFB es necesario haber aprobado TTFA",
            ))

        # ── Materias pertenecen a todas las carreras ───────────
        for mat_id in ["mat_GdP", "mat_GTI", "mat_TTFA", "mat_TTFB"]:
            if mat_id in entity_ids:
                for prog_id in ["prog_CEIA", "prog_CESE", "prog_CEIoT"]:
                    if prog_id in entity_ids:
                        rels.append(Relationship(
                            source_entity_id=mat_id,
                            target_entity_id=prog_id,
                            relation_type=RelationType.PERTENECE_A,
                            properties={"weight": 0.8},
                        ))

        # ── Plazos por nivel de carrera ────────────────────────
        for entity in entities:
            if entity.entity_type == EntityType.PROGRAMA:
                degree = entity.properties.get("degree_level", "")
                if degree == "especializacion":
                    rels.append(Relationship(
                        source_entity_id=entity.entity_id,
                        target_entity_id="plazo_10_bimestres",
                        relation_type=RelationType.TIENE_PLAZO,
                        properties={"plazo": "10 bimestres (2 años corridos)"},
                        source_text="Las especializaciones tienen un plazo de 10 bimestres",
                    ))
                elif degree == "maestria":
                    rels.append(Relationship(
                        source_entity_id=entity.entity_id,
                        target_entity_id="plazo_maestria",
                        relation_type=RelationType.TIENE_PLAZO,
                        properties={"plazo": "2+2 años"},
                        source_text="Las maestrías tienen un plazo de 2 años + 2 años para tesis",
                    ))

        # ── Títulos que otorga cada programa ───────────────────
        for entity in entities:
            if entity.entity_type == EntityType.PROGRAMA:
                title = entity.properties.get("title", "")
                if title:
                    rels.append(Relationship(
                        source_entity_id=entity.entity_id,
                        target_entity_id=f"titulo_{entity.name}",
                        relation_type=RelationType.OTORGA_TITULO,
                        properties={"titulo": title},
                    ))

        # ── Instituciones ──────────────────────────────────────
        if "inst_LSE" in entity_ids and "inst_FIUBA" in entity_ids:
            rels.append(Relationship(
                source_entity_id="inst_LSE",
                target_entity_id="inst_FIUBA",
                relation_type=RelationType.PERTENECE_A,
            ))
        if "inst_FIUBA" in entity_ids and "inst_UBA" in entity_ids:
            rels.append(Relationship(
                source_entity_id="inst_FIUBA",
                target_entity_id="inst_UBA",
                relation_type=RelationType.PERTENECE_A,
            ))

        return rels

    def _rule_based_extraction(
        self, text: str, entities: list[Entity]
    ) -> list[Relationship]:
        """Extracción de relaciones por patrones regex."""
        rels = []

        # Patrón: "para X es necesario/se requiere Y"
        req_patterns = [
            re.compile(
                r"para\s+(?:inscribir|cursar|aprobar)\s+(.+?)\s+(?:es necesario|necesit[aá]s?|deb[eé]s?|se requiere)\s+(.+?)(?:\.|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:requisito|condici[oó]n)\s+para\s+(.+?):\s*(.+?)(?:\.|$)",
                re.IGNORECASE,
            ),
        ]

        entity_names = {e.name.lower(): e for e in entities}
        for alias_list in [e.aliases for e in entities]:
            for alias in alias_list:
                matching = [e for e in entities if alias in e.aliases]
                if matching:
                    entity_names[alias.lower()] = matching[0]

        for pattern in req_patterns:
            for match in pattern.finditer(text):
                source_text = match.group(1).strip()
                target_text = match.group(2).strip()

                source_entity = self._find_entity_in_text(source_text, entity_names)
                target_entity = self._find_entity_in_text(target_text, entity_names)

                if source_entity and target_entity:
                    rels.append(Relationship(
                        source_entity_id=source_entity.entity_id,
                        target_entity_id=target_entity.entity_id,
                        relation_type=RelationType.ES_REQUISITO_PARA,
                        source_text=match.group(0)[:200],
                    ))

        # Patrón: "Art. N regula/establece X"
        for entity in entities:
            if entity.entity_type == EntityType.ARTICULO:
                content = entity.properties.get("content_preview", "").lower()
                for process_entity in entities:
                    if process_entity.entity_type == EntityType.PROCESO:
                        for alias in process_entity.aliases:
                            if alias.lower() in content:
                                rels.append(Relationship(
                                    source_entity_id=entity.entity_id,
                                    target_entity_id=process_entity.entity_id,
                                    relation_type=RelationType.REGULA,
                                ))
                                break

        return rels

    def _find_entity_in_text(
        self, text: str, entity_names: dict[str, Entity]
    ) -> Optional[Entity]:
        """Busca una entidad mencionada en un fragmento de texto."""
        text_lower = text.lower()
        for name, entity in entity_names.items():
            if name in text_lower:
                return entity
        return None

    def _deduplicate(self, relationships: list[Relationship]) -> list[Relationship]:
        """Elimina relaciones duplicadas."""
        seen = set()
        unique = []
        for rel in relationships:
            key = (rel.source_entity_id, rel.target_entity_id, rel.relation_type.value)
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        return unique
