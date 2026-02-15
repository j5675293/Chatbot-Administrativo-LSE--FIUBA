"""
Retrieval basado en grafo de conocimiento.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from difflib import SequenceMatcher

import networkx as nx

from src.graph_rag.entity_extractor import AcademicEntityExtractor, PROGRAM_DEFINITIONS, KNOWN_SUBJECTS
from src.graph_rag.graph_builder import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)


@dataclass
class GraphSearchResult:
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    subgraph_text: str = ""
    community_id: Optional[int] = None
    path_description: Optional[str] = None
    confidence: float = 0.0


class GraphRetriever:
    """Retrieval desde el grafo de conocimiento."""

    def __init__(
        self,
        graph_builder: KnowledgeGraphBuilder,
        entity_extractor: Optional[AcademicEntityExtractor] = None,
    ):
        self.graph_builder = graph_builder
        self.graph = graph_builder.graph
        self.entity_extractor = entity_extractor or AcademicEntityExtractor()

    def retrieve(self, query: str, top_k: int = 5) -> list[GraphSearchResult]:
        """Pipeline de retrieval por grafo."""
        results = []

        # 1. Identificar entidades en la query
        matched_nodes = self._match_query_entities(query)

        if not matched_nodes:
            logger.info(f"GraphRetriever: sin entidades encontradas en '{query[:60]}'")
            return []

        logger.info(f"GraphRetriever: nodos coincidentes: {matched_nodes}")

        # 2. Para cada entidad encontrada, expandir subgrafo
        for node_id in matched_nodes[:top_k]:
            subgraph = self.graph_builder.get_subgraph(node_id, depth=2)

            if subgraph.number_of_nodes() == 0:
                continue

            # 3. Generar contexto textual
            subgraph_text = self._subgraph_to_text(subgraph, node_id)

            # 4. Extraer entidades y relaciones del subgrafo
            entities_info = []
            for nid, data in subgraph.nodes(data=True):
                entities_info.append({
                    "id": nid,
                    "name": data.get("name", nid),
                    "type": data.get("entity_type", ""),
                    "properties": data.get("properties", {}),
                })

            rels_info = []
            for u, v, data in subgraph.edges(data=True):
                rels_info.append({
                    "source": self.graph.nodes[u].get("name", u),
                    "target": self.graph.nodes[v].get("name", v),
                    "type": data.get("relation_type", ""),
                    "source_text": data.get("source_text", ""),
                })

            # 5. Buscar paths entre entidades si hay más de una
            path_desc = None
            if len(matched_nodes) >= 2:
                path_desc = self._find_relevant_paths(
                    matched_nodes[0], matched_nodes[1]
                )

            confidence = min(len(entities_info) / 5.0, 1.0)

            results.append(GraphSearchResult(
                entities=entities_info,
                relationships=rels_info,
                subgraph_text=subgraph_text,
                path_description=path_desc,
                confidence=confidence,
            ))

        return results

    def _match_query_entities(self, query: str) -> list[str]:
        """Mapea términos de la query a nodos del grafo."""
        matched = []
        query_lower = query.lower()

        # Buscar programas por código y nombre
        for code, info in PROGRAM_DEFINITIONS.items():
            if code.lower() in query_lower:
                node_id = f"prog_{code}"
                if node_id in self.graph:
                    matched.append(node_id)
                continue
            for alias in info["aliases"]:
                if alias.lower() in query_lower:
                    node_id = f"prog_{code}"
                    if node_id in self.graph:
                        matched.append(node_id)
                    break

        # Buscar materias
        for code, info in KNOWN_SUBJECTS.items():
            if code.lower() in query_lower:
                node_id = f"mat_{code}"
                if node_id in self.graph:
                    matched.append(node_id)
                continue
            if info["full_name"].lower() in query_lower:
                node_id = f"mat_{code}"
                if node_id in self.graph:
                    matched.append(node_id)

        # Buscar procesos
        process_keywords = {
            "inscripcion": "proc_inscripcion",
            "inscribir": "proc_inscripcion",
            "baja": "proc_baja",
            "readmision": "proc_readmision",
            "readmisión": "proc_readmision",
            "prorroga": "proc_prorroga",
            "prórroga": "proc_prorroga",
            "defensa": "proc_defensa",
            "trabajo final": "proc_defensa",
        }
        for keyword, node_id in process_keywords.items():
            if keyword in query_lower and node_id in self.graph:
                matched.append(node_id)

        # Fuzzy matching contra todos los nodos si no hay matches exactos
        if not matched:
            matched = self._fuzzy_match(query_lower)

        return list(dict.fromkeys(matched))  # Dedup preserving order

    def _fuzzy_match(self, query_lower: str) -> list[str]:
        """Matching difuso contra nombres de nodos."""
        candidates = []
        for node_id, data in self.graph.nodes(data=True):
            name = data.get("name", "").lower()
            aliases = [a.lower() for a in data.get("aliases", [])]

            best_ratio = 0.0
            for term in [name] + aliases:
                ratio = SequenceMatcher(None, query_lower, term).ratio()
                best_ratio = max(best_ratio, ratio)

                # También buscar si el término está contenido en la query
                if term in query_lower or query_lower in term:
                    best_ratio = max(best_ratio, 0.8)

            if best_ratio > 0.5:
                candidates.append((node_id, best_ratio))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:5]]

    def _subgraph_to_text(self, subgraph: nx.DiGraph, center_node: str) -> str:
        """Convierte subgrafo a descripción en lenguaje natural."""
        lines = []

        center_data = subgraph.nodes.get(center_node, {})
        center_name = center_data.get("name", center_node)

        lines.append(f"Información sobre {center_name}:")

        # Propiedades del nodo central
        props = center_data.get("properties", {})
        if props.get("full_name"):
            lines.append(f"- Nombre completo: {props['full_name']}")
        if props.get("degree_level"):
            lines.append(f"- Nivel: {props['degree_level']}")
        if props.get("title"):
            lines.append(f"- Título: {props['title']}")

        # Relaciones salientes del nodo central
        for _, target, data in subgraph.out_edges(center_node, data=True):
            target_name = subgraph.nodes[target].get("name", target)
            rel_type = data.get("relation_type", "").replace("_", " ")
            source_text = data.get("source_text", "")
            rel_props = data.get("properties", {})

            if source_text:
                lines.append(f"- {rel_type}: {source_text}")
            elif rel_props.get("plazo"):
                lines.append(f"- {rel_type} {target_name}: {rel_props['plazo']}")
            else:
                lines.append(f"- {rel_type}: {target_name}")

        # Relaciones entrantes al nodo central
        for source, _, data in subgraph.in_edges(center_node, data=True):
            source_name = subgraph.nodes[source].get("name", source)
            rel_type = data.get("relation_type", "").replace("_", " ")
            source_text = data.get("source_text", "")

            if source_text:
                lines.append(f"- {source_name} {rel_type}: {source_text}")
            else:
                lines.append(f"- {source_name} {rel_type} {center_name}")

        # Relaciones entre vecinos (1 hop)
        for u, v, data in subgraph.edges(data=True):
            if u != center_node and v != center_node:
                u_name = subgraph.nodes[u].get("name", u)
                v_name = subgraph.nodes[v].get("name", v)
                rel_type = data.get("relation_type", "")
                source_text = data.get("source_text", "")
                if source_text:
                    lines.append(f"- Relación: {source_text}")

        return "\n".join(lines)

    def _find_relevant_paths(self, source_id: str, target_id: str) -> Optional[str]:
        """Describe el camino entre dos entidades."""
        path = self.graph_builder.get_path(source_id, target_id)
        if not path:
            return None

        descriptions = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            u_name = self.graph.nodes[u].get("name", u)
            v_name = self.graph.nodes[v].get("name", v)

            # Buscar arista en ambas direcciones
            if self.graph.has_edge(u, v):
                edge_data = self.graph.edges[u, v]
                rel_type = edge_data.get("relation_type", "relacionado con")
                descriptions.append(f"{u_name} --[{rel_type}]--> {v_name}")
            elif self.graph.has_edge(v, u):
                edge_data = self.graph.edges[v, u]
                rel_type = edge_data.get("relation_type", "relacionado con")
                descriptions.append(f"{v_name} --[{rel_type}]--> {u_name}")
            else:
                descriptions.append(f"{u_name} -- {v_name}")

        return " | ".join(descriptions)
