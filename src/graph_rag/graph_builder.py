"""
Construcción y gestión del grafo de conocimiento con NetworkX.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import pickle
import logging
from pathlib import Path
from typing import Optional

import networkx as nx

from src.graph_rag.entity_extractor import Entity, EntityType
from src.graph_rag.relationship_mapper import Relationship

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Construye y gestiona el grafo de conocimiento NetworkX."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(
        self, entities: list[Entity], relationships: list[Relationship]
    ) -> nx.DiGraph:
        """Construye el grafo desde entidades y relaciones."""
        self.graph = nx.DiGraph()

        for entity in entities:
            self.add_entity(entity)

        for relationship in relationships:
            self.add_relationship(relationship)

        self.merge_duplicate_entities()

        logger.info(
            f"Grafo construido: {self.graph.number_of_nodes()} nodos, "
            f"{self.graph.number_of_edges()} aristas"
        )
        return self.graph

    def add_entity(self, entity: Entity) -> None:
        """Agrega o actualiza un nodo en el grafo."""
        self.graph.add_node(
            entity.entity_id,
            name=entity.name,
            entity_type=entity.entity_type.value,
            aliases=entity.aliases,
            properties=entity.properties,
            source_document=entity.source_document,
        )

    def add_relationship(self, relationship: Relationship) -> None:
        """Agrega o actualiza una arista en el grafo."""
        # Verificar que ambos nodos existan (o crear placeholders)
        if relationship.source_entity_id not in self.graph:
            self.graph.add_node(
                relationship.source_entity_id,
                name=relationship.source_entity_id,
                entity_type="unknown",
            )
        if relationship.target_entity_id not in self.graph:
            self.graph.add_node(
                relationship.target_entity_id,
                name=relationship.target_entity_id,
                entity_type="unknown",
            )

        self.graph.add_edge(
            relationship.source_entity_id,
            relationship.target_entity_id,
            relation_type=relationship.relation_type.value,
            properties=relationship.properties,
            source_text=relationship.source_text,
        )

    def merge_duplicate_entities(self) -> None:
        """Fusiona entidades duplicadas basándose en aliases."""
        alias_map = {}  # alias -> canonical_id

        for node_id, data in self.graph.nodes(data=True):
            aliases = data.get("aliases", [])
            name = data.get("name", "")

            for alias in aliases + [name]:
                alias_lower = alias.lower().strip()
                if alias_lower and alias_lower not in alias_map:
                    alias_map[alias_lower] = node_id

    def get_subgraph(self, entity_id: str, depth: int = 2) -> nx.DiGraph:
        """Extrae subgrafo local alrededor de una entidad."""
        if entity_id not in self.graph:
            return nx.DiGraph()

        # BFS para encontrar nodos dentro del rango
        nodes = {entity_id}
        frontier = {entity_id}

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                # Vecinos salientes
                next_frontier.update(self.graph.successors(node))
                # Vecinos entrantes
                next_frontier.update(self.graph.predecessors(node))
            frontier = next_frontier - nodes
            nodes.update(frontier)

        return self.graph.subgraph(nodes).copy()

    def get_path(self, source_id: str, target_id: str) -> Optional[list[str]]:
        """Encuentra el camino más corto entre dos entidades."""
        try:
            # Usar grafo no dirigido para encontrar paths
            undirected = self.graph.to_undirected()
            path = nx.shortest_path(undirected, source_id, target_id)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_node_context(self, entity_id: str) -> str:
        """Genera contexto textual de un nodo a partir de sus vecinos."""
        if entity_id not in self.graph:
            return ""

        node_data = self.graph.nodes[entity_id]
        name = node_data.get("name", entity_id)
        entity_type = node_data.get("entity_type", "")
        props = node_data.get("properties", {})

        lines = [f"{name} ({entity_type})"]

        # Propiedades del nodo
        if props.get("full_name"):
            lines.append(f"  Nombre completo: {props['full_name']}")
        if props.get("degree_level"):
            lines.append(f"  Nivel: {props['degree_level']}")
        if props.get("title"):
            lines.append(f"  Título que otorga: {props['title']}")

        # Relaciones salientes
        for _, target, edge_data in self.graph.out_edges(entity_id, data=True):
            target_name = self.graph.nodes[target].get("name", target)
            rel_type = edge_data.get("relation_type", "")
            rel_props = edge_data.get("properties", {})
            source_text = edge_data.get("source_text", "")

            line = f"  -> {rel_type} -> {target_name}"
            if rel_props.get("plazo"):
                line += f" (plazo: {rel_props['plazo']})"
            if source_text:
                line += f" [{source_text[:80]}]"
            lines.append(line)

        # Relaciones entrantes
        for source, _, edge_data in self.graph.in_edges(entity_id, data=True):
            source_name = self.graph.nodes[source].get("name", source)
            rel_type = edge_data.get("relation_type", "")

            lines.append(f"  <- {rel_type} <- {source_name}")

        return "\n".join(lines)

    def get_all_node_contexts(self) -> dict[str, str]:
        """Genera contextos textuales para todos los nodos."""
        return {
            node_id: self.get_node_context(node_id)
            for node_id in self.graph.nodes()
        }

    def save(self, path: Path) -> None:
        """Guarda el grafo a disco (GraphML + pickle)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # GraphML (para visualización con herramientas externas)
        try:
            # GraphML no soporta dicts como atributos, simplificar
            simple_graph = nx.DiGraph()
            for node_id, data in self.graph.nodes(data=True):
                simple_data = {
                    "name": str(data.get("name", "")),
                    "entity_type": str(data.get("entity_type", "")),
                }
                simple_graph.add_node(node_id, **simple_data)

            for u, v, data in self.graph.edges(data=True):
                simple_data = {
                    "relation_type": str(data.get("relation_type", "")),
                }
                simple_graph.add_edge(u, v, **simple_data)

            nx.write_graphml(simple_graph, str(path / "knowledge_graph.graphml"))
        except Exception as e:
            logger.warning(f"No se pudo guardar GraphML: {e}")

        # Pickle (preserva todos los datos)
        with open(path / "knowledge_graph.pkl", "wb") as f:
            pickle.dump(self.graph, f)

        logger.info(f"Grafo guardado en {path}")

    def load(self, path: Path) -> None:
        """Carga el grafo desde disco."""
        path = Path(path)
        pkl_file = path / "knowledge_graph.pkl"

        if pkl_file.exists():
            with open(pkl_file, "rb") as f:
                self.graph = pickle.load(f)
            logger.info(
                f"Grafo cargado: {self.graph.number_of_nodes()} nodos, "
                f"{self.graph.number_of_edges()} aristas"
            )
        else:
            logger.warning(f"No se encontró grafo en {path}")

    def get_statistics(self) -> dict:
        """Retorna estadísticas del grafo."""
        if self.graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0}

        undirected = self.graph.to_undirected()

        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(undirected),
            "node_types": {},
            "edge_types": {},
        }

        # Contar por tipo de nodo
        for _, data in self.graph.nodes(data=True):
            etype = data.get("entity_type", "unknown")
            stats["node_types"][etype] = stats["node_types"].get(etype, 0) + 1

        # Contar por tipo de arista
        for _, _, data in self.graph.edges(data=True):
            rtype = data.get("relation_type", "unknown")
            stats["edge_types"][rtype] = stats["edge_types"].get(rtype, 0) + 1

        return stats
