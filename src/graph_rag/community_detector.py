"""
Detección de comunidades en el grafo de conocimiento.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import logging
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)


class CommunityDetector:
    """Detecta comunidades temáticas en el grafo de conocimiento usando Louvain."""

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.communities: list[set] = []
        self._node_to_community: dict[str, int] = {}

    def detect_communities(self, resolution: float = 1.0) -> list[set]:
        """Ejecuta detección de comunidades Louvain."""
        if self.graph.number_of_nodes() == 0:
            return []

        undirected = self.graph.to_undirected()

        try:
            import community as community_louvain
            partition = community_louvain.best_partition(
                undirected, resolution=resolution
            )
        except ImportError:
            # Fallback a algoritmo de NetworkX
            from networkx.algorithms.community import greedy_modularity_communities
            communities_gen = greedy_modularity_communities(undirected)
            partition = {}
            for idx, comm in enumerate(communities_gen):
                for node in comm:
                    partition[node] = idx

        # Organizar en lista de sets
        max_community = max(partition.values()) if partition else -1
        self.communities = [set() for _ in range(max_community + 1)]

        for node, comm_id in partition.items():
            self.communities[comm_id].add(node)
            self._node_to_community[node] = comm_id

        logger.info(f"Detectadas {len(self.communities)} comunidades")
        for i, comm in enumerate(self.communities):
            names = [
                self.graph.nodes[n].get("name", n)
                for n in list(comm)[:5]
            ]
            logger.info(f"  Comunidad {i}: {len(comm)} nodos ({', '.join(names)}...)")

        return self.communities

    def get_community_summary(self, community_id: int) -> str:
        """Genera resumen textual de una comunidad."""
        if community_id >= len(self.communities):
            return ""

        nodes = self.communities[community_id]
        lines = [f"Comunidad {community_id} ({len(nodes)} entidades):"]

        # Agrupar por tipo
        by_type: dict[str, list[str]] = {}
        for node_id in nodes:
            data = self.graph.nodes.get(node_id, {})
            etype = data.get("entity_type", "otro")
            name = data.get("name", node_id)
            by_type.setdefault(etype, []).append(name)

        for etype, names in sorted(by_type.items()):
            lines.append(f"  {etype}: {', '.join(names)}")

        # Relaciones internas
        internal_rels = []
        for u, v, data in self.graph.edges(data=True):
            if u in nodes and v in nodes:
                u_name = self.graph.nodes[u].get("name", u)
                v_name = self.graph.nodes[v].get("name", v)
                rel = data.get("relation_type", "")
                internal_rels.append(f"{u_name} --{rel}--> {v_name}")

        if internal_rels:
            lines.append(f"  Relaciones internas ({len(internal_rels)}):")
            for rel in internal_rels[:10]:
                lines.append(f"    {rel}")

        return "\n".join(lines)

    def get_community_for_entity(self, entity_id: str) -> Optional[int]:
        """Obtiene el ID de comunidad de una entidad."""
        return self._node_to_community.get(entity_id)

    def get_inter_community_bridges(self) -> list[tuple]:
        """Encuentra aristas que conectan diferentes comunidades."""
        bridges = []
        for u, v, data in self.graph.edges(data=True):
            comm_u = self._node_to_community.get(u)
            comm_v = self._node_to_community.get(v)
            if comm_u is not None and comm_v is not None and comm_u != comm_v:
                bridges.append((u, v, comm_u, comm_v, data))
        return bridges
