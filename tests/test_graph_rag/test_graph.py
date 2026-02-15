"""
Tests del sistema GraphRAG.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.graph_rag.entity_extractor import AcademicEntityExtractor, EntityType
from src.graph_rag.graph_builder import KnowledgeGraphBuilder
from src.graph_rag.community_detector import CommunityDetector


class TestEntityExtractor:
    """Tests de extracción de entidades."""

    def setup_method(self):
        self.extractor = AcademicEntityExtractor()

    def test_extract_program_entities(self):
        text = "La CEIA es una carrera de especialización en inteligencia artificial."
        entities = self.extractor.extract_entities(text, "CEIA.pdf")
        program_entities = [e for e in entities if e.entity_type == EntityType.PROGRAMA]
        assert len(program_entities) >= 1
        names = [e.name for e in program_entities]
        assert any("CEIA" in n for n in names)

    def test_extract_contact_entities(self):
        text = "Para consultas escribir a inscripcion.lse@fi.uba.ar"
        entities = self.extractor.extract_entities(text, "test.pdf")
        contacts = [e for e in entities if e.entity_type == EntityType.CONTACTO]
        assert len(contacts) >= 1

    def test_extract_deadline_entities(self):
        text = "El plazo máximo es de 10 bimestres para completar la carrera."
        entities = self.extractor.extract_entities(text, "Reglamento.pdf")
        deadlines = [e for e in entities if e.entity_type == EntityType.PLAZO]
        assert len(deadlines) >= 1

    def test_extract_article_entities(self):
        text = "Art. 3 La nota mínima de aprobación es 4."
        entities = self.extractor.extract_entities(text, "Reglamento.pdf")
        articles = [e for e in entities if e.entity_type == EntityType.ARTICULO]
        assert len(articles) >= 1


class TestKnowledgeGraphBuilder:
    """Tests del constructor de grafos."""

    def setup_method(self):
        self.builder = KnowledgeGraphBuilder()

    def test_build_empty(self):
        self.builder.build_graph([], [])
        assert self.builder.graph.number_of_nodes() == 0

    def test_add_entities(self):
        from src.graph_rag.entity_extractor import Entity, EntityType
        from src.graph_rag.relationship_mapper import Relationship, RelationType

        entities = [
            Entity(
                entity_id="prog-ceia",
                name="CEIA",
                entity_type=EntityType.PROGRAMA,
                properties={"titulo": "Especialista en IA"},
                source_document="CEIA.pdf",
            ),
            Entity(
                entity_id="prog-mia",
                name="MIA",
                entity_type=EntityType.PROGRAMA,
                properties={"titulo": "Magíster en IA"},
                source_document="MIA.pdf",
            ),
        ]

        relationships = [
            Relationship(
                source_entity_id="prog-ceia",
                target_entity_id="prog-mia",
                relation_type=RelationType.REQUIERE_EGRESO_DE,
                properties={"description": "CEIA es prerrequisito de MIA"},
            ),
        ]

        self.builder.build_graph(entities, relationships)
        assert self.builder.graph.number_of_nodes() == 2
        assert self.builder.graph.number_of_edges() == 1

    def test_get_subgraph(self):
        from src.graph_rag.entity_extractor import Entity, EntityType
        from src.graph_rag.relationship_mapper import Relationship, RelationType

        entities = [
            Entity("n1", "CEIA", EntityType.PROGRAMA, source_document="CEIA.pdf"),
            Entity("n2", "MIA", EntityType.PROGRAMA, source_document="MIA.pdf"),
            Entity("n3", "MCB", EntityType.PROGRAMA, source_document="MCB.pdf"),
        ]
        relationships = [
            Relationship("n1", "n2", RelationType.REQUIERE_EGRESO_DE),
            Relationship("n1", "n3", RelationType.REQUIERE_EGRESO_DE),
        ]

        self.builder.build_graph(entities, relationships)
        subgraph = self.builder.get_subgraph("n1", depth=1)
        assert len(subgraph.nodes()) == 3


class TestCommunityDetector:
    """Tests de detección de comunidades."""

    def test_detect_on_small_graph(self):
        import networkx as nx

        G = nx.Graph()
        G.add_edges_from([
            ("CEIA", "MIA"), ("CEIA", "MIAE"),
            ("CESE", "MIoT"), ("CEIoT", "MIoT"),
        ])

        for node in G.nodes():
            G.nodes[node]["name"] = node
            G.nodes[node]["type"] = "PROGRAMA"

        detector = CommunityDetector(G)
        communities = detector.detect_communities()

        # Debería encontrar al menos 1 comunidad
        assert len(communities) >= 1


class TestAntiHallucination:
    """Tests del motor anti-alucinación."""

    def test_should_abstain_out_of_scope(self):
        from src.hybrid.anti_hallucination import AntiHallucinationEngine

        engine = AntiHallucinationEngine()

        should, reason = engine.should_abstain(0.5, "¿Cuánto cuesta la carrera?")
        assert should is True

    def test_should_not_abstain_valid_question(self):
        from src.hybrid.anti_hallucination import AntiHallucinationEngine

        engine = AntiHallucinationEngine()

        should, reason = engine.should_abstain(0.8, "¿Cuál es la asistencia mínima?")
        assert should is False

    def test_fallback_contacts(self):
        from src.hybrid.anti_hallucination import AntiHallucinationEngine

        engine = AntiHallucinationEngine()

        contact = engine.get_fallback_contact("¿Cómo me inscribo?")
        assert "inscripcion" in contact

        contact = engine.get_fallback_contact("consulta sobre trabajo final")
        assert "direccion.posgrado" in contact

    def test_faithfulness_heuristic(self):
        from src.hybrid.anti_hallucination import AntiHallucinationEngine

        engine = AntiHallucinationEngine()

        context = "La CEIA requiere 75% de asistencia según Art. 2"
        answer = "La asistencia mínima en la CEIA es del 75% (Art. 2)"

        result = engine.check_faithfulness(answer, context)
        assert result.score > 0.5
