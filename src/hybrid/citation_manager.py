"""
Gestión de citaciones y atribución de fuentes.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    document_name: str
    page_number: int = 0
    section_title: str = ""
    text_snippet: str = ""
    citation_id: str = ""


class CitationManager:
    """Gestiona la atribución de fuentes en las respuestas."""

    def create_citations(self, sources: list[dict]) -> list[Citation]:
        """Crea citaciones numeradas desde las fuentes."""
        citations = []
        for i, source in enumerate(sources, 1):
            pages = source.get("page_numbers", [])
            page = pages[0] if pages else 0

            citations.append(Citation(
                document_name=source.get("document_name", "Desconocido"),
                page_number=page,
                section_title=source.get("section_title", ""),
                text_snippet=source.get("text_snippet", "")[:100],
                citation_id=f"[{i}]",
            ))
        return citations

    def format_citation_footer(self, citations: list[Citation]) -> str:
        """Genera pie de citaciones."""
        if not citations:
            return ""

        lines = ["\n📚 **Fuentes:**"]
        for citation in citations:
            line = f"{citation.citation_id} {citation.document_name}"
            if citation.section_title:
                line += f", {citation.section_title}"
            if citation.page_number > 0:
                line += f" (pág. {citation.page_number})"
            lines.append(line)

        return "\n".join(lines)

    def format_answer_with_citations(
        self, answer: str, citations: list[Citation]
    ) -> str:
        """Combina respuesta con pie de citaciones."""
        footer = self.format_citation_footer(citations)
        if footer:
            return f"{answer}\n{footer}"
        return answer
