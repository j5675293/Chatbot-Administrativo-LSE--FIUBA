"""
Estrategias de chunking para documentos académicos del LSE.
Tres estrategias: fixed-size, semántico por artículos, y FAQ Q&A pairs.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ChunkStrategy(Enum):
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    FAQ_QA_PAIRS = "faq_qa_pairs"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    document_name: str
    document_type: str
    page_numbers: list[int] = field(default_factory=list)
    section_title: str = ""
    chunk_index: int = 0
    strategy: str = ""
    metadata: dict = field(default_factory=dict)
    token_count: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "document_name": self.document_name,
            "document_type": self.document_type,
            "page_numbers": self.page_numbers,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "strategy": self.strategy,
            "metadata": self.metadata,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


class DocumentChunker:
    """Chunking multi-estrategia optimizado para documentos del LSE."""

    def __init__(
        self,
        fixed_chunk_size: int = 512,
        fixed_overlap: int = 128,
        max_chunk_tokens: int = 768,
        min_chunk_tokens: int = 50,
    ):
        self.fixed_chunk_size = fixed_chunk_size
        self.fixed_overlap = fixed_overlap
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens

    def chunk_document(
        self,
        text: str,
        document_name: str,
        document_type: str,
        pages_text: Optional[list[tuple[int, str]]] = None,
        strategy: Optional[ChunkStrategy] = None,
    ) -> list[Chunk]:
        """Selecciona y aplica la estrategia de chunking apropiada."""
        if strategy is None:
            strategy = self._select_strategy(document_type, text)

        logger.info(f"Chunking {document_name} con estrategia {strategy.value}")

        kwargs = {
            "text": text,
            "document_name": document_name,
            "document_type": document_type,
            "pages_text": pages_text,
        }

        if strategy == ChunkStrategy.FAQ_QA_PAIRS:
            chunks = self._chunk_faq_qa_pairs(**kwargs)
        elif strategy == ChunkStrategy.SEMANTIC:
            chunks = self._chunk_semantic(**kwargs)
            # Fallback a fixed si los chunks semánticos son demasiado grandes
            final_chunks = []
            for chunk in chunks:
                if chunk.token_count > self.max_chunk_tokens:
                    sub_chunks = self._chunk_fixed_size(
                        text=chunk.text,
                        document_name=document_name,
                        document_type=document_type,
                        section_prefix=chunk.section_title,
                    )
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)
            chunks = final_chunks
        else:
            chunks = self._chunk_fixed_size(**kwargs)

        # Asignar índices y token counts
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.token_count = self._estimate_tokens(chunk.text)
            chunk.strategy = strategy.value
            if not chunk.chunk_id:
                chunk.chunk_id = f"{document_name}_{i}_{uuid.uuid4().hex[:8]}"

        # Filtrar chunks muy pequeños
        chunks = [c for c in chunks if c.token_count >= self.min_chunk_tokens]

        logger.info(f"  -> {len(chunks)} chunks generados")
        return chunks

    def _select_strategy(self, document_type: str, text: str) -> ChunkStrategy:
        """Auto-selección de estrategia según tipo de documento."""
        if document_type == "faq":
            return ChunkStrategy.FAQ_QA_PAIRS
        if document_type == "reglamento":
            return ChunkStrategy.SEMANTIC
        if document_type == "resolucion":
            return ChunkStrategy.SEMANTIC
        if document_type == "programa":
            return ChunkStrategy.SEMANTIC
        return ChunkStrategy.FIXED_SIZE

    def _chunk_fixed_size(
        self,
        text: str,
        document_name: str,
        document_type: str,
        section_prefix: str = "",
        pages_text: Optional[list] = None,
    ) -> list[Chunk]:
        """Chunks de tamaño fijo con overlap, cortando en límites de oración."""
        sentences = self._split_sentences(text)
        chunks = []
        current_tokens = 0
        current_sentences = []
        overlap_sentences = []

        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)

            if current_tokens + sent_tokens > self.fixed_chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                if section_prefix:
                    chunk_text = f"[{section_prefix}]\n{chunk_text}"

                chunks.append(Chunk(
                    chunk_id="",
                    text=chunk_text,
                    document_name=document_name,
                    document_type=document_type,
                    section_title=section_prefix,
                ))

                # Calcular overlap
                overlap_tokens = 0
                overlap_sentences = []
                for s in reversed(current_sentences):
                    s_tokens = self._estimate_tokens(s)
                    if overlap_tokens + s_tokens > self.fixed_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens

                current_sentences = list(overlap_sentences)
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # Último chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if section_prefix:
                chunk_text = f"[{section_prefix}]\n{chunk_text}"
            chunks.append(Chunk(
                chunk_id="",
                text=chunk_text,
                document_name=document_name,
                document_type=document_type,
                section_title=section_prefix,
            ))

        return chunks

    def _chunk_semantic(
        self,
        text: str,
        document_name: str,
        document_type: str,
        pages_text: Optional[list] = None,
    ) -> list[Chunk]:
        """Chunking semántico que respeta estructura del documento."""
        chunks = []

        # Intentar dividir por artículos (Art. N)
        article_pattern = r"\n(?=Art\.\s*\d+)"
        sections = re.split(article_pattern, text)

        if len(sections) > 1:
            for section in sections:
                section = section.strip()
                if not section:
                    continue

                # Extraer título de sección
                title_match = re.match(r"(Art\.\s*\d+[^.]*\.?)", section)
                title = title_match.group(1).strip() if title_match else ""

                chunks.append(Chunk(
                    chunk_id="",
                    text=section,
                    document_name=document_name,
                    document_type=document_type,
                    section_title=title,
                ))
            return chunks

        # Intentar dividir por secciones con headers en mayúsculas
        header_pattern = r"\n(?=[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{5,}(?:\n|:))"
        sections = re.split(header_pattern, text)

        if len(sections) > 1:
            for section in sections:
                section = section.strip()
                if not section:
                    continue

                # Extraer título
                lines = section.split("\n")
                title = lines[0].strip() if lines else ""

                chunks.append(Chunk(
                    chunk_id="",
                    text=section,
                    document_name=document_name,
                    document_type=document_type,
                    section_title=title,
                ))
            return chunks

        # Intentar dividir por secciones numeradas (I., II., III. o 1., 2., 3.)
        numbered_pattern = r"\n(?=(?:[IVXLC]+\.|[0-9]+\.)\s+[A-ZÁÉÍÓÚÑ])"
        sections = re.split(numbered_pattern, text)

        if len(sections) > 1:
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                lines = section.split("\n")
                title = lines[0].strip() if lines else ""
                chunks.append(Chunk(
                    chunk_id="",
                    text=section,
                    document_name=document_name,
                    document_type=document_type,
                    section_title=title,
                ))
            return chunks

        # Fallback: un solo chunk grande (se subdividirá por fixed_size)
        chunks.append(Chunk(
            chunk_id="",
            text=text,
            document_name=document_name,
            document_type=document_type,
            section_title=document_name,
        ))
        return chunks

    def _chunk_faq_qa_pairs(
        self,
        text: str,
        document_name: str,
        document_type: str,
        pages_text: Optional[list] = None,
    ) -> list[Chunk]:
        """Chunking para FAQs: cada par Q&A es un chunk atómico."""
        chunks = []
        current_section = "General"

        # Detectar secciones (headers en mayúsculas o negrita)
        lines = text.split("\n")
        current_question = ""
        current_answer_lines = []
        in_answer = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_answer:
                    current_answer_lines.append("")
                continue

            # Detectar header de sección
            if (
                stripped.isupper()
                and len(stripped) > 5
                and "?" not in stripped
                and not stripped.startswith("•")
            ):
                current_section = stripped
                continue

            # Detectar header de sección tipo "SECCION / SUBSECCION"
            if re.match(r"^[A-ZÁÉÍÓÚÑ\s/]+$", stripped) and len(stripped) > 5:
                current_section = stripped
                continue

            # Detectar pregunta (empieza con bullet/guión/número y tiene ?)
            is_question = (
                ("?" in stripped)
                and (
                    stripped.startswith("•")
                    or stripped.startswith("-")
                    or stripped.startswith("–")
                    or re.match(r"^\d+[\.\)]\s", stripped)
                    or stripped.startswith("¿")
                )
            )

            if is_question:
                # Guardar Q&A anterior si existe
                if current_question and current_answer_lines:
                    answer_text = "\n".join(current_answer_lines).strip()
                    qa_text = (
                        f"[Sección: {current_section}]\n"
                        f"Pregunta: {current_question}\n"
                        f"Respuesta: {answer_text}"
                    )
                    chunks.append(Chunk(
                        chunk_id="",
                        text=qa_text,
                        document_name=document_name,
                        document_type=document_type,
                        section_title=current_section,
                        metadata={"question": current_question},
                    ))

                # Nueva pregunta
                current_question = re.sub(r"^[•\-–]\s*", "", stripped)
                current_question = re.sub(r"^\d+[\.\)]\s*", "", current_question)
                current_answer_lines = []
                in_answer = True
            elif in_answer:
                current_answer_lines.append(stripped)

        # Último Q&A
        if current_question and current_answer_lines:
            answer_text = "\n".join(current_answer_lines).strip()
            qa_text = (
                f"[Sección: {current_section}]\n"
                f"Pregunta: {current_question}\n"
                f"Respuesta: {answer_text}"
            )
            chunks.append(Chunk(
                chunk_id="",
                text=qa_text,
                document_name=document_name,
                document_type=document_type,
                section_title=current_section,
                metadata={"question": current_question},
            ))

        # Si no se detectaron Q&A, fallback a semántico
        if not chunks:
            logger.warning(f"No se detectaron Q&A en {document_name}, usando fallback semántico")
            return self._chunk_semantic(
                text=text,
                document_name=document_name,
                document_type=document_type,
            )

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Divide texto en oraciones respetando abreviaturas comunes."""
        # Proteger abreviaturas comunes
        text = re.sub(r"Art\.", "Art§", text)
        text = re.sub(r"Inc\.", "Inc§", text)
        text = re.sub(r"Dr\.", "Dr§", text)
        text = re.sub(r"Ing\.", "Ing§", text)
        text = re.sub(r"Esp\.", "Esp§", text)
        text = re.sub(r"Mag\.", "Mag§", text)

        # Dividir por puntos, signos de exclamación/interrogación
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])", text)

        # Restaurar abreviaturas
        sentences = [s.replace("§", ".") for s in sentences]

        # Filtrar vacías
        return [s.strip() for s in sentences if s.strip()]

    def _estimate_tokens(self, text: str) -> int:
        """Estimación aproximada de tokens (factor 1.3 para español)."""
        words = len(text.split())
        return int(words * 1.3)
