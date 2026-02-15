"""
HyDE - Hypothetical Document Embeddings.
Genera un documento hipotético como respuesta a la query,
luego usa su embedding para buscar documentos similares.

Referencia: Gao et al., 2022 - "Precise Zero-Shot Dense Retrieval without Relevance Labels"

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

HYDE_PROMPT_ES = (
    "Sos un experto en administración de posgrados universitarios del "
    "Laboratorio de Sistemas Embebidos (LSE) de FIUBA.\n\n"
    "Escribí un párrafo informativo que responda la siguiente pregunta de un "
    "estudiante. Usá un tono formal y proporcioná datos concretos (plazos, "
    "requisitos, porcentajes) como si fueras el documento oficial.\n\n"
    "Pregunta: {query}\n\n"
    "Respuesta hipotética (un párrafo):"
)


class HyDERetriever:
    """Retriever mejorado con Hypothetical Document Embeddings.

    Pipeline:
    1. Genera un documento hipotético con LLM
    2. Embeds el documento hipotético
    3. Busca chunks similares al documento hipotético
    4. Opcionalmente fusiona con búsqueda directa de la query
    """

    def __init__(
        self,
        llm_provider,
        embedding_model,
        vector_store,
        reranker=None,
        alpha: float = 0.6,
    ):
        """
        Args:
            llm_provider: Proveedor LLM para generar doc hipotético
            embedding_model: Modelo de embeddings
            vector_store: FAISS vector store
            reranker: Cross-encoder reranker opcional
            alpha: Peso del embedding HyDE vs query directa (0-1)
        """
        self.llm = llm_provider
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.reranker = reranker
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_fusion: bool = True,
        program_filter: Optional[str] = None,
    ) -> list:
        """Retrieval con HyDE.

        Args:
            query: Consulta del usuario
            top_k: Número de resultados
            use_fusion: Si True, combina embedding HyDE con embedding directo
            program_filter: Filtro por programa académico
        """
        # 1. Generar documento hipotético
        hypothetical_doc = self._generate_hypothetical(query)
        logger.info(f"HyDE doc generado ({len(hypothetical_doc)} chars)")

        # 2. Embedding del documento hipotético
        hyde_embedding = self.embedding_model.embed_query(hypothetical_doc)

        if use_fusion:
            # 3. Embedding directo de la query
            query_embedding = self.embedding_model.embed_query(query)

            # 4. Fusionar embeddings con peso alpha
            fused_embedding = (
                self.alpha * hyde_embedding
                + (1 - self.alpha) * query_embedding
            )
            # Normalizar
            norm = np.linalg.norm(fused_embedding)
            if norm > 0:
                fused_embedding = fused_embedding / norm

            search_embedding = fused_embedding
        else:
            search_embedding = hyde_embedding

        # 5. Búsqueda
        fetch_k = top_k * 3
        if program_filter:
            filter_meta = {"program_codes": [program_filter]}
            results = self.vector_store.search_with_filter(
                search_embedding, top_k=fetch_k, filter_metadata=filter_meta
            )
        else:
            results = self.vector_store.search_mmr(
                search_embedding, top_k=fetch_k, fetch_k=fetch_k * 2
            )

        if not results:
            return []

        # 6. Reranking contra la query original (no el doc hipotético)
        if self.reranker and len(results) > top_k:
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        return results

    def _generate_hypothetical(self, query: str) -> str:
        """Genera documento hipotético con LLM."""
        try:
            prompt = HYDE_PROMPT_ES.format(query=query)
            response = self.llm.generate(prompt)

            # Limpiar respuesta
            response = response.strip()
            if response.startswith("[Error"):
                logger.warning(f"LLM error en HyDE, usando query directa")
                return query

            return response[:1000]  # Limitar tamaño

        except Exception as e:
            logger.warning(f"Error generando doc hipotético: {e}")
            return query
