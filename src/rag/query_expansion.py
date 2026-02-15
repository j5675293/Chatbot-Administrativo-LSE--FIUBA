"""
Expansión de queries usando LLM y técnicas heurísticas.
Genera variantes de la consulta para mejorar el retrieval.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

QUERY_EXPANSION_PROMPT = (
    "Dada la siguiente pregunta sobre posgrados universitarios del LSE-FIUBA, "
    "generá 3 reformulaciones alternativas que busquen la misma información "
    "pero con diferentes palabras clave.\n\n"
    "Pregunta original: {query}\n\n"
    "Respondé SOLO con las 3 reformulaciones, una por línea, numeradas:\n"
    "1. ...\n2. ...\n3. ..."
)

# Sinónimos específicos del dominio LSE-FIUBA
DOMAIN_SYNONYMS = {
    "requisito": ["condición", "requerimiento", "exigencia"],
    "inscripción": ["matriculación", "registro", "ingreso"],
    "materia": ["asignatura", "curso", "cátedra"],
    "plazo": ["fecha límite", "tiempo máximo", "vencimiento"],
    "especialización": ["carrera de especialización", "posgrado"],
    "maestría": ["carrera de maestría", "máster"],
    "trabajo final": ["tesis", "proyecto final", "TFI"],
    "nota": ["calificación", "puntaje"],
    "bimestre": ["período", "cuatrimestre"],
    "asistencia": ["presencialidad", "concurrencia"],
    "aprobación": ["acreditación", "promoción"],
    "correlativa": ["prerrequisito", "requisito previo"],
    "baja": ["desvinculación", "retiro"],
    "prórroga": ["extensión", "ampliación de plazo"],
    "optativa": ["electiva", "materia opcional"],
    "director": ["tutor", "orientador"],
    "ceia": ["especialización en inteligencia artificial", "CEIA"],
    "cese": ["especialización en sistemas embebidos", "CESE"],
    "ceiot": ["especialización en internet de las cosas", "CEIoT"],
    "mia": ["maestría en inteligencia artificial", "MIA"],
    "gdp": ["gestión de proyectos", "GdP"],
}


class QueryExpander:
    """Expande queries para mejorar retrieval usando LLM y heurísticas."""

    def __init__(
        self,
        llm_provider=None,
        embedding_model=None,
        max_expansions: int = 3,
    ):
        self.llm = llm_provider
        self.embedding_model = embedding_model
        self.max_expansions = max_expansions

    def expand(self, query: str) -> list[str]:
        """Genera expansiones de la query. Devuelve [query_original, ...expansiones]."""
        expansions = [query]

        # Expansión con sinónimos del dominio
        synonym_exp = self._expand_with_synonyms(query)
        if synonym_exp:
            expansions.append(synonym_exp)

        # Expansión con LLM
        if self.llm:
            llm_expansions = self._expand_with_llm(query)
            expansions.extend(llm_expansions)

        # Limitar y deduplicar
        seen = set()
        unique = []
        for exp in expansions:
            normalized = exp.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(exp.strip())

        return unique[: self.max_expansions + 1]

    def expand_and_merge_results(
        self,
        query: str,
        retriever,
        top_k: int = 5,
        **retriever_kwargs,
    ) -> list:
        """Expande la query, ejecuta retrieval por cada variante y fusiona resultados."""
        expanded_queries = self.expand(query)
        all_results = {}

        for exp_query in expanded_queries:
            results = retriever.retrieve(
                query=exp_query, top_k=top_k, **retriever_kwargs
            )
            for r in results:
                key = r.text[:100]
                if key not in all_results:
                    all_results[key] = r
                else:
                    # Boost: si aparece en múltiples queries, aumentar score
                    all_results[key].score = max(
                        all_results[key].score, r.score
                    ) * 1.1

        # Ordenar por score y retornar top_k
        merged = sorted(all_results.values(), key=lambda r: r.score, reverse=True)
        return merged[:top_k]

    def _expand_with_synonyms(self, query: str) -> Optional[str]:
        """Reemplaza términos con sinónimos del dominio."""
        query_lower = query.lower()
        expanded = query

        for term, synonyms in DOMAIN_SYNONYMS.items():
            if term in query_lower:
                # Usar el primer sinónimo
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                expanded = pattern.sub(synonyms[0], expanded, count=1)
                if expanded.lower() != query.lower():
                    return expanded

        return None

    def _expand_with_llm(self, query: str) -> list[str]:
        """Genera reformulaciones con LLM."""
        try:
            prompt = QUERY_EXPANSION_PROMPT.format(query=query)
            response = self.llm.generate(prompt)

            expansions = []
            for line in response.strip().split("\n"):
                line = line.strip()
                # Eliminar numeración
                line = re.sub(r"^\d+[\.\)\-]\s*", "", line)
                if line and len(line) > 10 and line.lower() != query.lower():
                    expansions.append(line)

            return expansions[: self.max_expansions]

        except Exception as e:
            logger.warning(f"Error en expansión LLM: {e}")
            return []
