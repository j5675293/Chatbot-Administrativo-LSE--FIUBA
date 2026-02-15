"""
Memoria conversacional para el chatbot.
Mantiene contexto de la conversación con resumen y ventana deslizante.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

SUMMARY_PROMPT_ES = (
    "Resumí brevemente la siguiente conversación entre un estudiante y "
    "el asistente de posgrados del LSE-FIUBA. Enfocate en los temas "
    "consultados y las respuestas clave. Máximo 3 oraciones.\n\n"
    "Conversación:\n{conversation}\n\n"
    "Resumen:"
)

CONTEXTUALIZE_PROMPT_ES = (
    "Dada la siguiente conversación previa y una nueva pregunta del estudiante, "
    "reformulá la pregunta para que sea autocontenida (sin depender del contexto "
    "previo). Si la pregunta ya es autocontenida, devolvela tal cual.\n\n"
    "Resumen de conversación previa:\n{summary}\n\n"
    "Últimos mensajes:\n{recent_messages}\n\n"
    "Nueva pregunta: {question}\n\n"
    "Pregunta reformulada:"
)


@dataclass
class ConversationTurn:
    role: str  # "user" o "assistant"
    content: str
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class ConversationMemory:
    """Memoria conversacional con ventana deslizante y resumen progresivo.

    Estrategia:
    - Mantiene los últimos `window_size` turnos completos
    - Cuando se excede la ventana, comprime turnos viejos en un resumen
    - El resumen se usa como contexto para reformular preguntas
    """

    def __init__(
        self,
        llm_provider=None,
        window_size: int = 6,
        max_summary_length: int = 500,
    ):
        self.llm = llm_provider
        self.window_size = window_size
        self.max_summary_length = max_summary_length

        # Almacenamiento por sesión
        self._sessions: dict[str, dict] = defaultdict(
            lambda: {"turns": [], "summary": "", "topics": set()}
        )

    def add_turn(
        self, session_id: str, role: str, content: str, metadata: dict = None
    ) -> None:
        """Agrega un turno a la conversación."""
        session = self._sessions[session_id]
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        session["turns"].append(turn)

        # Extraer tópicos mencionados
        self._extract_topics(session, content)

        # Comprimir si excede la ventana
        if len(session["turns"]) > self.window_size * 2:
            self._compress(session_id)

    def get_chat_history(self, session_id: str) -> list[dict]:
        """Obtiene el historial formateado para el LLM."""
        session = self._sessions[session_id]
        messages = []

        # Agregar resumen como contexto si existe
        if session["summary"]:
            messages.append({
                "role": "system",
                "content": f"Resumen de conversación previa: {session['summary']}",
            })

        # Últimos turnos en la ventana
        recent = session["turns"][-self.window_size:]
        for turn in recent:
            messages.append({
                "role": turn.role,
                "content": turn.content,
            })

        return messages

    def contextualize_query(self, session_id: str, query: str) -> str:
        """Reformula la query teniendo en cuenta el contexto conversacional.

        Si la query parece depender de contexto previo (pronombres, referencias),
        usa LLM para reformularla.
        """
        session = self._sessions[session_id]

        # Si no hay historial, devolver tal cual
        if len(session["turns"]) < 2:
            return query

        # Detectar si necesita contextualización
        if not self._needs_contextualization(query):
            return query

        # Reformular con LLM
        if self.llm:
            return self._reformulate_with_llm(session_id, query)

        # Sin LLM: agregar tópicos recientes
        return self._reformulate_heuristic(session_id, query)

    def get_session_topics(self, session_id: str) -> set[str]:
        """Retorna los tópicos discutidos en la sesión."""
        return self._sessions[session_id]["topics"]

    def get_turn_count(self, session_id: str) -> int:
        """Retorna cantidad de turnos en la sesión."""
        return len(self._sessions[session_id]["turns"])

    def clear_session(self, session_id: str) -> None:
        """Limpia una sesión."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def _needs_contextualization(self, query: str) -> bool:
        """Detecta si la query tiene referencias que requieren contexto."""
        context_indicators = [
            # Pronombres demostrativos
            "eso", "esto", "esa", "este", "esta", "estos", "estas",
            # Pronombres personales referidos al tema
            "la misma", "el mismo", "lo mismo",
            # Referencias anafóricas
            "también", "además", "y la", "y el", "y los",
            # Preguntas continuativas
            "qué más", "algo más", "otra cosa",
            "y sobre", "y con respecto", "y en cuanto",
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in context_indicators)

    def _reformulate_with_llm(self, session_id: str, query: str) -> str:
        """Reformula query con LLM para hacerla autocontenida."""
        session = self._sessions[session_id]

        recent = session["turns"][-4:]
        recent_text = "\n".join(
            f"{'Estudiante' if t.role == 'user' else 'Asistente'}: {t.content[:200]}"
            for t in recent
        )

        try:
            prompt = CONTEXTUALIZE_PROMPT_ES.format(
                summary=session["summary"] or "(sin resumen previo)",
                recent_messages=recent_text,
                question=query,
            )
            reformulated = self.llm.generate(prompt)
            reformulated = reformulated.strip().strip('"')

            if reformulated and len(reformulated) > 5:
                logger.info(f"Query contextualizada: '{query}' -> '{reformulated}'")
                return reformulated

        except Exception as e:
            logger.warning(f"Error contextualizando query: {e}")

        return query

    def _reformulate_heuristic(self, session_id: str, query: str) -> str:
        """Reformulación heurística sin LLM."""
        session = self._sessions[session_id]
        topics = session["topics"]

        if topics:
            # Agregar el tópico más reciente a la query
            recent_topic = list(topics)[-1]
            return f"{query} (en relación a {recent_topic})"

        return query

    def _compress(self, session_id: str) -> None:
        """Comprime turnos antiguos en un resumen."""
        session = self._sessions[session_id]
        old_turns = session["turns"][: -self.window_size]
        session["turns"] = session["turns"][-self.window_size:]

        if self.llm and old_turns:
            conversation_text = "\n".join(
                f"{'Estudiante' if t.role == 'user' else 'Asistente'}: "
                f"{t.content[:200]}"
                for t in old_turns
            )

            try:
                prompt = SUMMARY_PROMPT_ES.format(conversation=conversation_text)
                new_summary = self.llm.generate(prompt)

                if session["summary"]:
                    session["summary"] = (
                        f"{session['summary']} {new_summary.strip()}"
                    )[: self.max_summary_length]
                else:
                    session["summary"] = new_summary.strip()[
                        : self.max_summary_length
                    ]

            except Exception as e:
                logger.warning(f"Error comprimiendo conversación: {e}")
        else:
            # Sin LLM: resumen simple
            user_msgs = [
                t.content[:100] for t in old_turns if t.role == "user"
            ]
            session["summary"] = "Temas consultados: " + "; ".join(user_msgs)

    def _extract_topics(self, session: dict, content: str) -> None:
        """Extrae tópicos del contenido para tracking."""
        content_lower = content.lower()
        topic_keywords = {
            "CEIA": ["ceia", "inteligencia artificial"],
            "CESE": ["cese", "sistemas embebidos"],
            "CEIoT": ["ceiot", "internet de las cosas"],
            "MIA": ["mia", "maestría en ia"],
            "MIAE": ["miae"],
            "MIoT": ["miot"],
            "MCB": ["mcb", "ciberseguridad"],
            "Reglamento": ["reglamento", "asistencia", "nota mínima", "bimestre"],
            "Inscripción": ["inscripción", "inscripci", "matricul"],
            "Trabajo Final": ["trabajo final", "tesis", "ttfa", "ttfb"],
            "GdP": ["gdp", "gestión de proyectos"],
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in content_lower for kw in keywords):
                session["topics"].add(topic)
