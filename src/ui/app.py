"""
Interfaz Streamlit del Chatbot Administrativo LSE-FIUBA.

Autor: Juan Ruiz Otondo - CEIA FIUBA
Carrera de Especialización en Inteligencia Artificial
Laboratorio de Sistemas Embebidos - FIUBA - UBA
"""

import sys
import time
from pathlib import Path

import streamlit as st
import requests

# Agregar root al path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Configuración de página ────────────────────────────────────
st.set_page_config(
    page_title="Chatbot Posgrados LSE-FIUBA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estado de sesión ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"


# ── Funciones auxiliares ───────────────────────────────────────

def call_api(question: str, mode: str = "hybrid", program_filter: str = None) -> dict:
    """Llama al endpoint de chat de la API."""
    url = f"{st.session_state.api_url}/api/v1/chat"
    payload = {
        "question": question,
        "mode": mode,
    }
    if program_filter and program_filter != "Todos":
        payload["program_filter"] = program_filter

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError:
        return {
            "answer": (
                "No se pudo conectar con la API. Asegurate de que el servidor "
                "esté corriendo con: `uvicorn src.api.main:app --reload`"
            ),
            "confidence": 0.0,
            "method": "error",
            "sources": [],
            "warnings": ["API no disponible"],
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "confidence": 0.0,
            "method": "error",
            "sources": [],
            "warnings": [str(e)],
        }


def call_comparison_api(question: str, program_filter: str = None) -> dict:
    """Llama al endpoint de comparación."""
    url = f"{st.session_state.api_url}/api/v1/chat/compare"
    payload = {"question": question}
    if program_filter and program_filter != "Todos":
        payload["program_filter"] = program_filter

    try:
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None


def display_confidence(confidence: float):
    """Muestra indicador visual de confianza."""
    if confidence >= 0.7:
        color = "green"
        label = "Alta"
    elif confidence >= 0.4:
        color = "orange"
        label = "Media"
    else:
        color = "red"
        label = "Baja"

    st.markdown(
        f"**Confianza:** :{color}[{label} ({confidence:.0%})]"
    )


def display_sources(sources: list):
    """Muestra las fuentes citadas."""
    if not sources:
        return

    with st.expander("📚 Fuentes consultadas", expanded=False):
        for i, source in enumerate(sources, 1):
            doc = source.get("document_name", "Desconocido")
            section = source.get("section_title", "")
            score = source.get("score", 0)
            snippet = source.get("text_snippet", "")
            source_type = source.get("source_type", "rag")

            icon = "📄" if source_type == "rag" else "🔗"
            st.markdown(f"{icon} **[{i}] {doc}**")
            if section:
                st.markdown(f"   Sección: {section}")
            st.markdown(f"   Relevancia: {score:.2f}")
            if snippet:
                st.caption(snippet[:200])
            st.divider()


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://www.fi.uba.ar/themes/flavor/images/logobb.png", width=200)
    st.title("⚙️ Configuración")

    st.session_state.api_url = st.text_input(
        "URL de la API",
        value=st.session_state.api_url,
    )

    mode = st.selectbox(
        "Modo de retrieval",
        options=["hybrid", "rag", "graph"],
        format_func=lambda x: {
            "hybrid": "Híbrido (RAG + GraphRAG)",
            "rag": "RAG Vectorial",
            "graph": "GraphRAG",
        }[x],
    )

    program_filter = st.selectbox(
        "Filtrar por programa",
        options=["Todos", "CEIA", "CESE", "CEIoT", "MIA", "MIAE", "MIoT", "MCB"],
    )

    comparison_mode = st.toggle("Modo comparación", value=False)

    st.divider()

    st.markdown("### 📋 Preguntas ejemplo")
    example_questions = [
        "¿Cuáles son los requisitos para la MIA?",
        "¿Cuál es el porcentaje mínimo de asistencia?",
        "¿Cómo me inscribo en GdP?",
        "¿Puedo cursar la MIA sin la CEIA?",
        "¿Cuál es el plazo máximo de la especialización?",
        "¿Qué requisitos tiene el TTFA?",
        "¿Cuál es la diferencia entre MIAE y MIA?",
    ]
    for q in example_questions:
        if st.button(q, key=f"example_{q[:20]}", use_container_width=True):
            st.session_state.example_question = q

    st.divider()
    st.markdown(
        "**Trabajo Final** de Juan Ruiz Otondo\n\n"
        "Carrera de Especialización en IA\n\n"
        "LSE - FIUBA - UBA"
    )

# ── Contenido principal ───────────────────────────────────────
st.title("🎓 Asistente Administrativo de Posgrados")
st.caption(
    "Laboratorio de Sistemas Embebidos - Facultad de Ingeniería - UBA"
)

# Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("metadata"):
            meta = message["metadata"]
            display_confidence(meta.get("confidence", 0))
            display_sources(meta.get("sources", []))
            if meta.get("warnings"):
                for w in meta["warnings"]:
                    st.warning(w)

# Input del usuario
prompt = st.chat_input("Hacé tu consulta sobre los posgrados del LSE...")

# Manejar pregunta ejemplo
if hasattr(st.session_state, "example_question"):
    prompt = st.session_state.example_question
    del st.session_state.example_question

if prompt:
    # Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    with st.chat_message("assistant"):
        if comparison_mode:
            with st.spinner("Comparando RAG vs GraphRAG vs Hybrid..."):
                comparison = call_comparison_api(prompt, program_filter)

            if comparison:
                tab_rag, tab_graph, tab_hybrid = st.tabs([
                    "📄 RAG Vectorial", "🔗 GraphRAG", "🔄 Híbrido"
                ])

                for tab, key, label in [
                    (tab_rag, "rag_answer", "RAG"),
                    (tab_graph, "graph_answer", "GraphRAG"),
                    (tab_hybrid, "hybrid_answer", "Híbrido"),
                ]:
                    with tab:
                        data = comparison[key]
                        st.markdown(data["answer"])
                        display_confidence(data["confidence"])
                        st.caption(
                            f"Tiempo: {data.get('processing_time_ms', 0):.0f}ms"
                        )
                        display_sources(data.get("sources", []))

                # Guardar la respuesta híbrida en el historial
                hybrid_data = comparison["hybrid_answer"]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": hybrid_data["answer"],
                    "metadata": hybrid_data,
                })
            else:
                st.error("Error al comparar. Verificar conexión con la API.")
        else:
            with st.spinner("Buscando en los documentos..."):
                response = call_api(prompt, mode, program_filter)

            answer = response.get("formatted_answer") or response.get("answer", "")
            st.markdown(answer)

            display_confidence(response.get("confidence", 0))
            display_sources(response.get("sources", []))

            for w in response.get("warnings", []):
                st.warning(w)

            if response.get("fallback_contacts"):
                st.info(
                    "📧 Contactos sugeridos: "
                    + ", ".join(response["fallback_contacts"])
                )

            st.caption(
                f"Método: {response.get('method', 'N/A')} | "
                f"Tiempo: {response.get('processing_time_ms', 0):.0f}ms"
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "metadata": response,
            })
