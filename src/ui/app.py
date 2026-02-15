"""
Interfaz Streamlit del Chatbot Administrativo LSE-FIUBA.
Incluye memoria conversacional, feedback de usuarios y diseño profesional.

Autor: Juan Ruiz Otondo - CEIA FIUBA
Carrera de Especialización en Inteligencia Artificial
Laboratorio de Sistemas Embebidos - FIUBA - UBA
"""

import sys
import time
import uuid
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

# ── CSS Personalizado ─────────────────────────────────────────
st.markdown("""
<style>
    /* Header institucional */
    .institutional-header {
        background: linear-gradient(135deg, #003366 0%, #004a8f 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .institutional-header h1 {
        color: white;
        font-size: 1.5rem;
        margin: 0;
    }
    .institutional-header p {
        color: #b8d4e8;
        font-size: 0.85rem;
        margin: 0;
    }
    /* Badges de confianza */
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    /* Sidebar */
    .sidebar-info {
        background: #f0f4f8;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        font-size: 0.85rem;
    }
    /* Feedback stars */
    .feedback-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Estado de sesión ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()


# ── Funciones auxiliares ───────────────────────────────────────

def call_api(
    question: str,
    mode: str = "hybrid",
    program_filter: str = None,
    session_id: str = None,
) -> dict:
    """Llama al endpoint de chat de la API."""
    url = f"{st.session_state.api_url}/api/v1/chat"
    payload = {
        "question": question,
        "mode": mode,
        "session_id": session_id or st.session_state.session_id,
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
                "esté corriendo con: `python run_api.py`"
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
    except Exception:
        return None


def submit_feedback(question: str, answer: str, rating: int, method: str,
                    confidence: float, is_correct: bool = None,
                    is_complete: bool = None, comment: str = "") -> bool:
    """Envía feedback a la API."""
    url = f"{st.session_state.api_url}/api/v1/feedback"
    payload = {
        "session_id": st.session_state.session_id,
        "question": question,
        "answer": answer,
        "rating": rating,
        "method": method,
        "confidence": confidence,
        "is_correct": is_correct,
        "is_complete": is_complete,
        "user_comment": comment,
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


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


def display_feedback_form(msg_index: int, question: str, answer: str,
                          method: str, confidence: float):
    """Muestra formulario de feedback para una respuesta."""
    feedback_key = f"fb_{msg_index}"

    if feedback_key in st.session_state.feedback_given:
        st.success("Gracias por tu feedback.", icon="✅")
        return

    with st.expander("📝 Valorar esta respuesta", expanded=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            rating = st.slider(
                "Rating", 1, 5, 3, key=f"rating_{msg_index}",
                help="1=Muy mala, 5=Excelente"
            )
            is_correct = st.radio(
                "Respuesta correcta?", ["Si", "No", "No sé"],
                horizontal=True, key=f"correct_{msg_index}"
            )
        with col2:
            is_complete = st.radio(
                "Respuesta completa?", ["Si", "No", "No sé"],
                horizontal=True, key=f"complete_{msg_index}"
            )
            comment = st.text_input(
                "Comentario (opcional)", key=f"comment_{msg_index}"
            )

        if st.button("Enviar feedback", key=f"btn_fb_{msg_index}"):
            correct_map = {"Si": True, "No": False, "No sé": None}
            success = submit_feedback(
                question=question,
                answer=answer,
                rating=rating,
                method=method,
                confidence=confidence,
                is_correct=correct_map[is_correct],
                is_complete=correct_map[is_complete],
                comment=comment,
            )
            if success:
                st.session_state.feedback_given.add(feedback_key)
                st.success("Feedback enviado correctamente.")
                st.rerun()
            else:
                st.error("No se pudo enviar el feedback. Verificar API.")


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    # Logo FIUBA
    st.image(
        "https://lse.posgrados.fi.uba.ar/sites/default/files/color/lse_posgrados_fi_uba-b8f2a14f/logo.png",
        width=220,
    )

    st.divider()
    st.markdown("### ⚙️ Configuración")

    st.session_state.api_url = st.text_input(
        "URL de la API",
        value=st.session_state.api_url,
    )

    mode = st.selectbox(
        "Modo de retrieval",
        options=["hybrid", "rag", "graph"],
        format_func=lambda x: {
            "hybrid": "🔄 Híbrido (RAG + GraphRAG)",
            "rag": "📄 RAG Vectorial",
            "graph": "🔗 GraphRAG",
        }[x],
    )

    program_filter = st.selectbox(
        "Filtrar por programa",
        options=["Todos", "CEIA", "CESE", "CEIoT", "MIA", "MIAE", "MIoT", "MCB"],
    )

    comparison_mode = st.toggle("Modo comparación", value=False)

    st.divider()

    # Técnicas activas
    st.markdown("### 🧠 Técnicas Activas")
    st.markdown("""
    <div class="sidebar-info">
    ✅ RAG + FAISS vectorial<br>
    ✅ GraphRAG + NetworkX<br>
    ✅ HyDE (Doc. Hipotético)<br>
    ✅ Query Expansion<br>
    ✅ Cross-Encoder Reranking<br>
    ✅ Anti-alucinación multi-capa<br>
    ✅ Memoria conversacional
    </div>
    """, unsafe_allow_html=True)

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

    # Controles de sesión
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Limpiar chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.feedback_given = set()
            st.rerun()
    with col_b:
        st.caption(f"Sesión: {st.session_state.session_id[:8]}...")

    st.divider()
    st.markdown(
        "**Trabajo Final** de Juan Ruiz Otondo\n\n"
        "Carrera de Especialización en IA\n\n"
        "LSE - FIUBA - UBA"
    )

# ── Header institucional ─────────────────────────────────────
st.markdown("""
<div class="institutional-header">
    <div>
        <h1>🎓 Asistente Administrativo de Posgrados</h1>
        <p>Laboratorio de Sistemas Embebidos (LSE) - Facultad de Ingeniería - Universidad de Buenos Aires</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Mostrar historial de mensajes ─────────────────────────────
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("metadata"):
            meta = message["metadata"]
            display_confidence(meta.get("confidence", 0))
            display_sources(meta.get("sources", []))
            if meta.get("warnings"):
                for w in meta["warnings"]:
                    st.warning(w)

            # Formulario de feedback solo para mensajes del asistente
            if message["role"] == "assistant" and meta.get("method") != "error":
                display_feedback_form(
                    msg_index=i,
                    question=st.session_state.messages[i - 1]["content"]
                    if i > 0 else "",
                    answer=message["content"],
                    method=meta.get("method", "hybrid"),
                    confidence=meta.get("confidence", 0),
                )

# ── Input del usuario ─────────────────────────────────────────
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
