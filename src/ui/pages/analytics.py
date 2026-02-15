"""
Dashboard de Analytics del Chatbot LSE-FIUBA.
Visualización de métricas, feedback y rendimiento del sistema.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import json
import sys
from pathlib import Path

import streamlit as st
import requests

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Analytics - Chatbot LSE-FIUBA",
    page_icon="📊",
    layout="wide",
)

API_URL = st.session_state.get("api_url", "http://localhost:8000")


def load_evaluation_report() -> dict:
    """Carga el reporte de evaluación desde archivo."""
    report_path = ROOT / "data" / "evaluation" / "evaluation_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def fetch_feedback_stats() -> dict:
    """Obtiene estadísticas de feedback desde la API."""
    try:
        response = requests.get(f"{API_URL}/api/v1/feedback/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


# ── Header ────────────────────────────────────────────────────
st.title("📊 Dashboard de Analytics")
st.caption("Métricas de rendimiento del Chatbot Administrativo LSE-FIUBA")

tab_eval, tab_ragas, tab_feedback, tab_system = st.tabs([
    "📈 Evaluación Comparativa",
    "🎯 Métricas RAGAS",
    "💬 Feedback de Usuarios",
    "⚙️ Sistema",
])

# ── Tab 1: Evaluación Comparativa ─────────────────────────────
with tab_eval:
    report = load_evaluation_report()

    if report is None:
        st.warning(
            "No se encontró reporte de evaluación. "
            "Ejecutar: `python run_evaluation.py`"
        )
    else:
        summary = report.get("summary", {})

        st.subheader("Resumen General")
        col1, col2, col3 = st.columns(3)

        methods = ["rag", "graph", "hybrid"]
        labels = ["RAG Vectorial", "GraphRAG", "Hybrid"]
        colors = ["blue", "orange", "green"]

        for col, method, label, color in zip(
            [col1, col2, col3], methods, labels, colors
        ):
            data = summary.get(method, {})
            with col:
                st.metric(label, f"{data.get('avg_keyword_hit', 0):.1%}",
                          delta=f"{data.get('wins', 0)} victorias")
                st.caption(
                    f"Confianza: {data.get('avg_confidence', 0):.1%} | "
                    f"Tiempo: {data.get('avg_time_ms', 0):.0f}ms"
                )

        # Tabla comparativa
        st.subheader("Comparación Detallada")
        comparison_data = {
            "Métrica": [
                "Keyword Hit Rate", "Confianza Promedio",
                "Tiempo (ms)", "Source Accuracy", "Victorias",
            ],
        }
        for method, label in zip(methods, labels):
            data = summary.get(method, {})
            comparison_data[label] = [
                f"{data.get('avg_keyword_hit', 0):.1%}",
                f"{data.get('avg_confidence', 0):.1%}",
                f"{data.get('avg_time_ms', 0):.0f}",
                f"{data.get('source_accuracy', 0):.1%}",
                str(data.get("wins", 0)),
            ]
        st.table(comparison_data)

        # Por categoría
        by_category = report.get("by_category", {})
        if by_category:
            st.subheader("Resultados por Categoría")
            for cat, cat_data in by_category.items():
                with st.expander(f"📁 {cat.title()} ({cat_data.get('count', 0)} preguntas)"):
                    cols = st.columns(3)
                    for col, method, label in zip(cols, methods, labels):
                        key = f"{method}_avg_keyword_hit"
                        val = cat_data.get(key, 0)
                        col.metric(label, f"{val:.1%}")

        # Abstención
        abstention = summary.get("abstention", {})
        if abstention.get("total", 0) > 0:
            st.subheader("Abstención (Preguntas fuera de dominio)")
            st.metric(
                "Tasa de abstención correcta",
                f"{abstention.get('accuracy', 0):.0%}",
                delta=f"{abstention.get('correct', 0)}/{abstention.get('total', 0)}"
            )

# ── Tab 2: Métricas RAGAS ─────────────────────────────────────
with tab_ragas:
    report = load_evaluation_report()

    if report is None:
        st.warning("No hay reporte de evaluación con métricas RAGAS.")
    else:
        ragas_data = report.get("ragas", {})
        if not ragas_data:
            st.info("Las métricas RAGAS no están disponibles en el reporte actual. "
                    "Re-ejecutar: `python run_evaluation.py`")
        else:
            st.subheader("Métricas RAGAS por Método")

            ragas_metrics = [
                ("Faithfulness", "avg_faithfulness",
                 "Porcentaje de claims respaldados por el contexto"),
                ("Answer Relevance", "avg_answer_relevance",
                 "Relevancia semántica de la respuesta a la pregunta"),
                ("Context Precision", "avg_context_precision",
                 "Porcentaje de contextos recuperados que son relevantes"),
                ("Context Recall", "avg_context_recall",
                 "Cobertura de la información necesaria en los contextos"),
                ("Overall Score", "avg_overall",
                 "Score ponderado global"),
            ]

            for metric_label, metric_key, description in ragas_metrics:
                st.markdown(f"**{metric_label}** - _{description}_")
                cols = st.columns(3)
                for col, method, label in zip(
                    cols, ["rag", "graph", "hybrid"],
                    ["RAG", "GraphRAG", "Hybrid"]
                ):
                    val = ragas_data.get(method, {}).get(metric_key, 0)
                    col.metric(label, f"{val:.1%}")
                st.divider()

# ── Tab 3: Feedback ───────────────────────────────────────────
with tab_feedback:
    feedback_stats = fetch_feedback_stats()

    if feedback_stats is None:
        st.warning(
            "No se pudo conectar con la API para obtener feedback. "
            "Asegurate de que la API esté corriendo."
        )
        # Intentar cargar desde archivo
        feedback_path = ROOT / "data" / "evaluation" / "feedback.json"
        if feedback_path.exists():
            with open(feedback_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            st.info(f"Cargadas {len(entries)} entradas de feedback desde archivo.")
            if entries:
                avg_rating = sum(e.get("rating", 0) for e in entries) / len(entries)
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Feedback", len(entries))
                col2.metric("Rating Promedio", f"{avg_rating:.1f}/5")
                col3.metric("Entradas recientes",
                            sum(1 for e in entries[-10:] if e.get("rating", 0) >= 4))
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Feedback", feedback_stats.get("total_entries", 0))
        col2.metric("Rating Promedio",
                     f"{feedback_stats.get('avg_rating', 0):.1f}/5")
        col3.metric("Tasa Correctas",
                     f"{feedback_stats.get('correct_rate', 0):.0%}")
        col4.metric("Tasa Completas",
                     f"{feedback_stats.get('complete_rate', 0):.0%}")

        # Distribución de ratings
        by_rating = feedback_stats.get("by_rating", {})
        if by_rating:
            st.subheader("Distribución de Ratings")
            chart_data = {
                "Rating": [f"{r} estrellas" for r in range(1, 6)],
                "Cantidad": [by_rating.get(str(r), 0) for r in range(1, 6)],
            }
            st.bar_chart(chart_data, x="Rating", y="Cantidad")

        # Por método
        by_method = feedback_stats.get("by_method", {})
        if by_method:
            st.subheader("Rating por Método de Retrieval")
            for method, data in by_method.items():
                st.markdown(
                    f"**{method.upper()}**: "
                    f"Rating {data.get('avg_rating', 0):.1f}/5 "
                    f"({data.get('count', 0)} respuestas)"
                )

        # Sugerencias de mejora
        suggestions = feedback_stats.get("improvement_suggestions", [])
        if suggestions:
            st.subheader("Sugerencias de Mejora")
            for suggestion in suggestions:
                st.warning(suggestion)

        # Preguntas peor valoradas
        low_rated = feedback_stats.get("low_rated_questions", [])
        if low_rated:
            st.subheader("Preguntas con Menor Valoración")
            for q in low_rated[:5]:
                with st.expander(
                    f"{'⭐' * q.get('rating', 1)} {q.get('question', '')[:80]}..."
                ):
                    st.markdown(f"**Método:** {q.get('method', 'N/A')}")
                    st.markdown(f"**Confianza:** {q.get('confidence', 0):.0%}")
                    if q.get("comment"):
                        st.markdown(f"**Comentario:** {q['comment']}")

# ── Tab 4: Sistema ────────────────────────────────────────────
with tab_system:
    st.subheader("Estado del Sistema")

    try:
        health = requests.get(f"{API_URL}/api/v1/health", timeout=5).json()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Estado",
                     "Online" if health.get("status") == "ok" else "Error")
        col2.metric("LLM Disponible",
                     "Si" if health.get("llm_available") else "No")
        col3.metric("Documentos", health.get("documents_loaded", 0))
        col4.metric("Vectores en FAISS", health.get("index_size", 0))

        st.metric("Nodos en Grafo", health.get("graph_nodes", 0))

    except Exception:
        st.error("No se pudo conectar con la API.")

    st.divider()

    st.subheader("Configuración Actual")
    st.markdown("""
    | Componente | Configuración |
    |---|---|
    | **LLM** | Ollama (llama3) / OpenAI API |
    | **Embeddings** | paraphrase-multilingual-MiniLM-L12-v2 (384 dims) |
    | **Vector DB** | FAISS IndexFlatIP |
    | **Grafo** | NetworkX + Louvain |
    | **Chunk Size** | 512 tokens, overlap 128 |
    | **Query Enhancement** | HyDE + Query Expansion |
    | **Anti-alucinación** | Faithfulness + Cross-ref + Abstención |
    | **Memoria** | Ventana deslizante + Resumen progresivo |
    """)
