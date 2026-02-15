# Chatbot Administrativo Inteligente - Posgrados LSE-FIUBA

**Trabajo Final** de la Carrera de Especialización en Inteligencia Artificial
Laboratorio de Sistemas Embebidos (LSE) - Facultad de Ingeniería - Universidad de Buenos Aires

**Autor:** Juan Ruiz Otondo

---

## Descripción

Agente administrativo inteligente (chatbot) basado en Procesamiento de Lenguaje Natural para la unidad de Posgrado del Laboratorio de Sistemas Embebidos (LSE) de FIUBA. El sistema responde consultas de estudiantes sobre reglamentos, carreras de especialización (CEIA, CESE, CEIoT), maestrías (MIA, MIAE, MIoT, MCB), procesos administrativos y preguntas frecuentes.

### Características principales

- **RAG Vectorial (FAISS):** Retrieval-Augmented Generation con base de datos vectorial para búsqueda semántica
- **GraphRAG (NetworkX):** Grafo de conocimiento con entidades académicas y sus relaciones
- **Sistema Híbrido:** Combinación inteligente de ambos sistemas con routing basado en tipo de consulta
- **Anti-alucinación multi-capa:** Verificación de fidelidad, abstención, cross-referencing
- **Citaciones automáticas:** Trazabilidad completa con fuentes y secciones
- **Pipeline automatizado:** Procesamiento incremental de nuevos documentos
- **Evaluación comparativa:** Framework de testing RAG vs GraphRAG vs Hybrid

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    Interfaz (Streamlit)                      │
├─────────────────────────────────────────────────────────────┤
│                      API (FastAPI)                           │
├──────────────┬──────────────┬───────────────────────────────┤
│              │   Answer     │                               │
│   Hybrid     │ Synthesizer  │   Anti-Hallucination Engine   │
│  Retriever   │  + Citation  │   (Faithfulness + Abstention) │
│              │   Manager    │                               │
├──────┬───────┴──────┬───────┴───────────────────────────────┤
│      │              │                                       │
│  RAG │   GraphRAG   │        LLM Provider                   │
│ FAISS│  NetworkX    │     (Ollama / OpenAI)                 │
│      │              │                                       │
├──────┴──────────────┴───────────────────────────────────────┤
│              Data Pipeline                                   │
│  PDF Extraction → Cleaning → Chunking → Metadata            │
└─────────────────────────────────────────────────────────────┘
```

## Estructura del proyecto

```
chatbot-lse-posgrados/
├── config/
│   └── settings.py              # Configuración centralizada (Pydantic)
├── data/
│   ├── raw/                     # PDFs originales
│   ├── processed/               # Chunks procesados (JSON)
│   ├── indexes/                 # Índice FAISS
│   ├── graphs/                  # Grafo de conocimiento (GraphML + Pickle)
│   └── evaluation/              # Reportes de evaluación
├── src/
│   ├── data_pipeline/           # Pipeline de procesamiento
│   │   ├── pdf_extractor.py     # Extracción dual (PyMuPDF + pdfplumber)
│   │   ├── text_cleaner.py      # Normalización en español
│   │   ├── chunker.py           # Chunking multi-estrategia
│   │   ├── metadata_extractor.py# Extracción de metadata académica
│   │   └── pipeline_orchestrator.py # Orquestador con detección de cambios
│   ├── rag/                     # RAG Vectorial
│   │   ├── embeddings.py        # Sentence-Transformers multilingual
│   │   ├── vector_store.py      # FAISS IndexFlatIP + MMR
│   │   ├── retriever.py         # Retriever con cross-encoder reranking
│   │   └── rag_chain.py         # Cadena RAG completa
│   ├── graph_rag/               # GraphRAG
│   │   ├── entity_extractor.py  # Extracción de 10 tipos de entidades
│   │   ├── relationship_mapper.py # 11 tipos de relaciones académicas
│   │   ├── graph_builder.py     # Constructor de grafo NetworkX
│   │   ├── graph_retriever.py   # Retrieval basado en grafo
│   │   └── community_detector.py# Detección de comunidades (Louvain)
│   ├── hybrid/                  # Sistema híbrido
│   │   ├── hybrid_retriever.py  # Combinación RAG + GraphRAG
│   │   ├── anti_hallucination.py# Motor anti-alucinación
│   │   ├── citation_manager.py  # Gestión de citaciones
│   │   └── answer_synthesizer.py# Síntesis de respuesta final
│   ├── llm/                     # Proveedores LLM
│   │   ├── llm_provider.py      # Abstracción Ollama/OpenAI
│   │   └── prompts.py           # Templates en español
│   ├── api/                     # API REST
│   │   ├── main.py              # App FastAPI
│   │   ├── schemas.py           # Modelos Pydantic
│   │   ├── dependencies.py      # Inyección de dependencias
│   │   └── routes/
│   │       ├── chat.py          # Endpoints /chat y /chat/compare
│   │       └── health.py        # Endpoints /health y /stats
│   ├── ui/                      # Interfaz
│   │   └── app.py               # Aplicación Streamlit
│   └── evaluation/              # Evaluación
│       ├── evaluator.py         # Evaluador comparativo
│       └── test_sets.py         # Conjunto de preguntas con ground truth
├── tests/                       # Tests unitarios e integración
│   ├── test_data_pipeline/
│   ├── test_rag/
│   ├── test_graph_rag/
│   ├── test_hybrid/
│   └── test_api/
├── run_pipeline.py              # Ejecutar pipeline de datos
├── run_api.py                   # Lanzar API
├── run_app.py                   # Lanzar interfaz Streamlit
├── run_evaluation.py            # Ejecutar evaluación comparativa
├── requirements.txt             # Dependencias
├── pytest.ini                   # Configuración de tests
├── .env.example                 # Variables de entorno template
└── .gitignore
```

## Instalación y Configuración

### Prerrequisitos

- Python 3.10+
- [Ollama](https://ollama.ai/) instalado (para LLM local gratuito)
- 4 GB de RAM mínimo (8 GB recomendado)

### Paso 1: Clonar e instalar dependencias

```bash
git clone https://github.com/<tu-usuario>/chatbot-lse-posgrados.git
cd chatbot-lse-posgrados
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### Paso 2: Configurar el LLM

```bash
# Instalar y correr Ollama (opción gratuita recomendada)
# Descargar desde https://ollama.ai/

# Descargar modelo (elegir uno):
ollama pull llama3          # 4.7 GB - Recomendado
ollama pull llama3:8b       # Variante 8B
ollama pull mistral         # 4.1 GB - Alternativa
```

### Paso 3: Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env según tu configuración
```

### Paso 4: Colocar los documentos PDF

Colocar los PDFs en el directorio `data/raw/`, o el pipeline los copiará automáticamente desde el directorio padre.

### Paso 5: Ejecutar el pipeline de datos

```bash
python run_pipeline.py
```

Esto ejecuta:
1. Extracción de texto y tablas de los PDFs
2. Limpieza y normalización del texto
3. Chunking inteligente (512 tokens, overlap 128)
4. Generación de embeddings (sentence-transformers multilingual)
5. Indexación en FAISS
6. Construcción del grafo de conocimiento (NetworkX)
7. Detección de comunidades (Louvain)

### Paso 6: Lanzar la API y la interfaz

```bash
# Terminal 1: API
python run_api.py

# Terminal 2: Interfaz
python run_app.py
```

Acceder a:
- **Chatbot:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

## Uso

### Interfaz Streamlit

La interfaz permite:
- Hacer consultas en lenguaje natural
- Seleccionar modo de retrieval (RAG / GraphRAG / Híbrido)
- Filtrar por programa académico
- Ver confianza de la respuesta
- Expandir fuentes citadas
- Activar modo comparación (RAG vs GraphRAG vs Hybrid)

### API REST

```bash
# Consulta simple
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuál es la asistencia mínima?", "mode": "hybrid"}'

# Comparación de métodos
curl -X POST http://localhost:8000/api/v1/chat/compare \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuáles son los requisitos de la MIA?"}'
```

## Evaluación

### Ejecutar evaluación completa

```bash
python run_evaluation.py
```

### Evaluación rápida (5 preguntas)

```bash
python run_evaluation.py --quick
```

### Evaluar por categoría

```bash
python run_evaluation.py --category factual
python run_evaluation.py --category procedural
python run_evaluation.py --category comparative
```

El reporte se genera en `data/evaluation/evaluation_report.json` e incluye:
- Keyword Hit Rate por método
- Confianza promedio
- Tiempo de respuesta
- Precisión de fuentes
- Abstención correcta (preguntas fuera de dominio)
- Desglose por categoría y dificultad

## Tests

```bash
# Todos los tests
pytest

# Tests rápidos (sin modelos ML)
pytest -m "not slow"

# Tests por módulo
pytest tests/test_data_pipeline/
pytest tests/test_rag/
pytest tests/test_graph_rag/
pytest tests/test_hybrid/
pytest tests/test_api/
```

## Agregar nuevos documentos

1. Colocar el nuevo PDF en `data/raw/`
2. Ejecutar: `python run_pipeline.py --doc nombre_del_archivo.pdf`
3. El pipeline procesará solo el nuevo documento (procesamiento incremental)

Para forzar reprocesamiento completo:
```bash
python run_pipeline.py --force
```

## Mecanismos anti-alucinación

1. **Verificación por embeddings:** Cada claim de la respuesta se compara semánticamente con el contexto fuente
2. **Cross-referencing:** Consistencia entre información de RAG y GraphRAG
3. **Abstención inteligente:** El sistema se abstiene cuando la confianza es baja o la pregunta está fuera de dominio
4. **Contactos de fallback:** Sugiere emails de contacto relevantes cuando no puede responder
5. **Citaciones obligatorias:** Toda respuesta incluye fuentes verificables

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| LLM | Ollama (llama3) / OpenAI API |
| Embeddings | sentence-transformers (multilingual-MiniLM-L12-v2) |
| Vector DB | FAISS (IndexFlatIP) |
| Graph DB | NetworkX + python-louvain |
| API | FastAPI + uvicorn |
| UI | Streamlit |
| PDF Processing | PyMuPDF + pdfplumber |
| Testing | pytest |

## Documentos procesados

| Documento | Tipo | Descripción |
|---|---|---|
| CEIA.pdf | Resolución | Plan de estudios - Esp. en Inteligencia Artificial |
| CESE.pdf | Resolución | Plan de estudios - Esp. en Sistemas Embebidos |
| CEIoT.pdf | Resolución | Plan de estudios - Esp. en Internet de las Cosas |
| MIAE.pdf | Resolución | Plan de estudios - Maestría en IA Embebida |
| MIoT.pdf | Resolución | Plan de estudios - Maestría en IoT |
| MCB.pdf | Resolución | Plan de estudios - Maestría en Ciberseguridad |
| MIA-AE1-Programa.pdf | Programa | Programa de materia MIA |
| Reglamento...2025.pdf | Reglamento | Reglamento de cursada y asistencia |
| FAQ - MIA.pdf | FAQ | Preguntas frecuentes MIA |
| FAQ - GdP...pdf | FAQ | Preguntas frecuentes GdP, GTI, TTFA, TTFB |
| FAQ - Optativas.pdf | FAQ | Preguntas frecuentes materias optativas |
| LSE-FIUBA-Trabajo-Final.pdf | Reglamento | Reglamento de trabajo final |
| Programa de Vinculación.pdf | Vinculación | Programa de vinculación profesional |

---

**Laboratorio de Sistemas Embebidos (LSE)** - Facultad de Ingeniería - Universidad de Buenos Aires
