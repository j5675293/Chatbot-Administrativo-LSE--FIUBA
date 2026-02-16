# 🤖 ASESOR ADMINISTRATIVO INTELIGENTE MEDIANTE PROCESAMEINTO DE LENGUAJE NATURAL - POSGRADO LSE-FIUBA

<img width="1875" height="866" alt="IA -LSE" src="https://github.com/user-attachments/assets/bf393332-4cd2-4bae-b878-e256184c3493" />

**Trabajo Final** de la Carrera de Especialización en Inteligencia Artificial
Laboratorio de Sistemas Embebidos (LSE) - Facultad de Ingeniería - Universidad de Buenos Aires

**Autor:** Juan Ruiz Otondo - a1702

---

## Descripción

Agente administrativo inteligente (chatbot) basado en Procesamiento de Lenguaje Natural (PLN) para la unidad de Posgrado del Laboratorio de Sistemas Embebidos (LSE) de FIUBA. El sistema responde consultas de la comunidad universitaria sobre reglamentos, carreras de especialización (CEIA, CESE, CEIoT), maestrías (MIA, MIAE, MIoT, MCB), procesos administrativos y preguntas frecuentes.
El sistema implementa una arquitectura en 5 capas que combina técnicas avanzadas de RAG (Retrieval-Augmented Generation) con GraphRAG y mecanismos anti-alucinación de triple verificación para garantizar respuestas precisas y verificables.

### Características principales

- **RAG Vectorial (FAISS):** Retrieval-Augmented Generation con base de datos vectorial para búsqueda semántica
- **GraphRAG (NetworkX):** Grafo de conocimiento con entidades académicas y sus relaciones
- **Sistema Híbrido:** Combinación inteligente de ambos sistemas con routing basado en tipo de consulta
- **HyDE (Hypothetical Document Embeddings):** Genera documentos hipotéticos para mejorar el retrieval
- **Query Expansion:** Expansión automática de consultas con sinónimos del dominio y LLM
- **Anti-alucinación multi-capa:** Verificación de fidelidad, abstención, cross-referencing
- **Memoria conversacional:** Ventana deslizante con resumen progresivo y contextualización de queries
- **Feedback Human-in-the-Loop:** Sistema de valoración y mejora continua basada en usuarios
- **Métricas RAGAS:** Evaluación con faithfulness, answer relevance, context precision y recall
- **Citaciones automáticas:** Trazabilidad completa con fuentes y secciones
- **Pipeline automatizado:** Procesamiento incremental de nuevos documentos
- **Evaluación comparativa:** Framework de testing RAG vs GraphRAG vs Hybrid
- **Analytics Dashboard:** Visualización de métricas y feedback del sistema
- **Docker Compose:** Despliegue completo con un solo comando

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│           Interfaz Streamlit (+ Analytics Dashboard)             │
├─────────────────────────────────────────────────────────────────┤
│                    API FastAPI + Feedback                        │
├──────────────┬──────────────┬───────────────────────────────────┤
│              │   Answer     │                                   │
│   Hybrid     │ Synthesizer  │   Anti-Hallucination Engine       │
│  Retriever   │  + Citation  │   (Faithfulness + Abstention +    │
│              │   Manager    │    Cross-Reference)               │
├──────┬───────┴──────┬───────┴───────────────────────────────────┤
│      │              │                                           │
│  RAG │   GraphRAG   │   Query Enhancement                      │
│ FAISS│  NetworkX    │   (HyDE + Query Expansion)               │
│      │              │                                           │
├──────┴──────────────┴───────────────────────────────────────────┤
│    Conversation Memory          LLM Provider                    │
│  (Window + Summary)          (Ollama / OpenAI)                  │
├─────────────────────────────────────────────────────────────────┤
│              Data Pipeline                                       │
│  PDF Extraction → Cleaning → Chunking → Metadata                │
├─────────────────────────────────────────────────────────────────┤
│          Evaluation (RAGAS + Benchmark + Feedback)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Flujo de Procesamiento de Consulta

```mermaid
graph TD
    A[👤 Usuario ingresa consulta] --> B[🖥️ Streamlit UI + Analytics]
    B -->|HTTP POST| C[⚡ FastAPI]
    
    C --> D1[📝 Conversation Memory]
    C --> D2[🔍 Query Enhancement]
    
    D1 -->|Contexto histórico| E[Reformulación contextual]
    D2 -->|HyDE + Expansions| E
    
    E --> F{🔀 Hybrid Retriever}
    
    F -->|Paralelo| G1[🔍 RAG/FAISSQuery + HyDE + Expansions]
    F -->|Paralelo| G2[🕸️ GraphRAG/NetworkXEntities + Relations]
    
    G1 --> H[📊 Reciprocal Rank FusionPesos Adaptativos]
    G2 --> H
    
    H --> I[✍️ Answer Synthesizer+ Citation Manager]
    
    I --> J[🛡️ Anti-Hallucination Engine]
    
    J --> K1[✅ Faithfulness CheckNLI Score ≥ 0.75]
    J --> K2[🔗 Cross-Reference0 conflicts]
    J --> K3[📊 AbstentionConfidence ≥ 0.65]
    
    K1 --> L{Pass All 3?}
    K2 --> L
    K3 --> L
    
    L -->|Sí| M[✅ Respuesta Aprobada]
    L -->|No| N[🚫 Abstención Honesta]
    
    M --> O[🖥️ Streamlit Renderiza]
    N --> O
    
    O --> P[👍 Feedback del Usuario]
    P -->|Rating ≤ 2| Q[📈 Failure Analysis→ Test Set]
    P -->|Rating ≥ 4| R[✅ Success Tracking]
    
    style F fill:#FFF3E0
    style G1 fill:#E3F2FD
    style G2 fill:#E8F5E9
    style J fill:#FFEBEE
    style L fill:#FFF9C4
    style M fill:#C8E6C9
    style N fill:#FFCDD2
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
│   └── evaluation/              # Reportes + Feedback
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
│   │   ├── rag_chain.py         # Cadena RAG completa
│   │   ├── hyde.py              # HyDE - Hypothetical Document Embeddings
│   │   └── query_expansion.py   # Expansión de queries con LLM y sinónimos
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
│   │   ├── answer_synthesizer.py# Síntesis de respuesta final
│   │   └── conversation_memory.py # Memoria conversacional
│   ├── llm/                     # Proveedores LLM
│   │   ├── llm_provider.py      # Abstracción Ollama/OpenAI
│   │   └── prompts.py           # Templates en español
│   ├── api/                     # API REST
│   │   ├── main.py              # App FastAPI
│   │   ├── schemas.py           # Modelos Pydantic
│   │   ├── dependencies.py      # Inyección de dependencias
│   │   └── routes/
│   │       ├── chat.py          # /chat, /chat/compare, /feedback
│   │       └── health.py        # /health y /stats
│   ├── ui/                      # Interfaz
│   │   ├── app.py               # Aplicación Streamlit principal
│   │   └── pages/
│   │       └── analytics.py     # Dashboard de analytics
│   └── evaluation/              # Evaluación
│       ├── evaluator.py         # Evaluador comparativo + RAGAS
│       ├── test_sets.py         # Conjunto de preguntas con ground truth
│       ├── ragas_metrics.py     # Métricas RAGAS
│       └── feedback.py          # Sistema de feedback
├── tests/                       # Tests unitarios e integración
├── Dockerfile                   # Imagen Docker
├── docker-compose.yml           # Orquestación de servicios
├── run_pipeline.py              # Ejecutar pipeline de datos
├── run_api.py                   # Lanzar API
├── run_app.py                   # Lanzar interfaz Streamlit
├── run_evaluation.py            # Ejecutar evaluación comparativa
├── requirements.txt             # Dependencias
├── pytest.ini                   # Configuración de tests
└── .env.example                 # Variables de entorno template
```

## 🔄 Flujo de Procesamiento de Consulta

```mermaid
graph TD
    A[👤 Usuario ingresa consulta] --> B[🖥️ Streamlit UI]
    B -->|HTTP POST| C[⚡ FastAPI /chat]
    C --> D{🔀 Hybrid Retriever}
    
    D -->|Paralelo| E[🔍 RAG/FAISSBúsqueda Vectorial]
    D -->|Paralelo| F[🕸️ GraphRAG/NetworkXBúsqueda en Grafo]
    
    E --> G[📊 Fusión RRF]
    F --> G
    
    G --> H[✍️ Answer Synthesizer]
    H --> I[📚 Citation Manager]
    I --> J[🛡️ Anti-Hallucination Engine]
    
    J -->|Faithfulness| K{✅ Score ≥ 0.6?}
    K -->|Sí| L[📤 Respuesta con citas]
    K -->|No| M[🚫 Abstención honesta]
    
    L --> N[🖥️ Streamlit renderiza]
    M --> N
    N --> O[👤 Usuario recibe respuesta]
    
    style D fill:#FFF3E0
    style E fill:#E3F2FD
    style F fill:#E8F5E9
    style J fill:#FFEBEE
    style K fill:#FFF9C4
```

## 🛠️ Stack Tecnológico

### **Backend & Core**
| Componente | Tecnología | Propósito |
|------------|------------|-----------|
| Framework API | `FastAPI` + `Uvicorn` | Servicios REST asíncronos |
| Validación | `Pydantic` | Schemas y configuración |
| Embeddings | `Sentence-Transformers` | Vectorización semántica multilingüe |
| Vector Search | `FAISS` | Búsqueda de similitud ultra-rápida |
| Graph Analysis | `NetworkX` | Análisis de grafo de conocimiento |
| Community Detection | `Louvain` | Clustering temático |
| Re-ranking | `Cross-Encoder` | Refinamiento de resultados |
| NLI | `DeBERTa-v3` | Verificación de fidelidad |
| PDF Processing | `PyMuPDF` + `pdfplumber` | Extracción dual de PDFs |
| OCR | `Tesseract` | Documentos escaneados |

### **LLM Providers**
| Modo | Proveedor | Modelos |
|------|-----------|---------|
| Local | `Ollama` | Llama 3.1 (70B), Mistral 7B |
| Cloud | `OpenAI` | GPT-4, GPT-4 Turbo |

### **Frontend**
| Componente | Tecnología |
|------------|------------|
| UI Framework | `Streamlit` |
| HTTP Client | `requests` |

### **Testing & Quality**
| Componente | Tecnología |
|------------|------------|
| Testing | `pytest` |
| Coverage | `pytest-cov` |
| Type Checking | `mypy` |

## 📊 Mapeo Arquitectura → Código

| Capa Arquitectónica | Directorio/Módulo | Archivos Principales |
|---------------------|-------------------|----------------------|
| **Capa 1**: Interfaz | `src/ui/` | `app.py`, `run_app.py` |
| **Capa 2**: API | `src/api/` | `main.py`, `schemas.py`, `routes/*`, `run_api.py` |
| **Capa 3**: Core | `src/rag/`<br/>`src/graph_rag/`<br/>`src/hybrid/` | `hybrid_retriever.py`<br/>`anti_hallucination.py`<br/>`answer_synthesizer.py` |
| **Capa 4**: LLM Provider | `src/llm/` | `llm_provider.py`, `prompts.py` |
| **Capa 5**: Data Pipeline | `src/data_pipeline/` | `pdf_extractor.py`<br/>`text_cleaner.py`<br/>`chunker.py`<br/>`pipeline_orchestrator.py` |


## Instalación y Configuración

### Prerrequisitos

- Python 3.10+
- [Ollama](https://ollama.ai/) instalado (para LLM local gratuito)
- 4 GB de RAM mínimo (8 GB recomendado)

### Opción A: Instalación manual

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

### Opción B: Docker Compose (recomendado para producción)

```bash
# Iniciar todos los servicios (Ollama + API + UI)
docker-compose up -d

# Ver logs
docker-compose logs -f api

# Ejecutar pipeline de datos
docker-compose run --rm pipeline

# Detener
docker-compose down
```

### Configurar el LLM

```bash
# Descargar modelo (elegir uno):
ollama pull llama3          # 4.7 GB - Recomendado
ollama pull llama3:8b       # Variante 8B
ollama pull mistral         # 4.1 GB - Alternativa
```

### Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env según tu configuración
```

### Colocar documentos y ejecutar pipeline

```bash
# Colocar PDFs en data/raw/
python run_pipeline.py
```

### Lanzar la API y la interfaz

```bash
# Terminal 1: API
python run_api.py

# Terminal 2: Interfaz
python run_app.py
```

Acceder a:
- **Chatbot:** http://localhost:8501
- **Analytics:** http://localhost:8501/analytics
- **API Docs:** http://localhost:8000/docs

## Uso

### Ejemplos de consultas

```
# Preguntas factuales
"¿Cuál es el porcentaje mínimo de asistencia requerido?"
"¿Qué título otorga la CEIA?"
"¿Cuántos bimestres dura la especialización?"

# Preguntas procedimentales
"¿Cómo me inscribo en Gestión de Proyectos?"
"¿Qué tengo que hacer para solicitar una prórroga?"
"¿Cómo es el proceso de defensa del trabajo final?"

# Preguntas comparativas (mejor con Hybrid/GraphRAG)
"¿Cuál es la diferencia entre MIAE y MIA?"
"¿Qué maestrías puedo hacer después de la CESE?"
"¿Cuáles son los requisitos de la MIA y qué especialización necesito?"

# Preguntas de contacto
"¿A quién contacto para dudas sobre inscripción?"
"¿Cuál es el email de gestión académica?"

# Preguntas con memoria conversacional
"¿Cuáles son los requisitos de la CEIA?"  →  (respuesta)
"¿Y cuántos bimestres dura?"              →  contextualiza automáticamente a CEIA

# Preguntas fuera de dominio (abstención correcta)
"¿Cuánto cuesta la carrera?" → Abstención + contacto de fallback
"¿Qué opinás sobre la UTN?" → Fuera de alcance
```

### API REST

```bash
# Consulta simple
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuál es la asistencia mínima?", "mode": "hybrid"}'

# Consulta con memoria conversacional
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Y cuántos bimestres dura?", "mode": "hybrid", "session_id": "sesion-1"}'

# Comparación de métodos
curl -X POST http://localhost:8000/api/v1/chat/compare \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuáles son los requisitos de la MIA?"}'

# Enviar feedback
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuál es la asistencia mínima?", "answer": "75%", "rating": 5, "is_correct": true}'

# Ver estadísticas de feedback
curl http://localhost:8000/api/v1/feedback/stats
```

## Evaluación

### Ejecutar evaluación completa (con RAGAS)

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

### Métricas incluidas

| Métrica | Descripción |
|---|---|
| **Keyword Hit Rate** | Porcentaje de palabras clave esperadas en la respuesta |
| **RAGAS Faithfulness** | Claims de la respuesta respaldados por el contexto |
| **RAGAS Answer Relevance** | Relevancia semántica respuesta-pregunta |
| **RAGAS Context Precision** | Porcentaje de contextos recuperados relevantes |
| **RAGAS Context Recall** | Cobertura de información necesaria en contextos |
| **Source Accuracy** | Coincidencia de fuentes esperadas vs recuperadas |
| **Abstención correcta** | Detección de preguntas fuera de dominio |
| **Tiempo de respuesta** | Latencia en milisegundos por método |

El reporte se genera en `data/evaluation/evaluation_report.json` y se visualiza en el dashboard de analytics.

### Benchmark de referencia: RAG vs GraphRAG vs Hybrid

| Tipo de pregunta | Mejor método | Razón |
|---|---|---|
| Datos específicos (nota mínima, plazos) | RAG | Información textual directa en los documentos |
| Relaciones entre programas (requisitos) | GraphRAG | Navegación por entidades y relaciones en el grafo |
| Comparaciones entre carreras | Hybrid | Combina texto descriptivo + estructura relacional |
| Contactos y emails | RAG | Datos puntuales en documentos FAQ |
| Caminos de formación (CESE → maestría) | GraphRAG | Paths entre nodos del grafo |
| Requisitos + descripción completa | Hybrid | Necesita ambas fuentes de información |

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

## Técnicas avanzadas implementadas

### HyDE (Hypothetical Document Embeddings)

Basado en [Gao et al., 2022]. En lugar de buscar directamente por la query del usuario, el sistema:
1. Genera un "documento hipotético" con el LLM que responde la pregunta
2. Usa el embedding de ese documento hipotético para buscar en FAISS
3. Fusiona el embedding HyDE con el embedding directo (alpha configurable)
4. Re-rankea contra la query original para mantener relevancia

Esto mejora el retrieval porque el documento hipotético tiene vocabulario más similar a los documentos reales que la query del usuario.

### Query Expansion

El sistema expande cada consulta de tres formas:
1. **Sinónimos del dominio:** Diccionario específico del LSE-FIUBA (ej: "requisito" → "condición", "materia" → "asignatura")
2. **Reformulaciones LLM:** Genera 3 variantes de la pregunta con diferentes palabras clave
3. **Fusión de resultados:** Combina y re-rankea resultados de todas las variantes

### Memoria conversacional

- **Ventana deslizante:** Mantiene los últimos N turnos de conversación
- **Resumen progresivo:** Comprime turnos viejos en un resumen con LLM
- **Contextualización:** Detecta pronombres y referencias anafóricas, reformula la query para que sea autocontenida
- **Tracking de tópicos:** Identifica programas y temas discutidos en la sesión

### Anti-alucinación multi-capa

7 capas de protección:
1. Verificación de fidelidad por embeddings (similitud claim-contexto)
2. Verificación de fidelidad por LLM (análisis de claims)
3. Verificación heurística (matching de datos específicos)
4. Cross-referencing RAG-GraphRAG (consistencia entre fuentes)
5. Abstención inteligente (confianza baja o fuera de dominio)
6. Contactos de fallback (sugiere emails relevantes)
7. Citaciones obligatorias (trazabilidad a fuentes)

## Casos de fallo conocidos y limitaciones

### Limitaciones del sistema

| Limitación | Descripción | Mitigación |
|---|---|---|
| **Dependencia de LLM** | La calidad depende del modelo LLM disponible | Fallback heurístico cuando LLM no está disponible |
| **Cobertura de documentos** | Solo responde sobre los 13 PDFs del corpus | Abstención + contacto de fallback para preguntas no cubiertas |
| **Idioma** | Optimizado para español rioplatense | Embeddings multilingües, pero prompts en español |
| **Actualización manual** | Los documentos deben actualizarse manualmente | Pipeline incremental con detección de cambios SHA-256 |
| **Latencia** | Embedding + LLM puede tomar 2-10 segundos | Cross-encoder reranking agrega latencia pero mejora precisión |
| **Información de costos** | No maneja información de aranceles | Abstención correcta para preguntas de costos |

### Casos de fallo documentados

1. **Preguntas ambiguas sin programa:** Cuando el usuario pregunta "¿cuáles son los requisitos?" sin especificar programa, el sistema puede mezclar información de múltiples carreras.
   - *Mitigación:* Usar filtro por programa en la UI o clarificar en la pregunta.

2. **Preguntas sobre regulaciones muy recientes:** Si el reglamento cambió después de los PDFs procesados, la información puede estar desactualizada.
   - *Mitigación:* Re-ejecutar pipeline cuando se actualicen documentos.

3. **Preguntas multi-hop complejas:** Consultas que requieren razonar sobre más de 3 saltos en el grafo pueden perder contexto.
   - *Mitigación:* GraphRAG con profundidad configurable; complementar con RAG.

4. **Tablas complejas en PDFs:** Algunas tablas de planes de estudio con formatos irregulares pueden no extraerse perfectamente.
   - *Mitigación:* Extracción dual PyMuPDF + pdfplumber con fallback.

5. **Preguntas en inglés:** El sistema responde en español aunque se pregunte en inglés; la calidad de retrieval puede disminuir.
   - *Mitigación:* Embeddings multilingües ayudan parcialmente.

### Evolución futura

- Fine-tuning del modelo de embeddings para el dominio académico
- Graph Neural Networks para node embeddings más expresivos
- Soporte multimodal (diagramas y tablas de los PDFs)
- Active learning con el feedback recolectado
- Migración a microservicios para escalabilidad

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| LLM | Ollama (llama3) / OpenAI API |
| Embeddings | sentence-transformers (multilingual-MiniLM-L12-v2, 384 dims) |
| Vector DB | FAISS (IndexFlatIP) |
| Graph DB | NetworkX + python-louvain |
| Query Enhancement | HyDE + Query Expansion + Cross-Encoder Reranking |
| Anti-alucinación | Faithfulness check + Cross-reference + Abstención |
| Memoria | Ventana deslizante + Resumen progresivo |
| Evaluación | RAGAS (Faithfulness, Answer Relevance, Context Precision, Recall) |
| Feedback | Human-in-the-Loop con almacenamiento JSON |
| API | FastAPI + uvicorn |
| UI | Streamlit (chat + analytics dashboard) |
| PDF Processing | PyMuPDF + pdfplumber |
| Deployment | Docker Compose (Ollama + API + UI) |
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
