"""
Conjuntos de preguntas de evaluación para el Chatbot LSE-FIUBA.
Incluye ground truth para validación automática.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

# ── Preguntas con respuesta esperada (ground truth) ──────────────
EVALUATION_QA_PAIRS = [
    # --- REGLAMENTO DE CURSADA ---
    {
        "id": "REG-001",
        "question": "¿Cuál es el porcentaje mínimo de asistencia requerido?",
        "expected_answer": "75%",
        "expected_keywords": ["75", "asistencia", "obligatoria"],
        "expected_source": "Reglamento de Cursada",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "REG-002",
        "question": "¿Cuál es la nota mínima para aprobar una materia?",
        "expected_answer": "4 (cuatro)",
        "expected_keywords": ["4", "cuatro", "mínima", "aprobar"],
        "expected_source": "Reglamento de Cursada",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "REG-003",
        "question": "¿Cuántos bimestres dura la cursada de una especialización?",
        "expected_answer": "La cursada se organiza en 5 bimestres por año, con clases de 8 semanas.",
        "expected_keywords": ["5", "bimestres", "8", "semanas"],
        "expected_source": "Reglamento de Cursada",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "REG-004",
        "question": "¿Cuál es el plazo máximo para completar la especialización?",
        "expected_answer": "10 bimestres",
        "expected_keywords": ["10", "bimestres"],
        "expected_source": "Reglamento de Cursada",
        "category": "factual",
        "difficulty": "medium",
    },
    {
        "id": "REG-005",
        "question": "¿Qué pasa si no apruebo una materia?",
        "expected_answer": "Se puede recursar la materia según las condiciones del Art. 4",
        "expected_keywords": ["recursar", "repetir", "Art"],
        "expected_source": "Reglamento de Cursada",
        "category": "procedural",
        "difficulty": "medium",
    },
    # --- PROGRAMAS / CARRERAS ---
    {
        "id": "PROG-001",
        "question": "¿Qué título otorga la CEIA?",
        "expected_answer": "Especialista en Inteligencia Artificial",
        "expected_keywords": ["Especialista", "Inteligencia Artificial"],
        "expected_source": "CEIA",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "PROG-002",
        "question": "¿La CEIA es presencial o a distancia?",
        "expected_answer": "A distancia",
        "expected_keywords": ["distancia"],
        "expected_source": "CEIA",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "PROG-003",
        "question": "¿Qué título otorga la CESE?",
        "expected_answer": "Especialista en Sistemas Embebidos",
        "expected_keywords": ["Especialista", "Sistemas Embebidos"],
        "expected_source": "CESE",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "PROG-004",
        "question": "¿Cuándo fue la primera cohorte de la MIA?",
        "expected_answer": "2025",
        "expected_keywords": ["2025"],
        "expected_source": "FAQ - MIA",
        "category": "factual",
        "difficulty": "medium",
    },
    # --- FAQ ---
    {
        "id": "FAQ-001",
        "question": "¿Cómo me inscribo en Gestión de Proyectos (GdP)?",
        "expected_answer": "Información sobre el proceso de inscripción en GdP",
        "expected_keywords": ["inscripci", "GdP", "Gestión"],
        "expected_source": "FAQ - GdP",
        "category": "procedural",
        "difficulty": "medium",
    },
    {
        "id": "FAQ-002",
        "question": "¿Puedo cursar la MIA sin haber hecho la CEIA?",
        "expected_answer": "Información sobre requisitos de la MIA y relación con CEIA",
        "expected_keywords": ["MIA", "CEIA", "requisit"],
        "expected_source": "FAQ - MIA",
        "category": "factual",
        "difficulty": "medium",
    },
    {
        "id": "FAQ-003",
        "question": "¿Cuántas materias optativas tiene la MIA?",
        "expected_answer": "Información sobre cantidad de optativas de la MIA",
        "expected_keywords": ["optativa", "MIA"],
        "expected_source": "FAQ - MIA",
        "category": "factual",
        "difficulty": "medium",
    },
    # --- PREGUNTAS MULTI-HOP (GraphRAG) ---
    {
        "id": "GRAPH-001",
        "question": "¿Cuáles son los requisitos de la MIA y qué especialización necesito cursar antes?",
        "expected_answer": "La MIA requiere tener la CEIA o CESE como base",
        "expected_keywords": ["MIA", "CEIA", "CESE", "requisit"],
        "expected_source": "FAQ - MIA",
        "category": "comparative",
        "difficulty": "hard",
    },
    {
        "id": "GRAPH-002",
        "question": "¿Qué diferencias hay entre la MIAE y la MIA?",
        "expected_answer": "Diferencias entre ambas maestrías",
        "expected_keywords": ["MIAE", "MIA", "diferencia"],
        "expected_source": "MIAE",
        "category": "comparative",
        "difficulty": "hard",
    },
    {
        "id": "GRAPH-003",
        "question": "Si tengo la CESE, ¿a qué maestrías puedo acceder?",
        "expected_answer": "Desde la CESE se puede acceder a varias maestrías",
        "expected_keywords": ["CESE", "maestría"],
        "expected_source": "CESE",
        "category": "comparative",
        "difficulty": "hard",
    },
    # --- PREGUNTAS FUERA DE DOMINIO (deben abstener) ---
    {
        "id": "OOD-001",
        "question": "¿Cuánto cuesta la carrera de especialización?",
        "expected_answer": "ABSTAIN",
        "expected_keywords": [],
        "expected_source": None,
        "category": "out_of_domain",
        "difficulty": "easy",
    },
    {
        "id": "OOD-002",
        "question": "¿Qué opinás sobre la carrera de ingeniería en la UTN?",
        "expected_answer": "ABSTAIN",
        "expected_keywords": [],
        "expected_source": None,
        "category": "out_of_domain",
        "difficulty": "easy",
    },
    # --- CONTACTOS ---
    {
        "id": "CONTACT-001",
        "question": "¿A quién contacto para dudas sobre inscripción?",
        "expected_answer": "inscripcion.lse@fi.uba.ar",
        "expected_keywords": ["inscripcion.lse@fi.uba.ar"],
        "expected_source": "FAQ",
        "category": "navigational",
        "difficulty": "easy",
    },
    {
        "id": "CONTACT-002",
        "question": "¿A quién contacto para consultas sobre el trabajo final?",
        "expected_answer": "direccion.posgrado.lse@fi.uba.ar",
        "expected_keywords": ["direccion.posgrado.lse@fi.uba.ar"],
        "expected_source": "FAQ",
        "category": "navigational",
        "difficulty": "easy",
    },
]


# ── Preguntas para comparación RAG vs GraphRAG ──────────────────
COMPARISON_QUESTIONS = [
    # Estas preguntas se esperan mejor con RAG
    {
        "question": "¿Cuál es la modalidad de asistencia requerida según el reglamento?",
        "expected_better": "rag",
        "reason": "Información textual directa del reglamento",
    },
    {
        "question": "¿Qué fundamentación tiene la CEIA como programa de posgrado?",
        "expected_better": "rag",
        "reason": "Texto descriptivo largo de la resolución",
    },
    # Estas preguntas se esperan mejor con GraphRAG
    {
        "question": "¿Qué carreras necesito completar para acceder a la MIA?",
        "expected_better": "graph",
        "reason": "Relaciones entre programas (prerequisitos)",
    },
    {
        "question": "¿Cuál es el camino desde la CESE hasta una maestría?",
        "expected_better": "graph",
        "reason": "Navegación por paths entre entidades",
    },
    # Estas preguntas se esperan mejor con Hybrid
    {
        "question": "¿Cuáles son los requisitos de la CEIA y qué materias incluye?",
        "expected_better": "hybrid",
        "reason": "Combinación de información textual y estructural",
    },
    {
        "question": "¿Qué opciones tengo después de terminar la especialización en IA?",
        "expected_better": "hybrid",
        "reason": "Necesita info descriptiva + relaciones entre programas",
    },
]
