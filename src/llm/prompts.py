"""
Templates de prompts en español para el chatbot administrativo del LSE-FIUBA.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

SYSTEM_PROMPT_ES = (
    "Sos un asistente administrativo virtual del Laboratorio de Sistemas "
    "Embebidos (LSE) de la Facultad de Ingeniería de la Universidad de Buenos "
    "Aires (FIUBA). Tu función es responder preguntas sobre los posgrados del "
    "LSE: las especializaciones (CEIA, CESE, CEIoT), las maestrías (MIA, MIAE, "
    "MIoT, MCB), y todos los procesos administrativos asociados.\n\n"
    "REGLAS IMPORTANTES:\n"
    "1. Respondé ÚNICAMENTE con información que se encuentra en el contexto "
    "proporcionado.\n"
    "2. Si no encontrás la información en el contexto, decí: 'No tengo "
    "información suficiente para responder esta pregunta. Te recomiendo "
    "contactar a [email relevante].'\n"
    "3. Citá siempre la fuente de tu respuesta (nombre del documento y "
    "sección).\n"
    "4. Usá lenguaje formal pero amigable, en español rioplatense (vos/tuteo).\n"
    "5. Si la pregunta es ambigua, pedí clarificación sobre a qué programa se "
    "refiere.\n"
    "6. NUNCA inventes información sobre plazos, requisitos o procesos "
    "administrativos.\n"
    "7. Si la pregunta involucra múltiples programas, aclarás las diferencias "
    "entre ellos.\n"
    "8. Cuando cites un artículo del reglamento, indicá el número de artículo.\n"
    "9. Si la respuesta involucra emails de contacto, incluilos en la respuesta."
)

RAG_QA_PROMPT_ES = (
    "Contexto recuperado de documentos oficiales del LSE-FIUBA:\n\n"
    "{context}\n\n"
    "---\n"
    "Pregunta del estudiante: {question}\n\n"
    "Instrucciones:\n"
    "- Respondé basándote EXCLUSIVAMENTE en el contexto proporcionado.\n"
    "- Citá el documento y la sección de donde obtenés la información usando "
    "el formato [Fuente: nombre_documento, sección].\n"
    "- Si el contexto no contiene la respuesta, indicalo claramente y sugerí "
    "un email de contacto relevante.\n"
    "- Formato: respuesta clara y concisa, seguida de las fuentes.\n\n"
    "Respuesta:"
)

ENTITY_EXTRACTION_PROMPT_ES = (
    "Dado el siguiente texto de un documento administrativo del LSE-FIUBA, "
    "extraé las entidades y sus relaciones.\n\n"
    "Texto:\n{text}\n\n"
    "Tipos de entidades a buscar:\n"
    "- PROGRAMA: nombres de carreras (CEIA, CESE, CEIoT, MIA, MIAE, MIoT, MCB)\n"
    "- MATERIA: nombres de asignaturas (ej: Gestión de Proyectos, Taller de "
    "Trabajo Final A)\n"
    "- REQUISITO: condiciones para inscripción o aprobación\n"
    "- PLAZO: fechas límite, duraciones (ej: 10 bimestres, 2 años)\n"
    "- PROCESO: procedimientos administrativos (inscripción, baja, readmisión)\n"
    "- CONTACTO: emails o medios de contacto\n"
    "- TITULO: títulos que se otorgan\n\n"
    "Formato de salida (JSON):\n"
    '{{\n'
    '  "entities": [\n'
    '    {{"name": "...", "type": "...", "properties": {{}}}}\n'
    '  ],\n'
    '  "relationships": [\n'
    '    {{"source": "...", "target": "...", "type": "...", '
    '"description": "..."}}\n'
    '  ]\n'
    '}}\n\n'
    "Extracción:"
)

FAITHFULNESS_CHECK_PROMPT_ES = (
    "Dada la siguiente respuesta y el contexto fuente, verificá si cada "
    "afirmación de la respuesta está respaldada por el contexto.\n\n"
    "Contexto:\n{context}\n\n"
    "Respuesta:\n{answer}\n\n"
    "Para cada afirmación en la respuesta, indicá:\n"
    "1. La afirmación\n"
    "2. Si está respaldada por el contexto (SI/NO)\n"
    "3. La evidencia del contexto que la respalda (si existe)\n\n"
    "Formato de salida (JSON):\n"
    '{{\n'
    '  "claims": [\n'
    '    {{\n'
    '      "claim": "...",\n'
    '      "supported": true/false,\n'
    '      "evidence": "..."\n'
    '    }}\n'
    '  ],\n'
    '  "overall_faithfulness": 0.0 a 1.0\n'
    '}}\n\n'
    "Verificación:"
)

QUERY_CLASSIFICATION_PROMPT_ES = (
    "Clasificá la siguiente pregunta sobre los posgrados del LSE-FIUBA en "
    "una de estas categorías:\n\n"
    "1. FACTUAL: pregunta sobre datos específicos (requisitos, plazos, "
    "materias, notas)\n"
    "2. PROCEDURAL: pregunta sobre cómo hacer algo (inscribirse, solicitar "
    "prórroga, pedir baja)\n"
    "3. COMPARATIVE: pregunta que compara programas o opciones\n"
    "4. EXPLORATORY: pregunta abierta sobre qué es o cómo funciona algo\n"
    "5. NAVIGATIONAL: pregunta sobre a quién contactar o dónde encontrar "
    "información\n\n"
    "Pregunta: {query}\n\n"
    "Respondé SOLO con la categoría (una palabra): "
)

ANSWER_SYNTHESIS_PROMPT_ES = (
    "Sos un asistente administrativo del LSE-FIUBA. Generá una respuesta "
    "basándote en la información recuperada de ambas fuentes.\n\n"
    "Información del sistema RAG (búsqueda por similitud):\n{rag_context}\n\n"
    "Información del grafo de conocimiento:\n{graph_context}\n\n"
    "Pregunta: {question}\n\n"
    "Instrucciones:\n"
    "- Combiná la información de ambas fuentes para dar la respuesta más "
    "completa posible.\n"
    "- Si hay contradicciones entre las fuentes, mencionalo.\n"
    "- Citá las fuentes usando [Fuente: documento, sección].\n"
    "- Si ninguna fuente tiene la información, indicalo claramente.\n\n"
    "Respuesta:"
)
