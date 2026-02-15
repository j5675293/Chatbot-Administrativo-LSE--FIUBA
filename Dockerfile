# Chatbot Administrativo LSE-FIUBA
# Autor: Juan Ruiz Otondo - CEIA FIUBA

FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para cachear dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Crear directorios de datos
RUN mkdir -p data/raw data/processed data/indexes data/graphs data/evaluation

# Variables de entorno por defecto
ENV PYTHONPATH=/app
ENV LLM_BACKEND=ollama
ENV LLM_MODEL=llama3
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

EXPOSE 8000 8501

# Comando por defecto: API
CMD ["python", "run_api.py"]
