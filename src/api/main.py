"""
Aplicación FastAPI del Chatbot Administrativo LSE-FIUBA.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import chat, health

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

app = FastAPI(
    title="Chatbot Administrativo LSE-FIUBA",
    description=(
        "API del agente administrativo inteligente para posgrados del "
        "Laboratorio de Sistemas Embebidos - FIUBA - UBA. "
        "Trabajo Final de Juan Ruiz Otondo, CEIA."
    ),
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(chat.router)
app.include_router(health.router)


@app.get("/")
async def root():
    return {
        "message": "Chatbot Administrativo LSE-FIUBA API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
