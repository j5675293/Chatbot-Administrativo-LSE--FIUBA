"""
Script para lanzar la API FastAPI del chatbot.

Autor: Juan Ruiz Otondo - CEIA FIUBA
Carrera de Especialización en Inteligencia Artificial
Laboratorio de Sistemas Embebidos - FIUBA - UBA

Uso:
    python run_api.py                   # Puerto 8000 (default)
    python run_api.py --port 8080       # Puerto personalizado
    python run_api.py --reload          # Con auto-reload (desarrollo)
"""

import sys
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("api")


def parse_args():
    parser = argparse.ArgumentParser(description="API del Chatbot LSE-FIUBA")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("API CHATBOT ADMINISTRATIVO LSE-FIUBA")
    logger.info("Autor: Juan Ruiz Otondo - CEIA FIUBA")
    logger.info("=" * 70)
    logger.info(f"Servidor: http://{args.host}:{args.port}")
    logger.info(f"Docs: http://localhost:{args.port}/docs")
    logger.info("=" * 70)

    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
