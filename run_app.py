"""
Script para lanzar la interfaz Streamlit del chatbot.

Autor: Juan Ruiz Otondo - CEIA FIUBA
Carrera de Especialización en Inteligencia Artificial
Laboratorio de Sistemas Embebidos - FIUBA - UBA

Uso:
    python run_app.py                   # Puerto 8501 (default)
    python run_app.py --port 8502       # Puerto personalizado
"""

import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interfaz Streamlit del Chatbot LSE-FIUBA"
    )
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument(
        "--api-url", type=str, default="http://localhost:8000",
        help="URL de la API backend",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("CHATBOT ADMINISTRATIVO LSE-FIUBA - Interfaz Streamlit")
    print("Autor: Juan Ruiz Otondo - CEIA FIUBA")
    print("=" * 70)
    print(f"Interfaz: http://localhost:{args.port}")
    print(f"API backend: {args.api_url}")
    print("=" * 70)
    print("\nAsegurate de que la API esté corriendo (python run_api.py)")

    app_path = ROOT / "src" / "ui" / "app.py"

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ])


if __name__ == "__main__":
    main()
