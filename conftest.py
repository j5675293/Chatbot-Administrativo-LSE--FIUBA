"""
Configuración global de pytest.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import sys
from pathlib import Path

# Asegurar que el root del proyecto esté en el path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
