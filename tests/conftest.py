"""Configuración común de la suite.

Los tests corren desde la raíz del proyecto con `python -m pytest -q`, pero
también desde otros cwd (IDE, tarea programada). La raíz se agrega al
sys.path para que `import src...` funcione en ambos casos.
"""
import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))
