"""
Punto de entrada serverless para Vercel.

Vercel (@vercel/python) detecta la variable `app` (aplicación ASGI) y la sirve.
Todas las rutas se enrutan aquí desde vercel.json.

NOTA: en Vercel la app corre en modo nube (sin Ollama/GPU, FS efímero). Ver
docs/VERCEL.md para las limitaciones y la configuración requerida.
"""
import os
import sys

# Permite importar app.py y config.py desde la raíz del repositorio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402  (la importación tardía es intencional)
