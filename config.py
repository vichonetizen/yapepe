import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Detección de entorno ─────────────────────────────────────────────────────────
# En Vercel (serverless) el único directorio escribible es /tmp, y es EFÍMERO entre
# invocaciones. Por eso la app detecta el entorno y ajusta dónde guarda sus datos.
IS_VERCEL = bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENV"))

BASE_DIR = Path(__file__).parent
DATA_DIR = Path("/tmp/pentamodal_data") if IS_VERCEL else (BASE_DIR / "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Carpeta de documentos fuente: local en el repo; en Vercel, dentro de /tmp.
DOCUMENTS_DIR = (DATA_DIR / "documents") if IS_VERCEL else (BASE_DIR / "documents")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
MODEL             = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
# Ruta única de la base de datos (todos los módulos la usan). Permite override por
# DATABASE_URL (p. ej. una DB externa Turso/Postgres para persistencia en Vercel).
DB_PATH           = str(DATA_DIR / "pentamodal.db")
DATABASE_URL      = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{DB_PATH}")
KG_FILE           = str(DATA_DIR / "knowledge_graph.json")
PORT              = int(os.getenv("PORT", "8000"))
HOST              = os.getenv("HOST", "127.0.0.1")
# Capa 4 — Autonomía: cada cuántas horas consolidar la memoria (0 = desactivado)
AUTONOMY_INTERVAL_HOURS = float(os.getenv("AUTONOMY_INTERVAL_HOURS", "6"))
