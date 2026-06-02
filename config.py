import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
MODEL             = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
DATABASE_URL      = f"sqlite+aiosqlite:///{DATA_DIR}/pentamodal.db"
KG_FILE           = str(DATA_DIR / "knowledge_graph.json")
PORT              = int(os.getenv("PORT", "8000"))
HOST              = os.getenv("HOST", "127.0.0.1")
# Capa 4 — Autonomía: cada cuántas horas consolidar la memoria (0 = desactivado)
AUTONOMY_INTERVAL_HOURS = float(os.getenv("AUTONOMY_INTERVAL_HOURS", "6"))
