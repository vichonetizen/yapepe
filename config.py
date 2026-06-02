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

# ── Capa 5 — Auto-reconstrucción (base estructural de los 5 documentos) ──────────
# El sistema se reconfigura solo sobre sustratos INSPECCIONABLES y REVERSIBLES
# (config/andamiaje y asociaciones del grafo), siempre contra un oráculo FIJO con
# checkpoint/rollback. Ver docs/CONSTITUCION_autoprogramacion.md.
RECON_ENABLED       = os.getenv("RECON_ENABLED", "1") not in ("0", "false", "False")
# Política de aplicación: "mixto" (auto-aplica bajo riesgo, propone lo demás),
# "auto" (auto-aplica todo lo seguro) o "propose" (todo queda como propuesta).
RECON_MODE          = os.getenv("RECON_MODE", "mixto")
# Presupuesto: máximo de ciclos de reconstrucción por arranque (0 = sin límite).
RECON_MAX_CYCLES    = int(os.getenv("RECON_MAX_CYCLES", "0"))
RECON_DIR           = DATA_DIR / "recon"
RECON_DIR.mkdir(parents=True, exist_ok=True)
RECON_CONFIG_FILE   = str(RECON_DIR / "config.json")        # parámetros reconfigurables
RECON_BASELINE_FILE = str(RECON_DIR / "baseline.json")      # línea base del oráculo
RECON_PROPOSALS_FILE = str(RECON_DIR / "proposals.json")    # propuestas pendientes
RECON_ATTEMPTS_FILE = str(RECON_DIR / "attempts.json")      # memoria de intentos
RECON_STATUS_FILE   = str(RECON_DIR / "status.json")        # último ciclo
CHECKPOINTS_DIR     = str(RECON_DIR / "checkpoints")        # estados reversibles

# ── Capa 6 — Bucle agente (ejecutar tareas, como un agente de codificación) ──────
# Implementa el bucle PLAN→ACT→OBSERVE→REFLECT→STOP (Vol. II §10.3) con
# herramientas y verificación. Ver docs/CONSTITUCION_autoprogramacion.md.
AGENT_WORKSPACE   = str((DATA_DIR / "agent_workspace"))     # único lugar escribible por el agente
AGENT_MAX_STEPS   = int(os.getenv("AGENT_MAX_STEPS", "12")) # presupuesto de pasos por tarea
AGENT_CMD_TIMEOUT = int(os.getenv("AGENT_CMD_TIMEOUT", "30"))  # seg. máx. por comando
# Raíz de lectura permitida para read_file/list_dir (por defecto, el propio repo).
AGENT_READ_ROOT   = str(BASE_DIR)
Path(AGENT_WORKSPACE).mkdir(parents=True, exist_ok=True)
