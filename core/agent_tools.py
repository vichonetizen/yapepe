"""
Capa 6 — REGISTRO DE HERRAMIENTAS del bucle agente.

"Un LLM por sí solo es la mitad generadora del bucle CEGIS sin la mitad
verificadora; todo el ingenio está en reconstruir esa mitad alrededor del modelo"
(Vol. I §4.1). Las herramientas son cómo el agente ACTÚA y OBSERVA el mundo real
en vez de solo predecir texto.

Cada herramienta declara un NIVEL DE RIESGO (autonomía proporcional al coste del
error, Vol. II §6.5):
  • "read"    : no modifica nada. Se ejecuta sin pedir permiso.
  • "write"   : escribe SOLO dentro del workspace aislado (data/agent_workspace).
  • "command" : ejecuta comandos del sistema. Requiere CONFIRMACIÓN humana salvo
                que el comando esté en la allowlist de solo-lectura.

SEGURIDAD (frontera datos/control, Vol. II §14.2): la salida de una herramienta es
DATO, nunca instrucción. El bucle nunca trata una observación como una orden a
obedecer. Las escrituras están confinadas por ruta; los secretos (.env, *.token)
no se pueden leer; los comandos destructivos están en una denylist inviolable.
"""
import asyncio
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import AGENT_WORKSPACE, AGENT_READ_ROOT, AGENT_CMD_TIMEOUT

# Comandos de solo-lectura que pueden correr sin confirmación.
SAFE_COMMAND_PREFIXES = (
    "echo", "ver", "whoami", "hostname", "date", "time",
    "python --version", "python -V", "pip --version", "node --version",
    "git status", "git log", "git diff", "git branch",
    "dir", "ls", "type", "cat", "pwd", "cd",
)
# Patrones SIEMPRE rechazados (destrucción / persistencia / red peligrosa).
DENY_COMMAND_PATTERNS = (
    r"\brm\b\s+-rf", r"\brmdir\b", r"\bdel\b", r"\bformat\b", r"\bmkfs\b",
    r"\bshutdown\b", r"\breboot\b", r":\(\)\s*\{", r">\s*/dev/sd",
    r"\bdd\b\s+if=", r"\bcurl\b.*\|\s*(sh|bash)", r"\bwget\b.*\|\s*(sh|bash)",
    r"\breg\b\s+delete", r"\bdiskpart\b", r"-rf\s+/",
)
# Lectura prohibida: secretos.
DENY_READ_NAMES = (".env", ".token")


def _within(root: str, target: str) -> bool:
    """¿target está dentro de root? (bloquea path traversal)."""
    root_abs = os.path.abspath(root)
    tgt_abs = os.path.abspath(target)
    return tgt_abs == root_abs or tgt_abs.startswith(root_abs + os.sep)


def _is_secret(path: str) -> bool:
    name = os.path.basename(path).lower()
    return name in DENY_READ_NAMES or name.endswith(".token") or name == ".env"


# ── Herramientas de LECTURA ──────────────────────────────────────────────────────
async def t_search_documents(query: str, **_) -> str:
    from core.semantic_memory import hybrid_search_documents
    hits = await hybrid_search_documents(str(query), top_k=4)
    hits = [h for h in hits if h["score"] > 0.05]
    if not hits:
        return "Sin resultados en el corpus."
    return "\n\n".join(f"[{h['filename']}] (score {h['score']})\n{h['content'][:500]}" for h in hits)


async def t_search_memory(query: str, **_) -> str:
    from sqlalchemy import text
    from core.semantic_memory import _get_engine
    terms = [t for t in re.findall(r"\b\w{4,}\b", str(query).lower())][:5]
    if not terms:
        return "Consulta vacía."
    engine = _get_engine()
    try:
        async with engine.connect() as conn:
            cond = " OR ".join(f"LOWER(content) LIKE :t{i}" for i in range(len(terms)))
            params = {f"t{i}": f"%{t}%" for i, t in enumerate(terms)}
            rows = (await conn.execute(text(
                f"SELECT category, content FROM memory_facts WHERE {cond} LIMIT 8"), params)).fetchall()
    except Exception:
        return "No hay memoria disponible."
    if not rows:
        return "Sin hechos relevantes en memoria."
    return "\n".join(f"• ({r[0]}) {r[1]}" for r in rows)


async def t_graph_neighbors(concept: str, kg=None, **_) -> str:
    if kg is None:
        return "Grafo no disponible."
    ctx = kg.get_context(str(concept))
    return ctx or f"'{concept}' no tiene asociaciones registradas todavía."


async def t_read_file(path: str, **_) -> str:
    target = path if os.path.isabs(path) else os.path.join(AGENT_READ_ROOT, path)
    if not _within(AGENT_READ_ROOT, target) and not _within(AGENT_WORKSPACE, target):
        return f"DENEGADO: '{path}' está fuera de la raíz de lectura permitida."
    if _is_secret(target):
        return "DENEGADO: no se permite leer archivos de secretos (.env / *.token)."
    if not os.path.isfile(target):
        return f"No existe el archivo: {path}"
    try:
        with open(target, encoding="utf-8", errors="replace") as f:
            data = f.read(8000)
        return data or "(archivo vacío)"
    except Exception as e:
        return f"Error leyendo: {e}"


async def t_list_dir(path: str = ".", **_) -> str:
    target = path if os.path.isabs(path) else os.path.join(AGENT_READ_ROOT, path)
    if not _within(AGENT_READ_ROOT, target) and not _within(AGENT_WORKSPACE, target):
        return f"DENEGADO: '{path}' está fuera de la raíz permitida."
    if not os.path.isdir(target):
        return f"No es un directorio: {path}"
    try:
        entries = sorted(os.listdir(target))[:100]
        return "\n".join(entries) or "(directorio vacío)"
    except Exception as e:
        return f"Error: {e}"


# ── Herramientas de ESCRITURA (solo workspace) ───────────────────────────────────
async def t_write_file(path: str, content: str = "", **_) -> str:
    target = os.path.join(AGENT_WORKSPACE, path)
    if not _within(AGENT_WORKSPACE, target):
        return "DENEGADO: solo se puede escribir dentro del workspace aislado."
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(str(content))
        rel = os.path.relpath(target, AGENT_WORKSPACE)
        return f"Escrito {len(str(content))} chars en workspace/{rel}"
    except Exception as e:
        return f"Error escribiendo: {e}"


# ── Herramienta de COMANDO (con confirmación) ────────────────────────────────────
def command_is_denied(cmd: str) -> bool:
    low = cmd.lower()
    return any(re.search(p, low) for p in DENY_COMMAND_PATTERNS)


def command_is_safe(cmd: str) -> bool:
    """¿Es un comando de solo-lectura que puede correr sin confirmación?"""
    c = cmd.strip().lower()
    return any(c == p or c.startswith(p + " ") for p in SAFE_COMMAND_PREFIXES)


async def t_run_command(command: str, **_) -> str:
    cmd = str(command).strip()
    if not cmd:
        return "Comando vacío."
    if command_is_denied(cmd):
        return "DENEGADO: comando destructivo/peligroso bloqueado por la denylist."
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd, cwd=AGENT_WORKSPACE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=AGENT_CMD_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            return f"Comando excedió el límite de {AGENT_CMD_TIMEOUT}s y fue terminado."
        text_out = (out or b"").decode("utf-8", errors="replace")[:4000]
        return f"(exit {proc.returncode})\n{text_out}" or "(sin salida)"
    except Exception as e:
        return f"Error ejecutando: {e}"


# ── Registro ─────────────────────────────────────────────────────────────────────
# risk: read | write | command. needs_approval lo decide el bucle según el riesgo.
TOOLS: dict[str, dict] = {
    "search_documents": {"fn": t_search_documents, "risk": "read", "args": ["query"],
        "desc": "Busca en el corpus de documentos (RAG). Úsalo para conocimiento de los textos."},
    "search_memory":   {"fn": t_search_memory, "risk": "read", "args": ["query"],
        "desc": "Busca hechos recordados sobre el usuario y el contexto."},
    "graph_neighbors": {"fn": t_graph_neighbors, "risk": "read", "args": ["concept"],
        "desc": "Devuelve conceptos asociados a uno dado en el grafo de conocimiento."},
    "read_file":       {"fn": t_read_file, "risk": "read", "args": ["path"],
        "desc": "Lee un archivo de texto dentro del repositorio o del workspace."},
    "list_dir":        {"fn": t_list_dir, "risk": "read", "args": ["path"],
        "desc": "Lista el contenido de un directorio permitido."},
    "write_file":      {"fn": t_write_file, "risk": "write", "args": ["path", "content"],
        "desc": "Escribe un archivo DENTRO del workspace aislado (data/agent_workspace)."},
    "run_command":     {"fn": t_run_command, "risk": "command", "args": ["command"],
        "desc": "Ejecuta un comando del sistema en el workspace. Los no triviales requieren tu confirmación."},
}


def tools_for(allow_write: bool, allow_commands: bool) -> dict:
    """Subconjunto de herramientas según los poderes concedidos a esta corrida."""
    out = {}
    for name, spec in TOOLS.items():
        if spec["risk"] == "write" and not allow_write:
            continue
        if spec["risk"] == "command" and not allow_commands:
            continue
        out[name] = spec
    return out


def describe_tools(tools: dict) -> str:
    lines = []
    for name, spec in tools.items():
        args = ", ".join(spec["args"])
        lines.append(f'- {name}({args}) [{spec["risk"]}]: {spec["desc"]}')
    return "\n".join(lines)
