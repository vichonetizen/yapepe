"""
Device access module — filesystem browse/read, system info, clipboard.
All filesystem paths are validated against ALLOWED_ROOTS to prevent traversal.
"""
import platform
import re
from pathlib import Path

# psutil y pyperclip se importan de forma perezosa dentro de las funciones que los
# usan: en entornos serverless (Vercel) pueden no estar disponibles, y así el módulo
# se importa igual sin romper el arranque de la app.

ALLOWED_ROOTS: list[Path] = [
    Path.home() / "Documents",
    Path.home() / "Desktop",
    Path.home() / "Downloads",
    Path.home() / "OneDrive" / "Escritorio",
    Path.home() / "OneDrive" / "Documentos",
    Path.home() / "OneDrive" / "Documents",
]

MAX_READ_BYTES = 500_000  # 500 KB

_TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".csv", ".html", ".css", ".yaml", ".yml",
    ".toml", ".log", ".xml", ".ini", ".cfg",
    ".sh", ".bat", ".ps1", ".rs", ".go", ".java", ".c", ".cpp",
}


# Archivos sensibles: nunca se listan como legibles ni se devuelven por read_file,
# aunque caigan dentro de una carpeta autorizada (defensa en profundidad).
#
# Criterio de detección (dos niveles, para equilibrar cobertura y falsos positivos):
#  - SUBSTRING para términos de alta señal: palabras que casi nunca aparecen en un
#    archivo inocente ("secret", "contraseña", "credencial", "adminsdk"...). Cubren
#    también sus plurales/derivados ("secretos", "credenciales") por contención.
#  - PALABRA COMPLETA para términos ambiguos ("key", "token", "clave"...): solo
#    cuentan si aparecen como palabra separada por . _ - o espacios en el nombre.
#    Así "token.json" o "claves_wifi.txt" se bloquean, pero archivos legítimos
#    como "tokenizer.py", "monkey.py", "keyboard.md" o "clavel.jpg" no.
_SECRET_NAMES = {"id_rsa", "id_ed25519", ".npmrc", ".pypirc", ".netrc", ".htpasswd"}
_SECRET_SUFFIXES = (".token", ".pem", ".key", ".pfx", ".kdbx", ".p12", ".ppk")
_SECRET_SUBSTRINGS = (
    "secret", "secreto",
    "credential", "credencial",
    "password", "passwd", "contraseña", "contrasena",
    "apikey", "privatekey",
    "serviceaccount", "service-account", "service_account", "adminsdk",
)
_SECRET_WORDS = {"key", "keys", "token", "tokens", "clave", "claves", "pwd"}


def _is_secret(p: Path) -> bool:
    name = p.name.lower()
    if name == ".env" or name.startswith(".env."):
        return True
    if name in _SECRET_NAMES or name.endswith(_SECRET_SUFFIXES):
        return True
    if any(s in name for s in _SECRET_SUBSTRINGS):
        return True
    palabras = re.split(r"[^a-z0-9áéíóúüñ]+", name)
    return any(w in _SECRET_WORDS for w in palabras)


def _resolve_safe(path: str) -> Path:
    """Resolve path and verify it falls under an allowed root."""
    p = Path(path).resolve()
    for root in ALLOWED_ROOTS:
        r = root.resolve()
        if r.exists() and (p == r or r in p.parents):
            return p
    raise PermissionError(f"Acceso denegado: '{path}' no está en una carpeta autorizada.")


def get_allowed_roots() -> list[str]:
    return [str(r) for r in ALLOWED_ROOTS if r.exists()]


def list_path(path: str) -> dict:
    p = _resolve_safe(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {path}")
    if p.is_file():
        return {
            "type": "file",
            "name": p.name,
            "path": str(p),
            "size": p.stat().st_size,
            "ext": p.suffix.lower(),
        }
    items = []
    for child in sorted(p.iterdir(), key=lambda c: (c.is_file(), c.name.lower())):
        try:
            s = child.stat()
            items.append({
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
                "size": s.st_size if child.is_file() else None,
                "ext": child.suffix.lower() if child.is_file() else None,
                "readable": (child.suffix.lower() in _TEXT_EXTENSIONS and not _is_secret(child)) if child.is_file() else None,
            })
        except (PermissionError, OSError):
            continue
    return {"type": "dir", "path": str(p), "items": items[:300]}


def read_file(path: str) -> str:
    p = _resolve_safe(path)
    if _is_secret(p):
        raise PermissionError("Lectura de archivos sensibles (claves/secretos) bloqueada.")
    if not p.is_file():
        raise ValueError("La ruta no es un archivo.")
    if p.suffix.lower() not in _TEXT_EXTENSIONS:
        raise ValueError(f"Extensión '{p.suffix}' no permitida para lectura directa. Usa el subidor de documentos para PDFs/DOCX.")
    size = p.stat().st_size
    if size > MAX_READ_BYTES:
        raise ValueError(f"Archivo muy grande ({size // 1024} KB). Máximo 500 KB para lectura directa.")
    return p.read_text(encoding="utf-8", errors="replace")


def get_sysinfo() -> dict:
    import psutil
    cpu = psutil.cpu_percent(interval=0.3)
    ram = psutil.virtual_memory()

    procs: list[dict] = []
    for proc in psutil.process_iter(["name", "pid", "memory_info"]):
        try:
            mi = proc.info["memory_info"]
            if mi:
                procs.append({
                    "name": proc.info["name"],
                    "pid": proc.info["pid"],
                    "ram_mb": round(mi.rss / 1024 ** 2, 1),
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    procs.sort(key=lambda x: x["ram_mb"], reverse=True)

    result: dict = {
        "cpu_percent": cpu,
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(ram.total / 1024 ** 3, 1),
        "ram_used_gb": round(ram.used / 1024 ** 3, 1),
        "ram_free_gb": round(ram.available / 1024 ** 3, 1),
        "ram_percent": ram.percent,
        "top_processes": procs[:12],
        "platform": platform.system(),
        "python": platform.python_version(),
    }
    try:
        disk = psutil.disk_usage("C:\\")
        result["disk_total_gb"] = round(disk.total / 1024 ** 3, 1)
        result["disk_used_gb"] = round(disk.used / 1024 ** 3, 1)
        result["disk_free_gb"] = round(disk.free / 1024 ** 3, 1)
        result["disk_percent"] = disk.percent
    except Exception:
        pass
    return result


def clipboard_read() -> str:
    try:
        import pyperclip
        return pyperclip.paste() or ""
    except Exception:
        return ""


def clipboard_write(text: str) -> None:
    import pyperclip
    pyperclip.copy(text)
