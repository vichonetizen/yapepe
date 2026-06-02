"""
Local-only auth token. Generated once, stored in data/local.token.
Injected into the served HTML so the frontend can include it on every request.
"""
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DATA_DIR

_TOKEN_FILE = DATA_DIR / "local.token"
_token: str | None = None


def get_token() -> str:
    global _token
    if _token:
        return _token
    # En Vercel (FS efímero/multi-instancia) un token fijo por variable de entorno
    # garantiza que la página servida y la validación usen el mismo token.
    env_token = os.getenv("LOCAL_TOKEN", "").strip()
    if env_token:
        _token = env_token
        return _token
    if _TOKEN_FILE.exists():
        candidate = _TOKEN_FILE.read_text().strip()
        if len(candidate) == 64:
            _token = candidate
            return _token
    _token = secrets.token_hex(32)
    _TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TOKEN_FILE.write_text(_token)
    return _token


def verify_token(candidate: str) -> bool:
    return bool(candidate) and secrets.compare_digest(candidate, get_token())
