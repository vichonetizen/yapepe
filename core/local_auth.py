"""
Local-only auth token. Generated once, stored in data/local.token.
Injected into the served HTML so the frontend can include it on every request.
"""
import secrets
from pathlib import Path

_TOKEN_FILE = Path("data/local.token")
_token: str | None = None


def get_token() -> str:
    global _token
    if _token:
        return _token
    if _TOKEN_FILE.exists():
        candidate = _TOKEN_FILE.read_text().strip()
        if len(candidate) == 64:
            _token = candidate
            return _token
    _token = secrets.token_hex(32)
    _TOKEN_FILE.parent.mkdir(exist_ok=True)
    _TOKEN_FILE.write_text(_token)
    return _token


def verify_token(candidate: str) -> bool:
    return bool(candidate) and secrets.compare_digest(candidate, get_token())
