"""
Provider configuration store: persists API keys and metadata in SQLite.
Supports built-in providers and custom OpenAI-compatible endpoints.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

_engine = None
_DB_PATH = "data/pentamodal.db"

BUILTIN_PROVIDERS = [
    {
        "slug": "google", "name": "Google AI Studio", "icon": "🌐",
        "description": "Tareas complejas y críticas. Fallback automático a Groq si cuota agotada.",
        "default_model": "gemini-2.0-flash", "base_url": "",
    },
    {
        "slug": "groq", "name": "Groq", "icon": "⚡",
        "description": "Tareas simples y medias. Ultra-rápido con capa gratuita generosa.",
        "default_model": "llama-3.3-70b-versatile", "base_url": "",
    },
    {
        "slug": "openai", "name": "OpenAI", "icon": "🟢",
        "description": "GPT-4o / GPT-4o-mini. Fallback flexible para cualquier complejidad.",
        "default_model": "gpt-4o-mini", "base_url": "",
    },
    {
        "slug": "anthropic", "name": "Anthropic Claude", "icon": "🎭",
        "description": "Claude Sonnet / Haiku. Razonamiento avanzado y escritura de alta calidad.",
        "default_model": "claude-sonnet-4-6", "base_url": "",
    },
]

_BUILTIN_SLUGS = {p["slug"] for p in BUILTIN_PROVIDERS}


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(f"sqlite+aiosqlite:///{_DB_PATH}", echo=False)
    return _engine


async def init_provider_store():
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS api_providers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                slug        TEXT UNIQUE NOT NULL,
                name        TEXT NOT NULL,
                description TEXT DEFAULT '',
                api_key     TEXT DEFAULT '',
                base_url    TEXT DEFAULT '',
                default_model TEXT DEFAULT '',
                icon        TEXT DEFAULT '🔑',
                is_builtin  INTEGER DEFAULT 0,
                is_active   INTEGER DEFAULT 1,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        for p in BUILTIN_PROVIDERS:
            await conn.execute(text("""
                INSERT OR IGNORE INTO api_providers
                    (slug, name, description, base_url, default_model, icon, is_builtin)
                VALUES (:slug, :name, :description, :base_url, :default_model, :icon, 1)
            """), p)


async def get_all_providers() -> list[dict]:
    """Return all providers. API key is masked (show only last 4 chars)."""
    engine = _get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(text(
            "SELECT id,slug,name,description,api_key,base_url,default_model,icon,is_builtin,is_active "
            "FROM api_providers ORDER BY is_builtin DESC, id"
        ))
        rows = result.fetchall()
    out = []
    for r in rows:
        key = r[4] or ""
        out.append({
            "id": r[0], "slug": r[1], "name": r[2], "description": r[3],
            "has_key": bool(key.strip()),
            "key_preview": ("•" * min(12, max(0, len(key) - 4)) + key[-4:]) if len(key) > 4 else ("•" * len(key)),
            "base_url": r[5], "default_model": r[6], "icon": r[7],
            "is_builtin": bool(r[8]), "is_active": bool(r[9]),
        })
    return out


async def get_provider_full(slug: str) -> dict | None:
    """Return single provider with full API key (for edit form)."""
    engine = _get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT slug,name,description,api_key,base_url,default_model,icon,is_builtin FROM api_providers WHERE slug=:s"),
            {"s": slug},
        )
        r = result.fetchone()
    if not r:
        return None
    return {
        "slug": r[0], "name": r[1], "description": r[2], "api_key": r[3],
        "base_url": r[4], "default_model": r[5], "icon": r[6], "is_builtin": bool(r[7]),
    }


async def upsert_provider(
    slug: str, name: str, description: str, api_key: str,
    base_url: str, default_model: str, icon: str,
) -> dict:
    engine = _get_engine()
    async with engine.begin() as conn:
        existing = (await conn.execute(
            text("SELECT id FROM api_providers WHERE slug=:s"), {"s": slug}
        )).fetchone()
        if existing:
            await conn.execute(text("""
                UPDATE api_providers
                SET name=:name, description=:desc, api_key=:key,
                    base_url=:url, default_model=:model, icon=:icon,
                    updated_at=CURRENT_TIMESTAMP
                WHERE slug=:slug
            """), {"slug": slug, "name": name, "desc": description, "key": api_key,
                   "url": base_url, "model": default_model, "icon": icon})
        else:
            await conn.execute(text("""
                INSERT INTO api_providers (slug,name,description,api_key,base_url,default_model,icon,is_builtin)
                VALUES (:slug,:name,:desc,:key,:url,:model,:icon,0)
            """), {"slug": slug, "name": name, "desc": description, "key": api_key,
                   "url": base_url, "model": default_model, "icon": icon})
    return {"slug": slug, "status": "saved"}


async def delete_provider(slug: str) -> dict:
    """Built-in providers: only clear the key. Custom providers: remove entirely."""
    engine = _get_engine()
    async with engine.begin() as conn:
        if slug in _BUILTIN_SLUGS:
            await conn.execute(
                text("UPDATE api_providers SET api_key='' WHERE slug=:s"), {"s": slug}
            )
            return {"slug": slug, "action": "key_cleared"}
        result = await conn.execute(
            text("DELETE FROM api_providers WHERE slug=:s AND is_builtin=0"), {"s": slug}
        )
        if result.rowcount == 0:
            return {"slug": slug, "action": "not_found"}
    return {"slug": slug, "action": "deleted"}


async def load_keys_into_client(ai_client) -> None:
    """Load all configured keys from DB into the AIClient instance."""
    engine = _get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(text(
            "SELECT slug, api_key, base_url, default_model FROM api_providers "
            "WHERE api_key != '' AND is_active=1"
        ))
        rows = result.fetchall()
    for slug, api_key, base_url, default_model in rows:
        if api_key.strip():
            ai_client.update_key(slug, api_key)
            if base_url or slug not in {"google", "groq", "openai", "anthropic"}:
                ai_client.register_custom(slug, api_key, base_url, default_model)
