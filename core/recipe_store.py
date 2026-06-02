"""
Capa 3 — Recetas de tareas.

Cuando el usuario califica bien una respuesta a una tarea no trivial, se guarda una
"receta": el disparador (lo que pidió) + el enfoque que funcionó. Ante una tarea
semánticamente similar en el futuro, la receta se recupera e inyecta en el prompt
para que el asistente **replique el enfoque exitoso**.

Esto cumple el objetivo de "replicar de manera efectiva una tarea solicitada":
el sistema aprende QUÉ enfoque funciona para QUÉ tipo de pedido, y lo reaplica.

Reutiliza el motor de embeddings de la Capa 1 (core/semantic_memory.embedder).
Si no hay embeddings, cae a búsqueda por palabras clave (LIKE).
"""
import re
import numpy as np
from sqlalchemy import text

from core.semantic_memory import (
    _get_engine, embedder, _to_blob, _from_blob, _normalize, MAX_EMBED_CHARS,
)

# Solo capturamos recetas de tareas con sustancia (no charla trivial)
_CAPTURE_COMPLEXITIES = {"media", "compleja", "critica"}
_MIN_RATING = 4.0


async def init_recipes_db():
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS task_recipes (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT,
                title        TEXT NOT NULL,
                trigger      TEXT NOT NULL,
                approach     TEXT NOT NULL,
                topics       TEXT DEFAULT '',
                complexity   TEXT DEFAULT '',
                rating       REAL DEFAULT 0,
                times_reused INTEGER DEFAULT 0,
                model        TEXT DEFAULT '',
                dim          INTEGER DEFAULT 0,
                vec          BLOB,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))


def _topics(s: str, limit: int = 6) -> str:
    found = re.findall(r'\b[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{5,}\b', s)
    seen, out = set(), []
    for w in found:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            out.append(wl)
        if len(out) >= limit:
            break
    return ", ".join(out)


async def add_recipe(
    session_id: str, trigger: str, approach: str,
    complexity: str = "", rating: float = 0.0,
) -> int | None:
    """Guarda una receta, embebiendo el disparador para recuperación semántica.
    Evita duplicados por título (primeros 60 chars del pedido)."""
    trigger = (trigger or "").strip()
    approach = (approach or "").strip()
    if len(trigger) < 8 or len(approach) < 20:
        return None

    title = trigger[:60]
    engine = _get_engine()

    async with engine.connect() as conn:
        dup = (await conn.execute(
            text("SELECT id FROM task_recipes WHERE title = :t LIMIT 1"), {"t": title}
        )).fetchone()
    if dup:
        return None

    # Embebido del disparador (si hay backend); si no, se guarda sin vector
    model, dim, blob = "", 0, None
    await embedder.resolve()
    if embedder.available():
        vecs = await embedder.embed([trigger[:MAX_EMBED_CHARS]])
        if vecs:
            v = np.asarray(vecs[0], dtype=np.float32)
            model, dim, blob = embedder.model_id, int(v.shape[0]), _to_blob(v)

    async with engine.begin() as conn:
        result = await conn.execute(text("""
            INSERT INTO task_recipes
                (session_id, title, trigger, approach, topics, complexity, rating, model, dim, vec)
            VALUES (:sid, :title, :trig, :appr, :top, :cx, :rating, :model, :dim, :vec)
        """), {
            "sid": session_id, "title": title, "trig": trigger[:2000],
            "appr": approach[:2000], "top": _topics(trigger + " " + approach),
            "cx": complexity, "rating": rating, "model": model, "dim": dim, "vec": blob,
        })
        return result.lastrowid


async def maybe_capture_recipe(
    session_id: str, user_request: str, assistant_response: str,
    complexity: str, rating: float,
) -> int | None:
    """Captura una receta solo si la tarea vale la pena y fue bien calificada."""
    if complexity not in _CAPTURE_COMPLEXITIES or rating < _MIN_RATING:
        return None
    return await add_recipe(
        session_id, user_request, assistant_response, complexity, rating
    )


async def search_recipes(query: str, top_k: int = 2, floor: float = 0.55) -> list[dict]:
    """Busca recetas aplicables a la consulta actual. Semántica si hay embeddings;
    si no, por palabras clave. Devuelve solo coincidencias relevantes (>= floor)."""
    query = (query or "").strip()
    if not query:
        return []

    await embedder.resolve()
    engine = _get_engine()

    if embedder.available():
        qv = await embedder.embed([query[:MAX_EMBED_CHARS]])
        if qv:
            q = np.asarray(qv[0], dtype=np.float32)
            q = q / (np.linalg.norm(q) + 1e-9)
            async with engine.connect() as conn:
                rows = (await conn.execute(text(
                    "SELECT id, title, trigger, approach, topics, rating, times_reused, vec "
                    "FROM task_recipes WHERE model = :m AND vec IS NOT NULL"
                ), {"m": embedder.model_id})).fetchall()
            scored = []
            for r in rows:
                v = _from_blob(r[7])
                if v.shape[0] != q.shape[0]:
                    continue
                sim = float(_normalize(v.reshape(1, -1))[0] @ q)
                if sim >= floor:
                    scored.append((sim, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {"id": r[0], "title": r[1], "trigger": r[2], "approach": r[3],
                 "topics": r[4], "rating": r[5], "times_reused": r[6], "score": round(s, 3)}
                for s, r in scored[:top_k]
            ]

    # Fallback por palabras clave
    terms = [t for t in re.findall(r'\b\w{4,}\b', query.lower())][:5]
    if not terms:
        return []
    async with engine.connect() as conn:
        cond = " OR ".join(f"(LOWER(trigger) LIKE :t{i} OR LOWER(topics) LIKE :t{i})"
                           for i in range(len(terms)))
        params = {f"t{i}": f"%{t}%" for i, t in enumerate(terms)}
        rows = (await conn.execute(text(
            f"SELECT id, title, trigger, approach, topics, rating, times_reused "
            f"FROM task_recipes WHERE {cond} ORDER BY rating DESC LIMIT :k"
        ), {**params, "k": top_k})).fetchall()
    return [
        {"id": r[0], "title": r[1], "trigger": r[2], "approach": r[3],
         "topics": r[4], "rating": r[5], "times_reused": r[6], "score": 0.0}
        for r in rows
    ]


async def mark_reused(recipe_ids: list[int]):
    if not recipe_ids:
        return
    engine = _get_engine()
    async with engine.begin() as conn:
        for rid in recipe_ids:
            await conn.execute(
                text("UPDATE task_recipes SET times_reused = times_reused + 1 WHERE id = :id"),
                {"id": rid},
            )


def build_recipe_context(recipes: list[dict]) -> str:
    """Formatea recetas para inyectar en el system prompt."""
    if not recipes:
        return ""
    lines = ["Recetas de tareas previas que funcionaron (replica el enfoque cuando aplique):"]
    for r in recipes:
        lines.append(
            f"• Pedido similar: «{r['trigger'][:120]}»\n"
            f"  Enfoque que funcionó: {r['approach'][:350]}"
        )
    return "\n".join(lines)


async def get_all_recipes(limit: int = 100) -> list[dict]:
    engine = _get_engine()
    async with engine.connect() as conn:
        rows = (await conn.execute(text(
            "SELECT id, title, trigger, approach, topics, complexity, rating, times_reused, created_at "
            "FROM task_recipes ORDER BY created_at DESC LIMIT :k"
        ), {"k": limit})).fetchall()
    return [
        {"id": r[0], "title": r[1], "trigger": r[2], "approach": r[3], "topics": r[4],
         "complexity": r[5], "rating": r[6], "times_reused": r[7], "created_at": str(r[8])}
        for r in rows
    ]


async def delete_recipe(rid: int) -> bool:
    engine = _get_engine()
    async with engine.begin() as conn:
        result = await conn.execute(text("DELETE FROM task_recipes WHERE id = :id"), {"id": rid})
        return result.rowcount > 0


async def get_recipe_stats() -> dict:
    engine = _get_engine()
    async with engine.connect() as conn:
        total = (await conn.execute(text("SELECT COUNT(*) FROM task_recipes"))).scalar() or 0
        reused = (await conn.execute(
            text("SELECT COALESCE(SUM(times_reused),0) FROM task_recipes")
        )).scalar() or 0
    return {"total": total, "total_reused": reused}
