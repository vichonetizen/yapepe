"""
Capa 4 — Autonomía / auto-mantenimiento.

Proceso periódico (y disparable a mano) que mantiene la memoria sana SIN
intervención del usuario, para que el sistema sea usable durante mucho tiempo:

  1. Consolida hechos de memoria   — fusiona duplicados, suma frecuencias.
  2. Poda ruido                    — elimina hechos de bajo valor y antiguos.
  3. Deduplica aprendizajes (PAC)  — quita aprendizajes casi idénticos.
  4. Completa el índice semántico  — embebe los chunks pendientes.
  5. Mantiene el grafo de conocimiento — poda y guarda.

Es 100% determinista (no llama a ningún LLM): no consume cuota ni GPU más allá
de embeber lo pendiente. Pensado para correr en bucle cada N horas.
"""
from datetime import datetime

from sqlalchemy import text

from core.semantic_memory import _get_engine, index_document_chunks

# Estado del último ciclo (para /autonomy/status)
_last_run: dict = {"at": None, "report": None}


async def ensure_indexes():
    """Crea índices útiles para que la app siga rápida cuando la DB crece.
    Idempotente — se llama una vez al arrancar."""
    engine = _get_engine()
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role)",
        "CREATE INDEX IF NOT EXISTS idx_facts_category ON memory_facts(category)",
        "CREATE INDEX IF NOT EXISTS idx_learnings_created ON conversation_learnings(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(doc_id)",
        "CREATE INDEX IF NOT EXISTS idx_recipes_model ON task_recipes(model)",
    ]
    async with engine.begin() as conn:
        for s in statements:
            try:
                await conn.execute(text(s))
            except Exception:
                pass


async def _consolidate_facts(conn) -> dict:
    """Fusiona hechos de memoria con el mismo contenido (misma categoría)."""
    rows = (await conn.execute(text(
        "SELECT id, category, content, confidence, frequency FROM memory_facts"
    ))).fetchall()

    groups: dict[tuple, list] = {}
    for r in rows:
        key = (r[1], (r[2] or "").strip().lower())
        groups.setdefault(key, []).append(r)

    merged = 0
    for key, items in groups.items():
        if len(items) < 2:
            continue
        # Conservar el de mayor confianza; sumar frecuencias en él, borrar el resto
        items.sort(key=lambda x: (x[3], x[4]), reverse=True)
        keep = items[0]
        total_freq = sum(it[4] for it in items)
        best_conf = max(it[3] for it in items)
        await conn.execute(text(
            "UPDATE memory_facts SET frequency=:f, confidence=:c, updated_at=:u WHERE id=:id"
        ), {"f": total_freq, "c": best_conf, "u": datetime.utcnow(), "id": keep[0]})
        for it in items[1:]:
            await conn.execute(text("DELETE FROM memory_facts WHERE id=:id"), {"id": it[0]})
            merged += 1
    return {"facts_merged": merged}


async def _prune_noise(conn) -> dict:
    """Elimina hechos de un solo uso, baja confianza y antiguos (>30 días)."""
    result = await conn.execute(text(
        "DELETE FROM memory_facts "
        "WHERE frequency = 1 AND confidence < 0.5 "
        "AND updated_at < datetime('now', '-30 days')"
    ))
    return {"facts_pruned": result.rowcount or 0}


async def _dedupe_learnings(conn) -> dict:
    """Quita aprendizajes PAC casi idénticos (mismo prefijo de 120 chars)."""
    result = await conn.execute(text(
        "DELETE FROM conversation_learnings "
        "WHERE id NOT IN ("
        "  SELECT MIN(id) FROM conversation_learnings GROUP BY SUBSTR(content, 1, 120)"
        ")"
    ))
    return {"learnings_deduped": result.rowcount or 0}


async def consolidate_memory(kg=None) -> dict:
    """Ejecuta un ciclo completo de consolidación. Devuelve un informe."""
    report: dict = {}
    engine = _get_engine()

    async with engine.begin() as conn:
        report.update(await _consolidate_facts(conn))
        report.update(await _prune_noise(conn))
        report.update(await _dedupe_learnings(conn))

    # Completar el índice semántico (chunks sin embedding)
    try:
        report["chunks_indexed"] = await index_document_chunks()
    except Exception:
        report["chunks_indexed"] = 0

    # Mantener el grafo de conocimiento
    if kg is not None:
        try:
            kg._prune_if_needed()
            kg.save()
            report["kg_nodes"] = kg.graph.number_of_nodes()
        except Exception:
            pass

    _last_run["at"] = datetime.utcnow().isoformat()
    _last_run["report"] = report
    return report


def status() -> dict:
    return {"last_run": _last_run["at"], "last_report": _last_run["report"]}
