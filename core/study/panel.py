"""
panel.py — Panel de progreso y criterio "listo para prueba" (Ola 4, pista integrada).

Resume el dominio del estudiante por corpus: % de conceptos dominados (cards con
>=2 repasos correctos), repasos pendientes hoy, y la señal de readiness — el
documento está "listo para prueba" cuando >=85% de los conceptos clave están
dominados (criterio operativo del proyecto).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import text

from .store import get_engine, ensure_migrated
from . import fsrs

READINESS_THRESHOLD = 0.85


async def _concept_ids_for_corpus(corpus_ids: list[str]) -> set[str]:
    """Conceptos cuyo source_chunks toca alguno de los documentos del corpus."""
    want_docs = {str(d) for d in corpus_ids}
    engine = get_engine()
    async with engine.connect() as conn:
        concept_rows = (await conn.execute(
            text("SELECT concept_id, source_chunks FROM study_concepts"))).fetchall()
        chunk_rows = (await conn.execute(
            text("SELECT id, doc_id FROM document_chunks"))).fetchall()
    chunk_doc = {str(cid): str(did) for cid, did in chunk_rows}
    out: set[str] = set()
    for concept_id, sc in concept_rows:
        try:
            chunks = json.loads(sc or "[]")
        except (json.JSONDecodeError, TypeError):
            chunks = []
        if any(chunk_doc.get(str(ch)) in want_docs for ch in chunks):
            out.add(concept_id)
    return out


async def progress(corpus_ids: list[str], now: Optional[datetime] = None) -> dict:
    await ensure_migrated()
    concept_ids = await _concept_ids_for_corpus(corpus_ids)
    total = len(concept_ids)
    cards = [c for c in await fsrs.all_cards() if c.concept_id in concept_ids]
    mastered = [c for c in cards if fsrs.is_mastered(c)]
    due = [c for c in await fsrs.due_cards(now) if c.concept_id in concept_ids]
    pct = (len(mastered) / total) if total else 0.0
    return {
        "total_concepts": total,
        "cards": len(cards),
        "mastered": len(mastered),
        "mastery_pct": round(pct, 3),
        "due_today": len(due),
        "ready_for_exam": pct >= READINESS_THRESHOLD,
        "readiness_threshold": READINESS_THRESHOLD,
    }
