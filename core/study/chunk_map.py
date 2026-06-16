"""
chunk_map.py — Acceso a los chunks del document_store como objetos Chunk de EstudIA.

Centraliza el SQL crudo sobre document_chunks (ya migrada con page/section_path)
para que el resto del paquete trabaje con el contrato Chunk y nunca toque la tabla
directamente. Los doc_id de pentamodal son enteros (documents.id); aquí se manejan
como str hacia afuera.
"""

from __future__ import annotations

import json
import re

from sqlalchemy import text

from .store import get_engine
from .models import Chunk


def _to_int_ids(ids) -> list[int]:
    out = []
    for d in ids:
        try:
            out.append(int(d))
        except (TypeError, ValueError):
            continue
    return out


def _row_to_chunk(row) -> Chunk:
    cid, doc_id, content, page, section_path = row
    try:
        sp = json.loads(section_path) if section_path else []
    except (json.JSONDecodeError, TypeError):
        sp = []
    return Chunk(chunk_id=str(cid), doc_id=str(doc_id), text=content or "",
                 page=page, section_path=sp)


async def get_chunks_for_docs(doc_ids: list[str], limit: int = 2000) -> list[Chunk]:
    """Todos los chunks de los documentos indicados (orden de lectura)."""
    int_ids = _to_int_ids(doc_ids)
    if not int_ids:
        return []
    ph = ",".join(f":d{i}" for i in range(len(int_ids)))
    params = {f"d{i}": v for i, v in enumerate(int_ids)}
    params["lim"] = limit
    engine = get_engine()
    async with engine.connect() as conn:
        res = await conn.execute(
            text(f"""
                SELECT id, doc_id, content, page, section_path
                FROM document_chunks
                WHERE doc_id IN ({ph})
                ORDER BY doc_id, chunk_index
                LIMIT :lim
            """),
            params,
        )
        return [_row_to_chunk(r) for r in res.fetchall()]


async def get_chunks_by_ids(chunk_ids: list[str]) -> list[Chunk]:
    """Chunks por id (preserva el orden de entrada; omite ids inexistentes)."""
    int_ids = _to_int_ids(chunk_ids)
    if not int_ids:
        return []
    ph = ",".join(f":i{i}" for i in range(len(int_ids)))
    params = {f"i{i}": v for i, v in enumerate(int_ids)}
    engine = get_engine()
    async with engine.connect() as conn:
        res = await conn.execute(
            text(f"SELECT id, doc_id, content, page, section_path "
                 f"FROM document_chunks WHERE id IN ({ph})"),
            params,
        )
        by_id = {r[0]: _row_to_chunk(r) for r in res.fetchall()}
    return [by_id[i] for i in int_ids if i in by_id]


async def get_chunk(chunk_id: str) -> Chunk | None:
    try:
        cid = int(chunk_id)
    except (TypeError, ValueError):
        return None
    engine = get_engine()
    async with engine.connect() as conn:
        res = await conn.execute(
            text("SELECT id, doc_id, content, page, section_path FROM document_chunks WHERE id = :id"),
            {"id": cid},
        )
        row = res.fetchone()
    return _row_to_chunk(row) if row else None


async def topic_chunks(doc_ids: list[str], topic: str, top_k: int = 12) -> list[Chunk]:
    """Chunks de los documentos rankeados por solapamiento léxico con `topic`.

    Ranking lexical simple (v1): cuenta de términos del topic presentes en el chunk.
    El ranking semántico (hybrid_search) se cableará en una ola posterior."""
    chunks = await get_chunks_for_docs(doc_ids)
    terms = [t for t in re.findall(r"[\wáéíóúüñ]+", (topic or "").lower()) if len(t) > 2]
    if not terms:
        return chunks[:top_k]
    scored: list[tuple[int, Chunk]] = []
    for c in chunks:
        low = c.text.lower()
        s = sum(low.count(t) for t in terms)
        if s > 0:
            scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    ranked = [c for _, c in scored[:top_k]]
    return ranked or chunks[:top_k]
