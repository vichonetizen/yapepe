"""
grounding.py — LA COMPUERTA DE LA REGLA DURA de EstudIA (pista integrada).

Única autoridad para emitir el "sello de cita". Todo lo que el motor de estudio
produzca (Question, GradeResult, Comparison, Debate, Edge, Concept) debe pasar por
aquí antes de salir. Dos niveles de garantía:

  • assert_grounded(obj)  — estructural y SIN I/O: el objeto trae `evidence`
    (o equivalente) NO vacía. Barato; se usa en bucles de generación.
  • assert_resolvable(obj) — con I/O: cada chunk_id de la evidencia EXISTE en
    document_chunks y (opcionalmente) pertenece al conjunto de chunks que se
    recuperaron para esa tarea. Esto bloquea la alucinación de citas por el LLM:
    nunca confiamos en que el modelo "inventó" un id válido.

Sin evidencia resoluble, no se emite. Si falta, se marca N/D — no se rellena a ojo.
"""

from __future__ import annotations

import json
from typing import Iterable, Optional

from sqlalchemy import text

from .store import get_engine
from .models import Citation


class UngroundedError(Exception):
    """Se intentó emitir algo sin evidencia válida/resoluble. Regla dura violada."""


# --------------------------------------------------------------------------- #
# Nivel 1 — estructural (sin I/O)
# --------------------------------------------------------------------------- #
def _evidence_of(obj) -> list:
    """Extrae la evidencia de cualquier contrato del motor de estudio."""
    # Question / GradeResult / Edge.evidence (lista o str)
    ev = getattr(obj, "evidence", None)
    if ev is not None:
        return list(ev) if isinstance(ev, (list, tuple, set)) else [ev]
    # Concept.source_chunks
    sc = getattr(obj, "source_chunks", None)
    if sc is not None:
        return list(sc)
    return []


def assert_grounded(obj) -> None:
    """Rechaza objetos sin evidencia estructural. No toca la base de datos.

    Para Comparison/Debate exige además que CADA dimensión/posición traiga su
    propia evidencia (la regla dura extendida del spec)."""
    # Comparison: cada dimension con evidence
    dims = getattr(obj, "dimensions", None)
    if dims is not None:
        if not dims or any(not d.get("evidence") for d in dims):
            raise UngroundedError(
                "Comparison sin evidencia en alguna dimensión: no se emite."
            )
        return
    # Debate: cada position con evidence
    positions = getattr(obj, "positions", None)
    if positions is not None:
        if not positions or any(not p.get("evidence") for p in positions):
            raise UngroundedError(
                "Debate sin evidencia en alguna posición: no se emite."
            )
        return
    # Resto (Question, GradeResult IA, Edge, Concept)
    if not _evidence_of(obj):
        raise UngroundedError(
            f"{type(obj).__name__} sin evidencia: regla dura violada, no se emite."
        )


def is_grounded(obj) -> bool:
    try:
        assert_grounded(obj)
        return True
    except UngroundedError:
        return False


# --------------------------------------------------------------------------- #
# Nivel 2 — resolución contra la fuente (con I/O)
# --------------------------------------------------------------------------- #
async def resolve_evidence(chunk_ids: Iterable[str], quote_chars: int = 240) -> list[Citation]:
    """Resuelve chunk_ids -> [Citation] consultando document_chunks.

    Lanza UngroundedError si algún id no existe (cita colgada/alucinada)."""
    ids = [str(c) for c in chunk_ids]
    if not ids:
        raise UngroundedError("resolve_evidence recibió evidencia vacía.")

    # IDs internos son enteros autoincrement; descartamos no numéricos de una.
    int_ids = []
    for c in ids:
        try:
            int_ids.append(int(c))
        except (TypeError, ValueError):
            raise UngroundedError(f"chunk_id no válido (no numérico): {c!r}")

    placeholders = ",".join(f":id{i}" for i in range(len(int_ids)))
    params = {f"id{i}": v for i, v in enumerate(int_ids)}

    engine = get_engine()
    async with engine.connect() as conn:
        res = await conn.execute(
            text(f"""
                SELECT dc.id, dc.doc_id, dc.content, dc.page, dc.section_path, d.filename
                FROM document_chunks dc
                JOIN documents d ON dc.doc_id = d.id
                WHERE dc.id IN ({placeholders})
            """),
            params,
        )
        rows = {r[0]: r for r in res.fetchall()}

    missing = [c for c in int_ids if c not in rows]
    if missing:
        raise UngroundedError(
            f"Citas no resolubles (chunk_ids inexistentes): {missing}. "
            "Posible alucinación de cita; se descarta."
        )

    citations: list[Citation] = []
    for cid in int_ids:
        _id, doc_id, content, page, section_path, filename = rows[cid]
        try:
            sp = json.loads(section_path) if section_path else []
        except (json.JSONDecodeError, TypeError):
            sp = []
        citations.append(Citation(
            chunk_id=str(_id),
            doc_id=str(doc_id),
            filename=filename,
            page=page,
            section_path=sp,
            quote=(content or "")[:quote_chars],
        ))
    return citations


async def assert_resolvable(obj, allowed_chunk_ids: Optional[Iterable[str]] = None) -> list[Citation]:
    """Garantía fuerte anti-alucinación: la evidencia es estructural Y resoluble,
    y (si se pasa allowed_chunk_ids) está DENTRO del conjunto recuperado para la
    tarea. Devuelve las citas resueltas para poder renderizarlas."""
    assert_grounded(obj)
    ev = _evidence_of(obj)
    if allowed_chunk_ids is not None:
        allowed = {str(c) for c in allowed_chunk_ids}
        fuera = [str(c) for c in ev if str(c) not in allowed]
        if fuera:
            raise UngroundedError(
                f"Evidencia fuera del corpus recuperado para esta tarea: {fuera}. "
                "El LLM citó chunks que no se le entregaron; se descarta."
            )
    return await resolve_evidence(ev)


# --------------------------------------------------------------------------- #
# Render de citas
# --------------------------------------------------------------------------- #
def build_citation_block(citations: list[Citation]) -> str:
    """Bloque de citas legible para adjuntar a una respuesta del tutor."""
    if not citations:
        return ""
    lines = ["Fuentes:"]
    for c in citations:
        quote = c.quote.strip().replace("\n", " ")
        if len(quote) > 160:
            quote = quote[:157] + "…"
        lines.append(f"  • {c.label()} «{quote}»")
    return "\n".join(lines)
