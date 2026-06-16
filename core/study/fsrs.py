"""
fsrs.py — Repetición espaciada de EstudIA (Ola 4, pista integrada).

Programador FSRS-lite derivado de SM-2: aritmética pura y determinista (corre en la
Máquina B de 8 GB, sin LLM). Gestiona due/interval/ease/lapses/reps por Card. El
criterio de dominio del proyecto se implementa aquí: un concepto queda "dominado"
con >=2 repasos correctos a dificultad creciente (cada acierto sube card.difficulty
y reps; un fallo resetea reps y suma lapse).

Las fechas se pasan por parámetro `now` para que las pruebas sean deterministas.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import text

from .store import get_engine, ensure_migrated
from .models import Card
from . import chunk_map

MASTERY_REPS = 2   # criterio de dominio: >=2 repasos correctos a dificultad creciente
PASS_QUALITY = 3   # quality < 3 = fallo (escala 0..5)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def score_to_quality(score: float) -> int:
    """Mapea score 0..1 (de GradeResult) a quality 0..5 (escala SM-2)."""
    return max(0, min(5, round(score * 5)))


def review(card: Card, score: float, now: Optional[datetime] = None) -> Card:
    """Aplica un repaso (in place) y devuelve la card actualizada."""
    now = now or _now()
    q = score_to_quality(score)
    if q < PASS_QUALITY:
        card.reps = 0
        card.lapses += 1
        card.interval = 1
        card.ease = max(1.3, card.ease - 0.2)
    else:
        card.reps += 1
        if card.reps == 1:
            card.interval = 1
        elif card.reps == 2:
            card.interval = 6
        else:
            card.interval = max(1, round(card.interval * card.ease))
        card.ease = max(1.3, card.ease + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))
        card.difficulty = min(5, card.difficulty + 1)   # dificultad creciente
    card.last_review = now.isoformat()
    card.due = (now + timedelta(days=card.interval)).isoformat()
    return card


def is_mastered(card: Card) -> bool:
    """Dominado: >=2 repasos correctos consecutivos (a dificultad creciente)."""
    return card.reps >= MASTERY_REPS


# --------------------------------------------------------------------------- #
# Persistencia de cards (study_cards)
# --------------------------------------------------------------------------- #
_COLS = "card_id, concept_id, front, back, due, interval, ease, lapses, reps, last_review, difficulty"


def _row_to_card(r) -> Card:
    return Card(card_id=r[0], concept_id=r[1], front=r[2], back=r[3], due=r[4],
                interval=r[5] or 0, ease=r[6] if r[6] is not None else 2.5,
                lapses=r[7] or 0, reps=r[8] or 0, last_review=r[9],
                difficulty=r[10] or 3)


async def upsert_card(card: Card) -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("""INSERT OR REPLACE INTO study_cards
                    (card_id, concept_id, front, back, due, interval, ease, lapses, reps, last_review, difficulty)
                    VALUES (:id,:cid,:f,:b,:due,:iv,:ea,:lp,:rp,:lr,:df)"""),
            {"id": card.card_id, "cid": card.concept_id, "f": card.front, "b": card.back,
             "due": card.due, "iv": card.interval, "ea": card.ease, "lp": card.lapses,
             "rp": card.reps, "lr": card.last_review, "df": card.difficulty})


async def get_card(card_id: str) -> Optional[Card]:
    engine = get_engine()
    async with engine.connect() as conn:
        r = (await conn.execute(
            text(f"SELECT {_COLS} FROM study_cards WHERE card_id = :id"),
            {"id": card_id})).fetchone()
    return _row_to_card(r) if r else None


async def all_cards() -> list[Card]:
    engine = get_engine()
    async with engine.connect() as conn:
        rows = (await conn.execute(text(f"SELECT {_COLS} FROM study_cards"))).fetchall()
    return [_row_to_card(r) for r in rows]


async def due_cards(now: Optional[datetime] = None) -> list[Card]:
    """Cards nunca repasadas (due NULL) o vencidas (due <= now)."""
    now = (now or _now()).isoformat()
    engine = get_engine()
    async with engine.connect() as conn:
        rows = (await conn.execute(
            text(f"SELECT {_COLS} FROM study_cards WHERE due IS NULL OR due <= :n ORDER BY due IS NOT NULL, due"),
            {"n": now})).fetchall()
    return [_row_to_card(r) for r in rows]


async def ensure_cards_for_corpus(corpus_ids: list[str]) -> int:
    """Crea una Card por concepto del corpus (1:1) si no existe. Devuelve cuántas creó."""
    await ensure_migrated()
    import json
    engine = get_engine()
    async with engine.connect() as conn:
        concepts = (await conn.execute(
            text("SELECT concept_id, label, source_chunks FROM study_concepts"))).fetchall()
        existing = {r[0] for r in (await conn.execute(
            text("SELECT card_id FROM study_cards"))).fetchall()}

    created = 0
    for concept_id, label, sc in concepts:
        if concept_id in existing:
            continue
        try:
            chunk_ids = json.loads(sc or "[]")
        except (json.JSONDecodeError, TypeError):
            chunk_ids = []
        back = ""
        if chunk_ids:
            ch = await chunk_map.get_chunk(str(chunk_ids[0]))
            back = (ch.text[:240] if ch else "")
        card = Card(card_id=concept_id, concept_id=concept_id,
                    front=f"Explica el concepto: «{label}»", back=back, difficulty=3)
        await upsert_card(card)
        created += 1
    return created
