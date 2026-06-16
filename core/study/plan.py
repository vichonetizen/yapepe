"""
plan.py — Plan de aprendizaje dinámico (Ola 4, pista integrada).

Prioriza lo que toca repasar hoy (cards vencidas por FSRS) y lo que aún no se domina
(reps < criterio). Es la agenda que el panel muestra y que la ola adaptativa (Ola 5)
reordenará según ErrorClusters.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Optional

from . import fsrs


async def build_plan(now: Optional[datetime] = None, limit: int = 20) -> dict:
    due = await fsrs.due_cards(now)
    cards = await fsrs.all_cards()
    to_learn = [c for c in cards if not fsrs.is_mastered(c) and c not in due]
    queue = due + to_learn
    return {
        "due_today": len(due),
        "to_learn": len(to_learn),
        "next": asdict(queue[0]) if queue else None,
        "queue": [asdict(c) for c in queue[:limit]],
    }
