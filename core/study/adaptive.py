"""
adaptive.py — Aprendizaje adaptativo de EstudIA (Ola 5, pista integrada).

Cierra el lazo de mejora con la señal de desempeño:
  • adjust_difficulty(card): a partir de reps/lapses propone subir/bajar dificultad
    y emite un MasteryUpdate auditable (señal de "refuerzo").
  • cluster_errors(): agrupa de forma NO supervisada los conceptos que el estudiante
    falla juntos (co-ocurrencia en respuestas erradas, union-find) → ErrorClusters.
    Sin sklearn: clustering por co-fallo, explicable y determinista.
  • reorder(): prioriza el plan poniendo primero los conceptos de los clusters y los
    no dominados.

Lee la traza ya persistida (study_results.gaps) y las cards (FSRS). No reentrena
modelos: la "recompensa" es el dominio medido.
"""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy import text

from .store import get_engine, ensure_migrated
from .models import MasteryUpdate, ErrorCluster
from . import fsrs


def _level(reps: int, lapses: int) -> float:
    """Nivel de dominio 0..1 a partir de aciertos consecutivos y fallos."""
    return round(reps / (reps + lapses + 1), 3)


def adjust_difficulty(card) -> MasteryUpdate:
    """Propone ajuste de dificultad según la tendencia de la card (no muta la card)."""
    prev = _level(card.reps, card.lapses)
    if card.reps >= fsrs.MASTERY_REPS and card.lapses == 0:
        action = "subir dificultad"
        new = min(1.0, prev + 0.1)
    elif card.lapses > 0 and card.reps == 0:
        action = "reforzar (bajar dificultad)"
        new = max(0.0, prev - 0.1)
    else:
        action = "mantener"
        new = prev
    return MasteryUpdate(concept_id=card.concept_id, prev_level=prev, new_level=new,
                         trigger=f"reps={card.reps}, lapses={card.lapses}", action=action)


# --------------------------------------------------------------------------- #
# Clustering de errores (no supervisado, por co-fallo)
# --------------------------------------------------------------------------- #
async def _failed_gap_sets(corpus_ids: Optional[list[str]] = None) -> list[list[str]]:
    """Listas de concept_ids fallados juntos (gaps de GradeResults con score<0.6)."""
    engine = get_engine()
    async with engine.connect() as conn:
        rows = (await conn.execute(
            text("SELECT gaps FROM study_results WHERE score < 0.6"))).fetchall()
    sets = []
    for (gaps,) in rows:
        try:
            ids = [str(g) for g in json.loads(gaps or "[]")]
        except (json.JSONDecodeError, TypeError):
            ids = []
        if ids:
            sets.append(ids)
    return sets


def _union_find(gap_sets: list[list[str]]) -> dict[str, list[str]]:
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for s in gap_sets:
        for c in s:
            find(c)
        for c in s[1:]:
            union(s[0], c)

    groups: dict[str, list[str]] = {}
    for c in parent:
        groups.setdefault(find(c), []).append(c)
    return groups


async def cluster_errors(corpus_ids: Optional[list[str]] = None,
                         persist: bool = True) -> list[ErrorCluster]:
    await ensure_migrated()
    gap_sets = await _failed_gap_sets(corpus_ids)
    if not gap_sets:
        return []
    groups = _union_find(gap_sets)

    clusters: list[ErrorCluster] = []
    for root, concept_ids in groups.items():
        concept_set = set(concept_ids)
        freq = sum(1 for s in gap_sets if concept_set & set(s))
        cluster_id = "cl_" + sorted(concept_ids)[0]
        clusters.append(ErrorCluster(
            cluster_id=cluster_id,
            concept_ids=sorted(concept_ids),
            error_type="dominio insuficiente (fallos recurrentes)",
            frequency=freq,
            recommended_action=("repasar y comparar estos conceptos relacionados; "
                                "subir su prioridad en el plan"),
        ))
    clusters.sort(key=lambda c: -c.frequency)

    if persist:
        engine = get_engine()
        async with engine.begin() as conn:
            for cl in clusters:
                await conn.execute(
                    text("""INSERT OR REPLACE INTO study_error_clusters
                            (cluster_id, concept_ids, error_type, frequency, recommended_action)
                            VALUES (:id,:c,:t,:f,:a)"""),
                    {"id": cl.cluster_id, "c": json.dumps(cl.concept_ids),
                     "t": cl.error_type, "f": cl.frequency, "a": cl.recommended_action})
    return clusters


# --------------------------------------------------------------------------- #
# Reordenamiento del plan
# --------------------------------------------------------------------------- #
async def reorder(corpus_ids: Optional[list[str]] = None) -> dict:
    """Reordena las cards: conceptos en ErrorClusters primero, luego no dominados,
    luego dominados. Devuelve el orden y los clusters/updates."""
    clusters = await cluster_errors(corpus_ids)
    cluster_concepts = {c for cl in clusters for c in cl.concept_ids}
    cards = await fsrs.all_cards()

    def rank(card) -> tuple:
        if card.concept_id in cluster_concepts:
            bucket = 0
        elif not fsrs.is_mastered(card):
            bucket = 1
        else:
            bucket = 2
        return (bucket, -card.lapses, -((card.reps + 1) ** -1))

    ordered = sorted(cards, key=rank)
    updates = [adjust_difficulty(c) for c in cards]
    return {
        "clusters": [cl.__dict__ for cl in clusters],
        "order": [c.concept_id for c in ordered],
        "updates": [u.__dict__ for u in updates],
    }
