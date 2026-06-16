"""
Prueba de la Ola 5 de EstudIA (pista integrada) — aprendizaje adaptativo.

Contra DB/grafo temporales:
  1. cluster_errors agrupa conceptos fallados juntos (union-find sobre gaps).
  2. adjust_difficulty emite MasteryUpdate (sube tras racha, baja tras lapse).
  3. reorder pone los conceptos de los clusters antes que los dominados.
"""

import asyncio
import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402

from core.study import store, extract, fsrs, adaptive, concept_graph  # noqa: E402
from core.study.models import Card  # noqa: E402

_CHUNKS = [
    "La autopoiesis es la organización de lo vivo según Maturana y Varela.",
    "Maturana sostiene que la cognición es un fenómeno biológico del vivir.",
    "La deriva estructural acopla al organismo con su medio continuamente.",
]


def _ok(cond: bool, msg: str) -> bool:
    print(("  [OK]   " if cond else "  [FALLA] ") + msg)
    return cond


async def _run() -> bool:
    passed = True
    tmp_db = os.path.join(tempfile.gettempdir(), "estudia_test_adapt.db")
    tmp_graph = os.path.join(tempfile.gettempdir(), "estudia_test_adapt_graph.json")
    for f in (tmp_db, tmp_graph):
        if os.path.exists(f):
            os.remove(f)
    store.use_db(tmp_db)
    concept_graph.use_path(tmp_graph)
    await store.migrate()

    engine = store.get_engine()
    async with engine.begin() as conn:
        r = await conn.execute(
            text("INSERT INTO documents (filename, file_size, chunk_count) VALUES ('Maturana.pdf',0,:n)"),
            {"n": len(_CHUNKS)})
        doc_id = str(r.lastrowid)
        for i, c in enumerate(_CHUNKS):
            await conn.execute(
                text("INSERT INTO document_chunks (doc_id, chunk_index, content, page) VALUES (:d,:i,:c,:p)"),
                {"d": int(doc_id), "i": i, "c": c, "p": i + 1})

    await extract.extract_corpus([doc_id])
    await fsrs.ensure_cards_for_corpus([doc_id])
    cards = await fsrs.all_cards()
    concept_ids = [c.concept_id for c in cards]
    assert len(concept_ids) >= 4, "se necesitan >=4 conceptos para la prueba"
    c1, c2, c3, c_master = concept_ids[0], concept_ids[1], concept_ids[2], concept_ids[3]

    # Simula fallos: c1+c2 juntos, c2+c3 juntos -> deben quedar en UN cluster.
    import json
    async with engine.begin() as conn:
        for gaps in ([c1, c2], [c2, c3], [c1, c2]):
            await conn.execute(
                text("INSERT INTO study_results (session_id,q_id,user_answer,score,verdict,feedback,evidence,gaps) "
                     "VALUES ('s','q','x',0.0,'Incorrecto','f','[]',:g)"),
                {"g": json.dumps(gaps)})

    # 1. cluster_errors --------------------------------------------------------
    clusters = await adaptive.cluster_errors([doc_id])
    passed &= _ok(len(clusters) >= 1, f"cluster_errors produce clusters ({len(clusters)})")
    big = max(clusters, key=lambda c: len(c.concept_ids))
    passed &= _ok({c1, c2, c3} <= set(big.concept_ids),
                  "los conceptos co-fallados quedan en un mismo cluster")
    passed &= _ok(big.frequency >= 2 and bool(big.recommended_action),
                  "el cluster trae frecuencia y acción recomendada")

    # 2. adjust_difficulty -----------------------------------------------------
    up_win = adaptive.adjust_difficulty(Card("k", "k", "f", "b", reps=2, lapses=0))
    up_lose = adaptive.adjust_difficulty(Card("k", "k", "f", "b", reps=0, lapses=2))
    passed &= _ok(up_win.action == "subir dificultad" and up_win.new_level >= up_win.prev_level,
                  "adjust_difficulty sube tras racha de aciertos")
    passed &= _ok("reforzar" in up_lose.action and up_lose.new_level <= up_lose.prev_level,
                  "adjust_difficulty baja/refuerza tras lapse")

    # 3. reorder ---------------------------------------------------------------
    mcard = await fsrs.get_card(c_master)
    fsrs.review(mcard, 1.0); fsrs.review(mcard, 1.0)   # dominado, fuera de clusters
    await fsrs.upsert_card(mcard)
    res = await adaptive.reorder([doc_id])
    order = res["order"]
    passed &= _ok(order.index(c1) < order.index(c_master) and order.index(c2) < order.index(c_master),
                  "reorder pone los conceptos del cluster antes que el dominado")

    for f in (tmp_db, tmp_graph):
        try:
            os.remove(f)
        except OSError:
            pass
    return passed


def main() -> bool:
    print("== Adaptativo EstudIA (integrada): Ola 5 ==")
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
