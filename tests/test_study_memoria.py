"""
Prueba de la Ola 4 de EstudIA (pista integrada) — memoria: FSRS, panel, examen, plan.

Contra DB y grafo temporales, determinista (now fijo):
  1. fsrs.review: acierto sube interval/reps; 2 aciertos -> dominado; fallo resetea + lapse.
  2. ensure_cards_for_corpus crea una card por concepto.
  3. panel.progress: readiness pasa a True al dominar >=85% de conceptos.
  4. exam.start_exam cubre conceptos; submit_exam (respuestas correctas) puntúa alto.
  5. plan.build_plan lista cards vencidas / por aprender.
"""

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402

from core.study import store, extract, fsrs, panel, plan, exam, concept_graph  # noqa: E402
from core.study import llm as _llm  # noqa: E402
from core.study.models import Card  # noqa: E402

_llm.provider_available = lambda: False  # test determinista: fuerza ruta offline

_NOW = datetime(2026, 6, 15, tzinfo=timezone.utc)
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
    tmp_db = os.path.join(tempfile.gettempdir(), "estudia_test_mem.db")
    tmp_graph = os.path.join(tempfile.gettempdir(), "estudia_test_mem_graph.json")
    for f in (tmp_db, tmp_graph):
        if os.path.exists(f):
            os.remove(f)
    store.use_db(tmp_db)
    concept_graph.use_path(tmp_graph)
    await store.migrate()

    engine = store.get_engine()
    async with engine.begin() as conn:
        r = await conn.execute(
            text("INSERT INTO documents (filename, file_size, chunk_count) VALUES (:f,0,:n)"),
            {"f": "Maturana.pdf", "n": len(_CHUNKS)})
        doc_id = str(r.lastrowid)
        for i, c in enumerate(_CHUNKS):
            await conn.execute(
                text("INSERT INTO document_chunks (doc_id, chunk_index, content, page) VALUES (:d,:i,:c,:p)"),
                {"d": int(doc_id), "i": i, "c": c, "p": i + 1})

    # 1. FSRS determinista -----------------------------------------------------
    card = Card(card_id="c1", concept_id="c1", front="f", back="b")
    fsrs.review(card, 1.0, now=_NOW)
    passed &= _ok(card.reps == 1 and card.interval == 1 and card.due is not None,
                  "fsrs: 1er acierto -> reps=1, interval=1, due fijado")
    fsrs.review(card, 1.0, now=_NOW)
    passed &= _ok(card.reps == 2 and card.interval == 6 and fsrs.is_mastered(card),
                  "fsrs: 2º acierto -> reps=2, interval=6, DOMINADO")
    fsrs.review(card, 0.0, now=_NOW)
    passed &= _ok(card.reps == 0 and card.lapses == 1 and not fsrs.is_mastered(card),
                  "fsrs: fallo -> reps=0, lapse+1, ya no dominado")

    # 2. cards por concepto ----------------------------------------------------
    await extract.extract_corpus([doc_id])
    created = await fsrs.ensure_cards_for_corpus([doc_id])
    passed &= _ok(created >= 3, f"ensure_cards crea una card por concepto ({created})")

    # 3. panel: readiness ------------------------------------------------------
    p0 = await panel.progress([doc_id], now=_NOW)
    passed &= _ok(not p0["ready_for_exam"] and p0["mastery_pct"] == 0.0,
                  "panel: sin repasos, no listo para prueba")
    for c in await fsrs.all_cards():
        fsrs.review(c, 1.0, now=_NOW)
        fsrs.review(c, 1.0, now=_NOW)
        await fsrs.upsert_card(c)
    p1 = await panel.progress([doc_id], now=_NOW)
    passed &= _ok(p1["ready_for_exam"] and p1["mastery_pct"] >= 0.85,
                  f"panel: tras dominar, listo para prueba ({p1['mastery_pct']:.0%})")

    # 4. examen ----------------------------------------------------------------
    ex = await exam.start_exam([doc_id], coverage_target=0.5, n=6)
    passed &= _ok(ex["num_questions"] >= 1, f"exam.start genera preguntas ({ex['num_questions']})")
    passed &= _ok(0.0 <= ex["coverage"] <= 1.0, f"exam reporta cobertura ({ex['coverage']:.0%})")
    answers = {q.q_id: q.answer_key for q in ex["questions"]}
    rep = await exam.submit_exam(ex["exam_id"], answers)
    passed &= _ok(rep["score"] >= 0.5, f"exam.submit con respuestas correctas puntúa alto ({rep['score']:.0%})")

    # 5. plan ------------------------------------------------------------------
    pl = await plan.build_plan(now=_NOW)
    passed &= _ok("queue" in pl and isinstance(pl["queue"], list), "plan.build_plan devuelve cola de repaso")

    for f in (tmp_db, tmp_graph):
        try:
            os.remove(f)
        except OSError:
            pass
    return passed


def main() -> bool:
    print("== Memoria EstudIA (integrada): Ola 4 (FSRS/examen/panel/plan) ==")
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
