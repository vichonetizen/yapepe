"""
Prueba del motor de estudio v1 de EstudIA (pista integrada) — Ola 2.

Extremo a extremo SIN claves de API ni embeddings, contra una DB temporal:
  1. focus.retrieve(ALL) trae los chunks del corpus; TOPIC filtra por tema.
  2. generate() (fallback) produce preguntas, TODAS ancladas y dentro del corpus.
  3. session.start_session + submit_answer califican y PERSISTEN en study_results.
  4. La evidencia de las preguntas es resoluble a una cita real (doc_id/página).
  5. El router /study/* importa y expone rutas.
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

from core.study import store, focus, generate, grade, session, grounding  # noqa: E402
from core.study.models import StudyRequest, Focus  # noqa: E402


def _ok(cond: bool, msg: str) -> bool:
    print(("  [OK]   " if cond else "  [FALLA] ") + msg)
    return cond


_CHUNKS = [
    "La autopoiesis es la organización de lo vivo según Maturana y Varela.",
    "Un sistema autopoiético se produce a sí mismo de manera continua y cerrada.",
    "El conocer es efectivo en el dominio de la acción del observador humano.",
    "La deriva estructural acopla al organismo con su medio a lo largo del tiempo.",
]


async def _run() -> bool:
    passed = True
    tmp = os.path.join(tempfile.gettempdir(), "estudia_test_motor.db")
    if os.path.exists(tmp):
        os.remove(tmp)
    store.use_db(tmp)
    await store.migrate()

    engine = store.get_engine()
    async with engine.begin() as conn:
        r = await conn.execute(
            text("INSERT INTO documents (filename, file_size, chunk_count) VALUES (:f, 0, :n)"),
            {"f": "Maturana - El árbol del conocimiento.pdf", "n": len(_CHUNKS)},
        )
        doc_id = str(r.lastrowid)
        for i, c in enumerate(_CHUNKS):
            await conn.execute(
                text("INSERT INTO document_chunks (doc_id, chunk_index, content, page, section_path) "
                     "VALUES (:d, :i, :c, :p, :sp)"),
                {"d": int(doc_id), "i": i, "c": c, "p": 10 + i, "sp": '["Cap. 1"]'},
            )

    # 1. focus -----------------------------------------------------------------
    req_all = StudyRequest(corpus_ids=[doc_id], focus=Focus.ALL)
    chunks_all, mode_all = await focus.retrieve(req_all)
    passed &= _ok(len(chunks_all) == len(_CHUNKS) and mode_all == "all",
                  f"focus ALL trae los {len(_CHUNKS)} chunks")
    passed &= _ok(all(c.page is not None for c in chunks_all), "los chunks traen página")

    req_topic = StudyRequest(corpus_ids=[doc_id], focus=Focus.TOPIC, topic="autopoiesis")
    chunks_topic, mode_topic = await focus.retrieve(req_topic)
    passed &= _ok(0 < len(chunks_topic) <= len(_CHUNKS) and
                  all("autopoi" in c.text.lower() for c in chunks_topic),
                  "focus TOPIC filtra a chunks del tema")

    # COMPARE sin conceptos clave degrada honestamente a topic (con aviso)
    _, mode_cmp = await focus.retrieve(StudyRequest(corpus_ids=[doc_id], focus=Focus.COMPARE, topic="conocer"))
    passed &= _ok("compare" in mode_cmp and ("topic" in mode_cmp or "sin conceptos" in mode_cmp),
                  f"COMPARE sin conceptos clave degrada con aviso (modo: {mode_cmp})")

    # 2. generación anclada ----------------------------------------------------
    qs = generate.generate(req_all, chunks_all, max_questions=6)
    allowed = {c.chunk_id for c in chunks_all}
    passed &= _ok(len(qs) >= 1, f"generate() produce preguntas ({len(qs)})")
    passed &= _ok(all(q.is_grounded() for q in qs), "todas ancladas (evidence no vacía)")
    passed &= _ok(all(all(e in allowed for e in q.evidence) for q in qs),
                  "toda evidencia dentro del corpus recuperado")

    # 3. sesión + calificación persistida -------------------------------------
    started = await session.start_session(req_all, n=4)
    sid = started["session_id"]
    q0 = started["questions"][0]
    # respuesta correcta a la primera (V/F 'verdadero' o cloze con su answer_key)
    correct_answer = q0.answer_key
    gr = await session.submit_answer(sid, q0.q_id, correct_answer)
    passed &= _ok(gr.score == 1.0 and bool(gr.evidence),
                  "submit_answer correcto -> score 1.0 con evidencia")
    results = await session.session_results(sid)
    passed &= _ok(len(results) == 1 and results[0]["q_id"] == q0.q_id,
                  "el resultado se PERSISTE en study_results")

    # 4. cita resoluble --------------------------------------------------------
    cites = await grounding.resolve_evidence(q0.evidence)
    passed &= _ok(cites[0].page is not None and "Maturana" in cites[0].filename,
                  "la evidencia de la pregunta resuelve a una cita real")

    # 5. router ----------------------------------------------------------------
    from core.study.api import router as study_router
    paths = {r.path for r in study_router.routes}
    passed &= _ok({"/study/session", "/study/grade", "/study/generate"} <= paths,
                  "el router expone /study/session, /study/grade, /study/generate")

    try:
        os.remove(tmp)
    except OSError:
        pass
    return passed


def main() -> bool:
    print("== Motor de estudio v1 EstudIA (integrada): Ola 2 ==")
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
