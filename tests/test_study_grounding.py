"""
Prueba de la Ola 0 de EstudIA (pista integrada) — cimientos de cita y persistencia.

Verifica:
  1. La migración es IDEMPOTENTE (corre 2 veces sin error) y añade page/section_path.
  2. resolve_evidence() resuelve un chunk real a una Citation con doc_id/page.
  3. resolve_evidence() de un chunk_id inexistente lanza UngroundedError (anti-cita-colgada).
  4. assert_grounded(Question(evidence=[])) lanza UngroundedError; con evidencia, no.
  5. assert_resolvable() rechaza evidencia FUERA del corpus recuperado (anti-alucinación).
  6. Las 8 dataclasses nuevas (Card..ErrorCluster) importan sin error.

Usa una DB temporal: NO toca data/pentamodal.db.
"""

import asyncio
import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")  # consola Windows cp1252 -> utf-8
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402

from core.study import store, grounding  # noqa: E402
from core.study.models import (  # noqa: E402
    Question, QuestionType, Card, Exam, ConceptMap, Hierarchy,
    Comparison, Debate, MasteryUpdate, ErrorCluster,
)


def _ok(cond: bool, msg: str) -> bool:
    print(("  [OK]   " if cond else "  [FALLA] ") + msg)
    return cond


async def _run() -> bool:
    passed = True
    tmp = os.path.join(tempfile.gettempdir(), "estudia_test_ola0.db")
    if os.path.exists(tmp):
        os.remove(tmp)
    store.use_db(tmp)

    # 1. Migración idempotente -------------------------------------------------
    s1 = await store.migrate()
    s2 = await store.migrate()  # segunda vez: no debe romper ni re-añadir
    passed &= _ok("page" in s1["chunks_columns_added"], "migrate() añade columna page")
    passed &= _ok(s2["chunks_columns_added"] == [], "migrate() es idempotente (2ª corrida no añade nada)")
    hc = await store.healthcheck()
    passed &= _ok(hc["chunks_have_page"] and hc["chunks_have_section_path"],
                  "document_chunks tiene page y section_path tras migrar")
    passed &= _ok(len(s1["study_tables_ensured"]) == 8, "se crean las 8 tablas study_*")

    # Sembrar un documento + chunk con página real ----------------------------
    engine = store.get_engine()
    async with engine.begin() as conn:
        r = await conn.execute(
            text("INSERT INTO documents (filename, file_size, chunk_count) VALUES (:f, 0, 1)"),
            {"f": "Maturana - El árbol del conocimiento.pdf"},
        )
        doc_id = r.lastrowid
        r2 = await conn.execute(
            text("INSERT INTO document_chunks (doc_id, chunk_index, content, page, section_path) "
                 "VALUES (:d, 0, :c, :p, :sp)"),
            {"d": doc_id, "c": "El conocer es un fenómeno biológico (autopoiesis).",
             "p": 42, "sp": '["Cap. 1", "Conocer"]'},
        )
        chunk_id = str(r2.lastrowid)

    # 2. Resolución de cita ----------------------------------------------------
    cites = await grounding.resolve_evidence([chunk_id])
    c0 = cites[0]
    passed &= _ok(c0.page == 42 and c0.doc_id == str(doc_id) and "Maturana" in c0.filename,
                  "resolve_evidence() devuelve Citation con doc_id, filename y página")
    passed &= _ok(c0.section_path == ["Cap. 1", "Conocer"], "section_path se deserializa correctamente")

    # 3. Cita colgada ----------------------------------------------------------
    try:
        await grounding.resolve_evidence(["99999999"])
        passed &= _ok(False, "chunk_id inexistente DEBE lanzar UngroundedError")
    except grounding.UngroundedError:
        passed &= _ok(True, "chunk_id inexistente lanza UngroundedError (anti-cita-colgada)")

    # 4. Regla dura estructural ------------------------------------------------
    q_bad = Question(q_id="q1", type=QuestionType.OPEN, prompt="¿?", answer_key="x",
                     rationale="", evidence=[])
    q_good = Question(q_id="q2", type=QuestionType.OPEN, prompt="¿?", answer_key="x",
                      rationale="", evidence=[chunk_id])
    try:
        grounding.assert_grounded(q_bad)
        passed &= _ok(False, "Question(evidence=[]) DEBE lanzar UngroundedError")
    except grounding.UngroundedError:
        passed &= _ok(True, "Question(evidence=[]) lanza UngroundedError")
    passed &= _ok(grounding.is_grounded(q_good), "Question con evidencia pasa assert_grounded")

    # 5. Anti-alucinación: evidencia fuera del corpus recuperado ---------------
    q_hallu = Question(q_id="q3", type=QuestionType.OPEN, prompt="¿?", answer_key="x",
                       rationale="", evidence=["123456"])  # id no entregado
    try:
        await grounding.assert_resolvable(q_hallu, allowed_chunk_ids={chunk_id})
        passed &= _ok(False, "evidencia fuera del corpus DEBE lanzar UngroundedError")
    except grounding.UngroundedError:
        passed &= _ok(True, "assert_resolvable rechaza evidencia fuera del corpus recuperado")
    resolved = await grounding.assert_resolvable(q_good, allowed_chunk_ids={chunk_id})
    passed &= _ok(len(resolved) == 1, "assert_resolvable devuelve la cita cuando la evidencia es válida")

    # 6. Dataclasses nuevas existen --------------------------------------------
    _ = [Card("c", "k", "f", "b"), Exam("e", ["d"]), ConceptMap(nodes=["k"]),
         Hierarchy(root_concept="k"), Comparison(concept_ids=["a", "b"]),
         Debate(topic="t"), MasteryUpdate("k", 0.1, 0.2, "t", "a"),
         ErrorCluster("cl", ["k"], "tipo")]
    passed &= _ok(True, "Card/Exam/ConceptMap/Hierarchy/Comparison/Debate/MasteryUpdate/ErrorCluster importan")

    # Comparison/Debate hard rule
    comp_bad = Comparison(concept_ids=["a", "b"], dimensions=[{"dimension": "d", "evidence": []}])
    passed &= _ok(not comp_bad.is_grounded(), "Comparison con dimensión sin evidencia NO está grounded")

    try:
        os.remove(tmp)
    except OSError:
        pass
    return passed


def main() -> bool:
    print("== Ola 0 EstudIA (integrada): cimientos de cita y persistencia ==")
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
