"""
exam.py — Modo "rendir prueba" (Ola 4, pista integrada).

start_exam genera un examen cuyas preguntas cubren >=coverage_target de los
conceptos clave del corpus; las preguntas quedan FIJADAS (ancladas) en la sesión de
examen. submit_exam califica de forma DETERMINISTA y no recupera nada en vivo para
el alumno (modo read-only: no llama hybrid_search ni la memoria del estudiante) —
así se evalúa lo que el alumno sabe, no lo que el sistema le sopla.
"""

from __future__ import annotations

import json
import uuid
from typing import Optional

from sqlalchemy import text

from .store import get_engine, ensure_migrated
from .models import StudyRequest, Focus
from . import chunk_map, generate as _generate, grade as _grade

# Exámenes vivos del proceso (preguntas ancladas). La fila study_exams persiste meta.
_EXAMS: dict[str, dict] = {}


async def _concepts_with_chunks(corpus_ids: list[str]) -> dict[str, list[str]]:
    want_docs = {str(d) for d in corpus_ids}
    engine = get_engine()
    async with engine.connect() as conn:
        concept_rows = (await conn.execute(
            text("SELECT concept_id, source_chunks FROM study_concepts"))).fetchall()
        chunk_rows = (await conn.execute(
            text("SELECT id, doc_id FROM document_chunks"))).fetchall()
    chunk_doc = {str(cid): str(did) for cid, did in chunk_rows}
    out: dict[str, list[str]] = {}
    for concept_id, sc in concept_rows:
        try:
            chunks = json.loads(sc or "[]")
        except (json.JSONDecodeError, TypeError):
            chunks = []
        in_corpus = [str(ch) for ch in chunks if chunk_doc.get(str(ch)) in want_docs]
        if in_corpus:
            out[concept_id] = in_corpus
    return out


async def start_exam(corpus_ids: list[str], coverage_target: float = 0.85,
                     n: int = 10, time_limit: Optional[int] = None) -> dict:
    await ensure_migrated()
    concepts = await _concepts_with_chunks(corpus_ids)
    # Reúne los chunks que anclan a los conceptos clave.
    chunk_ids = sorted({cid for ids in concepts.values() for cid in ids})
    chunks = await chunk_map.get_chunks_by_ids(chunk_ids)
    req = StudyRequest(corpus_ids=corpus_ids, focus=Focus.ALL)
    questions = await _generate.agenerate(req, chunks, max_questions=max(n, len(concepts)))

    # Cobertura: conceptos cuyo source_chunk aparece en la evidencia de alguna pregunta.
    cited = {str(e) for q in questions for e in q.evidence}
    covered = {cid for cid, ids in concepts.items() if set(ids) & cited}
    coverage = (len(covered) / len(concepts)) if concepts else 0.0

    exam_id = str(uuid.uuid4())
    _EXAMS[exam_id] = {"questions": {q.q_id: q for q in questions},
                       "order": [q.q_id for q in questions]}
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("""INSERT INTO study_exams
                    (exam_id, corpus_ids, coverage_target, time_limit, is_submitted, questions)
                    VALUES (:id,:c,:ct,:tl,0,:q)"""),
            {"id": exam_id, "c": json.dumps(corpus_ids), "ct": coverage_target,
             "tl": time_limit, "q": json.dumps([q.q_id for q in questions])})

    return {
        "exam_id": exam_id,
        "num_questions": len(questions),
        "num_key_concepts": len(concepts),
        "coverage": round(coverage, 3),
        "coverage_target": coverage_target,
        "meets_target": coverage >= coverage_target,
        "time_limit": time_limit,
        "questions": questions,   # el caller serializa
    }


async def submit_exam(exam_id: str, answers: dict[str, str]) -> dict:
    exam = _EXAMS.get(exam_id)
    if not exam:
        raise KeyError(f"Examen desconocido o expirado: {exam_id}")
    results = []
    total = 0.0
    gaps: set[str] = set()
    for q_id, q in exam["questions"].items():
        ans = answers.get(q_id, "")
        gr = _grade.grade(q, ans)   # determinista, sin recuperación en vivo (read-only)
        total += gr.score
        gaps.update(gr.gaps)
        results.append(gr)

    n = len(exam["questions"])
    score = (total / n) if n else 0.0
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE study_exams SET is_submitted = 1 WHERE exam_id = :id"),
            {"id": exam_id})
    return {
        "exam_id": exam_id,
        "score": round(score, 3),
        "num_questions": n,
        "gaps": sorted(gaps),
        "results": results,   # el caller serializa
    }
