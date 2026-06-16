"""
session.py — Orquestador de la sesión de estudio (pista integrada).

Flujo: start_session(StudyRequest) -> genera preguntas ancladas y las guarda en una
sesión en memoria (proceso); next_question() las sirve en orden; submit_answer()
califica y PERSISTE el resultado en study_results (durabilidad). La traza usa el
mismo data/pentamodal.db; no se crea log paralelo.
"""

from __future__ import annotations

import json
import uuid
from typing import Callable, Optional

from sqlalchemy import text

from .store import get_engine
from .models import StudyRequest, Question, GradeResult
from . import focus as _focus, generate as _generate, grade as _grade

# Sesiones vivas del proceso. Las preguntas (con su evidencia) se guardan aquí para
# poder calificar después sin re-generar; los resultados sí se persisten en SQLite.
_SESSIONS: dict[str, dict] = {}


async def start_session(req: StudyRequest,
                        llm: Optional[Callable] = None,
                        n: int = 5) -> dict:
    chunks, mode = await _focus.retrieve(req)
    questions = _generate.generate(req, chunks, llm=llm, max_questions=n)
    sid = str(uuid.uuid4())
    _SESSIONS[sid] = {
        "questions": {q.q_id: q for q in questions},
        "order": [q.q_id for q in questions],
        "pos": 0,
        "allowed": {c.chunk_id for c in chunks},
    }
    return {
        "session_id": sid,
        "focus_mode": mode,
        "num_questions": len(questions),
        "questions": questions,        # el caller serializa
    }


def get_question(session_id: str, q_id: str) -> Optional[Question]:
    sess = _SESSIONS.get(session_id)
    return sess["questions"].get(q_id) if sess else None


def next_question(session_id: str) -> Optional[Question]:
    sess = _SESSIONS.get(session_id)
    if not sess or sess["pos"] >= len(sess["order"]):
        return None
    qid = sess["order"][sess["pos"]]
    sess["pos"] += 1
    return sess["questions"][qid]


async def submit_answer(session_id: str,
                        q_id: str,
                        answer: str,
                        llm: Optional[Callable] = None) -> GradeResult:
    sess = _SESSIONS.get(session_id)
    if not sess:
        raise KeyError(f"Sesión desconocida: {session_id}")
    question = sess["questions"].get(q_id)
    if question is None:
        raise KeyError(f"Pregunta desconocida en la sesión: {q_id}")
    result = _grade.grade(question, answer, llm=llm)
    await _persist_result(session_id, result)
    return result


async def _persist_result(session_id: str, gr: GradeResult) -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("""INSERT INTO study_results
                    (session_id, q_id, user_answer, score, verdict, feedback, evidence, gaps)
                    VALUES (:s, :q, :ua, :sc, :v, :fb, :ev, :gp)"""),
            {"s": session_id, "q": gr.q_id, "ua": gr.user_answer, "sc": gr.score,
             "v": gr.verdict, "fb": gr.feedback,
             "ev": json.dumps(gr.evidence), "gp": json.dumps(gr.gaps)},
        )


async def session_results(session_id: str) -> list[dict]:
    engine = get_engine()
    async with engine.connect() as conn:
        res = await conn.execute(
            text("""SELECT q_id, user_answer, score, verdict, feedback, evidence, gaps, created_at
                    FROM study_results WHERE session_id = :s ORDER BY id"""),
            {"s": session_id},
        )
        rows = res.fetchall()
    out = []
    for q_id, ua, sc, v, fb, ev, gp, ts in rows:
        out.append({"q_id": q_id, "user_answer": ua, "score": sc, "verdict": v,
                    "feedback": fb, "evidence": json.loads(ev or "[]"),
                    "gaps": json.loads(gp or "[]"), "created_at": str(ts)})
    return out
