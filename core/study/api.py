"""
api.py — Router FastAPI de EstudIA (/study/*), pista integrada.

Se incluye en app.py con una sola línea (app.include_router). Convierte los
dataclasses de models.py a dict para la respuesta y traduce UngroundedError a
HTTP 422 con mensaje en español. Todos los endpoints aseguran la migración Ola 0
de forma perezosa (idempotente) antes de operar.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import store, session as study_session, focus as study_focus, generate as study_generate
from .models import StudyRequest, Focus, Question, GradeResult
from .grounding import UngroundedError, resolve_evidence

router = APIRouter(prefix="/study", tags=["estudia"])


# --------------------------------------------------------------------------- #
# Serialización
# --------------------------------------------------------------------------- #
def _q_dict(q: Question) -> dict:
    d = asdict(q)
    d["type"] = q.type.value
    return d


def _gr_dict(gr: GradeResult) -> dict:
    return asdict(gr)


# --------------------------------------------------------------------------- #
# Modelos de entrada
# --------------------------------------------------------------------------- #
class StudyRequestIn(BaseModel):
    corpus_ids: list[str]
    focus: str = "all"
    topic: Optional[str] = None
    key_concepts: list[str] = []
    key_authors: list[str] = []
    difficulty: int = 3
    n: int = 5


class AnswerIn(BaseModel):
    session_id: str
    q_id: str
    answer: str


def _to_request(body: StudyRequestIn) -> StudyRequest:
    try:
        focus = Focus(body.focus)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"focus inválido: {body.focus}")
    return StudyRequest(
        corpus_ids=body.corpus_ids, focus=focus, topic=body.topic,
        key_concepts=body.key_concepts, key_authors=body.key_authors,
        difficulty=body.difficulty,
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.post("/migrate")
async def study_migrate():
    return await store.migrate()


@router.get("/health")
async def study_health():
    await store.ensure_migrated()
    return await store.healthcheck()


@router.post("/generate")
async def study_generate_endpoint(body: StudyRequestIn):
    await store.ensure_migrated()
    req = _to_request(body)
    chunks, mode = await study_focus.retrieve(req)
    if not chunks:
        raise HTTPException(status_code=404,
                            detail="No hay chunks para ese corpus. ¿Documento indexado?")
    questions = study_generate.generate(req, chunks, max_questions=body.n)
    return {"focus_mode": mode, "num_questions": len(questions),
            "questions": [_q_dict(q) for q in questions]}


@router.post("/session")
async def study_session_start(body: StudyRequestIn):
    await store.ensure_migrated()
    req = _to_request(body)
    res = await study_session.start_session(req, n=body.n)
    return {"session_id": res["session_id"], "focus_mode": res["focus_mode"],
            "num_questions": res["num_questions"],
            "questions": [_q_dict(q) for q in res["questions"]]}


@router.post("/grade")
async def study_grade_endpoint(body: AnswerIn):
    await store.ensure_migrated()
    try:
        gr = await study_session.submit_answer(body.session_id, body.q_id, body.answer)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except UngroundedError as e:
        raise HTTPException(status_code=422, detail=f"Regla dura: {e}")
    return _gr_dict(gr)


@router.get("/results/{session_id}")
async def study_results(session_id: str):
    await store.ensure_migrated()
    return {"results": await study_session.session_results(session_id)}


@router.post("/cite")
async def study_cite(chunk_ids: list[str]):
    """Resuelve chunk_ids -> citas trazables (doc_id, filename, página, cita)."""
    await store.ensure_migrated()
    try:
        cites = await resolve_evidence(chunk_ids)
    except UngroundedError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"citations": [asdict(c) for c in cites]}
