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

from . import (store, session as study_session, focus as study_focus,
               generate as study_generate, extract as study_extract,
               structures as study_structures, ingest_bridge as study_ingest,
               fsrs as study_fsrs, plan as study_plan, panel as study_panel,
               exam as study_exam)
from .models import StudyRequest, Focus, Question, GradeResult
from .grounding import UngroundedError, resolve_evidence
from .concept_graph import ConceptGraph

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


class IngestIn(BaseModel):
    filename: str   # nombre dentro de documents/ o ruta absoluta


class ExtractIn(BaseModel):
    corpus_ids: list[str]


class CompareIn(BaseModel):
    concept_a: str
    concept_b: str


class DebateIn(BaseModel):
    topic: str
    corpus_ids: list[str]


class EnsureCardsIn(BaseModel):
    corpus_ids: list[str]


class ReviewIn(BaseModel):
    card_id: str
    score: float   # 0..1 (de un GradeResult)


class ExamStartIn(BaseModel):
    corpus_ids: list[str]
    coverage_target: float = 0.85
    n: int = 10
    time_limit: Optional[int] = None


class ExamSubmitIn(BaseModel):
    exam_id: str
    answers: dict[str, str]


class PanelIn(BaseModel):
    corpus_ids: list[str]


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
    questions = await study_generate.agenerate(req, chunks, max_questions=body.n)
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


# --------------------------------------------------------------------------- #
# Ola 1 — Comprensión: conceptos y grafo
# --------------------------------------------------------------------------- #
@router.post("/ingest")
async def study_ingest_endpoint(body: IngestIn):
    """Ingiere un documento del corpus con número de página por chunk (PDF)."""
    await store.ensure_migrated()
    try:
        return await study_ingest.ingest_path(body.filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/extract")
async def study_extract_endpoint(body: ExtractIn):
    """Extrae conceptos/aristas anclados del corpus y construye el grafo."""
    await store.ensure_migrated()
    return await study_extract.extract_corpus(body.corpus_ids)


@router.get("/conceptmap")
async def study_conceptmap(max_edges: int = 60):
    """Mapa conceptual (Mermaid) fiel al grafo de conceptos."""
    await store.ensure_migrated()
    g = ConceptGraph.load()
    return {"render": "mermaid", "mermaid": g.to_mermaid(max_edges),
            "num_nodes": len(g.nodes), "num_edges": len(g.edges)}


@router.get("/hierarchy")
async def study_hierarchy(root: str, max_levels: int = 5):
    """Jerarquía BFS de conceptos desde un concepto raíz."""
    await store.ensure_migrated()
    g = ConceptGraph.load()
    levels = g.bfs_levels(root, max_levels=max_levels)
    return {"root": root, "num_levels": len(levels),
            "levels": [[g.label_of(k) for k in lvl] for lvl in levels]}


@router.post("/compare")
async def study_compare(body: CompareIn):
    """Comparativa de 2 conceptos con evidencia de ambas fuentes."""
    await store.ensure_migrated()
    try:
        comp = await study_structures.comparison(body.concept_a, body.concept_b)
    except UngroundedError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return asdict(comp)


@router.post("/debate")
async def study_debate(body: DebateIn):
    """Debate cross-document: posiciones por documento sobre un tema, con cita."""
    await store.ensure_migrated()
    try:
        deb = await study_structures.debate(body.topic, body.corpus_ids)
    except UngroundedError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return asdict(deb)


# --------------------------------------------------------------------------- #
# Ola 4 — Memoria: FSRS, plan, panel, examen
# --------------------------------------------------------------------------- #
@router.post("/cards/ensure")
async def study_cards_ensure(body: EnsureCardsIn):
    """Crea una card por concepto del corpus (para repaso espaciado)."""
    await store.ensure_migrated()
    created = await study_fsrs.ensure_cards_for_corpus(body.corpus_ids)
    return {"created": created}


@router.post("/review")
async def study_review(body: ReviewIn):
    """Registra un repaso de card y aplica FSRS (due/interval/ease/lapses)."""
    await store.ensure_migrated()
    card = await study_fsrs.get_card(body.card_id)
    if card is None:
        raise HTTPException(status_code=404, detail=f"Card desconocida: {body.card_id}")
    study_fsrs.review(card, body.score)
    await study_fsrs.upsert_card(card)
    d = asdict(card)
    d["mastered"] = study_fsrs.is_mastered(card)
    return d


@router.get("/plan")
async def study_plan_endpoint():
    """Plan de aprendizaje: cards vencidas hoy + lo no dominado."""
    await store.ensure_migrated()
    return await study_plan.build_plan()


@router.post("/panel")
async def study_panel_endpoint(body: PanelIn):
    """Panel de progreso: % dominio, repasos pendientes, 'listo para prueba'."""
    await store.ensure_migrated()
    return await study_panel.progress(body.corpus_ids)


@router.post("/exam/start")
async def study_exam_start(body: ExamStartIn):
    """Inicia un examen que cubre >=coverage_target de los conceptos clave."""
    await store.ensure_migrated()
    res = await study_exam.start_exam(body.corpus_ids, body.coverage_target,
                                      body.n, body.time_limit)
    return {**{k: v for k, v in res.items() if k != "questions"},
            "questions": [_q_dict(q) for q in res["questions"]]}


@router.post("/exam/submit")
async def study_exam_submit(body: ExamSubmitIn):
    """Califica un examen (determinista, read-only) y devuelve el informe."""
    await store.ensure_migrated()
    try:
        res = await study_exam.submit_exam(body.exam_id, body.answers)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {**{k: v for k, v in res.items() if k != "results"},
            "results": [_gr_dict(g) for g in res["results"]]}
