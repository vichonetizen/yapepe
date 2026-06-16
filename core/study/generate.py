"""
generate.py — Generación de preguntas ANCLADAS (pista integrada).

Única puerta de emisión: aplica la regla dura (descarta preguntas sin evidencia o
que citen chunks fuera del corpus recuperado). Con LLM inyectable genera preguntas
ricas; sin LLM degrada a plantillas cloze/V-F derivadas del chunk (evidencia = ese
chunk), para funcionar offline. El cableado de ai_client se hará en una ola posterior.
"""

from __future__ import annotations

import re
import uuid
from typing import Callable, Optional

from .models import Question, QuestionType, StudyRequest, Chunk
from .grounding import is_grounded


_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a",
    "y", "o", "u", "e", "que", "en", "con", "por", "para", "es", "son", "se", "su",
    "sus", "lo", "como", "más", "pero", "sin", "sobre", "entre", "este", "esta",
    "esto", "ese", "esa", "eso", "the", "of", "and", "to", "in", "is", "no", "ni",
}


def _words(text: str) -> list[str]:
    return re.findall(r"\b[\wáéíóúüñÁÉÍÓÚÜÑ]+\b", text)


def estimate_difficulty(text: str) -> int:
    n = len(_words(text))
    sentences = max(1, len(re.findall(r"[.!?]", text)))
    score = 1 + min(4, n // 18) + (1 if sentences >= 3 else 0)
    return max(1, min(5, score))


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.split()) >= 4]


def _salient_term(text: str) -> Optional[str]:
    caps = [w for w in _words(text) if w[:1].isupper() and w.lower() not in _STOPWORDS and len(w) > 3]
    if caps:
        return max(caps, key=len)
    content = [w for w in _words(text) if w.lower() not in _STOPWORDS and len(w) > 4]
    return max(content, key=len) if content else None


def _fallback_questions(chunk: Chunk) -> list[Question]:
    qs: list[Question] = []
    sents = _sentences(chunk.text)
    if not sents:
        return qs
    base = sents[0]
    term = _salient_term(base)
    if term:
        blanked = re.sub(rf"\b{re.escape(term)}\b", "_____", base, count=1)
        if "_____" in blanked:
            qs.append(Question(
                q_id=str(uuid.uuid4()), type=QuestionType.CLOZE,
                prompt=f"Completa: {blanked}", answer_key=term,
                rationale="Término extraído directamente del fragmento fuente.",
                evidence=[chunk.chunk_id], concept_ids=list(chunk.concepts),
                difficulty=estimate_difficulty(base),
            ))
    qs.append(Question(
        q_id=str(uuid.uuid4()), type=QuestionType.TF,
        prompt=f"¿Verdadero o falso? {base}", answer_key="verdadero",
        rationale="La afirmación aparece literalmente en el fragmento fuente.",
        evidence=[chunk.chunk_id], concept_ids=list(chunk.concepts),
        difficulty=estimate_difficulty(base),
    ))
    return qs


def _coerce_question(d: dict) -> Optional[Question]:
    try:
        return Question(
            q_id=str(d.get("q_id") or uuid.uuid4()),
            type=QuestionType(d.get("type", "open")),
            prompt=str(d["prompt"]),
            answer_key=str(d.get("answer_key", "")),
            rationale=str(d.get("rationale", "")),
            evidence=[str(e) for e in (d.get("evidence") or [])],
            concept_ids=[str(c) for c in (d.get("concept_ids") or [])],
            options=d.get("options"),
            difficulty=int(d.get("difficulty", 3)),
        )
    except (KeyError, ValueError, TypeError):
        return None


def generate(req: StudyRequest,
             chunks: list[Chunk],
             llm: Optional[Callable[[StudyRequest, list[Chunk]], list[dict]]] = None,
             max_questions: int = 10) -> list[Question]:
    """Devuelve SOLO preguntas ancladas. Las no ancladas se descartan en silencio."""
    allowed = {c.chunk_id for c in chunks}
    candidates: list[Question] = []
    if llm is not None:
        try:
            raw = llm(req, chunks) or []
        except Exception:
            raw = []
        for d in raw:
            q = _coerce_question(d)
            if q is not None:
                candidates.append(q)
    else:
        for ch in chunks:
            candidates.extend(_fallback_questions(ch))

    emitted: list[Question] = []
    for q in candidates:
        if not is_grounded(q):
            continue
        if any(str(e) not in allowed for e in q.evidence):
            continue
        emitted.append(q)
        if len(emitted) >= max_questions:
            break
    return emitted


async def agenerate(req: StudyRequest, chunks: list[Chunk],
                    max_questions: int = 10) -> list[Question]:
    """Genera con LLM (ai_client) si hay proveedor; si no, cae a la ruta offline.
    El gate de la regla dura se aplica igual en ambos caminos."""
    from . import llm  # import diferido: evita coste si no se usa
    raw = await llm.agenerate_questions(req, chunks, n=max_questions)
    allowed = {c.chunk_id for c in chunks}
    emitted: list[Question] = []
    for d in raw:
        q = _coerce_question(d)
        if q and is_grounded(q) and all(str(e) in allowed for e in q.evidence):
            emitted.append(q)
            if len(emitted) >= max_questions:
                break
    if emitted:
        return emitted
    return generate(req, chunks, llm=None, max_questions=max_questions)
