"""
grade.py — Calificación ANCLADA (pista integrada).

MCQ/V-F/cloze deterministas (sin LLM). OPEN con LLM inyectable o, en su defecto,
similitud léxica contra answer_key. Toda GradeResult hereda la evidencia de la
Question anclada: ningún feedback sale sin cita.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Optional

from .models import Question, QuestionType, GradeResult
from .grounding import assert_grounded


_TRUE = {"verdadero", "v", "true", "t", "si", "sí", "cierto", "1"}
_FALSE = {"falso", "f", "false", "no", "incorrecto", "0"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _content_words(s: str) -> set[str]:
    return {w for w in _norm(s).split() if len(w) > 3}


def _jaccard(a: str, b: str) -> float:
    wa, wb = _content_words(a), _content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _verdict(score: float) -> str:
    if score >= 0.85:
        return "Correcto"
    if score >= 0.5:
        return "Parcialmente correcto"
    return "Incorrecto"


def grade(question: Question,
          user_answer: str,
          llm: Optional[Callable[[Question, str], dict]] = None) -> GradeResult:
    ua = _norm(user_answer)
    key = _norm(question.answer_key)
    score = 0.0
    feedback = ""

    if question.type == QuestionType.TF:
        def to_bool(x: str) -> Optional[bool]:
            if x in _TRUE:
                return True
            if x in _FALSE:
                return False
            return None
        ub, kb = to_bool(ua), to_bool(key)
        score = 1.0 if (ub is not None and ub == kb) else 0.0
        feedback = ("Correcto." if score else
                    f"La respuesta esperada era «{question.answer_key}». {question.rationale}")

    elif question.type == QuestionType.MCQ:
        correct = ua == key
        if not correct and question.options:
            for i, opt in enumerate(question.options):
                letter = chr(ord("a") + i)
                if _norm(opt) == key and (ua == _norm(opt) or ua == letter):
                    correct = True
                    break
        score = 1.0 if correct else 0.0
        feedback = ("Correcto." if score else
                    f"La opción correcta era «{question.answer_key}». {question.rationale}")

    elif question.type == QuestionType.CLOZE:
        score = 1.0 if (ua == key or (key and key in ua)) else 0.0
        feedback = ("Correcto." if score else
                    f"El término esperado era «{question.answer_key}». {question.rationale}")

    else:  # OPEN
        if llm is not None:
            try:
                res = llm(question, user_answer) or {}
                score = max(0.0, min(1.0, float(res.get("score", 0.0))))
                feedback = str(res.get("feedback", "")) or question.rationale
            except Exception:
                llm = None
        if llm is None:
            score = round(_jaccard(user_answer, question.answer_key), 2)
            feedback = (f"Tu respuesta cubre ~{int(score * 100)}% de los puntos clave "
                        f"esperados. Referencia: {question.rationale}")

    gaps = list(question.concept_ids) if score < 0.6 else []
    result = GradeResult(
        q_id=question.q_id, user_answer=user_answer, score=score,
        verdict=_verdict(score), feedback=feedback,
        evidence=list(question.evidence), gaps=gaps,
    )
    assert_grounded(result)
    return result
