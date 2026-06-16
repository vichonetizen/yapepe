"""
Prueba del cableado de LLM de EstudIA (pista integrada) — Ola "ai_client".

Sin proveedor real (no se necesita API key):
  1. parse_questions_json descarta evidencia inventada o vacía (anti-alucinación).
  2. build_gen_prompt incrusta los chunk ids permitidos.
  3. agenerate() cae a la ruta offline cuando no hay proveedor (degradación elegante).
  4. agrade() determinista (V/F) y agrade() OPEN offline heredan la evidencia.
"""

import asyncio
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.study import generate, grade, llm  # noqa: E402
from core.study.models import StudyRequest, Question, QuestionType, Chunk  # noqa: E402


def _ok(cond: bool, msg: str) -> bool:
    print(("  [OK]   " if cond else "  [FALLA] ") + msg)
    return cond


async def _run() -> bool:
    passed = True
    chunks = [
        Chunk(chunk_id="1", doc_id="d1", page=1,
              text="La autopoiesis es la organización de lo vivo según Maturana."),
        Chunk(chunk_id="2", doc_id="d1", page=2,
              text="El conocer es efectivo en el dominio de la acción del observador."),
    ]
    allowed = {"1", "2"}

    # 1. parser anti-alucinación ----------------------------------------------
    raw = ('[{"type":"tf","prompt":"p","answer_key":"verdadero","evidence":["1"]},'
           '{"type":"open","prompt":"q","answer_key":"x","evidence":["999"]},'
           '{"type":"open","prompt":"r","answer_key":"y","evidence":[]}]')
    parsed = llm.parse_questions_json(raw, allowed)
    passed &= _ok(len(parsed) == 1 and parsed[0]["evidence"] == ["1"],
                  "parse_questions_json descarta evidencia inventada/vacía")

    # 2. prompt con ids --------------------------------------------------------
    _system, user = llm.build_gen_prompt(StudyRequest(corpus_ids=["d1"]), chunks)
    passed &= _ok("[1]" in user and "[2]" in user, "el prompt incrusta los chunk ids")

    # 3. agenerate cae a offline sin proveedor --------------------------------
    orig = llm.provider_available
    llm.provider_available = lambda: False
    try:
        qs = await generate.agenerate(StudyRequest(corpus_ids=["d1"]), chunks, max_questions=4)
    finally:
        llm.provider_available = orig
    passed &= _ok(len(qs) >= 1 and all(q.is_grounded() for q in qs),
                  "agenerate() degrada a offline (preguntas ancladas)")

    # 4. agrade determinista + OPEN offline -----------------------------------
    tf = Question(q_id="t", type=QuestionType.TF, prompt="¿V o F?",
                  answer_key="verdadero", rationale="r", evidence=["1"])
    gr = await grade.agrade(tf, "verdadero")
    passed &= _ok(gr.score == 1.0 and gr.evidence == ["1"], "agrade V/F determinista hereda evidencia")

    llm.provider_available = lambda: False
    try:
        op = Question(q_id="o", type=QuestionType.OPEN, prompt="¿Qué es la autopoiesis?",
                      answer_key="la organización de lo vivo", rationale="r",
                      evidence=["1"], concept_ids=["autopoiesis"])
        gro = await grade.agrade(op, "la organización de lo vivo")
    finally:
        llm.provider_available = orig
    passed &= _ok(gro.evidence == ["1"], "agrade OPEN offline hereda evidencia")

    return passed


def main() -> bool:
    print("== Cableado LLM EstudIA (integrada): offline-safe ==")
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
