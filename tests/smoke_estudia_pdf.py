"""
Smoke test extremo a extremo de EstudIA con un PDF REAL del corpus (ruta OFFLINE).

Demuestra el flujo completo sobre contenido real, sin API key ni red:
  PDF real -> chunks -> extract (conceptos/grafo) -> sesión (pregunta anclada)
  -> calificación -> cita resoluble -> mapa conceptual.

Usa DB y grafo temporales: NO toca data/pentamodal.db. Fuerza la ruta offline.
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

from core.study import store, extract, session, structures, concept_graph, grounding, llm  # noqa: E402
from core.study.models import StudyRequest, Focus  # noqa: E402
from core import document_store  # noqa: E402
from config import DOCUMENTS_DIR  # noqa: E402


_CANDIDATES = [
    "Del Ser al Hacer by Humberto Maturana R..pdf",
    "El sentido de lo humano by Humberto Maturana Romesín.pdf",
    "Aldous Huxley - Las puertas de la percepción.pdf",
    "Emociones y Lenguaje en Educación y Política by Humberto Maturana.pdf",
]


def _ok(cond: bool, msg: str) -> bool:
    print(("  [OK]   " if cond else "  [FALLA] ") + msg)
    return cond


async def _ingest(doc_dir: str, max_chunks: int = 15):
    for fn in _CANDIDATES:
        path = os.path.join(doc_dir, fn)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                data = f.read()
            txt = await document_store.extract_text(fn, data)
        except Exception as e:
            print(f"  · {fn}: no extraíble ({e}); pruebo otro…")
            continue
        chunks = document_store._chunk_text(txt)[:max_chunks]
        if not chunks:
            continue
        engine = store.get_engine()
        async with engine.begin() as conn:
            r = await conn.execute(
                text("INSERT INTO documents (filename, file_size, chunk_count) VALUES (:f,:s,:n)"),
                {"f": fn, "s": len(data), "n": len(chunks)})
            did = str(r.lastrowid)
            for i, c in enumerate(chunks):
                await conn.execute(
                    text("INSERT INTO document_chunks (doc_id, chunk_index, content) VALUES (:d,:i,:c)"),
                    {"d": int(did), "i": i, "c": c})
        return did, fn, len(chunks)
    return None, None, 0


async def _run() -> bool:
    passed = True
    llm.provider_available = lambda: False  # fuerza ruta offline (sin red)

    tmp_db = os.path.join(tempfile.gettempdir(), "estudia_smoke.db")
    tmp_graph = os.path.join(tempfile.gettempdir(), "estudia_smoke_graph.json")
    for f in (tmp_db, tmp_graph):
        if os.path.exists(f):
            os.remove(f)
    store.use_db(tmp_db)
    concept_graph.use_path(tmp_graph)
    await store.migrate()

    doc_id, fn, n = await _ingest(str(DOCUMENTS_DIR))
    if not doc_id:
        print("  [SKIP] No se encontró un PDF de texto extraíble en documents/.")
        return True
    print(f"\n  Documento: {fn}  ({n} chunks)\n")
    passed &= _ok(n > 0, "PDF real ingerido a chunks")

    res = await extract.extract_corpus([doc_id])
    passed &= _ok(res["concepts"] >= 1, f"extracción de conceptos del PDF real ({res['concepts']})")

    started = await session.start_session(StudyRequest(corpus_ids=[doc_id], focus=Focus.ALL), n=3)
    passed &= _ok(started["num_questions"] >= 1, f"sesión genera preguntas ({started['num_questions']})")
    if started["questions"]:
        q = started["questions"][0]
        cites = await grounding.resolve_evidence(q.evidence)
        print(f"\n  ── Pregunta de muestra ({q.type.value}) ──")
        print(f"  {q.prompt}")
        print(f"  Respuesta esperada: {q.answer_key}")
        print(f"  {grounding.build_citation_block(cites)}\n")
        passed &= _ok(q.is_grounded() and cites[0].filename == fn,
                      "la pregunta cita el documento real")

        gr = await session.submit_answer(started["session_id"], q.q_id, q.answer_key)
        passed &= _ok(bool(gr.evidence), "calificación con evidencia (cierre del loop)")

    cm = await structures.concept_map(max_edges=8)
    print("  ── Mapa conceptual (primeras líneas Mermaid) ──")
    print("\n".join("  " + ln for ln in cm["mermaid"].splitlines()[:6]))
    passed &= _ok(cm["mermaid"].startswith("graph TD"), "mapa conceptual generado")

    for f in (tmp_db, tmp_graph):
        try:
            os.remove(f)
        except OSError:
            pass
    return passed


def main() -> bool:
    print("== SMOKE EstudIA extremo a extremo con PDF real (offline) ==")
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
