"""
Smoke de EstudIA con Ollama LOCAL (sin API key) — prueba la ruta LLM real.

Requiere Ollama corriendo con algún modelo de chat. Verifica:
  1. llm detecta y registra Ollama como proveedor.
  2. Conectividad real: el modelo local responde.
  3. generate.agenerate() produce preguntas ANCLADAS (vía Ollama; si el modelo
     devuelve algo no usable, cae a offline — el gate de evidencia se respeta igual).
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

from core.study import store, extract, generate, concept_graph, chunk_map, llm, ingest_bridge  # noqa: E402
from core.study.models import StudyRequest, Focus  # noqa: E402
from config import DOCUMENTS_DIR  # noqa: E402

_CANDIDATES = [
    "Educacion_en_tiempos_de_Pandemia.pdf",
    "ARTICULO_RETOS_EN_LA_EDUCACION_EN_TIEMPO.pdf",
    "La_pandemia_virtualidad_en_la_educacion.pdf",
    "Pensar-la-educacion.pdf",
]


def _ok(cond: bool, msg: str) -> bool:
    print(("  [OK]   " if cond else "  [FALLA] ") + msg)
    return cond


async def _run() -> bool:
    passed = True
    prov = llm.current_provider()
    print(f"  Proveedor detectado: {prov}")
    if not prov:
        print("  [SKIP] No hay proveedor LLM (Ollama no corre). Nada que probar aquí.")
        return True
    passed &= _ok(prov[0] == "ollama", f"Ollama registrado como proveedor (modelo {prov[1]})")

    print("  Probando conectividad con el modelo local (puede tardar la 1ª vez)…")
    txt = await llm._complete("Responde solo con la palabra OK.", "Di OK.", complexity="simple")
    print(f"  Respuesta del modelo: {txt[:80]!r}")
    passed &= _ok(bool(txt.strip()) and not txt.startswith("⚠️"), "el modelo local responde (conectividad real)")

    # Ingesta de un PDF de texto + generación real
    tmp_db = os.path.join(tempfile.gettempdir(), "estudia_ollama.db")
    tmp_graph = os.path.join(tempfile.gettempdir(), "estudia_ollama_graph.json")
    for f in (tmp_db, tmp_graph):
        if os.path.exists(f):
            os.remove(f)
    store.use_db(tmp_db)
    concept_graph.use_path(tmp_graph)
    await store.migrate()

    doc_id = None
    for fn in _CANDIDATES:
        if os.path.exists(os.path.join(str(DOCUMENTS_DIR), fn)):
            res = await ingest_bridge.ingest_path(fn)
            if res.get("doc_id"):
                doc_id = res["doc_id"]
                print(f"  Documento: {fn} (chunks={res['chunks']})")
                break
    if not doc_id:
        print("  [SKIP] Sin PDF de texto para generar.")
        return passed

    await extract.extract_corpus([doc_id])
    chunks = (await chunk_map.get_chunks_for_docs([doc_id]))[:4]
    print("  Generando preguntas con el modelo local…")
    qs = await generate.agenerate(StudyRequest(corpus_ids=[doc_id], focus=Focus.ALL),
                                  chunks, max_questions=3)
    for q in qs[:3]:
        print(f"\n  · ({q.type.value}) {q.prompt[:160]}")
        print(f"    clave: {q.answer_key[:80]}  | evidencia: {q.evidence}")
    allowed = {c.chunk_id for c in chunks}
    passed &= _ok(len(qs) >= 1 and all(q.is_grounded() for q in qs)
                  and all(all(e in allowed for e in q.evidence) for q in qs),
                  "agenerate produce preguntas ancladas dentro del corpus")

    for f in (tmp_db, tmp_graph):
        try:
            os.remove(f)
        except OSError:
            pass
    return passed


def main() -> bool:
    print("== SMOKE EstudIA con Ollama local (sin API key) ==")
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
