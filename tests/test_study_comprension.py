"""
Prueba de la Ola 1 de EstudIA (pista integrada) — comprensión: conceptos y grafo.

Contra DB y grafo temporales (no toca producción):
  1. extract_corpus() produce conceptos y aristas desde los chunks.
  2. REGLA DURA: todo study_concepts.source_chunks NO vacío; todo study_edges.evidence NO vacío.
  3. El grafo persiste, expone Mermaid y BFS por niveles.
  4. chunks_for_concepts() + get_chunks_by_ids() resuelven concepto -> chunks.
  5. focus(MAP/HIERARCHY) con key_concepts selecciona vía grafo.
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

from core.study import store, extract, chunk_map, focus, concept_graph  # noqa: E402
from core.study.concept_graph import ConceptGraph  # noqa: E402
from core.study.models import StudyRequest, Focus  # noqa: E402


def _ok(cond: bool, msg: str) -> bool:
    print(("  [OK]   " if cond else "  [FALLA] ") + msg)
    return cond


_CHUNKS = [
    "La autopoiesis es la organización de lo vivo según Maturana y Varela.",
    "Maturana sostiene que la autopoiesis define al sistema vivo como red cerrada.",
    "El observador y la cognición aparecen en la biología de Maturana y Varela.",
    "La cognición es un fenómeno biológico ligado a la organización autopoiética.",
]


async def _run() -> bool:
    passed = True
    tmp_db = os.path.join(tempfile.gettempdir(), "estudia_test_comp.db")
    tmp_graph = os.path.join(tempfile.gettempdir(), "estudia_test_graph.json")
    for f in (tmp_db, tmp_graph):
        if os.path.exists(f):
            os.remove(f)
    store.use_db(tmp_db)
    concept_graph.use_path(tmp_graph)
    await store.migrate()

    engine = store.get_engine()
    async with engine.begin() as conn:
        r = await conn.execute(
            text("INSERT INTO documents (filename, file_size, chunk_count) VALUES (:f,0,:n)"),
            {"f": "Maturana - El árbol del conocimiento.pdf", "n": len(_CHUNKS)})
        doc_id = str(r.lastrowid)
        for i, c in enumerate(_CHUNKS):
            await conn.execute(
                text("INSERT INTO document_chunks (doc_id, chunk_index, content, page, section_path) "
                     "VALUES (:d,:i,:c,:p,:sp)"),
                {"d": int(doc_id), "i": i, "c": c, "p": 30 + i, "sp": '["Cap. 1"]'})

    # 1. extracción ------------------------------------------------------------
    res = await extract.extract_corpus([doc_id])
    passed &= _ok(res["concepts"] >= 3, f"extract produce conceptos ({res['concepts']})")
    passed &= _ok(res["edges"] >= 1, f"extract produce aristas de co-ocurrencia ({res['edges']})")

    # 2. regla dura en la persistencia ----------------------------------------
    async with engine.connect() as conn:
        bad_c = (await conn.execute(text(
            "SELECT COUNT(*) FROM study_concepts WHERE source_chunks IS NULL "
            "OR source_chunks = '[]' OR source_chunks = ''"))).scalar()
        bad_e = (await conn.execute(text(
            "SELECT COUNT(*) FROM study_edges WHERE evidence IS NULL OR evidence = ''"))).scalar()
    passed &= _ok(bad_c == 0, "0 conceptos sin source_chunks (regla dura)")
    passed &= _ok(bad_e == 0, "0 aristas sin evidence (regla dura)")

    # 3. grafo: persistencia, mermaid, BFS ------------------------------------
    g = ConceptGraph.load(tmp_graph)
    passed &= _ok(len(g.nodes) == res["concepts"], "el grafo persiste y recarga los nodos")
    mer = g.to_mermaid()
    passed &= _ok(mer.startswith("graph TD") and "-->" in mer, "exporta Mermaid con aristas")
    root_key = max(g.nodes, key=lambda k: g.nodes[k]["freq"])
    levels = g.bfs_levels(g.label_of(root_key))
    passed &= _ok(len(levels) >= 1 and levels[0] == [root_key], "bfs_levels parte de la raíz")

    # 4. concepto -> chunks ----------------------------------------------------
    chunk_ids = await extract.chunks_for_concepts([g.label_of(root_key)])
    passed &= _ok(len(chunk_ids) >= 1, "chunks_for_concepts resuelve chunks del concepto")
    chs = await chunk_map.get_chunks_by_ids(chunk_ids)
    passed &= _ok(len(chs) == len(chunk_ids) and all(c.page is not None for c in chs),
                  "get_chunks_by_ids recupera los chunks con página")

    # 5. focus vía grafo -------------------------------------------------------
    req_map = StudyRequest(corpus_ids=[doc_id], focus=Focus.MAP,
                           key_concepts=[g.label_of(root_key)])
    sel, mode = await focus.retrieve(req_map)
    passed &= _ok(len(sel) >= 1 and "grafo" in mode, f"focus MAP usa el grafo (modo: {mode})")

    for f in (tmp_db, tmp_graph):
        try:
            os.remove(f)
        except OSError:
            pass
    return passed


def main() -> bool:
    print("== Comprensión EstudIA (integrada): Ola 1 (conceptos + grafo) ==")
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
