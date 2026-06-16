"""
Prueba de la Ola 3 de EstudIA (pista integrada) — estructuras pedagógicas.

Contra DB y grafo temporales, con DOS documentos:
  1. comparison(a, b): cada dimensión con evidencia de AMBOS conceptos.
  2. debate(topic): posiciones de >=2 documentos, cada una con cita.
  3. hierarchy(root): Hierarchy con niveles desde la raíz.
  4. concept_map(): Mermaid + aristas con evidencia.
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

from core.study import store, extract, structures, concept_graph  # noqa: E402


def _ok(cond: bool, msg: str) -> bool:
    print(("  [OK]   " if cond else "  [FALLA] ") + msg)
    return cond


_DOCS = {
    "Maturana - El árbol del conocimiento.pdf": [
        "La autopoiesis es la organización de lo vivo según Maturana.",
        "En la biología de Maturana la cognición es un fenómeno del vivir.",
    ],
    "Varela - De cuerpo presente.pdf": [
        "La cognición enactiva según Varela surge de la corporalidad.",
        "El concepto de autopoiesis es central en la obra de Varela.",
    ],
}


async def _run() -> bool:
    passed = True
    tmp_db = os.path.join(tempfile.gettempdir(), "estudia_test_estruct.db")
    tmp_graph = os.path.join(tempfile.gettempdir(), "estudia_test_estruct_graph.json")
    for f in (tmp_db, tmp_graph):
        if os.path.exists(f):
            os.remove(f)
    store.use_db(tmp_db)
    concept_graph.use_path(tmp_graph)
    await store.migrate()

    engine = store.get_engine()
    doc_ids = []
    async with engine.begin() as conn:
        for fn, chunks in _DOCS.items():
            r = await conn.execute(
                text("INSERT INTO documents (filename, file_size, chunk_count) VALUES (:f,0,:n)"),
                {"f": fn, "n": len(chunks)})
            did = str(r.lastrowid)
            doc_ids.append(did)
            for i, c in enumerate(chunks):
                await conn.execute(
                    text("INSERT INTO document_chunks (doc_id, chunk_index, content, page, section_path) "
                         "VALUES (:d,:i,:c,:p,'[]')"),
                    {"d": int(did), "i": i, "c": c, "p": i + 1})

    await extract.extract_corpus(doc_ids)

    # 1. comparison ------------------------------------------------------------
    comp = await structures.comparison("Maturana", "Varela")
    passed &= _ok(comp.is_grounded(), "comparison: toda dimensión con evidencia")
    passed &= _ok(len(comp.dimensions[0]["evidence"]) >= 2,
                  "comparison: la dimensión cita evidencia de ambos conceptos")

    # 2. debate (cross-document) ----------------------------------------------
    deb = await structures.debate("autopoiesis", doc_ids)
    passed &= _ok(len(deb.positions) >= 2, f"debate abarca >=2 documentos ({len(deb.positions)})")
    passed &= _ok(all(p["evidence"] for p in deb.positions), "cada posición lleva evidencia")
    passed &= _ok(len({p["author"] for p in deb.positions}) >= 2, "posiciones de autores/documentos distintos")

    # 3. hierarchy -------------------------------------------------------------
    g = concept_graph.ConceptGraph.load(tmp_graph)
    root = max(g.nodes, key=lambda k: g.nodes[k]["freq"])
    h = await structures.hierarchy(g.label_of(root))
    passed &= _ok(len(h.levels) >= 1 and h.levels[0] == [root], "hierarchy parte de la raíz")

    # 4. concept_map -----------------------------------------------------------
    cm = await structures.concept_map()
    passed &= _ok(cm["mermaid"].startswith("graph TD") and len(cm["edges"]) >= 1,
                  "concept_map exporta Mermaid con aristas (con evidencia)")
    passed &= _ok(all(e["evidence"] for e in cm["edges"]), "toda arista del mapa tiene evidencia")

    for f in (tmp_db, tmp_graph):
        try:
            os.remove(f)
        except OSError:
            pass
    return passed


def main() -> bool:
    print("== Estructuras EstudIA (integrada): Ola 3 ==")
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
