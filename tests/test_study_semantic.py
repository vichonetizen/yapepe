"""
Prueba de la recuperación SEMÁNTICA de EstudIA (pista integrada).

Con un embebedor falso determinista (no usa Ollama):
  1. rank_by_similarity ordena por similitud semántica.
  2. topic_chunks usa el ranking semántico cuando hay embeddings.
  3. El interruptor ESTUDIA_EMBED=off desactiva el semántico (cae a léxico).
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

from core.study import store, chunk_map, embed  # noqa: E402
from core.study.models import Chunk  # noqa: E402

_KW = ["autopoiesis", "cognicion", "celula", "mercado", "economia"]


def _fake_embed(texts):
    def vec(t):
        tl = t.lower()
        v = [float(tl.count(k)) for k in _KW]
        return v if any(v) else [0.001] * len(_KW)
    return [vec(t) for t in texts]


def _ok(cond, msg):
    print(("  [OK]   " if cond else "  [FALLA] ") + msg)
    return cond


async def _run():
    passed = True
    embed.set_embedder_for_tests(_fake_embed)
    os.environ["ESTUDIA_EMBED"] = "auto"

    chunks = [
        Chunk(chunk_id="1", doc_id="d", text="La autopoiesis define la célula viva."),
        Chunk(chunk_id="2", doc_id="d", text="El mercado y la economía financiera global."),
        Chunk(chunk_id="3", doc_id="d", text="La cognición y la autopoiesis en biología."),
    ]
    ranked = await embed.rank_by_similarity("autopoiesis celula", chunks, top_k=3)
    passed &= _ok(ranked is not None and ranked[0].chunk_id in ("1", "3"),
                  "rank_by_similarity pone un chunk de autopoiesis primero")
    passed &= _ok(ranked[-1].chunk_id == "2", "el chunk de mercado/economía queda último")
    passed &= _ok(await embed.available(), "embedder disponible (fake)")

    # topic_chunks vía DB usa el semántico
    tmp = os.path.join(tempfile.gettempdir(), "estudia_sem.db")
    if os.path.exists(tmp):
        os.remove(tmp)
    store.use_db(tmp)
    await store.migrate()
    engine = store.get_engine()
    async with engine.begin() as conn:
        r = await conn.execute(text("INSERT INTO documents (filename,file_size,chunk_count) VALUES ('d.pdf',0,3)"))
        did = r.lastrowid
        for i, c in enumerate(chunks):
            await conn.execute(text("INSERT INTO document_chunks (doc_id,chunk_index,content) VALUES (:d,:i,:c)"),
                               {"d": did, "i": i, "c": c.text})
    tc = await chunk_map.topic_chunks([str(did)], "autopoiesis celula", top_k=2)
    passed &= _ok(len(tc) == 2 and "mercado" not in tc[0].text.lower(),
                  "topic_chunks rankea por semántica (primero autopoiesis, no mercado)")

    # interruptor off -> None (cae a léxico en el llamador)
    os.environ["ESTUDIA_EMBED"] = "off"
    passed &= _ok((await embed.rank_by_similarity("x", chunks)) is None,
                  "ESTUDIA_EMBED=off desactiva el ranking semántico")
    os.environ["ESTUDIA_EMBED"] = "auto"

    try:
        os.remove(tmp)
    except OSError:
        pass
    return passed


def main():
    print("== Recuperación semántica EstudIA (integrada) ==")
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
