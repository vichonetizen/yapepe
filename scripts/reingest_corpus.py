"""
reingest_corpus.py — Re-ingiere TODO el corpus de documents/ con número de página y
OCR (escaneados), reemplazando los uploads antiguos sin páginas.

Por cada PDF: borra filas previas con ese filename (evita duplicados) → ingiere con
ingest_bridge (páginas nativas + OCR donde haga falta) → al final extrae conceptos
del corpus completo y crea las cards FSRS. Opera sobre la DB real (data/pentamodal.db).

Pensado para correr EN SEGUNDO PLANO con la app detenida (evita contención SQLite).
Imprime progreso línea a línea (flush) para monitorizar.
"""

import asyncio
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")  # consola Windows cp1252 -> utf-8
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402

from core.study import store, ingest_bridge, extract, fsrs  # noqa: E402
from config import DOCUMENTS_DIR  # noqa: E402


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(str(msg).encode("ascii", "replace").decode("ascii"), flush=True)


async def _delete_existing(filename):
    engine = store.get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(
            "DELETE FROM document_chunks WHERE doc_id IN (SELECT id FROM documents WHERE filename=:f)"),
            {"f": filename})
        await conn.execute(text("DELETE FROM documents WHERE filename=:f"), {"f": filename})


async def _existing_chunks(filename):
    engine = store.get_engine()
    async with engine.connect() as conn:
        n = (await conn.execute(text(
            "SELECT COALESCE(SUM(chunk_count),0) FROM documents WHERE filename=:f"),
            {"f": filename})).scalar() or 0
    return int(n)


async def _doc_id(filename):
    engine = store.get_engine()
    async with engine.connect() as conn:
        r = (await conn.execute(text(
            "SELECT id FROM documents WHERE filename=:f ORDER BY id DESC LIMIT 1"),
            {"f": filename})).scalar()
    return str(r) if r else None


async def main():
    await store.migrate()
    log(f"OCR backend: {ingest_bridge.ocr_backend()}")
    pdfs = sorted(f for f in os.listdir(str(DOCUMENTS_DIR)) if f.lower().endswith(".pdf"))
    log(f"== Re-ingesta de {len(pdfs)} PDFs (con páginas/OCR) ==")

    new_ids, t0 = [], time.time()
    for i, fn in enumerate(pdfs, 1):
        st = time.time()
        try:
            # Reanudable: si ya está ingerido con contenido real (>1 chunk), saltar.
            if await _existing_chunks(fn) > 1:
                did = await _doc_id(fn)
                if did:
                    new_ids.append(did)
                log(f"[{i}/{len(pdfs)}] = {fn[:48]} (ya ingerido, salto)")
                continue
            await _delete_existing(fn)
            res = await ingest_bridge.ingest_path(fn)
            dt = time.time() - st
            if res.get("doc_id"):
                new_ids.append(res["doc_id"])
                log(f"[{i}/{len(pdfs)}] ✓ {fn[:48]} → {res['chunks']} frag · "
                    f"{res.get('pages_with_text')} pág.texto · {res.get('pages_needing_ocr',0)} OCR · {dt:.0f}s")
            else:
                log(f"[{i}/{len(pdfs)}] ⚠ {fn[:48]} → sin texto ({res.get('note','')}) · {dt:.0f}s")
        except Exception as e:
            log(f"[{i}/{len(pdfs)}] ✗ {fn[:48]} → {e}")

    log(f"-- Ingesta terminada en {(time.time()-t0)/60:.1f} min · {len(new_ids)} docs con contenido --")
    if new_ids:
        log("Extrayendo conceptos del corpus completo…")
        r = await extract.extract_corpus(new_ids)
        log(f"  conceptos={r['concepts']} aristas={r['edges']} chunks={r['chunks']}")
        created = await fsrs.ensure_cards_for_corpus(new_ids)
        log(f"  cards creadas: {created}")
    log("== LISTO ==")


if __name__ == "__main__":
    asyncio.run(main())
