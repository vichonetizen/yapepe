"""
Prueba de la ingesta con número de página (pista integrada).

Contra DB temporal, sobre PDFs REALES del corpus:
  1. ingest_path() de un PDF con texto puebla `page` en los chunks (cita con página).
  2. La cita resuelta trae número de página real (grounding.resolve_evidence).
  3. Un PDF sin capa de texto reporta pages_needing_ocr > 0 (hook OCR honesto).

Si no hay ningún PDF de texto en el corpus, marca SKIP (no falla).
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

from core.study import store, ingest_bridge, grounding  # noqa: E402
from config import DOCUMENTS_DIR  # noqa: E402


_TEXT_CANDIDATES = [
    "Educacion_en_tiempos_de_Pandemia.pdf",
    "ARTICULO_RETOS_EN_LA_EDUCACION_EN_TIEMPO.pdf",
    "La_pandemia_virtualidad_en_la_educacion.pdf",
    "Pensar-la-educacion.pdf",
    "Educacion_universitaria_y_cuerpos_en_pan.pdf",
    "el saber didactico Camilioni.pdf",
    "Emociones y Lenguaje en Educación y Política by Humberto Maturana.pdf",
]
_SCANNED_CANDIDATE = "Del Ser al Hacer by Humberto Maturana R..pdf"


def _ok(cond: bool, msg: str) -> bool:
    print(("  [OK]   " if cond else "  [FALLA] ") + msg)
    return cond


async def _run() -> bool:
    passed = True
    tmp = os.path.join(tempfile.gettempdir(), "estudia_test_ingest.db")
    if os.path.exists(tmp):
        os.remove(tmp)
    store.use_db(tmp)
    await store.migrate()

    # 1+2. PDF con texto -> páginas pobladas + cita con página -----------------
    chosen = None
    for fn in _TEXT_CANDIDATES:
        p = os.path.join(str(DOCUMENTS_DIR), fn)
        if not os.path.exists(p):
            continue
        res = await ingest_bridge.ingest_path(fn)
        if res.get("doc_id") and (res.get("pages_with_text") or 0) >= 2:
            chosen = res
            break

    if chosen is None:
        print("  [SKIP] No se encontró un PDF de texto multipágina en documents/.")
    else:
        print(f"  Documento: {chosen['filename']}  "
              f"(páginas={chosen['pages']}, chunks={chosen['chunks']}, "
              f"con_texto={chosen['pages_with_text']}, ocr_pend={chosen['pages_needing_ocr']})")
        engine = store.get_engine()
        async with engine.connect() as conn:
            total = (await conn.execute(text(
                "SELECT COUNT(*) FROM document_chunks WHERE doc_id = :d"),
                {"d": int(chosen["doc_id"])})).scalar()
            with_page = (await conn.execute(text(
                "SELECT COUNT(*) FROM document_chunks WHERE doc_id = :d AND page IS NOT NULL"),
                {"d": int(chosen["doc_id"])})).scalar()
            sample = (await conn.execute(text(
                "SELECT id FROM document_chunks WHERE doc_id = :d AND page IS NOT NULL LIMIT 1"),
                {"d": int(chosen["doc_id"])})).scalar()
        passed &= _ok(total > 0 and with_page == total,
                      f"todos los chunks ({with_page}/{total}) traen número de página")
        cites = await grounding.resolve_evidence([str(sample)])
        passed &= _ok(cites[0].page is not None and cites[0].filename == chosen["filename"],
                      f"la cita resuelve con página real (pág. {cites[0].page})")

    # 3. PDF escaneado -> reporta páginas que requieren OCR --------------------
    p = os.path.join(str(DOCUMENTS_DIR), _SCANNED_CANDIDATE)
    if os.path.exists(p):
        res2 = await ingest_bridge.ingest_path(_SCANNED_CANDIDATE)
        passed &= _ok(res2.get("pages_needing_ocr", 0) > 0,
                      f"PDF escaneado reporta pages_needing_ocr={res2.get('pages_needing_ocr')}")
    else:
        print("  [SKIP] PDF escaneado de referencia no presente.")

    try:
        os.remove(tmp)
    except OSError:
        pass
    return passed


def main() -> bool:
    print("== Ingesta con número de página EstudIA (integrada) ==")
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
