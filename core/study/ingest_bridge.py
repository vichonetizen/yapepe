"""
ingest_bridge.py — Ingesta con número de página + hook de OCR (pista integrada).

Cierra el gap del inventario: el RAG existente no guardaba `page`. Aquí se extrae el
PDF PÁGINA POR PÁGINA (pypdf, siempre disponible) y cada chunk queda con su número de
página real → las citas pueden decir «pág. N». Las páginas sin capa de texto (PDFs
escaneados) se intentan por OCR SI hay backend local; si no, se cuentan como
`pages_needing_ocr` y NO se insertan chunks vacíos (regla NO INVENTAR).

OCR es opt-in y desacoplado: requiere pytesseract + un renderizador (pdf2image o
PyMuPDF). Hoy no están instalados, así que el camino nativo funciona y el OCR queda
dormido sin romper nada. 100% offline / local-first.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sqlalchemy import text  # noqa: E402

from config import DOCUMENTS_DIR  # noqa: E402
from .store import get_engine, ensure_migrated  # noqa: E402
from core import document_store  # extract_text/_chunk_text son sin estado  # noqa: E402


def ocr_backend() -> str | None:
    """Nombre del backend de OCR disponible, o None. Requiere OCR + renderizador."""
    try:
        import pytesseract  # noqa: F401
    except Exception:
        return None
    for renderer in ("pdf2image", "fitz"):
        try:
            __import__(renderer)
            return f"pytesseract+{renderer}"
        except Exception:
            continue
    return None


def _ocr_page(data: bytes, page_no: int) -> str:
    """OCR de una página (best-effort). Devuelve '' si no hay backend."""
    try:
        import pytesseract
        try:
            from pdf2image import convert_from_bytes
            imgs = convert_from_bytes(data, first_page=page_no, last_page=page_no)
            if not imgs:
                return ""
            return pytesseract.image_to_string(imgs[0], lang="spa+eng") or ""
        except Exception:
            import fitz  # PyMuPDF
            from PIL import Image
            doc = fitz.open(stream=data, filetype="pdf")
            pix = doc[page_no - 1].get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return pytesseract.image_to_string(img, lang="spa+eng") or ""
    except Exception:
        return ""


def _extract_pages_pdf(data: bytes) -> list[str]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return [(p.extract_text() or "") for p in reader.pages]


async def ingest_pdf(filename: str, data: bytes, *, min_words_per_page: int = 4,
                     chunk_size: int = 400, overlap: int = 80,
                     allow_ocr: bool = True) -> dict:
    """Ingiere un PDF con número de página por chunk. Devuelve un resumen."""
    await ensure_migrated()
    pages = _extract_pages_pdf(data)
    backend = ocr_backend() if allow_ocr else None

    rows: list[tuple[int, str, int]] = []   # (chunk_index, content, page)
    pages_with_text = 0
    pages_needing_ocr = 0
    idx = 0
    for page_no, ptext in enumerate(pages, start=1):
        ptext = (ptext or "").strip()
        if len(ptext.split()) < min_words_per_page:
            if backend:
                ptext = _ocr_page(data, page_no).strip()
            if len(ptext.split()) < min_words_per_page:
                pages_needing_ocr += 1
                continue
        pages_with_text += 1
        for ch in document_store._chunk_text(ptext, chunk_size, overlap):
            rows.append((idx, ch, page_no))
            idx += 1

    if not rows:
        return {"doc_id": None, "filename": filename, "pages": len(pages),
                "chunks": 0, "pages_with_text": 0,
                "pages_needing_ocr": pages_needing_ocr, "ocr_backend": backend,
                "note": "Sin capa de texto extraíble; requiere OCR para estudiar el cuerpo."}

    engine = get_engine()
    async with engine.begin() as conn:
        r = await conn.execute(
            text("INSERT INTO documents (filename, file_size, chunk_count) VALUES (:f,:s,:n)"),
            {"f": filename, "s": len(data), "n": len(rows)})
        doc_id = str(r.lastrowid)
        for ci, content, page in rows:
            await conn.execute(
                text("INSERT INTO document_chunks (doc_id, chunk_index, content, page) "
                     "VALUES (:d,:i,:c,:p)"),
                {"d": int(doc_id), "i": ci, "c": content, "p": page})

    return {"doc_id": doc_id, "filename": filename, "pages": len(pages),
            "chunks": len(rows), "pages_with_text": pages_with_text,
            "pages_needing_ocr": pages_needing_ocr, "ocr_backend": backend}


async def ingest_path(path_or_name: str) -> dict:
    """Ingiere un archivo del corpus (PDF con páginas; otros formatos sin página)."""
    await ensure_migrated()
    p = Path(path_or_name)
    if not p.is_absolute():
        p = Path(DOCUMENTS_DIR) / path_or_name
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo: {p}")
    data = p.read_bytes()
    if p.suffix.lower() == ".pdf":
        return await ingest_pdf(p.name, data)

    # Formatos sin paginación nativa (docx/txt/md/csv): un solo flujo, page=None.
    txt = await document_store.extract_text(p.name, data)
    chunks = document_store._chunk_text(txt)
    if not chunks:
        return {"doc_id": None, "filename": p.name, "pages": 0, "chunks": 0,
                "pages_with_text": 0, "pages_needing_ocr": 0, "ocr_backend": None,
                "note": "Documento vacío o sin texto."}
    engine = get_engine()
    async with engine.begin() as conn:
        r = await conn.execute(
            text("INSERT INTO documents (filename, file_size, chunk_count) VALUES (:f,:s,:n)"),
            {"f": p.name, "s": len(data), "n": len(chunks)})
        doc_id = str(r.lastrowid)
        for i, ch in enumerate(chunks):
            await conn.execute(
                text("INSERT INTO document_chunks (doc_id, chunk_index, content) VALUES (:d,:i,:c)"),
                {"d": int(doc_id), "i": i, "c": ch})
    return {"doc_id": doc_id, "filename": p.name, "pages": None, "chunks": len(chunks),
            "pages_with_text": None, "pages_needing_ocr": 0, "ocr_backend": None,
            "note": "Formato sin paginación nativa (page=N/D)."}
