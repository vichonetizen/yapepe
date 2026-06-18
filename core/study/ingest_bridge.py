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

import glob
import io
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sqlalchemy import text  # noqa: E402

from config import DOCUMENTS_DIR  # noqa: E402
from .store import get_engine, ensure_migrated  # noqa: E402
from core import document_store  # extract_text/_chunk_text son sin estado  # noqa: E402


def _find_tesseract() -> str | None:
    p = shutil.which("tesseract")
    if p:
        return p
    for c in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(c):
            return c
    return None


def _find_poppler() -> str | None:
    pat = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Microsoft",
                       "WinGet", "Packages", "oschwartz10612.Poppler*",
                       "poppler-*", "Library", "bin")
    for d in glob.glob(pat):
        if os.path.exists(os.path.join(d, "pdftoppm.exe")):
            return d
    if shutil.which("pdftoppm"):
        return os.path.dirname(shutil.which("pdftoppm"))
    return None


def _find_tessdata() -> str | None:
    """Carpeta tessdata de usuario con idiomas extra (p. ej. spa) si existe."""
    cand = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "data", "tessdata")
    if os.path.exists(os.path.join(cand, "spa.traineddata")):
        return cand
    return None


_TESS = _find_tesseract()
_POPPLER = _find_poppler()
_TESSDATA = _find_tessdata()


def ocr_backend() -> str | None:
    """Nombre del backend de OCR disponible, o None. Requiere el MOTOR tesseract
    (no solo el wrapper) + pdf2image (con Poppler)."""
    if not _TESS:
        return None
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = _TESS
        pytesseract.get_tesseract_version()
    except Exception:
        return None
    try:
        import pdf2image  # noqa: F401
    except Exception:
        return None
    return f"tesseract@{os.path.basename(os.path.dirname(_TESS))}+poppler"


def _ocr_lang() -> str:
    """Idiomas para OCR: español+inglés si el spa está disponible, si no inglés."""
    return "spa+eng" if _TESSDATA else "eng"


def _ocr_page(data: bytes, page_no: int) -> str:
    """OCR de una página (best-effort). Devuelve '' si no hay backend/falla."""
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        if _TESS:
            pytesseract.pytesseract.tesseract_cmd = _TESS
        kw = {"first_page": page_no, "last_page": page_no, "dpi": 150}
        if _POPPLER:
            kw["poppler_path"] = _POPPLER
        imgs = convert_from_bytes(data, **kw)
        if not imgs:
            return ""
        config = f'--tessdata-dir "{_TESSDATA}"' if _TESSDATA else ""
        try:
            return pytesseract.image_to_string(imgs[0], lang=_ocr_lang(), config=config) or ""
        except Exception:
            return pytesseract.image_to_string(imgs[0]) or ""   # fallback eng/default
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
