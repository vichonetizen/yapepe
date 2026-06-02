"""
Document store: ingest → chunk → BM25 retrieval
Supported: .txt .md .csv .pdf .docx
"""
import re
import math
from collections import Counter
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

from config import DB_PATH, DOCUMENTS_DIR

_engine = None
_DB_PATH = DB_PATH
_has_chunks: bool | None = None  # None = unknown, False = empty, True = has data


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(f"sqlite+aiosqlite:///{_DB_PATH}", echo=False)
    return _engine


async def init_documents_db():
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_size INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            )
        """))


def _tokenize(s: str) -> list[str]:
    return re.findall(r'\b[a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9]{2,}\b', s.lower())


def _chunk_text(text: str, chunk_size: int = 550, overlap: int = 100) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []

    for para in paragraphs:
        words = para.split()
        if len(current) + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            current = current[-overlap:]
        current.extend(words)

    if current:
        chunks.append(" ".join(current))

    # Fallback: pure word-window split when no paragraph breaks
    if not chunks:
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

    return [c for c in chunks if c.strip()]


async def extract_text(filename: str, file_bytes: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext in ('.txt', '.md', '.csv'):
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return file_bytes.decode('latin-1', errors='replace')
    elif ext == '.pdf':
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            if not pages:
                raise ValueError("El PDF no contiene texto extraíble (puede ser imagen escaneada).")
            return "\n\n".join(pages)
        except ImportError:
            raise ValueError("pypdf no instalado. Ejecuta: pip install pypdf")
    elif ext in ('.docx',):
        try:
            from docx import Document
            import io
            doc = Document(io.BytesIO(file_bytes))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise ValueError("python-docx no instalado. Ejecuta: pip install python-docx")
    else:
        raise ValueError(f"Formato no soportado: {ext}. Válidos: .txt .md .pdf .docx .csv")


async def add_document(filename: str, file_bytes: bytes) -> dict:
    global _has_chunks
    text_content = await extract_text(filename, file_bytes)
    chunks = _chunk_text(text_content)
    if not chunks:
        raise ValueError("El documento está vacío o no contiene texto extraíble.")

    engine = _get_engine()
    async with engine.begin() as conn:
        result = await conn.execute(
            text("INSERT INTO documents (filename, file_size, chunk_count) VALUES (:fn, :fs, :cc)"),
            {"fn": filename, "fs": len(file_bytes), "cc": len(chunks)},
        )
        doc_id = result.lastrowid
        for i, chunk in enumerate(chunks):
            await conn.execute(
                text("INSERT INTO document_chunks (doc_id, chunk_index, content) VALUES (:d, :i, :c)"),
                {"d": doc_id, "i": i, "c": chunk},
            )

    _has_chunks = True
    return {"id": doc_id, "filename": filename, "chunks": len(chunks), "chars": len(text_content)}


async def list_documents() -> list[dict]:
    engine = _get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT id, filename, file_size, chunk_count, created_at FROM documents ORDER BY created_at DESC")
        )
        rows = result.fetchall()
    return [
        {"id": r[0], "filename": r[1], "file_size": r[2], "chunk_count": r[3], "created_at": str(r[4])}
        for r in rows
    ]


async def delete_document(doc_id: int) -> bool:
    global _has_chunks
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM document_chunks WHERE doc_id = :id"), {"id": doc_id})
        result = await conn.execute(text("DELETE FROM documents WHERE id = :id"), {"id": doc_id})
    _has_chunks = None  # invalidate — may now be empty
    return result.rowcount > 0


async def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """BM25-inspired retrieval over document chunks with SQL pre-filtering."""
    global _has_chunks
    if not query.strip():
        return []

    # Fast path: skip expensive BM25 if no documents exist
    if _has_chunks is None:
        engine = _get_engine()
        async with engine.connect() as conn:
            count = (await conn.execute(text("SELECT COUNT(*) FROM document_chunks"))).scalar() or 0
        _has_chunks = count > 0
    if not _has_chunks:
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return []

    # Deduplicate terms; limit to 6 to avoid SQL parameter explosion
    unique_terms = list(dict.fromkeys(query_terms))[:6]

    engine = _get_engine()
    async with engine.connect() as conn:
        # Pre-filter via LIKE so we never load the full corpus into RAM
        conditions = " OR ".join(f"dc.content LIKE :t{i}" for i in range(len(unique_terms)))
        params = {f"t{i}": f"%{t}%" for i, t in enumerate(unique_terms)}
        result = await conn.execute(
            text(f"""
                SELECT dc.content, d.filename
                FROM document_chunks dc
                JOIN documents d ON dc.doc_id = d.id
                WHERE {conditions}
                LIMIT 200
            """),
            params,
        )
        rows = result.fetchall()

    if not rows:
        return []

    query_counter = Counter(query_terms)
    N = len(rows)

    tokenized = [(Counter(_tokenize(content)), content, filename) for content, filename in rows]

    df: Counter = Counter()
    for tokens, _, _ in tokenized:
        for term in query_counter:
            if term in tokens:
                df[term] += 1

    idf = {
        t: math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1)
        for t in query_counter
    }

    avg_len = sum(sum(t.values()) for t, *_ in tokenized) / max(N, 1)
    k1, b = 1.5, 0.75

    scored: list[tuple[float, str, str]] = []
    for tokens, content, filename in tokenized:
        doc_len = sum(tokens.values())
        score = 0.0
        for term in query_counter:
            tf = tokens.get(term, 0)
            if tf > 0:
                score += idf[term] * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * doc_len / max(avg_len, 1))
                )
        if score > 0:
            scored.append((score, content, filename))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [
        {"content": c, "filename": f, "score": round(s, 3)}
        for s, c, f in scored[:top_k]
    ]


async def get_doc_count() -> dict:
    engine = _get_engine()
    async with engine.connect() as conn:
        docs = (await conn.execute(text("SELECT COUNT(*) FROM documents"))).scalar() or 0
        chunks = (await conn.execute(text("SELECT COUNT(*) FROM document_chunks"))).scalar() or 0
    return {"documents": docs, "chunks": chunks}
