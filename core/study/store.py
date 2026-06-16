"""
store.py — Persistencia de EstudIA (pista integrada) sobre la MISMA SQLite de
pentamodal-app (data/pentamodal.db).

Responsabilidades de la Ola 0:
  1. Migración IDEMPOTENTE y retrocompatible de `document_chunks`: añade las
     columnas `page` y `section_path` que el RAG existente no guarda y que la
     trazabilidad obligatoria de EstudIA necesita (cierra el gap del inventario).
  2. Crear las tablas propias de EstudIA (study_*), sin tocar las existentes.

Todo CREATE/ALTER es seguro de correr N veces. Las nuevas columnas son NULLABLE,
así que el RAG y el chat actuales siguen funcionando sin cambios.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402
from sqlalchemy import text  # noqa: E402

from config import DB_PATH  # noqa: E402

# Ruta de la DB y motor perezoso. `use_db()` permite redirigir a una DB temporal
# en pruebas (mismo patrón que semantic_memory).
_db_path: str = DB_PATH
_engine = None


def use_db(path: str) -> None:
    """Solo para pruebas: redirige el almacenamiento a otra DB y resetea el motor."""
    global _db_path, _engine
    _db_path = path
    _engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(f"sqlite+aiosqlite:///{_db_path}", echo=False)
    return _engine


async def _column_names(conn, table: str) -> set[str]:
    """Columnas existentes de una tabla (vacío si la tabla no existe)."""
    res = await conn.execute(text(f"PRAGMA table_info({table})"))
    return {row[1] for row in res.fetchall()}


# Tablas propias de EstudIA. JSON se guarda como TEXT (json.dumps).
_STUDY_TABLES = {
    "study_concepts": """
        CREATE TABLE IF NOT EXISTS study_concepts (
            concept_id    TEXT PRIMARY KEY,
            label         TEXT NOT NULL,
            definition    TEXT,
            aliases       TEXT,            -- json: [str]
            source_chunks TEXT NOT NULL    -- json: [chunk_id]  (regla dura: no vacía)
        )
    """,
    "study_authors": """
        CREATE TABLE IF NOT EXISTS study_authors (
            author_id     TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            theses        TEXT,            -- json: [str]
            source_chunks TEXT             -- json: [chunk_id]
        )
    """,
    "study_edges": """
        CREATE TABLE IF NOT EXISTS study_edges (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            src       TEXT NOT NULL,
            dst       TEXT NOT NULL,
            type      TEXT NOT NULL,        -- Relation
            evidence  TEXT NOT NULL         -- chunk_id (regla dura)
        )
    """,
    "study_cards": """
        CREATE TABLE IF NOT EXISTS study_cards (
            card_id     TEXT PRIMARY KEY,
            concept_id  TEXT NOT NULL,
            front       TEXT NOT NULL,
            back        TEXT NOT NULL,
            due         TEXT,
            interval    INTEGER DEFAULT 0,
            ease        REAL DEFAULT 2.5,
            lapses      INTEGER DEFAULT 0,
            reps        INTEGER DEFAULT 0,
            last_review TEXT,
            difficulty  INTEGER DEFAULT 3
        )
    """,
    "study_results": """
        CREATE TABLE IF NOT EXISTS study_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            q_id        TEXT,
            user_answer TEXT,
            score       REAL,
            verdict     TEXT,
            feedback    TEXT,
            evidence    TEXT,               -- json: [chunk_id]
            gaps        TEXT,               -- json: [concept_id]
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "study_exams": """
        CREATE TABLE IF NOT EXISTS study_exams (
            exam_id         TEXT PRIMARY KEY,
            corpus_ids      TEXT NOT NULL,  -- json: [doc_id]
            coverage_target REAL DEFAULT 0.85,
            time_limit      INTEGER,
            is_submitted    INTEGER DEFAULT 0,
            questions       TEXT,           -- json: [q_id]
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "study_student_state": """
        CREATE TABLE IF NOT EXISTS study_student_state (
            concept_id  TEXT PRIMARY KEY,
            level       REAL DEFAULT 0.0,   -- 0..1 (probabilidad de retención)
            last_review TEXT,
            due         TEXT,
            lapses      INTEGER DEFAULT 0
        )
    """,
    "study_error_clusters": """
        CREATE TABLE IF NOT EXISTS study_error_clusters (
            cluster_id         TEXT PRIMARY KEY,
            concept_ids        TEXT NOT NULL,  -- json: [concept_id]
            error_type         TEXT,
            frequency          INTEGER DEFAULT 0,
            recommended_action TEXT
        )
    """,
}


async def migrate() -> dict:
    """Aplica la migración Ola 0. Idempotente. Devuelve un resumen de lo hecho."""
    engine = get_engine()
    summary = {"chunks_columns_added": [], "study_tables_ensured": []}

    async with engine.begin() as conn:
        # Asegura que existan las tablas base del RAG (por si se corre antes que
        # init_documents_db en un entorno limpio). CREATE IF NOT EXISTS = inocuo.
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

        # Migración retrocompatible: añade page / section_path SOLO si faltan.
        cols = await _column_names(conn, "document_chunks")
        if "page" not in cols:
            await conn.execute(text("ALTER TABLE document_chunks ADD COLUMN page INTEGER"))
            summary["chunks_columns_added"].append("page")
        if "section_path" not in cols:
            await conn.execute(text("ALTER TABLE document_chunks ADD COLUMN section_path TEXT"))
            summary["chunks_columns_added"].append("section_path")

        # Tablas propias de EstudIA.
        for name, ddl in _STUDY_TABLES.items():
            await conn.execute(text(ddl))
            summary["study_tables_ensured"].append(name)

    return summary


async def healthcheck() -> dict:
    """Estado mínimo del almacén de estudio (para /study/panel y diagnóstico)."""
    engine = get_engine()
    async with engine.connect() as conn:
        cols = await _column_names(conn, "document_chunks")
        concepts = (await conn.execute(text("SELECT COUNT(*) FROM study_concepts"))).scalar() or 0
        cards = (await conn.execute(text("SELECT COUNT(*) FROM study_cards"))).scalar() or 0
    return {
        "chunks_have_page": "page" in cols,
        "chunks_have_section_path": "section_path" in cols,
        "study_concepts": concepts,
        "study_cards": cards,
    }
