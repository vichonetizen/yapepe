"""
PAC — Protocolo de Aprendizaje Continuo
Extrae, almacena y recupera aprendizajes de conversaciones y documentos.

Ciclo por cada intercambio:
  1. RECUPERACIÓN  — búsqueda BM25 sobre aprendizajes previos
  2. SÍNTESIS      — el AI responde con citas de fuentes
  3. CONSOLIDACIÓN — extracción heurística de nuevos aprendizajes → DB
"""
import re
import math
from collections import Counter
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

from config import DB_PATH
_engine = None
_DB_PATH = DB_PATH

# ── Heuristic patterns ─────────────────────────────────────────────────────────

_FACT_PATTERNS = [
    # Declaraciones: X es/son/tiene/consiste...
    r'[A-ZÁÉÍÓÚ][^.!?\n]{10,120}(?:es |son |fue |fueron |tiene |tienen |consiste en |se define como |significa )[^.!?\n]{8,100}[.!?]',
    # Conclusiones explícitas
    r'(?:por lo tanto|en conclusión|en resumen|esto significa|la clave es)[^.!?\n]{15,150}[.!?]',
    # Puntos numerados/con viñeta (suelen ser hechos concretos)
    r'(?:^|\n)\s*[-•*\d][.)]\s*([A-ZÁÉÍÓÚ][^.\n]{20,120})',
]

_TOPIC_PATTERNS = [
    r'\b[A-ZÁÉÍÓÚ][a-záéíóúüñ]{2,}(?:\s+[A-ZÁÉÍÓÚ][a-záéíóúüñ]{2,}){0,2}\b',
    r'\b(?:API|AI|ML|IA|CPU|GPU|RAG|LLM|SDK|SQL|JSON|HTTP|REST|NLP|CNN|RNN)\b',
    r'\b\w{5,}(?:ción|sión|dad|ismo|ista|aje|logía|grafía|tura)\b',
]

_DOC_CITE_RE = re.compile(r'\[📄\s*([^\]]{3,60})\]')
_PREV_CITE_RE = re.compile(r'\[🧠[^\]]*\]')


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(f"sqlite+aiosqlite:///{_DB_PATH}", echo=False)
    return _engine


# ── DB init ────────────────────────────────────────────────────────────────────

async def init_learnings_db():
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversation_learnings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                content     TEXT NOT NULL,
                topics      TEXT DEFAULT '',
                source      TEXT DEFAULT 'conversation',
                doc_refs    TEXT DEFAULT '',
                confidence  REAL DEFAULT 0.7,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_topics(text: str, limit: int = 6) -> list[str]:
    found: set[str] = set()
    for pat in _TOPIC_PATTERNS:
        for m in re.findall(pat, text):
            t = m.strip()
            if 3 < len(t) < 45:
                found.add(t)
    return list(found)[:limit]


def _extract_facts(text: str, limit: int = 5) -> list[str]:
    facts: list[str] = []
    for pat in _FACT_PATTERNS:
        for m in re.findall(pat, text, re.MULTILINE | re.IGNORECASE):
            s = m.strip()
            if isinstance(s, tuple):
                s = s[0].strip()
            if 25 < len(s) < 300:
                facts.append(s)
    return list(dict.fromkeys(facts))[:limit]   # deduplicate preserving order


def _tokenize(s: str) -> list[str]:
    return re.findall(r'\b[a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9]{2,}\b', s.lower())


# ── Core: extract + store ──────────────────────────────────────────────────────

async def extract_and_store_learning(
    session_id: str,
    user_msg: str,
    assistant_msg: str,
    doc_hits: list[dict] | None = None,
) -> list[dict]:
    """
    Extrae aprendizajes de un intercambio y los persiste.
    Llamar de forma asíncrona (fire-and-forget) tras cada respuesta.
    """
    learnings: list[dict] = []

    # 1. Facts declared by the user
    for fact in _extract_facts(user_msg, limit=2):
        learnings.append({
            "session_id": session_id,
            "content": fact,
            "topics": ", ".join(_extract_topics(fact + " " + user_msg)),
            "source": "usuario",
            "doc_refs": "",
            "confidence": 0.85,
        })

    # 2. Key conclusions from AI response
    ai_facts = _extract_facts(assistant_msg, limit=3)
    doc_refs_in_resp = ", ".join(_DOC_CITE_RE.findall(assistant_msg))
    for fact in ai_facts:
        learnings.append({
            "session_id": session_id,
            "content": fact,
            "topics": ", ".join(_extract_topics(fact + " " + user_msg)),
            "source": "ia_sintetizado",
            "doc_refs": doc_refs_in_resp,
            "confidence": 0.75,
        })

    # 3. Document fragments that were actually used
    if doc_hits:
        for hit in doc_hits[:2]:
            chunk = hit["content"][:250].strip()
            if chunk:
                learnings.append({
                    "session_id": session_id,
                    "content": f"[📄 {hit['filename']}] {chunk}",
                    "topics": ", ".join(_extract_topics(chunk)),
                    "source": "documento",
                    "doc_refs": hit["filename"],
                    "confidence": 0.90,
                })

    if not learnings:
        return []

    engine = _get_engine()
    async with engine.begin() as conn:
        for l in learnings:
            # Avoid storing near-duplicate content
            existing = (await conn.execute(
                text("SELECT id FROM conversation_learnings WHERE content=:c LIMIT 1"),
                {"c": l["content"][:300]},
            )).fetchone()
            if not existing:
                await conn.execute(text("""
                    INSERT INTO conversation_learnings
                        (session_id, content, topics, source, doc_refs, confidence)
                    VALUES (:sid, :content, :topics, :source, :doc_refs, :confidence)
                """), {
                    "sid": l["session_id"], "content": l["content"][:500],
                    "topics": l["topics"], "source": l["source"],
                    "doc_refs": l["doc_refs"], "confidence": l["confidence"],
                })

    return learnings


# ── Retrieval ──────────────────────────────────────────────────────────────────

async def search_learnings(query: str, top_k: int = 4) -> list[dict]:
    """BM25-inspired search over stored learnings with SQL pre-filtering."""
    if not query.strip():
        return []

    q_terms = Counter(_tokenize(query))
    if not q_terms:
        return []

    unique_terms = list(dict.fromkeys(q_terms))[:5]

    engine = _get_engine()
    async with engine.connect() as conn:
        # Pre-filter by content+topics LIKE to avoid loading 300+ rows into RAM
        conditions = " OR ".join(
            f"(content LIKE :t{i} OR topics LIKE :t{i})" for i in range(len(unique_terms))
        )
        params = {f"t{i}": f"%{t}%" for i, t in enumerate(unique_terms)}
        result = await conn.execute(text(
            f"SELECT id, content, topics, source, doc_refs, confidence, created_at "
            f"FROM conversation_learnings WHERE {conditions} ORDER BY created_at DESC LIMIT 80"
        ), params)
        rows = result.fetchall()

    if not rows:
        return []

    N = len(rows)
    tokenized = [
        (Counter(_tokenize(r[1] + " " + (r[2] or ""))), r)
        for r in rows
    ]

    df: Counter = Counter()
    for tokens, _ in tokenized:
        for t in q_terms:
            if t in tokens:
                df[t] += 1

    idf = {
        t: math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1)
        for t in q_terms
    }
    avg_len = sum(sum(tk.values()) for tk, _ in tokenized) / max(N, 1)
    k1, b = 1.5, 0.75

    scored: list[tuple[float, tuple]] = []
    for tokens, row in tokenized:
        dl = sum(tokens.values())
        score = sum(
            idf[t] * tokens[t] * (k1 + 1) / (tokens[t] + k1 * (1 - b + b * dl / max(avg_len, 1)))
            for t in q_terms if tokens.get(t, 0) > 0
        )
        if score > 0:
            scored.append((score, row))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [
        {
            "id": r[0], "content": r[1], "topics": r[2] or "",
            "source": r[3], "doc_refs": r[4] or "", "confidence": r[5],
            "created_at": str(r[6]), "score": round(s, 3),
        }
        for s, r in scored[:top_k]
    ]


async def get_all_learnings(limit: int = 200) -> list[dict]:
    engine = _get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(text(
            "SELECT id, session_id, content, topics, source, doc_refs, confidence, created_at "
            "FROM conversation_learnings ORDER BY created_at DESC LIMIT :lim"
        ), {"lim": limit})
        rows = result.fetchall()
    return [
        {
            "id": r[0], "session_id": r[1], "content": r[2], "topics": r[3] or "",
            "source": r[4], "doc_refs": r[5] or "", "confidence": r[6], "created_at": str(r[7]),
        }
        for r in rows
    ]


async def delete_learning(lid: int) -> bool:
    engine = _get_engine()
    async with engine.begin() as conn:
        result = await conn.execute(
            text("DELETE FROM conversation_learnings WHERE id=:id"), {"id": lid}
        )
        return result.rowcount > 0


async def delete_all_learnings() -> int:
    engine = _get_engine()
    async with engine.begin() as conn:
        result = await conn.execute(text("DELETE FROM conversation_learnings"))
        return result.rowcount


async def get_learning_stats() -> dict:
    engine = _get_engine()
    async with engine.connect() as conn:
        total = (await conn.execute(text(
            "SELECT COUNT(*) FROM conversation_learnings"
        ))).scalar() or 0
        src_rows = (await conn.execute(text(
            "SELECT source, COUNT(*) FROM conversation_learnings GROUP BY source"
        ))).fetchall()
    return {
        "total": total,
        "by_source": {r[0]: r[1] for r in src_rows},
    }


def build_pac_context(learnings: list[dict]) -> str:
    """Format retrieved learnings for system prompt injection."""
    if not learnings:
        return ""
    src_icon = {"usuario": "👤", "ia_sintetizado": "💡", "documento": "📄"}
    lines = []
    for l in learnings:
        icon = src_icon.get(l["source"], "🧠")
        lines.append(f"{icon} {l['content'][:200]}")
    return "\n".join(lines)
