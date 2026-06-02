"""
Memoria semántica — Capa 1 del asistente Pentamodal.

Permite RELACIONAR IDEAS POR SIGNIFICADO, no solo por palabras compartidas.
BM25 (document_store / learning_engine) solo conecta textos que comparten términos
exactos. Los embeddings conectan "autopoiesis" con "sistema que se auto-construye"
aunque no compartan ni una palabra. Eso es lo que hace que el asistente "piense".

Backends de embeddings, en orden de preferencia:
  1. Ollama  (nomic-embed-text)        — LOCAL, sin API key, ideal para GPU de 2 GB
  2. OpenAI  (text-embedding-3-small)  — nube, alta calidad
  3. Google  (text-embedding-004)      — nube
  4. Ninguno → degradación elegante a BM25 (la app nunca se rompe)

Los vectores se persisten en SQLite (BLOB float32) y la similitud coseno se
calcula con numpy. Para un corpus de cientos/miles de chunks, la búsqueda por
fuerza bruta es instantánea — no hace falta una base vectorial dedicada.
"""
import asyncio
import json as _json
import urllib.request
import numpy as np

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_DB_PATH = "data/pentamodal.db"
_engine = None

# Los modelos de embeddings tienen ventana de contexto limitada (nomic-embed-text
# ≈ 2048 tokens). Chunks de >~4800 caracteres la exceden y el embed falla. Truncamos
# a un tamaño seguro: los primeros ~3000 caracteres capturan el tema del fragmento.
MAX_EMBED_CHARS = 3000


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(f"sqlite+aiosqlite:///{_DB_PATH}", echo=False)
    return _engine


def _reset_engine_for_tests(db_path: str):
    """Solo para pruebas: redirige el almacenamiento a una DB temporal."""
    global _DB_PATH, _engine
    _DB_PATH = db_path
    _engine = None


# ── Motor de embeddings pluggable ────────────────────────────────────────────────

class Embedder:
    """Resuelve un backend de embeddings disponible y vectoriza texto.

    Diseñado para fallar de forma segura: si ningún backend está disponible,
    `available()` devuelve False y el resto del sistema usa BM25.
    """

    def __init__(self):
        self._keys: dict[str, str] = {}
        self._ollama_url = "http://localhost:11434"
        self._backend: str | None = None   # 'ollama' | 'openai' | 'google' | None
        self._model = ""
        self._resolved = False
        # Inyectable en pruebas: callable(list[str]) -> list[list[float]] | None
        self._fake = None

    def configure(self, keys: dict, ollama_url: str = ""):
        """Llamar al iniciar la app con las claves cargadas en AIClient."""
        self._keys = dict(keys or {})
        if ollama_url:
            base = ollama_url.rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            self._ollama_url = base
        self._resolved = False  # forzar re-resolución con la nueva config

    def set_fake_embedder(self, fn):
        """Solo pruebas: inyecta un embebedor determinista."""
        self._fake = fn
        self._backend = "fake" if fn else None
        self._model = "fake-embed" if fn else ""
        self._resolved = True

    def _probe_ollama(self) -> bool:
        try:
            with urllib.request.urlopen(self._ollama_url + "/api/tags", timeout=2) as r:
                _json.loads(r.read())
                return True
        except Exception:
            return False

    async def resolve(self):
        if self._resolved:
            return
        # Preferencia local-first: Ollama cubre el objetivo "GPU 2 GB sin internet".
        ollama_ok = await asyncio.get_event_loop().run_in_executor(None, self._probe_ollama)
        if ollama_ok:
            self._backend, self._model = "ollama", "nomic-embed-text"
        elif self._keys.get("openai"):
            self._backend, self._model = "openai", "text-embedding-3-small"
        elif self._keys.get("google"):
            self._backend, self._model = "google", "gemini-embedding-001"
        else:
            self._backend, self._model = None, ""
        self._resolved = True

    @property
    def model_id(self) -> str:
        """Etiqueta usada para versionar vectores. Si cambia el modelo, hay que reindexar."""
        return f"{self._backend}:{self._model}" if self._backend else ""

    def available(self) -> bool:
        return self._backend is not None

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Devuelve un vector por texto, o None si no hay backend / falla."""
        if not texts:
            return []
        if self._fake is not None:
            return self._fake(list(texts))

        await self.resolve()
        if not self._backend:
            return None
        try:
            if self._backend in ("ollama", "openai"):
                from openai import AsyncOpenAI
                if self._backend == "ollama":
                    client = AsyncOpenAI(base_url=self._ollama_url + "/v1", api_key="ollama")
                else:
                    client = AsyncOpenAI(api_key=self._keys["openai"])
                resp = await client.embeddings.create(model=self._model, input=list(texts))
                return [list(d.embedding) for d in resp.data]
            elif self._backend == "google":
                from google import genai
                client = genai.Client(api_key=self._keys["google"])
                # gemini-embedding-001 acepta una lista de textos en una sola llamada
                r = await client.aio.models.embed_content(model=self._model, contents=list(texts))
                return [[float(x) for x in e.values] for e in r.embeddings]
        except Exception:
            return None
        return None


embedder = Embedder()


# ── Almacén de vectores ──────────────────────────────────────────────────────────

def _to_blob(vec) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return m / norms


async def init_semantic_db():
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id INTEGER PRIMARY KEY,
                doc_id   INTEGER,
                model    TEXT NOT NULL,
                dim      INTEGER NOT NULL,
                vec      BLOB NOT NULL
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chunk_emb_model ON chunk_embeddings(model)"
        ))


# ── Indexación ─────────────────────────────────────────────────────────────────

async def index_document_chunks(batch_size: int = 16, max_batches: int = 400) -> int:
    """Embebe los chunks de document_chunks que aún no tienen vector para el modelo
    actual. Idempotente y reanudable: se puede llamar varias veces sin duplicar.

    batch_size=16 por defecto: los chunks son grandes (~1k tokens) y los free tiers
    de nube limitan los tokens por petición. Si un lote falla (p. ej. 429 de cuota),
    se detiene dejando el progreso parcial guardado — la próxima llamada continúa
    desde donde quedó. Devuelve el número de chunks indexados en esta llamada."""
    await embedder.resolve()
    if not embedder.available():
        return 0

    model = embedder.model_id
    engine = _get_engine()
    indexed = 0

    for _ in range(max_batches):
        async with engine.connect() as conn:
            rows = (await conn.execute(text("""
                SELECT dc.id, dc.doc_id, dc.content
                FROM document_chunks dc
                LEFT JOIN chunk_embeddings ce
                       ON ce.chunk_id = dc.id AND ce.model = :model
                WHERE ce.chunk_id IS NULL
                LIMIT :batch
            """), {"model": model, "batch": batch_size})).fetchall()

        if not rows:
            break

        texts = [(r[2] or "")[:MAX_EMBED_CHARS] for r in rows]
        vectors = await embedder.embed(texts)
        if not vectors:
            # Un chunk problemático puede tumbar el lote entero. Reintentamos uno por
            # uno (truncando más en el segundo intento) y saltamos los irrecuperables,
            # para no frenar la indexación del resto del corpus.
            ok_rows, vectors = [], []
            for r in rows:
                t = (r[2] or "")[:MAX_EMBED_CHARS]
                v = await embedder.embed([t])
                if not v:
                    v = await embedder.embed([t[:1200]])
                if v:
                    ok_rows.append(r)
                    vectors.append(v[0])
            if not ok_rows:
                break  # backend realmente caído → progreso guardado, reanudable
            rows = ok_rows

        async with engine.begin() as conn:
            for (chunk_id, doc_id, _content), vec in zip(rows, vectors):
                v = np.asarray(vec, dtype=np.float32)
                await conn.execute(text("""
                    INSERT OR REPLACE INTO chunk_embeddings (chunk_id, doc_id, model, dim, vec)
                    VALUES (:cid, :did, :model, :dim, :vec)
                """), {"cid": chunk_id, "did": doc_id, "model": model,
                       "dim": int(v.shape[0]), "vec": _to_blob(v)})
                indexed += 1

    return indexed


async def index_doc(doc_id: int) -> int:
    """Indexa los chunks de un documento recién subido (fire-and-forget desde el upload)."""
    await embedder.resolve()
    if not embedder.available():
        return 0
    model = embedder.model_id
    engine = _get_engine()
    async with engine.connect() as conn:
        rows = (await conn.execute(text("""
            SELECT dc.id, dc.doc_id, dc.content
            FROM document_chunks dc
            LEFT JOIN chunk_embeddings ce
                   ON ce.chunk_id = dc.id AND ce.model = :model
            WHERE dc.doc_id = :did AND ce.chunk_id IS NULL
        """), {"model": model, "did": doc_id})).fetchall()
    if not rows:
        return 0
    vectors = await embedder.embed([(r[2] or "")[:MAX_EMBED_CHARS] for r in rows])
    if not vectors:
        return 0
    async with engine.begin() as conn:
        for (chunk_id, did, _c), vec in zip(rows, vectors):
            v = np.asarray(vec, dtype=np.float32)
            await conn.execute(text("""
                INSERT OR REPLACE INTO chunk_embeddings (chunk_id, doc_id, model, dim, vec)
                VALUES (:cid, :did, :model, :dim, :vec)
            """), {"cid": chunk_id, "did": did, "model": model,
                   "dim": int(v.shape[0]), "vec": _to_blob(v)})
    return len(rows)


# ── Búsqueda ─────────────────────────────────────────────────────────────────────

async def semantic_search_documents(query: str, top_k: int = 5) -> list[dict] | None:
    """Búsqueda por significado sobre todos los chunks indexados.
    Devuelve None (no [] ) cuando no hay embeddings disponibles, para que el llamador
    distinga 'sin backend' de 'sin resultados' y haga fallback a BM25."""
    if not query.strip():
        return None
    await embedder.resolve()
    if not embedder.available():
        return None

    qv = await embedder.embed([query[:MAX_EMBED_CHARS]])
    if not qv:
        return None
    q = np.asarray(qv[0], dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)

    model = embedder.model_id
    engine = _get_engine()
    async with engine.connect() as conn:
        rows = (await conn.execute(text("""
            SELECT ce.vec, dc.content, d.filename
            FROM chunk_embeddings ce
            JOIN document_chunks dc ON dc.id = ce.chunk_id
            JOIN documents d        ON d.id  = ce.doc_id
            WHERE ce.model = :model
        """), {"model": model})).fetchall()

    if not rows:
        return []

    mats = np.vstack([_from_blob(r[0]) for r in rows])
    if mats.shape[1] != q.shape[0]:
        return None  # dimensión incompatible (modelo cambió) → reindexar
    mats = _normalize(mats)
    sims = mats @ q  # coseno (vectores normalizados)

    order = np.argsort(-sims)[:top_k]
    return [
        {"content": rows[i][1], "filename": rows[i][2], "score": float(sims[i])}
        for i in order
    ]


def _key(hit: dict) -> tuple:
    return (hit.get("filename", ""), (hit.get("content", "") or "")[:160])


def _minmax(values: list[float]) -> dict:
    if not values:
        return {}
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return {"lo": lo, "span": span}


async def hybrid_search_documents(
    query: str, top_k: int = 5, alpha: float = 0.5, sem_floor: float = 0.45,
) -> list[dict]:
    """Combina BM25 (léxico) + semántico (significado).

    alpha pondera léxico vs. semántico (0.5 = mitad y mitad).
    sem_floor descarta resultados solo-semánticos con coseno bajo (ruido).
    Si no hay embeddings, devuelve exactamente lo que daría BM25 (degradación total)."""
    from core.document_store import search_documents

    bm = await search_documents(query, top_k=top_k * 3)
    sem = await semantic_search_documents(query, top_k=top_k * 3)

    if sem is None:
        return bm[:top_k]  # sin backend semántico → comportamiento original intacto

    # Normalizar cada lista a 0..1 para mezclarlas en la misma escala
    bm_scores = [h["score"] for h in bm]
    sem_scores = [h["score"] for h in sem]
    bm_n = _minmax(bm_scores)
    sem_n = _minmax(sem_scores)

    merged: dict[tuple, dict] = {}

    for h in bm:
        norm = (h["score"] - bm_n["lo"]) / bm_n["span"] if bm_n else 0.0
        merged[_key(h)] = {**h, "_bm": norm, "_sem": 0.0}

    for h in sem:
        if h["score"] < sem_floor and _key(h) not in merged:
            continue  # solo-semántico de baja relevancia → descartar ruido
        norm = (h["score"] - sem_n["lo"]) / sem_n["span"] if sem_n else 0.0
        k = _key(h)
        if k in merged:
            merged[k]["_sem"] = norm
        else:
            merged[k] = {**h, "_bm": 0.0, "_sem": norm}

    out = []
    for h in merged.values():
        blended = alpha * h["_bm"] + (1 - alpha) * h["_sem"]
        out.append({
            "content": h["content"], "filename": h["filename"],
            "score": round(blended, 4),
            "bm25": round(h["_bm"], 4), "semantic": round(h["_sem"], 4),
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]


# ── Estado / utilidades ──────────────────────────────────────────────────────────

async def maybe_autoindex_on_start() -> int:
    """Backfill automático al arrancar SOLO si el backend es local (Ollama):
    es gratis e ilimitado. Con backends de nube no se indexa solo para no
    consumir cuota sin que el usuario lo pida (usar POST /semantic/reindex)."""
    await embedder.resolve()
    if embedder._backend == "ollama":
        return await index_document_chunks()
    return 0


async def status() -> dict:
    await embedder.resolve()
    engine = _get_engine()
    async with engine.connect() as conn:
        try:
            indexed = (await conn.execute(
                text("SELECT COUNT(*) FROM chunk_embeddings WHERE model = :m"),
                {"m": embedder.model_id},
            )).scalar() or 0
        except Exception:
            indexed = 0
        try:
            total_chunks = (await conn.execute(
                text("SELECT COUNT(*) FROM document_chunks")
            )).scalar() or 0
        except Exception:
            total_chunks = 0
    return {
        "available": embedder.available(),
        "backend": embedder._backend,
        "model": embedder.model_id,
        "indexed_chunks": indexed,
        "total_chunks": total_chunks,
        "pending": max(0, total_chunks - indexed) if embedder.available() else total_chunks,
    }
