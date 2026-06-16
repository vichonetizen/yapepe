"""
embed.py — Recuperación semántica para EstudIA (pista integrada).

Reusa el Embedder del proyecto, que auto-resuelve **bge-m3 vía Ollama local**
(multilingüe, fuerte en español) sin API key. Rankea chunks por similitud coseno
con la consulta. Si no hay backend de embeddings, devuelve None y el llamador cae
a ranking léxico (degradación elegante).
"""

from __future__ import annotations

import os

import numpy as np

from core.semantic_memory import Embedder
from .models import Chunk

# ollama_url por defecto (http://localhost:11434); resolve() elige bge-m3 si está.
_embedder = Embedder()


def _enabled() -> bool:
    """Interruptor ESTUDIA_EMBED (auto|off): 'off' desactiva el ranking semántico."""
    return os.getenv("ESTUDIA_EMBED", "auto").strip().lower() not in ("0", "off", "false", "no")


def set_embedder_for_tests(fn) -> None:
    """Solo pruebas: inyecta un embebedor determinista (texts -> vectores)."""
    _embedder.set_fake_embedder(fn)


async def available() -> bool:
    if not _enabled():
        return False
    await _embedder.resolve()
    return _embedder.available()


async def rank_by_similarity(query: str, chunks: list[Chunk], top_k: int = 12):
    """Devuelve los chunks ordenados por similitud semántica, o None si no hay
    backend de embeddings disponible (para que el llamador use léxico)."""
    if not _enabled():
        return None
    if not chunks:
        return []
    vecs = await _embedder.embed([query] + [c.text for c in chunks])
    if not vecs or len(vecs) != len(chunks) + 1:
        return None
    q = np.asarray(vecs[0], dtype=float)
    M = np.asarray(vecs[1:], dtype=float)
    qn = q / (np.linalg.norm(q) + 1e-9)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    sims = Mn @ qn
    order = np.argsort(-sims)[:top_k]
    return [chunks[int(i)] for i in order]
