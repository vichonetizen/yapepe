"""
focus.py — Selector de corpus por enfoque (Focus enum).

retrieve(StudyRequest) decide QUÉ chunks alimentan al generador:
  • ALL    -> todos los chunks del corpus.
  • TOPIC  -> chunks rankeados por el tema (lexical v1).
  • COMPARE / HIERARCHY / MAP -> requieren el grafo de conceptos tipado (Ola 1).
    Hasta que exista, degradan de forma honesta a TOPIC (si hay tema) o ALL,
    en vez de fallar. `retrieve` informa el modo efectivo aplicado.
"""

from __future__ import annotations

from .models import StudyRequest, Focus, Chunk
from . import chunk_map

# Enfoques que aún dependen del grafo de conceptos (Ola 1).
_NEEDS_GRAPH = {Focus.COMPARE, Focus.HIERARCHY, Focus.MAP}


async def retrieve(req: StudyRequest, top_k: int = 12) -> tuple[list[Chunk], str]:
    """Devuelve (chunks, modo_efectivo). modo_efectivo documenta cualquier
    degradación aplicada respecto del focus pedido."""
    if req.focus == Focus.ALL:
        return await chunk_map.get_chunks_for_docs(req.corpus_ids), "all"

    if req.focus == Focus.TOPIC:
        if req.topic:
            return await chunk_map.topic_chunks(req.corpus_ids, req.topic, top_k), "topic"
        return await chunk_map.get_chunks_for_docs(req.corpus_ids), "all (topic sin tema)"

    if req.focus in _NEEDS_GRAPH:
        if req.topic:
            chunks = await chunk_map.topic_chunks(req.corpus_ids, req.topic, top_k)
            return chunks, f"{req.focus.value}→topic (grafo pendiente, Ola 1)"
        chunks = await chunk_map.get_chunks_for_docs(req.corpus_ids)
        return chunks, f"{req.focus.value}→all (grafo pendiente, Ola 1)"

    return await chunk_map.get_chunks_for_docs(req.corpus_ids), "all"
