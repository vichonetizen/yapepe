"""
extract.py — Comprensión: extracción de conceptos/aristas anclados (pista integrada).

Construye el grafo de conceptos desde los chunks del corpus y lo persiste, además
de poblar study_concepts y study_edges. Regla dura del contrato: todo Concept lleva
source_chunks no vacía y todo Edge lleva evidence (chunk_id).

Sin LLM (esta ola) usa heurística: conceptos = términos capitalizados + términos de
contenido salientes; aristas = co-ocurrencia dentro del mismo chunk (relation
"related", evidence = ese chunk). El tipado fino de relaciones (defines/is-a/…) y
los autores se enriquecerán al cablear el LLM.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from typing import Callable, Optional

from sqlalchemy import text

from .store import get_engine, ensure_migrated
from . import chunk_map
from .concept_graph import ConceptGraph, normalize_concept


_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a",
    "y", "o", "u", "e", "que", "en", "con", "por", "para", "es", "son", "se", "su",
    "sus", "lo", "como", "mas", "más", "pero", "sin", "sobre", "entre", "este",
    "esta", "esto", "ese", "esa", "eso", "esos", "esas", "estos", "estas", "ser",
    "the", "of", "and", "to", "in", "is", "no", "ni", "donde", "cuando", "porque",
    "cual", "cuales", "segun", "según", "cada", "todo", "toda", "todos", "todas",
}


def _extract_concepts(text_in: str, max_per_chunk: int = 8) -> list[str]:
    """Heurística: capitalizados (no inicio de frase) + contenido largo salientes."""
    found: dict[str, str] = {}  # key -> surface

    # Capitalizados internos (nombres propios, términos): omite la 1ª palabra de cada oración.
    for sent in re.split(r"(?<=[.!?])\s+", text_in):
        toks = re.findall(r"\b[\wáéíóúüñÁÉÍÓÚÜÑ]+\b", sent)
        for i, w in enumerate(toks):
            if i == 0:
                continue
            if w[:1].isupper() and len(w) > 3 and w.lower() not in _STOPWORDS:
                found.setdefault(normalize_concept(w), w)

    # Términos de contenido largos (incluye tecnicismos en minúscula, p.ej. autopoiesis).
    freq: dict[str, tuple[int, str]] = {}
    for w in re.findall(r"\b[\wáéíóúüñ]{7,}\b", text_in.lower()):
        if w in _STOPWORDS:
            continue
        k = normalize_concept(w)
        n, surface = freq.get(k, (0, w))
        freq[k] = (n + 1, surface)
    for k, (n, surface) in sorted(freq.items(), key=lambda kv: -kv[1][0]):
        found.setdefault(k, surface)

    # Orden estable, recorta.
    return list(found.values())[:max_per_chunk]


async def extract_corpus(doc_ids: list[str],
                         llm: Optional[Callable] = None,
                         persist: bool = True) -> dict:
    """Extrae conceptos/aristas del corpus -> grafo persistido + tablas study_*."""
    await ensure_migrated()
    chunks = await chunk_map.get_chunks_for_docs(doc_ids)
    g = ConceptGraph()

    for ch in chunks:
        concepts = _extract_concepts(ch.text)
        for c in concepts:
            g.add_concept(c, ch.chunk_id)
        # Aristas de co-ocurrencia dentro del chunk (evidencia = ese chunk).
        keys = list(dict.fromkeys(normalize_concept(c) for c in concepts))
        for a_label, b_label in combinations(concepts, 2):
            if normalize_concept(a_label) != normalize_concept(b_label):
                g.add_edge(a_label, b_label, "related", ch.chunk_id)

    if persist:
        await _persist_concepts(g)
        g.save()

    return {
        "chunks": len(chunks),
        "concepts": len(g.nodes),
        "edges": len(g.edges),
        "graph_path": _safe_graph_path(),
    }


def _safe_graph_path() -> str:
    from .concept_graph import _graph_path
    return _graph_path


async def _persist_concepts(g: ConceptGraph) -> None:
    """Guarda conceptos (source_chunks NO vacía) y aristas (evidence) en SQLite."""
    engine = get_engine()
    async with engine.begin() as conn:
        for key, node in g.nodes.items():
            if not node["source_chunks"]:
                continue  # regla dura: no persistir conceptos sin anclaje
            await conn.execute(
                text("""INSERT OR REPLACE INTO study_concepts
                        (concept_id, label, definition, aliases, source_chunks)
                        VALUES (:id, :lab, NULL, :al, :sc)"""),
                {"id": key, "lab": node["label"], "al": json.dumps([]),
                 "sc": json.dumps(sorted(node["source_chunks"]))},
            )
        # Reemplaza aristas derivadas de esta corrida.
        for e in g.edges:
            if not e["evidence"]:
                continue
            await conn.execute(
                text("""INSERT INTO study_edges (src, dst, type, evidence)
                        VALUES (:s, :d, :t, :ev)"""),
                {"s": e["src"], "d": e["dst"], "t": e["relation"], "ev": e["evidence"]},
            )


async def chunks_for_concepts(concept_keys: list[str]) -> list[str]:
    """Union de source_chunks de los conceptos indicados (resuelve por clave canónica)."""
    keys = {normalize_concept(k) for k in concept_keys if normalize_concept(k)}
    if not keys:
        return []
    engine = get_engine()
    async with engine.connect() as conn:
        rows = (await conn.execute(
            text("SELECT concept_id, source_chunks FROM study_concepts"))).fetchall()
    seen: list[str] = []
    for cid, sc in rows:
        if cid not in keys:
            continue
        try:
            for ch in json.loads(sc or "[]"):
                if ch not in seen:
                    seen.append(ch)
        except (json.JSONDecodeError, TypeError):
            continue
    return seen


async def concepts_for_chunks(chunk_ids: list[str]) -> dict[str, list[str]]:
    """Mapa chunk_id -> [concept_id] (consultando study_concepts.source_chunks)."""
    engine = get_engine()
    async with engine.connect() as conn:
        rows = (await conn.execute(
            text("SELECT concept_id, source_chunks FROM study_concepts"))).fetchall()
    want = {str(c) for c in chunk_ids}
    out: dict[str, list[str]] = {c: [] for c in want}
    for cid, sc in rows:
        try:
            chunks = json.loads(sc or "[]")
        except (json.JSONDecodeError, TypeError):
            chunks = []
        for ch in chunks:
            if ch in want:
                out[ch].append(cid)
    return out
