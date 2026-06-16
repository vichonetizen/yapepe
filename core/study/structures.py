"""
structures.py — Estructuras pedagógicas de EstudIA (Ola 3, pista integrada).

Genera mapa conceptual, jerarquía, comparativa y debate, SIEMPRE anclados:
  • comparison(a, b): cada dimensión lleva evidencia de AMBOS conceptos.
  • debate(topic): cada posición lleva evidencia y abarca varios documentos.
  • hierarchy(root) / concept_map(): derivados fieles del grafo de conceptos.

Sin LLM (esta ola) las dimensiones/posiciones se construyen con extractos anclados
del corpus; con un LLM inyectable se enriquecen sin perder el anclaje (todo pasa por
grounding.assert_grounded antes de emitirse).
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from .models import ConceptMap, Hierarchy, Comparison, Debate
from .concept_graph import ConceptGraph, normalize_concept
from . import chunk_map, extract as _extract
from .grounding import assert_grounded, resolve_evidence, UngroundedError


async def concept_map(max_edges: int = 60) -> dict:
    """Mapa conceptual (Mermaid + listas) fiel al grafo persistido."""
    g = ConceptGraph.load()
    return {
        "render": "mermaid",
        "mermaid": g.to_mermaid(max_edges),
        "nodes": [{"concept_id": k, "label": v["label"]} for k, v in g.nodes.items()],
        "edges": [{"src": e["src"], "dst": e["dst"], "relation": e["relation"],
                   "evidence": e["evidence"]} for e in g.edges[:max_edges]],
    }


async def hierarchy(root_label: str, max_levels: int = 5) -> Hierarchy:
    g = ConceptGraph.load()
    levels = g.bfs_levels(root_label, max_levels=max_levels)
    return Hierarchy(root_concept=normalize_concept(root_label), levels=levels)


async def comparison(concept_a: str, concept_b: str,
                     llm: Optional[Callable] = None) -> Comparison:
    """Tabla comparativa con evidencia de AMBOS conceptos (regla dura extendida)."""
    a_ids = await _extract.chunks_for_concepts([concept_a])
    b_ids = await _extract.chunks_for_concepts([concept_b])
    if not a_ids or not b_ids:
        raise UngroundedError(
            f"Faltan chunks fuente para comparar «{concept_a}» / «{concept_b}»."
        )
    a_ch = (await chunk_map.get_chunks_by_ids(a_ids[:1]))[0]
    b_ch = (await chunk_map.get_chunks_by_ids(b_ids[:1]))[0]

    dimensions = [{
        "dimension": "Definición / contexto en la fuente",
        "a_value": a_ch.text[:220],
        "b_value": b_ch.text[:220],
        "evidence": [a_ch.chunk_id, b_ch.chunk_id],   # de AMBOS
    }]
    shared = set(a_ids) & set(b_ids)
    if shared:
        sh = (await chunk_map.get_chunks_by_ids(list(shared)[:1]))[0]
        dimensions.append({
            "dimension": "Aparición conjunta (co-ocurrencia)",
            "a_value": "Ambos conceptos aparecen en el mismo fragmento.",
            "b_value": sh.text[:220],
            "evidence": [sh.chunk_id, sh.chunk_id],
        })

    comp = Comparison(
        concept_ids=[normalize_concept(concept_a), normalize_concept(concept_b)],
        dimensions=dimensions,
        summary=f"Comparación anclada de «{concept_a}» y «{concept_b}» según el corpus.",
    )
    assert_grounded(comp)
    return comp


async def debate(topic: str, corpus_ids: list[str],
                 llm: Optional[Callable] = None) -> Debate:
    """Posiciones de distintos documentos sobre `topic`, cada una con cita."""
    chunks = await chunk_map.get_chunks_for_docs(corpus_ids)
    terms = [t for t in re.findall(r"[\wáéíóúüñ]+", topic.lower()) if len(t) > 2]

    best_per_doc: dict[str, tuple[int, object]] = {}
    for c in chunks:
        score = sum(c.text.lower().count(t) for t in terms) if terms else 1
        if score <= 0:
            continue
        cur = best_per_doc.get(c.doc_id)
        if cur is None or score > cur[0]:
            best_per_doc[c.doc_id] = (score, c)

    positions = []
    for doc_id, (_score, c) in best_per_doc.items():
        cite = (await resolve_evidence([c.chunk_id]))[0]
        positions.append({
            "author": cite.filename,
            "stance": c.text[:220],
            "evidence": [c.chunk_id],
        })

    deb = Debate(
        topic=topic,
        positions=positions,
        synthesis=(f"{len(positions)} fuente(s) abordan «{topic}». "
                   "Contrasta las posiciones citadas para la síntesis."),
    )
    assert_grounded(deb)   # cada posición con evidencia
    return deb
