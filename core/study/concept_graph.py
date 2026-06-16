"""
concept_graph.py — Grafo de conceptos de EstudIA (pista integrada).

Grafo ligero (sin dependencia de networkx para evitar choques de versión): nodos =
conceptos normalizados con sus chunks fuente; aristas = relaciones con evidencia
(chunk_id) y peso. Persiste en JSON (mismo patrón node-link que usa el KG del
proyecto, pero formato propio y estable). Provee BFS por niveles y export a Mermaid.

La deduplicación usa una clave canónica (NFKD + minúsculas + plural→singular),
réplica desacoplada de associations.normalize_concept para no acoplarnos a internals
privados (riesgo señalado en el diseño).
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import DATA_DIR  # noqa: E402

_graph_path: str = str(DATA_DIR / "study_graph.json")


def use_path(path: str) -> None:
    """Solo para pruebas: redirige el archivo de persistencia del grafo."""
    global _graph_path
    _graph_path = path


def normalize_concept(s: str) -> str:
    """Clave canónica: sin acentos, minúsculas, plural→singular simple (es)."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > 4:
        if s.endswith("es"):
            s = s[:-2]
        elif s.endswith("s"):
            s = s[:-1]
    return s


class ConceptGraph:
    def __init__(self) -> None:
        # key -> {"label": surface, "freq": int, "source_chunks": set[str]}
        self.nodes: dict[str, dict] = {}
        # lista de {"src","dst","relation","evidence","weight"}
        self.edges: list[dict] = []
        self._edge_idx: dict[tuple, int] = {}

    # -- construcción ------------------------------------------------------- #
    def add_concept(self, label: str, source_chunk: str | None = None) -> str:
        key = normalize_concept(label)
        if not key:
            return ""
        node = self.nodes.get(key)
        if node is None:
            node = {"label": label.strip(), "freq": 0, "source_chunks": set()}
            self.nodes[key] = node
        node["freq"] += 1
        if source_chunk is not None:
            node["source_chunks"].add(str(source_chunk))
        return key

    def add_edge(self, src_label: str, dst_label: str, relation: str,
                 evidence: str, weight: int = 1) -> None:
        a, b = normalize_concept(src_label), normalize_concept(dst_label)
        if not a or not b or a == b:
            return
        self.add_concept(src_label)
        self.add_concept(dst_label)
        ekey = (a, b, relation)
        idx = self._edge_idx.get(ekey)
        if idx is None:
            self.edges.append({"src": a, "dst": b, "relation": relation,
                               "evidence": str(evidence), "weight": weight})
            self._edge_idx[ekey] = len(self.edges) - 1
        else:
            self.edges[idx]["weight"] += weight

    # -- consulta ----------------------------------------------------------- #
    def _adjacency(self, relations: set[str] | None = None) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {k: set() for k in self.nodes}
        for e in self.edges:
            if relations and e["relation"] not in relations:
                continue
            adj.setdefault(e["src"], set()).add(e["dst"])
            adj.setdefault(e["dst"], set()).add(e["src"])  # co-ocurrencia: no dirigida
        return adj

    def bfs_levels(self, root_label: str, relations: set[str] | None = None,
                   max_levels: int = 5) -> list[list[str]]:
        root = normalize_concept(root_label)
        if root not in self.nodes:
            return []
        adj = self._adjacency(relations)
        seen = {root}
        levels = [[root]]
        frontier = [root]
        while frontier and len(levels) < max_levels:
            nxt = []
            for n in frontier:
                for m in sorted(adj.get(n, ())):
                    if m not in seen:
                        seen.add(m)
                        nxt.append(m)
            if not nxt:
                break
            levels.append(nxt)
            frontier = nxt
        return levels

    def label_of(self, key: str) -> str:
        return self.nodes.get(key, {}).get("label", key)

    def to_mermaid(self, max_edges: int = 60) -> str:
        ids = {k: f"n{i}" for i, k in enumerate(self.nodes)}
        lines = ["graph TD"]
        for k, nid in ids.items():
            safe = self.label_of(k).replace('"', "'")
            lines.append(f'  {nid}["{safe}"]')
        for e in sorted(self.edges, key=lambda x: -x["weight"])[:max_edges]:
            if e["src"] in ids and e["dst"] in ids:
                rel = e["relation"]
                lines.append(f'  {ids[e["src"]]} -->|{rel}| {ids[e["dst"]]}')
        return "\n".join(lines)

    # -- persistencia ------------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "nodes": [{"key": k, "label": v["label"], "freq": v["freq"],
                       "source_chunks": sorted(v["source_chunks"])}
                      for k, v in self.nodes.items()],
            "edges": self.edges,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptGraph":
        g = cls()
        for n in d.get("nodes", []):
            g.nodes[n["key"]] = {"label": n["label"], "freq": n.get("freq", 1),
                                 "source_chunks": set(n.get("source_chunks", []))}
        for e in d.get("edges", []):
            g.edges.append(e)
            g._edge_idx[(e["src"], e["dst"], e["relation"])] = len(g.edges) - 1
        return g

    def save(self, path: str | None = None) -> None:
        p = path or _graph_path
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | None = None) -> "ConceptGraph":
        p = path or _graph_path
        if not os.path.exists(p):
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
