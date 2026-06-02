import json
import re
from pathlib import Path
import networkx as nx
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import KG_FILE

MAX_KG_NODES = 400
SAVE_EVERY_N = 5

STOP_WORDS = {
    'el', 'la', 'los', 'las', 'de', 'del', 'en', 'y', 'a', 'un', 'una',
    'que', 'es', 'se', 'no', 'con', 'por', 'para', 'como', 'más', 'su',
    'the', 'is', 'are', 'was', 'be', 'to', 'of', 'and', 'or', 'it',
    'this', 'that', 'with', 'for', 'as', 'an', 'in', 'on', 'at', 'by',
    'from', 'but', 'not', 'have', 'has', 'had', 'do', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'can', 'also', 'its',
}


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._dirty = 0
        self._load()

    def _load(self):
        path = Path(KG_FILE)
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data, directed=True, multigraph=False)
            except Exception:
                self.graph = nx.DiGraph()

    def save(self):
        path = Path(KG_FILE)
        path.parent.mkdir(exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nx.node_link_data(self.graph), f, ensure_ascii=False, indent=2)

    def _extract_concepts(self, text: str) -> list[str]:
        proper = re.findall(r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]{2,}\b', text)
        tech = re.findall(r'\b[a-z]{2,}_[a-z_]+\b|\b[a-z]{2,}[A-Z][a-zA-Z]+\b', text)
        long_words = [
            w.lower() for w in re.findall(r'\b[a-záéíóúA-ZÁÉÍÓÚ]{6,}\b', text)
            if w.lower() not in STOP_WORDS
        ]
        combined = list({c for c in (proper + tech + long_words) if c.lower() not in STOP_WORDS})
        return combined[:12]

    def update_from_conversation(self, user_text: str, assistant_text: str):
        full_text = user_text + " " + assistant_text
        concepts = self._extract_concepts(full_text)

        for concept in concepts:
            if concept not in self.graph:
                self.graph.add_node(concept, weight=1, type="concept")
            else:
                self.graph.nodes[concept]["weight"] = self.graph.nodes[concept].get("weight", 1) + 1

        sentences = re.split(r'[.!?;\n]', full_text)
        for sentence in sentences:
            present = [c for c in concepts if c in sentence or c.lower() in sentence.lower()]
            for i, c1 in enumerate(present):
                for c2 in present[i + 1:]:
                    if c1 != c2:
                        if self.graph.has_edge(c1, c2):
                            self.graph[c1][c2]["weight"] = self.graph[c1][c2].get("weight", 1) + 1
                        else:
                            self.graph.add_edge(c1, c2, weight=1, relation="relacionado_con")

        self._dirty += 1
        if self._dirty >= SAVE_EVERY_N:
            self._prune_if_needed()
            self.save()
            self._dirty = 0

    def _max_nodes(self) -> int:
        """Tope de nodos: lo puede auto-ajustar la Capa 5 (recon_config), con
        MAX_KG_NODES como defecto si aún no hay config."""
        try:
            import core.recon_config as recon_config
            return int(recon_config.get("kg_max_nodes"))
        except Exception:
            return MAX_KG_NODES

    def _prune_if_needed(self):
        cap = self._max_nodes()
        if self.graph.number_of_nodes() <= cap:
            return
        sorted_nodes = sorted(
            self.graph.nodes(data=True),
            key=lambda x: x[1].get("weight", 1),
            reverse=True,
        )
        keep = {n for n, _ in sorted_nodes[:cap]}
        self.graph.remove_nodes_from([n for n in list(self.graph.nodes()) if n not in keep])

    def get_context(self, query: str, max_concepts: int = 6) -> str:
        query_concepts = self._extract_concepts(query)
        if not query_concepts or self.graph.number_of_nodes() == 0:
            return ""

        relevant_nodes = set()
        for concept in query_concepts:
            for node in self.graph.nodes():
                if concept.lower() in node.lower() or node.lower() in concept.lower():
                    relevant_nodes.add(node)
                    neighbors = list(self.graph.successors(node)) + list(self.graph.predecessors(node))
                    relevant_nodes.update(neighbors[:3])

        if not relevant_nodes:
            top_nodes = sorted(
                self.graph.nodes(data=True),
                key=lambda x: x[1].get("weight", 1),
                reverse=True,
            )
            relevant_nodes = {n for n, _ in top_nodes[:max_concepts]}

        lines = []
        for node in list(relevant_nodes)[:max_concepts]:
            if node in self.graph:
                succs = list(self.graph.successors(node))[:3]
                if succs:
                    lines.append(f"- {node} → {', '.join(succs)}")
                else:
                    lines.append(f"- {node} (concepto conocido)")

        return "\n".join(lines) if lines else ""

    def to_vis_data(self) -> dict:
        nodes = []
        edges = []

        top_nodes = sorted(
            self.graph.nodes(data=True),
            key=lambda x: x[1].get("weight", 1),
            reverse=True,
        )[:80]
        node_ids = {n: i for i, (n, _) in enumerate(top_nodes)}

        for node, data in top_nodes:
            w = data.get("weight", 1)
            nodes.append({
                "id": node_ids[node],
                "label": node,
                "value": min(w * 3, 30),
                "title": f"{node} (frecuencia: {w})",
            })

        seen_edges = set()
        for src, dst, data in self.graph.edges(data=True):
            if src in node_ids and dst in node_ids:
                key = (node_ids[src], node_ids[dst])
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append({
                        "from": node_ids[src],
                        "to": node_ids[dst],
                        "label": data.get("relation", ""),
                        "value": data.get("weight", 1),
                    })

        return {"nodes": nodes, "edges": edges}
