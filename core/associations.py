"""
Capa 5 — MOTOR DE ASOCIACIONES NEURONALES CONTINUAS.

Es el sustrato 2 (memoria/grafo) de Vol. III §18.2: inspeccionable, de bajo
riesgo y reversible. Hace lo que el usuario pidió —"asociaciones constantes y
autónomas, sin que se lo pida"— tejiendo sobre el grafo de conocimiento vínculos
que la conversación no creó explícitamente pero que se siguen de su estructura:

  • PUENTES TRANSITIVOS:  si A→B y B→C son fuertes pero no existe A→C, se infiere
    la asociación indirecta A→C (relación "asociado_indirecto"). Es el equivalente
    a "razonar" un vínculo de segundo orden entre conceptos.
  • RECÍPROCOS:           si A→B es fuerte pero no existe B→A, se crea el vínculo
    de vuelta (relación "asociado_reciproco"), densificando la red.

El motor solo PROPONE operaciones sobre una COPIA del grafo; quien las aplica y
evalúa contra el oráculo es el bucle maestro (core/self_reconstruction.py). Como
solo AÑADE aristas entre nodos ya existentes, nunca puede inflar el número de
nodos ni romper el tope del grafo: es seguro por construcción.

Determinista: cero LLM, cero red, cero GPU.

AMPLIACIÓN (limpieza/optimización automática): además de TEJER vínculos, el motor
MANTIENE el grafo —fusiona conceptos duplicados (el mismo concepto escrito con
tildes/mayúsculas/plural distintos pasa a ser un solo nodo, más conectado) y poda
nodos aislados (ruido)—. Reconstrucción, optimización y limpieza dejan de ser
tareas separadas: son una sola transformación del grafo, evaluada de una vez
contra el oráculo (que ahora premia también la "limpieza", `kg_cleanliness`).
"""
import unicodedata


def normalize_concept(s: str) -> str:
    """Clave canónica de un concepto: sin tildes, en minúsculas y sin plural simple.
    Permite detectar que 'Cognición', 'cognicion' y 'cogniciones' son lo mismo."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower().strip()
    if len(s) > 4 and s.endswith("es"):
        s = s[:-2]
    elif len(s) > 3 and s.endswith("s"):
        s = s[:-1]
    return s


def propose(graph, *, bridge_min_weight: int = 2, backlink_min_weight: int = 3,
            max_new_edges: int = 40) -> list[dict]:
    """Examina el grafo y devuelve asociaciones candidatas (no las aplica).
    Cada op: {kind, src, dst, weight, relation}."""
    if graph is None or graph.number_of_nodes() == 0:
        return []

    candidates: list[dict] = []

    # 1. Puentes transitivos A→B→C  ⇒  A→C
    for a in list(graph.nodes()):
        for b in graph.successors(a):
            w_ab = graph[a][b].get("weight", 1)
            if w_ab < bridge_min_weight:
                continue
            for c in graph.successors(b):
                if c == a or c == b:
                    continue
                w_bc = graph[b][c].get("weight", 1)
                if w_bc < bridge_min_weight:
                    continue
                if graph.has_edge(a, c):
                    continue
                candidates.append({
                    "kind": "bridge", "src": a, "dst": c,
                    "weight": min(w_ab, w_bc),
                    "relation": "asociado_indirecto",
                })

    # 2. Recíprocos de aristas fuertes A→B  ⇒  B→A
    for a, b, data in list(graph.edges(data=True)):
        w = data.get("weight", 1)
        if w < backlink_min_weight:
            continue
        if graph.has_edge(b, a):
            continue
        candidates.append({
            "kind": "backlink", "src": b, "dst": a,
            "weight": max(1, w - 1),
            "relation": "asociado_reciproco",
        })

    # Deduplicar (puede haber el mismo puente por varios caminos) y priorizar por
    # fuerza: las asociaciones más respaldadas primero (explotación, §13.2.1).
    seen: set = set()
    unique: list[dict] = []
    for op in sorted(candidates, key=lambda o: o["weight"], reverse=True):
        key = (op["src"], op["dst"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(op)

    return unique[:max_new_edges]


def apply(graph, ops: list[dict]) -> int:
    """Aplica las operaciones de asociación a un grafo (in place). Devuelve cuántas
    aristas nuevas se crearon. Idempotente: no duplica aristas existentes."""
    added = 0
    for op in ops:
        src, dst = op["src"], op["dst"]
        if not graph.has_node(src) or not graph.has_node(dst):
            continue
        if graph.has_edge(src, dst):
            continue
        graph.add_edge(src, dst, weight=op.get("weight", 1),
                       relation=op.get("relation", "asociado"))
        added += 1
    return added


# ── LIMPIEZA / OPTIMIZACIÓN del grafo ────────────────────────────────────────────
def count_issues(graph) -> dict:
    """Cuenta el "desorden" del grafo: nodos aislados y conceptos duplicados.
    Es la señal que alimenta la sub-capacidad de limpieza del oráculo."""
    if graph is None or graph.number_of_nodes() == 0:
        return {"isolated": 0, "dup_extra": 0}
    isolated = sum(1 for n in graph.nodes() if graph.degree(n) == 0)
    groups: dict[str, int] = {}
    for n in graph.nodes():
        groups[normalize_concept(n)] = groups.get(normalize_concept(n), 0) + 1
    dup_extra = sum(c - 1 for c in groups.values() if c > 1)
    return {"isolated": isolated, "dup_extra": dup_extra}


def plan_cleanup(graph, *, max_merges: int = 25, prune_isolated: bool = True) -> dict:
    """Plan de limpieza: qué duplicados fusionar y qué aislados podar.
    No muta el grafo; devuelve un plan aplicable y reversible (vía checkpoint)."""
    if graph is None or graph.number_of_nodes() == 0:
        return {"merges": [], "prune_isolated": prune_isolated}
    groups: dict[str, list] = {}
    for n in graph.nodes():
        groups.setdefault(normalize_concept(n), []).append(n)
    merges = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        # Conservar el nodo más "pesado" (más frecuente); fusionar el resto en él.
        members.sort(key=lambda n: graph.nodes[n].get("weight", 1), reverse=True)
        merges.append({"keep": members[0], "drop": members[1:]})
    merges = merges[:max_merges]
    return {"merges": merges, "prune_isolated": prune_isolated}


def apply_cleanup(graph, plan: dict) -> dict:
    """Aplica el plan de limpieza in place. Fusiona duplicados (redirigiendo sus
    aristas al nodo conservado → más conexiones) y poda aislados. Devuelve stats."""
    merged_nodes = redirected = 0
    for m in plan.get("merges", []):
        keep = m["keep"]
        if keep not in graph:
            continue
        for d in m["drop"]:
            if d == keep or d not in graph:
                continue
            for succ in list(graph.successors(d)):
                if succ != keep and not graph.has_edge(keep, succ):
                    graph.add_edge(keep, succ, **graph[d][succ]); redirected += 1
            for pred in list(graph.predecessors(d)):
                if pred != keep and not graph.has_edge(pred, keep):
                    graph.add_edge(pred, keep, **graph[pred][d]); redirected += 1
            graph.nodes[keep]["weight"] = (graph.nodes[keep].get("weight", 1)
                                           + graph.nodes[d].get("weight", 1))
            graph.remove_node(d); merged_nodes += 1
    pruned = 0
    if plan.get("prune_isolated"):
        isolated = [n for n in list(graph.nodes()) if graph.degree(n) == 0]
        graph.remove_nodes_from(isolated)
        pruned = len(isolated)
    return {"merged_nodes": merged_nodes, "redirected_edges": redirected, "pruned_isolated": pruned}


def stats(graph) -> dict:
    """Resumen de la densidad asociativa del grafo (para /recon/associations)."""
    if graph is None or graph.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "edges_per_node": 0.0,
                "inferred": 0, "reciprocal": 0}
    inferred = reciprocal = 0
    for _, _, d in graph.edges(data=True):
        rel = d.get("relation", "")
        if rel == "asociado_indirecto":
            inferred += 1
        elif rel == "asociado_reciproco":
            reciprocal += 1
    n = graph.number_of_nodes()
    e = graph.number_of_edges()
    return {
        "nodes": n, "edges": e,
        "edges_per_node": round(e / n, 3) if n else 0.0,
        "inferred": inferred, "reciprocal": reciprocal,
    }
