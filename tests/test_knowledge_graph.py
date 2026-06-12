"""
Prueba de la Capa — Grafo de conocimiento (core/knowledge_graph.py).

Valida el MECANISMO real del grafo sobre una instancia AISLADA:
  - extracción de conceptos (_extract_concepts: nombres propios, snake/camelCase,
    palabras largas; descarta stop-words; tope de 12),
  - actualización desde una conversación (update_from_conversation: crea nodos con
    peso/tipo, refuerza pesos repetidos y teje aristas entre co-ocurrentes),
  - recuperación de contexto (get_context: string con los conceptos relevantes),
  - poda (_prune_if_needed: respeta el tope de nodos quedándose con los de más peso).

No toca el knowledge_graph.json real: se redirige KG_FILE a una ruta temporal y se
borra al terminar; además se resetea la caché de recon_config que consulta la poda.

Ejecutar:  python tests/test_knowledge_graph.py
"""
import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.knowledge_graph as kg
import core.recon_config as recon_config


def _ok(cond, label):
    print(("  OK  " if cond else "  XX  ") + label)
    if not cond:
        raise AssertionError(label)


def main():
    # ── AISLAMIENTO ───────────────────────────────────────────────────────────
    # KnowledgeGraph._load()/save() leen el módulo-global KG_FILE. Lo apuntamos a
    # una ruta temporal inexistente para que la instancia nazca vacía y nunca
    # toque el grafo real. Guardamos el original para restaurarlo.
    _orig_kg_file = kg.KG_FILE
    _orig_max_nodes = kg.MAX_KG_NODES
    tmp_dir = tempfile.mkdtemp(prefix="kg_test_")
    tmp_kg_file = os.path.join(tmp_dir, "kg_test.json")
    kg.KG_FILE = tmp_kg_file
    # La poda consulta recon_config.get("kg_max_nodes"); reseteamos su caché para
    # no arrastrar estado de una corrida previa.
    recon_config.reset_cache_for_tests()

    try:
        print("\n[1] Instancia aislada nace vacía (no carga el grafo real)")
        g = kg.KnowledgeGraph()
        _ok(g.graph.number_of_nodes() == 0, "grafo nuevo sin nodos")
        _ok(g.graph.number_of_edges() == 0, "grafo nuevo sin aristas")
        _ok(not os.path.exists(tmp_kg_file),
            "no se crea archivo en el constructor (KG_FILE temporal inexistente)")

        print("\n[2] Extracción de conceptos (_extract_concepts)")
        concepts = g._extract_concepts(
            "Maturana describe la autopoiesis y el sistema autopoietico vivo."
        )
        cl = {c.lower() for c in concepts}
        _ok("maturana" in cl, "captura nombre propio 'Maturana'")
        _ok("autopoiesis" in cl, "captura palabra larga 'autopoiesis'")
        _ok(all(c.lower() not in kg.STOP_WORDS for c in concepts),
            "no incluye stop-words")
        _ok("la" not in cl and "el" not in cl,
            "descarta palabras cortas/stop-words ('la', 'el')")
        _ok(len(concepts) <= 12, "respeta el tope de 12 conceptos")

        # Tope duro: con muchísimas palabras largas distintas no excede 12.
        many = " ".join(f"conceptolargo{i:02d}" for i in range(40))
        _ok(len(g._extract_concepts(many)) <= 12,
            "nunca devuelve más de 12 conceptos aunque haya más candidatos")

        # snake_case / camelCase los reconoce la rama 'tech'.
        tech = {c for c in g._extract_concepts("usa get_context y luego getContext")}
        _ok("get_context" in tech, "reconoce snake_case 'get_context'")
        _ok("getContext" in tech, "reconoce camelCase 'getContext'")

        print("\n[3] update_from_conversation: crea nodos con peso y tipo")
        g.update_from_conversation(
            "Quiero entender la autopoiesis",
            "La autopoiesis es la autoorganizacion de un sistema vivo",
        )
        _ok("autopoiesis" in g.graph, "el concepto 'autopoiesis' quedó como nodo")
        node = g.graph.nodes["autopoiesis"]
        _ok(node.get("type") == "concept", "el nodo tiene type='concept'")
        _ok(node.get("weight", 0) >= 1, "el nodo tiene weight >= 1")

        print("\n[4] Refuerzo de peso al repetir un concepto")
        w_before = g.graph.nodes["autopoiesis"]["weight"]
        g.update_from_conversation(
            "Mas sobre autopoiesis",
            "La autopoiesis se sostiene a si misma",
        )
        w_after = g.graph.nodes["autopoiesis"]["weight"]
        _ok(w_after > w_before,
            f"el peso del concepto crece al repetirse ({w_before} -> {w_after})")

        print("\n[5] Aristas entre conceptos co-ocurrentes (relacionado_con)")
        # Frase con dos conceptos largos juntos -> debe aparecer una arista.
        edges_before = g.graph.number_of_edges()
        g.update_from_conversation(
            "Relaciona neurociencia con cognicion",
            "La neurociencia estudia la cognicion del cerebro",
        )
        _ok(g.graph.number_of_edges() > edges_before,
            "se crean aristas entre conceptos que co-ocurren")
        # Comprobar que al menos un par co-ocurrente quedó conectado en algún sentido.
        pair_ok = (
            g.graph.has_edge("neurociencia", "cognicion")
            or g.graph.has_edge("cognicion", "neurociencia")
        )
        _ok(pair_ok, "existe arista entre 'neurociencia' y 'cognicion'")
        # La arista lleva la relación declarada por el módulo.
        if g.graph.has_edge("neurociencia", "cognicion"):
            rel = g.graph["neurociencia"]["cognicion"].get("relation")
        else:
            rel = g.graph["cognicion"]["neurociencia"].get("relation")
        _ok(rel == "relacionado_con", "la arista usa relation='relacionado_con'")

        print("\n[6] get_context: recupera conceptos relevantes a la consulta")
        ctx = g.get_context("Hablame de la autopoiesis")
        _ok(isinstance(ctx, str), "get_context devuelve un string")
        _ok("autopoiesis" in ctx.lower(),
            "el contexto menciona el concepto consultado")

        # Sin conceptos extraíbles de la query -> string vacío.
        _ok(g.get_context("el la de y") == "",
            "query sin conceptos -> contexto vacío")

        # Grafo vacío -> contexto vacío aunque la query tenga conceptos.
        empty = kg.KnowledgeGraph()
        _ok(empty.get_context("autopoiesis") == "",
            "grafo vacío -> contexto vacío")

        print("\n[7] _prune_if_needed: respeta el tope quedándose con los de más peso")
        gp = kg.KnowledgeGraph()
        # Forzamos un tope pequeño e independiente de recon_config para esta prueba.
        gp._max_nodes = lambda: 3
        # Nodos con pesos crecientes; los de menor peso deben caer al podar.
        for i, w in enumerate([1, 2, 3, 4, 5]):
            gp.graph.add_node(f"n{i}", weight=w, type="concept")
        _ok(gp.graph.number_of_nodes() == 5, "preparados 5 nodos antes de podar")
        gp._prune_if_needed()
        _ok(gp.graph.number_of_nodes() == 3,
            "tras podar quedan exactamente 'cap' (3) nodos")
        survivors = set(gp.graph.nodes())
        _ok(survivors == {"n2", "n3", "n4"},
            "sobreviven los 3 nodos de mayor peso (n2,n3,n4); caen los flojos")

        # Por debajo del tope, podar no cambia nada.
        gp._max_nodes = lambda: 100
        before = gp.graph.number_of_nodes()
        gp._prune_if_needed()
        _ok(gp.graph.number_of_nodes() == before,
            "por debajo del tope la poda es no-op")

        print("\n[8] save() escribe en la ruta temporal, no en el grafo real")
        g.save()
        _ok(os.path.exists(tmp_kg_file),
            "save() crea el archivo en la ruta temporal aislada")
        _ok(os.path.basename(tmp_kg_file) != os.path.basename(_orig_kg_file)
            or os.path.dirname(tmp_kg_file) != os.path.dirname(_orig_kg_file),
            "la ruta temporal no es el knowledge_graph.json real")

        print("\nEXITO: KNOWLEDGE_GRAPH VALIDADO")

    finally:
        # ── LIMPIEZA: restaurar estado global y borrar lo temporal ───────────────
        kg.KG_FILE = _orig_kg_file
        kg.MAX_KG_NODES = _orig_max_nodes
        recon_config.reset_cache_for_tests()
        try:
            if os.path.exists(tmp_kg_file):
                os.remove(tmp_kg_file)
            os.rmdir(tmp_dir)
        except OSError:
            pass


if __name__ == "__main__":
    main()
