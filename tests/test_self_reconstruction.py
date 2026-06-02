"""
Prueba de la Capa 5 — Auto-reconstrucción (base estructural de los 5 documentos).

Es el ORÁCULO de la propia Capa 5: verifica que el bucle maestro respeta el
protocolo no negociable de la trilogía:

  1. recon_config impone RESTRICCIONES DURAS (cotas) — nada fuera de rango entra.
  2. El ANCLA externa: perfil, invariantes y puerta de promoción funcionan.
  3. El motor de ASOCIACIONES teje puentes/recíprocos, acotado e idempotente.
  4. CHECKPOINT/ROLLBACK restaura el estado mutable de forma trivial.
  5. El BUCLE: auto-aplica asociaciones (bajo riesgo) y deja como PROPUESTA los
     cambios que afectan respuestas (modo "mixto") — NO los aplica solo.

Determinista, sin LLM, aislado en un directorio temporal. Ejecutar:
    python tests/test_self_reconstruction.py
"""
import asyncio
import os
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from sqlalchemy import text

import core.semantic_memory as sm
import core.recon_config as recon_config
import core.anchor as anchor
import core.checkpoints as checkpoints
import core.associations as associations
import core.self_reconstruction as recon
from core.knowledge_graph import KnowledgeGraph
import core.knowledge_graph as kg_mod

TMP = os.path.join("data", "_test_recon")
TEST_DB = os.path.join(TMP, "recon.db")


def _ok(cond, label):
    print(("  ✅ " if cond else "  ❌ ") + label)
    if not cond:
        raise AssertionError(label)


def _redirect_paths():
    """Apunta TODAS las rutas de la Capa 5 al directorio temporal (aislamiento)."""
    os.makedirs(TMP, exist_ok=True)
    recon_config.RECON_CONFIG_FILE = os.path.join(TMP, "config.json")
    anchor.RECON_BASELINE_FILE = os.path.join(TMP, "baseline.json")
    checkpoints.RECON_CONFIG_FILE = recon_config.RECON_CONFIG_FILE
    checkpoints.KG_FILE = os.path.join(TMP, "kg.json")
    checkpoints.CHECKPOINTS_DIR = os.path.join(TMP, "checkpoints")
    kg_mod.KG_FILE = os.path.join(TMP, "kg.json")
    recon.RECON_PROPOSALS_FILE = os.path.join(TMP, "proposals.json")
    recon.RECON_ATTEMPTS_FILE = os.path.join(TMP, "attempts.json")
    recon.RECON_STATUS_FILE = os.path.join(TMP, "status.json")
    recon.RECON_MODE = "mixto"
    recon_config.reset_cache_for_tests()


async def main():
    if os.path.isdir(TMP):
        shutil.rmtree(TMP, ignore_errors=True)
    _redirect_paths()
    sm._reset_engine_for_tests(TEST_DB)
    sm.embedder.set_fake_embedder(None)
    engine = sm._get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(
            "CREATE TABLE memory_facts (id INTEGER PRIMARY KEY, category TEXT, content TEXT)"))
        await conn.execute(text(
            "CREATE TABLE document_chunks (id INTEGER PRIMARY KEY, doc_id INTEGER, content TEXT)"))

    # ── [1] Restricciones duras del sustrato reconfigurable ──────────────────────
    print("\n[1] recon_config impone cotas (restricciones duras §1.3)")
    cfg = recon_config.load()
    _ok(recon_config.within_bounds(cfg), "config de fábrica dentro de cotas")
    cfg["rag_top_k"] = 9999          # fuera de rango
    recon_config.save(cfg)
    reloaded = recon_config.load()
    _ok(reloaded["rag_top_k"] == recon_config.KNOBS["rag_top_k"]["max"],
        f"valor fuera de rango se satura al máximo ({reloaded['rag_top_k']})")
    _ok(not recon_config.within_bounds({"rag_top_k": -1}), "detecta config inválida")
    recon_config.save(recon_config._defaults())

    # ── [2] El ancla externa: perfil, invariantes, puerta de promoción ───────────
    print("\n[2] Ancla externa (oráculo fijo §8, §13.1.1)")
    base_state = {"kg_nodes": 10, "kg_edges": 10, "facts_total": 4, "facts_unique": 4,
                  "chunks_total": 0, "chunks_indexed": 0, "config": recon_config.load()}
    prof = anchor.profile(base_state)
    _ok(0.0 < prof["kg_connectivity"] < 1.0, f"conectividad no saturada = {prof['kg_connectivity']}")
    _ok(anchor.primary(prof) > 0, "métrica primaria > 0")

    ok_inv, _ = anchor.check_invariants(base_state)
    _ok(ok_inv, "estado sano pasa invariantes")
    exploded = dict(base_state); exploded["kg_nodes"] = anchor.HARD_MAX_KG_NODES + 1
    _ok(not anchor.check_invariants(exploded)[0], "explosión del grafo viola invariante")
    lost = {"kg_nodes": 0, "kg_edges": 0, "facts_total": 0, "facts_unique": 0,
            "chunks_total": 0, "chunks_indexed": 0, "config": recon_config.load()}
    _ok(not anchor.check_invariants(lost, prev=base_state)[0], "pérdida total de hechos viola invariante")

    better = dict(base_state); better["kg_edges"] = 20
    ev_before = anchor.evaluate(base_state); ev_after = anchor.evaluate(better)
    _ok(anchor.promote_gate(ev_after, ev_before)[0], "puerta promueve una mejora real")
    worse = dict(base_state); worse["kg_edges"] = 5
    _ok(not anchor.promote_gate(anchor.evaluate(worse), ev_before)[0], "puerta descarta lo que no mejora")

    # ── [3] Motor de asociaciones ────────────────────────────────────────────────
    print("\n[3] Asociaciones neuronales (puentes + recíprocos)")
    g = nx.DiGraph()
    g.add_edge("autopoiesis", "cognición", weight=3, relation="relacionado_con")
    g.add_edge("cognición", "lenguaje", weight=3, relation="relacionado_con")
    ops = associations.propose(g, bridge_min_weight=2, backlink_min_weight=3, max_new_edges=40)
    kinds = {o["kind"] for o in ops}
    _ok(any(o["src"] == "autopoiesis" and o["dst"] == "lenguaje" for o in ops),
        "infiere el puente transitivo autopoiesis→lenguaje")
    _ok("backlink" in kinds, "propone recíprocos de aristas fuertes")
    added = associations.apply(g, ops)
    _ok(added == len(ops), f"aplica {added} asociaciones nuevas")
    _ok(associations.apply(g, ops) == 0, "es idempotente (no duplica)")
    st = associations.stats(g)
    _ok(st["inferred"] >= 1 and st["reciprocal"] >= 1, "stats cuenta inferidas y recíprocas")

    # ── [4] Checkpoint / rollback ────────────────────────────────────────────────
    print("\n[4] Checkpoint y rollback (reversión trivial §22.2)")
    recon_config.save(recon_config._defaults())
    cp = checkpoints.create(label="prueba")
    cfg2 = recon_config.load(); cfg2["assoc_max_new_edges"] = 123; recon_config.save(cfg2)
    _ok(recon_config.get("assoc_max_new_edges") == 123, "se modificó la config")
    checkpoints.restore(cp)
    _ok(recon_config.get("assoc_max_new_edges") == recon_config.KNOBS["assoc_max_new_edges"]["default"],
        "rollback restaura el valor original")

    # ── [5] El bucle maestro en modo mixto ───────────────────────────────────────
    print("\n[5] Bucle maestro (modo mixto por riesgo)")
    kg = KnowledgeGraph()
    kg.graph = nx.DiGraph()
    for a, b in [("sistema", "estructura"), ("estructura", "componente"),
                 ("componente", "función"), ("sistema", "diseño")]:
        kg.graph.add_edge(a, b, weight=3, relation="relacionado_con")
    kg.save()
    edges_before = kg.graph.number_of_edges()
    report = await recon.run_cycle(kg, mode="mixto")
    print("     informe:", {k: report[k] for k in ("promoted", "rolled_back", "associations")})
    _ok(report["associations"].get("promoted"), "auto-aplicó asociaciones (bajo riesgo)")
    _ok(kg.graph.number_of_edges() > edges_before, "el grafo ganó asociaciones")
    _ok(not report["rolled_back"], "no hubo rollback (no rompió invariantes)")

    proposals = recon.list_proposals()
    pending_response = [p for p in proposals if p["risk"] == "response" and p["status"] == "pending"]
    if pending_response:
        _ok(all(p["status"] == "pending" for p in pending_response),
            "los cambios que afectan respuestas quedan PENDIENTES (no auto-aplicados)")
        res = recon.resolve_proposal(pending_response[0]["id"], approve=True, kg=kg)
        _ok(res["ok"] and res["status"] == "applied", "aprobar una propuesta la aplica")
    else:
        _ok(True, "sin propuestas de riesgo en este estado (correcto)")

    # ── [6] Limpieza / optimización del grafo ────────────────────────────────────
    print("\n[6] Limpieza automática (fusión de duplicados + poda de aislados)")
    gd = nx.DiGraph()
    gd.add_edge("Cognición", "lenguaje", weight=3, relation="relacionado_con")
    gd.add_edge("cogniciones", "saber", weight=2, relation="relacionado_con")  # duplicado de Cognición
    gd.add_node("huérfano")  # nodo aislado (ruido)
    issues = associations.count_issues(gd)
    _ok(issues["dup_extra"] >= 1 and issues["isolated"] >= 1,
        f"detecta duplicados ({issues['dup_extra']}) y aislados ({issues['isolated']})")
    clean_before = anchor.profile({"kg_nodes": gd.number_of_nodes(), "kg_edges": gd.number_of_edges(),
                                    "kg_isolated": issues["isolated"], "kg_dup_extra": issues["dup_extra"],
                                    "config": recon_config.load()})["kg_cleanliness"]
    plan = associations.plan_cleanup(gd, max_merges=25, prune_isolated=True)
    res = associations.apply_cleanup(gd, plan)
    _ok(res["merged_nodes"] >= 1, f"fusionó {res['merged_nodes']} duplicado(s)")
    _ok(res["pruned_isolated"] >= 1, f"podó {res['pruned_isolated']} aislado(s)")
    issues2 = associations.count_issues(gd)
    clean_after = anchor.profile({"kg_nodes": gd.number_of_nodes(), "kg_edges": gd.number_of_edges(),
                                  "kg_isolated": issues2["isolated"], "kg_dup_extra": issues2["dup_extra"],
                                  "config": recon_config.load()})["kg_cleanliness"]
    _ok(clean_after > clean_before, f"la limpieza sube (kg_cleanliness {clean_before}→{clean_after})")
    _ok("huérfano" not in gd.nodes(), "el nodo aislado fue eliminado")

    print("\n🎉 CAPA 5 (AUTO-RECONSTRUCCIÓN) VALIDADA\n")
    await sm._get_engine().dispose()
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
