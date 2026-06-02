"""
Capa 5 — EL ANCLA EXTERNA (oráculo fijo, invariantes y presupuesto).

Esta es la pieza no negociable de toda la trilogía (Vol. III §20, §22; plan
operativo §1.2, §1.3, §10). Es lo que el sistema puede LEER pero NUNCA reescribir:

  • ORÁCULO PERFILADO (§13.1.1): en vez de un solo número devuelve un VECTOR de
    salud por sub-capacidad, para que la exploración sepa dónde hay margen.
  • INVARIANTES INVIOLABLES (§1.3): propiedades que ningún cambio puede romper.
    Se comprueban FUERA del bucle de auto-edición.
  • PUERTA DE PROMOCIÓN (§8): un cambio se promueve SOLO si mejora la línea base,
    no introduce regresión y no viola invariantes ni restricciones duras.
  • LÍNEA BASE: la referencia contra la que se mide "¿mejoró?".

Los PESOS del perfil y las COTAS de los invariantes están CODIFICADOS AQUÍ como
constantes, no en core/recon_config.py. Es deliberado: si el sistema pudiera
editar su propio examen, "siempre aprobaría" (Goodhart sobre uno mismo, Vol. III
§21.2). El ancla es externa precisamente porque está fuera de lo reconfigurable.

Es 100% determinista: no llama a ningún LLM (respeta cuota y la GPU de 2 GB).
"""
import json
import os
import sys

from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RECON_BASELINE_FILE
import core.recon_config as recon_config
from core.semantic_memory import _get_engine

# ── Constantes del oráculo (FIJAS — el sistema no las puede tocar) ───────────────
# Pesos de cada sub-capacidad en la métrica primaria. Suman 1.0.
PROFILE_WEIGHTS = {
    "memory_health":   0.18,   # memoria sin duplicados ni ruido
    "kg_connectivity": 0.22,   # densidad de asociaciones del grafo (lo que el usuario llama "asociaciones neuronales")
    "kg_cleanliness":  0.15,   # grafo limpio: sin nodos aislados ni conceptos duplicados
    "doc_coverage":    0.22,   # fragmentos indexados / totales
    "index_freshness": 0.13,   # 1.0 si no hay pendientes de indexar
    "config_sanity":   0.10,   # parámetros dentro de sus cotas
}
# Mejora mínima neta sobre la línea base para promover (evita aceptar ruido, §8).
MIN_IMPROVEMENT = 0.001
# Caída de la primaria que se considera regresión a vigilar (§8 VIGILANCIA).
REGRESSION_TOLERANCE = 0.02
# Tope ABSOLUTO de nodos del grafo: invariante duro independiente de kg_max_nodes.
HARD_MAX_KG_NODES = 5000


# ── Estado del sistema (lo que el oráculo mide) ──────────────────────────────────
async def snapshot_state(kg=None) -> dict:
    """Reúne las señales que el oráculo necesita. Tolerante a tablas ausentes
    (para poder evaluarse en un entorno de pruebas mínimo)."""
    st: dict = {
        "kg_nodes": 0, "kg_edges": 0, "kg_isolated": 0, "kg_dup_extra": 0,
        "facts_total": 0, "facts_unique": 0,
        "chunks_total": 0, "chunks_indexed": 0,
        "config": recon_config.load(),
    }
    if kg is not None:
        try:
            st["kg_nodes"] = kg.graph.number_of_nodes()
            st["kg_edges"] = kg.graph.number_of_edges()
            import core.associations as associations
            issues = associations.count_issues(kg.graph)
            st["kg_isolated"] = issues["isolated"]
            st["kg_dup_extra"] = issues["dup_extra"]
        except Exception:
            pass

    engine = _get_engine()
    async with engine.connect() as conn:
        st["facts_total"]    = await _scalar(conn, "SELECT COUNT(*) FROM memory_facts")
        st["facts_unique"]   = await _scalar(
            conn, "SELECT COUNT(*) FROM (SELECT 1 FROM memory_facts GROUP BY category, LOWER(content))")
        st["chunks_total"]   = await _scalar(conn, "SELECT COUNT(*) FROM document_chunks")
        st["chunks_indexed"] = await _scalar(
            conn, "SELECT COUNT(DISTINCT chunk_id) FROM chunk_embeddings")
    return st


async def _scalar(conn, sql: str) -> int:
    try:
        return (await conn.execute(text(sql))).scalar() or 0
    except Exception:
        return 0


# ── El oráculo perfilado (§13.1.1) ───────────────────────────────────────────────
def profile(state: dict) -> dict:
    """Vector de salud por sub-capacidad, cada componente en [0,1]."""
    facts_total = state.get("facts_total", 0)
    facts_unique = state.get("facts_unique", 0)
    chunks_total = state.get("chunks_total", 0)
    chunks_indexed = state.get("chunks_indexed", 0)
    nodes = state.get("kg_nodes", 0)
    edges = state.get("kg_edges", 0)

    # memoria sana = proporción de hechos no duplicados
    memory_health = 1.0 if facts_total == 0 else round(facts_unique / facts_total, 4)

    # conectividad del grafo: aristas por nodo, normalizada a un objetivo de ~3.
    # Es la medida directa de "cuántas asociaciones por concepto" existen.
    if nodes == 0:
        kg_connectivity = 0.0
    else:
        kg_connectivity = round(min(1.0, (edges / nodes) / 3.0), 4)

    # limpieza del grafo: penaliza nodos aislados y conceptos duplicados.
    isolated = state.get("kg_isolated", 0)
    dup_extra = state.get("kg_dup_extra", 0)
    if nodes == 0:
        kg_cleanliness = 1.0
    else:
        kg_cleanliness = round(max(0.0, 1.0 - (isolated + dup_extra) / nodes), 4)

    doc_coverage = 1.0 if chunks_total == 0 else round(chunks_indexed / chunks_total, 4)
    index_freshness = 1.0 if chunks_total == 0 else round(
        min(1.0, chunks_indexed / chunks_total), 4)
    config_sanity = 1.0 if recon_config.within_bounds(state.get("config", {})) else 0.0

    return {
        "memory_health":   memory_health,
        "kg_connectivity": kg_connectivity,
        "kg_cleanliness":  kg_cleanliness,
        "doc_coverage":    doc_coverage,
        "index_freshness": index_freshness,
        "config_sanity":   config_sanity,
    }


def primary(prof: dict) -> float:
    """Métrica primaria escalar = media ponderada del perfil (con pesos FIJOS)."""
    return round(sum(prof.get(k, 0.0) * w for k, w in PROFILE_WEIGHTS.items()), 6)


# ── Invariantes inviolables (§1.3) ───────────────────────────────────────────────
def check_invariants(state: dict, prev: dict | None = None) -> tuple[bool, list[str]]:
    """Comprueba propiedades que ningún cambio puede violar. Devuelve (ok, fallos)."""
    fails: list[str] = []

    # 1. El grafo no puede explotar más allá del tope absoluto.
    if state.get("kg_nodes", 0) > HARD_MAX_KG_NODES:
        fails.append(f"kg_nodes>{HARD_MAX_KG_NODES}")

    # 2. Los parámetros deben estar dentro de sus restricciones duras.
    if not recon_config.within_bounds(state.get("config", {})):
        fails.append("config fuera de cotas")

    # 3. Sin pérdida de datos: un cambio no puede vaciar la memoria/corpus
    #    que existía antes (no destruir conocimiento previo).
    if prev is not None:
        if prev.get("facts_total", 0) > 0 and state.get("facts_total", 0) == 0:
            fails.append("se perdieron todos los hechos")
        if prev.get("chunks_total", 0) > 0 and state.get("chunks_total", 0) == 0:
            fails.append("se perdió todo el corpus")
        # El grafo no debe perder más del 50% de sus nodos de golpe.
        pn = prev.get("kg_nodes", 0)
        if pn > 0 and state.get("kg_nodes", 0) < pn * 0.5:
            fails.append("colapso del grafo (>50% nodos perdidos)")

    return (len(fails) == 0, fails)


# ── Puerta de promoción (§8) ─────────────────────────────────────────────────────
def evaluate(state: dict, prev: dict | None = None, baseline: float | None = None) -> dict:
    """Evalúa un estado contra el oráculo fijo. NO decide aún promover; reporta."""
    prof = profile(state)
    prim = primary(prof)
    inv_ok, inv_fails = check_invariants(state, prev)
    base = baseline if baseline is not None else load_baseline()
    regression = (prim < base - REGRESSION_TOLERANCE)
    return {
        "profile": prof,
        "primary": prim,
        "baseline": base,
        "invariants_ok": inv_ok,
        "invariant_failures": inv_fails,
        "regression": regression,
    }


def promote_gate(eval_after: dict, eval_before: dict) -> tuple[bool, str]:
    """¿Se debe promover el cambio? SOLO si (a) mejora estrictamente la primaria,
    (b) sin regresión, (c) sin violar invariantes (§8)."""
    if not eval_after["invariants_ok"]:
        return False, "invariante violado: " + ", ".join(eval_after["invariant_failures"])
    if eval_after["regression"]:
        return False, "regresión detectada"
    if eval_after["primary"] < eval_before["primary"] + MIN_IMPROVEMENT:
        return False, "no mejora la línea base (cambio descartado)"
    return True, "mejora verificada"


def deficit_ranking(prof: dict) -> list[tuple[str, float]]:
    """Sub-capacidades ordenadas por margen de mejora (§13.2.1): dónde duele más."""
    return sorted(((k, round(1.0 - prof.get(k, 0.0), 4)) for k in PROFILE_WEIGHTS),
                  key=lambda x: x[1], reverse=True)


# ── Línea base (referencia de "¿mejoró?") ────────────────────────────────────────
def load_baseline() -> float:
    if os.path.exists(RECON_BASELINE_FILE):
        try:
            with open(RECON_BASELINE_FILE, encoding="utf-8") as f:
                return float(json.load(f).get("primary", 0.0))
        except Exception:
            return 0.0
    return 0.0


def save_baseline(primary_value: float, prof: dict | None = None) -> None:
    """Actualiza la línea base. Solo debe llamarse tras una promoción legítima:
    la base sube, nunca se fija a dedo (anti-autoengaño, §13.5)."""
    os.makedirs(os.path.dirname(RECON_BASELINE_FILE), exist_ok=True)
    with open(RECON_BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump({"primary": round(primary_value, 6), "profile": prof or {}}, f,
                  ensure_ascii=False, indent=2)
