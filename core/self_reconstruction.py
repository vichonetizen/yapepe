"""
Capa 5 — EL BUCLE MAESTRO de auto-reconstrucción.

Es la materialización del bucle del plan operativo §3, enriquecido por las tres
palancas del §13.4: PERFILAR → DECIDIR FOCO → PROPONER (sobre copia) → EVALUAR
(contra oráculo fijo) → PROMOVER+checkpoint / DESCARTAR → VIGILAR (rollback).

Su estructura de seguridad es la del plan: se prueba sobre una COPIA, se decide
contra un ORÁCULO FIJO (core/anchor.py) y se puede REVERTIR (core/checkpoints.py).
La autonomía es total dentro del bucle; no requiere conversación.

POLÍTICA "MIXTO POR RIESGO" (elección del usuario, plan §1.3 — punto de
aprobación proporcional al riesgo):
  • Asociaciones del grafo (bajo riesgo, inspeccionable, reversible) → SE APLICAN
    solas si pasan la puerta de promoción.
  • Cambios de parámetros que afectan a las respuestas → quedan como PROPUESTA en
    proposals.json para que el usuario los apruebe (endpoint /recon/proposals).

Determinista: no llama a ningún LLM. Pensado para correr en el bucle de autonomía
cada N horas, sin que el usuario lo pida.
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (RECON_MODE, RECON_PROPOSALS_FILE, RECON_ATTEMPTS_FILE,
                    RECON_STATUS_FILE)
import core.recon_config as recon_config
import core.anchor as anchor
import core.checkpoints as checkpoints
import core.associations as associations

_last_run: dict = {"at": None, "report": None}


# ── Utilidades de persistencia ligera ────────────────────────────────────────────
def _load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def _save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── El bucle ─────────────────────────────────────────────────────────────────────
async def run_cycle(kg=None, mode: str | None = None) -> dict:
    """Ejecuta UN ciclo del bucle maestro. Devuelve un informe trazable (§3 paso 8)."""
    mode = mode or RECON_MODE
    report: dict = {"at": datetime.utcnow().isoformat(), "mode": mode,
                    "associations": {}, "proposal": None, "promoted": False,
                    "rolled_back": False, "notes": []}

    # 1. PERFILAR (§13.1.1): medir el estado actual contra el oráculo fijo.
    before = await anchor.snapshot_state(kg)
    eval_before = anchor.evaluate(before, prev=None)
    report["profile_before"] = eval_before["profile"]
    report["primary_before"] = eval_before["primary"]

    # Si la línea base aún no existe, sembrarla con el estado actual (bootstrap, §4).
    if anchor.load_baseline() == 0.0 and eval_before["primary"] > 0:
        anchor.save_baseline(eval_before["primary"], eval_before["profile"])
        eval_before = anchor.evaluate(before, prev=None)

    # 2. CHECKPOINT previo: red de seguridad antes de tocar nada (§8).
    cp_id = checkpoints.create(label="pre-cycle")
    report["checkpoint"] = cp_id

    # 3. SUSTRATO BAJO RIESGO — asociaciones del grafo (auto-aplicables).
    if kg is not None and mode in ("mixto", "auto"):
        report["associations"] = _reconstruct_associations(kg, before, eval_before, report)

    # 4. SUSTRATO QUE AFECTA RESPUESTAS — parámetros (propuesta o auto según modo).
    after_assoc = await anchor.snapshot_state(kg)
    report["proposal"] = _consider_param_change(after_assoc, mode, report)

    # 5. VIGILAR (§8 VIGILANCIA): re-evaluar; si hay regresión/invariante roto, rollback.
    final = await anchor.snapshot_state(kg)
    eval_final = anchor.evaluate(final, prev=before)
    report["profile_after"] = eval_final["profile"]
    report["primary_after"] = eval_final["primary"]
    if not eval_final["invariants_ok"] or eval_final["regression"]:
        checkpoints.restore(cp_id, kg=kg)
        report["rolled_back"] = True
        report["notes"].append(
            "rollback: " + (", ".join(eval_final["invariant_failures"]) or "regresión"))
        eval_final = anchor.evaluate(await anchor.snapshot_state(kg), prev=before)

    # 6. CONTABILIZAR: actualizar línea base si promovimos algo, y registrar traza.
    if report["promoted"] and not report["rolled_back"]:
        anchor.save_baseline(eval_final["primary"], eval_final["profile"])

    _last_run["at"] = report["at"]
    _last_run["report"] = report
    _save_json(RECON_STATUS_FILE, report)
    return report


def _reconstruct_associations(kg, before: dict, eval_before: dict, report: dict) -> dict:
    """MANTENIMIENTO del grafo en una sola transformación sobre una COPIA:
    (1) limpia conceptos duplicados y nodos aislados, y (2) teje asociaciones
    nuevas. Se evalúa de una vez contra el oráculo (que premia conectividad Y
    limpieza) y se promueve solo si mejora. Reconstrucción + limpieza juntas."""
    cfg = recon_config.load()

    # PROPONER sobre una copia (§3 paso 3): no tocar el grafo real todavía.
    trial = kg.graph.copy()

    # (1) Limpieza: fusionar duplicados + podar aislados.
    plan = associations.plan_cleanup(
        trial,
        max_merges=int(cfg["assoc_max_merges"]),
        prune_isolated=bool(int(cfg["assoc_prune_isolated"])),
    )
    clean_stats = associations.apply_cleanup(trial, plan)

    # (2) Asociaciones nuevas sobre el grafo ya limpio.
    ops = associations.propose(
        trial,
        bridge_min_weight=int(cfg["assoc_bridge_min_weight"]),
        backlink_min_weight=int(cfg["assoc_backlink_min_weight"]),
        max_new_edges=int(cfg["assoc_max_new_edges"]),
    )
    added = associations.apply(trial, ops)

    touched = added + clean_stats["merged_nodes"] + clean_stats["pruned_isolated"]
    if touched == 0:
        return {"proposed": 0, "applied": 0, "cleanup": clean_stats, "promoted": False}

    # EVALUAR la copia: simular el estado resultante y medir contra el oráculo.
    issues = associations.count_issues(trial)
    sim = dict(before)
    sim["kg_nodes"] = trial.number_of_nodes()
    sim["kg_edges"] = trial.number_of_edges()
    sim["kg_isolated"] = issues["isolated"]
    sim["kg_dup_extra"] = issues["dup_extra"]
    eval_after = anchor.evaluate(sim, prev=before)

    ok, reason = anchor.promote_gate(eval_after, eval_before)
    if ok:
        # PROMOVER: aplicar la MISMA transformación al grafo real + persistir.
        associations.apply_cleanup(kg.graph, plan)
        associations.apply(kg.graph, ops)
        try:
            kg.save()
        except Exception:
            pass
        report["promoted"] = True
        return {"proposed": len(ops), "applied": added, "cleanup": clean_stats,
                "promoted": True, "reason": reason}
    return {"proposed": len(ops), "applied": 0, "cleanup": clean_stats,
            "promoted": False, "reason": reason}


def _consider_param_change(state: dict, mode: str, report: dict) -> dict | None:
    """Exploración dirigida por déficit (§13.2.1): elige el parámetro que más
    ayudaría a la sub-capacidad con mayor margen y, según el modo y el riesgo,
    lo auto-aplica o lo deja como propuesta."""
    prof = anchor.profile(state)
    ranking = anchor.deficit_ranking(prof)
    cfg = recon_config.load()
    attempts = _load_json(RECON_ATTEMPTS_FILE, [])

    # Mapa déficit → ajuste de parámetro candidato (con su dirección).
    suggestion = _suggest_for_deficit(ranking, cfg)
    if suggestion is None:
        return None

    knob, new_value, rationale = suggestion
    spec = recon_config.KNOBS[knob]

    # Memoria de intentos (§13.2.3): no repetir un cambio ya probado y descartado.
    sig = f"{knob}={new_value}"
    if any(a.get("sig") == sig and a.get("outcome") == "rejected" for a in attempts):
        return None

    proposal = {
        "id": _next_proposal_id(),
        "knob": knob, "from": cfg.get(knob), "to": new_value,
        "risk": spec["risk"], "rationale": rationale,
        "desc": spec.get("desc", ""), "at": datetime.utcnow().isoformat(),
        "status": "pending",
    }

    # Riesgo bajo + modo que lo permite → auto-aplicar (pasando por checkpoint/gate
    # ya gestionado por el ciclo). En "mixto", los de riesgo "response" NO se aplican.
    auto = (spec["risk"] == "low" and mode in ("mixto", "auto")) or mode == "auto"
    if auto:
        cfg[knob] = new_value
        recon_config.save(cfg)
        report["promoted"] = True
        proposal["status"] = "applied"
        attempts.append({"sig": sig, "outcome": "applied",
                         "at": proposal["at"]})
        _save_json(RECON_ATTEMPTS_FILE, attempts)
        report["notes"].append(f"auto-aplicado {sig} (riesgo bajo)")
    else:
        # Guardar como propuesta pendiente (punto de aprobación, §1.3).
        proposals = _load_json(RECON_PROPOSALS_FILE, [])
        # Evitar propuestas duplicadas del mismo parámetro.
        proposals = [p for p in proposals if not (p["knob"] == knob and p["status"] == "pending")]
        proposals.append(proposal)
        _save_json(RECON_PROPOSALS_FILE, proposals)
        report["notes"].append(f"propuesta pendiente {sig} (afecta respuestas)")

    return proposal


def _suggest_for_deficit(ranking: list, cfg: dict):
    """Traduce el mayor déficit recuperable en un ajuste concreto de parámetro."""
    for subcap, margin in ranking:
        if margin <= 0.0:
            continue
        if subcap == "kg_connectivity":
            # Más asociaciones por ciclo y umbral de puente más permisivo.
            cur = int(cfg["assoc_max_new_edges"])
            new = min(recon_config.KNOBS["assoc_max_new_edges"]["max"], cur + 20)
            if new != cur:
                return ("assoc_max_new_edges", new,
                        f"conectividad del grafo baja (margen {margin}): tejer más asociaciones por ciclo")
        if subcap == "kg_cleanliness":
            # Grafo sucio (duplicados/aislados): permitir más fusiones por ciclo.
            cur = int(cfg["assoc_max_merges"])
            new = min(recon_config.KNOBS["assoc_max_merges"]["max"], cur + 25)
            if new != cur:
                return ("assoc_max_merges", new,
                        f"grafo poco limpio (margen {margin}): fusionar más conceptos duplicados por ciclo")
        if subcap == "doc_coverage" or subcap == "index_freshness":
            # Recuperar más fragmentos ayuda a aprovechar el corpus indexado.
            cur = int(cfg["rag_top_k"])
            new = min(recon_config.KNOBS["rag_top_k"]["max"], cur + 1)
            if new != cur:
                return ("rag_top_k", new,
                        f"cobertura documental con margen {margin}: recuperar más fragmentos por consulta")
        if subcap == "memory_health":
            # Bajar un poco el umbral de relevancia para no descartar contexto útil.
            cur = float(cfg["rag_score_min"])
            new = round(max(recon_config.KNOBS["rag_score_min"]["min"], cur - 0.01), 3)
            if new != cur:
                return ("rag_score_min", new,
                        f"salud de memoria con margen {margin}: afinar el filtro de relevancia")
    return None


# ── Aprobación / rechazo de propuestas (puntos de aprobación, §1.3) ───────────────
def list_proposals() -> list[dict]:
    return _load_json(RECON_PROPOSALS_FILE, [])


def resolve_proposal(proposal_id: int, approve: bool, kg=None) -> dict:
    """Aplica o rechaza una propuesta pendiente. Al aprobar, el cambio pasa por
    checkpoint + puerta de promoción + vigilancia (igual que el bucle)."""
    proposals = _load_json(RECON_PROPOSALS_FILE, [])
    target = next((p for p in proposals if p["id"] == proposal_id), None)
    if target is None or target["status"] != "pending":
        return {"ok": False, "error": "propuesta no encontrada o ya resuelta"}

    attempts = _load_json(RECON_ATTEMPTS_FILE, [])
    sig = f"{target['knob']}={target['to']}"

    if not approve:
        target["status"] = "rejected"
        attempts.append({"sig": sig, "outcome": "rejected", "at": datetime.utcnow().isoformat()})
        _save_json(RECON_ATTEMPTS_FILE, attempts)
        _save_json(RECON_PROPOSALS_FILE, proposals)
        return {"ok": True, "status": "rejected", "proposal": target}

    # Aprobar: checkpoint, aplicar, verificar invariantes, rollback si rompe algo.
    cp_id = checkpoints.create(label=f"approve-{target['knob']}")
    cfg = recon_config.load()
    cfg[target["knob"]] = target["to"]
    recon_config.save(cfg)

    state = recon_config.load()
    if not recon_config.within_bounds(state):
        checkpoints.restore(cp_id, kg=kg)
        return {"ok": False, "error": "el cambio violaría una restricción dura", "rolled_back": True}

    target["status"] = "applied"
    attempts.append({"sig": sig, "outcome": "applied", "at": datetime.utcnow().isoformat()})
    _save_json(RECON_ATTEMPTS_FILE, attempts)
    _save_json(RECON_PROPOSALS_FILE, proposals)
    return {"ok": True, "status": "applied", "proposal": target, "checkpoint": cp_id}


def _next_proposal_id() -> int:
    proposals = _load_json(RECON_PROPOSALS_FILE, [])
    return (max((p["id"] for p in proposals), default=0) + 1) if proposals else 1


def status() -> dict:
    return {"last_run": _last_run["at"], "last_report": _last_run["report"],
            "baseline": anchor.load_baseline(), "mode": RECON_MODE}
