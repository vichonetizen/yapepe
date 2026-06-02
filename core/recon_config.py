"""
Capa 5 — El sustrato reconfigurable (andamiaje, grado 1).

Almacén de los ÚNICOS parámetros que el sistema puede ajustarse a sí mismo.
Es código/datos inspeccionables, versionables y reversibles → el sustrato donde,
según los documentos (Vol. III §18.1, §22.4), la autonomía plena es segura.

Cada parámetro declara:
  - default      : valor de fábrica.
  - min/max      : RESTRICCIÓN DURA (plan operativo §1.3). El motor nunca puede
                   salirse de aquí; un valor fuera de rango es un invariante roto.
  - risk         : "low"      → el motor puede auto-aplicarlo sin preguntar.
                   "response" → afecta a las respuestas; en modo "mixto" queda
                                como PROPUESTA que el usuario aprueba (§1.3 punto
                                de aprobación proporcional al riesgo).

NOTA DE DISEÑO (frontera datos/control, Vol. II §14.2): los pesos del oráculo y
los invariantes NO viven aquí, sino en core/anchor.py (el ancla externa), porque
si el sistema pudiera editar su propio examen, la auto-mejora colapsaría en
"aprobarse a sí mismo" (Vol. III §21.2). Este archivo es lo que el sistema PUEDE
tocar; el ancla es lo que NO.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RECON_CONFIG_FILE

# Especificación de los parámetros ajustables. Es la "constitución" de qué se
# puede mover y dentro de qué límites.
KNOBS: dict[str, dict] = {
    # — Recuperación documental (afecta a respuestas → propuesta) —
    "rag_top_k":             {"default": 5,    "min": 2,   "max": 10,   "risk": "response",
                              "desc": "Cuántos fragmentos de documentos se recuperan por consulta."},
    "rag_score_min":         {"default": 0.05, "min": 0.0, "max": 0.30, "risk": "response",
                              "desc": "Relevancia mínima para incluir un fragmento."},
    "kg_max_nodes":          {"default": 400,  "min": 100, "max": 1000, "risk": "response",
                              "desc": "Tamaño máximo del grafo de conocimiento."},
    # — Motor de asociaciones del grafo (bajo riesgo → auto-aplicable) —
    "assoc_max_new_edges":   {"default": 40,   "min": 5,   "max": 200,  "risk": "low",
                              "desc": "Asociaciones nuevas que se pueden tejer por ciclo."},
    "assoc_bridge_min_weight": {"default": 2,  "min": 1,   "max": 10,   "risk": "low",
                              "desc": "Fuerza mínima de A→B y B→C para inferir el puente A→C."},
    "assoc_backlink_min_weight": {"default": 3, "min": 1,  "max": 10,   "risk": "low",
                              "desc": "Fuerza mínima de A→B para crear el recíproco B→A."},
    # — Limpieza / optimización automática del grafo (bajo riesgo → auto) —
    "assoc_max_merges":      {"default": 25,   "min": 0,   "max": 200,  "risk": "low",
                              "desc": "Conceptos duplicados que se pueden fusionar por ciclo."},
    "assoc_prune_isolated":  {"default": 1,    "min": 0,   "max": 1,    "risk": "low",
                              "desc": "Podar nodos aislados (sin conexiones) del grafo: 1=sí, 0=no."},
}

_cache: dict | None = None


def _defaults() -> dict:
    return {k: spec["default"] for k, spec in KNOBS.items()}


def _clamp(name: str, value):
    """Aplica la restricción dura. Devuelve el valor saneado dentro de [min,max]."""
    spec = KNOBS.get(name)
    if spec is None:
        return value
    try:
        v = type(spec["default"])(value)
    except (ValueError, TypeError):
        return spec["default"]
    return max(spec["min"], min(spec["max"], v))


def load() -> dict:
    """Carga la config (creándola con valores de fábrica si no existe).
    Saneada SIEMPRE contra las cotas: ningún valor corrupto puede entrar."""
    global _cache
    cfg = _defaults()
    if os.path.exists(RECON_CONFIG_FILE):
        try:
            with open(RECON_CONFIG_FILE, encoding="utf-8") as f:
                saved = json.load(f)
            for k in cfg:
                if k in saved:
                    cfg[k] = _clamp(k, saved[k])
        except Exception:
            pass
    else:
        save(cfg)
    _cache = cfg
    return dict(cfg)


def save(cfg: dict) -> None:
    """Guarda la config saneada contra las cotas."""
    global _cache
    clean = {k: _clamp(k, cfg.get(k, KNOBS[k]["default"])) for k in KNOBS}
    os.makedirs(os.path.dirname(RECON_CONFIG_FILE), exist_ok=True)
    with open(RECON_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    _cache = clean


def get(name: str):
    """Lectura cacheada de un parámetro (con fallback al default de fábrica)."""
    global _cache
    if _cache is None:
        load()
    return _cache.get(name, KNOBS.get(name, {}).get("default"))


def within_bounds(cfg: dict) -> bool:
    """Invariante: ¿todos los valores están dentro de su restricción dura?"""
    for k, spec in KNOBS.items():
        if k not in cfg:
            return False
        v = cfg[k]
        if not isinstance(v, (int, float)) or v < spec["min"] or v > spec["max"]:
            return False
    return True


def reset_cache_for_tests():
    global _cache
    _cache = None
