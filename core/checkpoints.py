"""
Capa 5 — Gestor de CHECKPOINTS y ROLLBACK.

"El checkpoint/rollback no es solo seguridad: es lo que hace EXPLORABLE la
auto-edición. Permite intentar cambios sabiendo que cualquier degradación es
reversible sin pérdida" (plan operativo §8; Vol. III §22.2).

Guarda el estado MUTABLE que la Capa 5 puede tocar —los parámetros
reconfigurables y el grafo de conocimiento— como una copia triv'almente
restaurable. La reversión es copiar de vuelta: barata y total.

La capacidad de revertir reside AQUÍ, en un componente que el bucle de
auto-edición no deshabilita (corrigibilidad, §22.2).
"""
import json
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CHECKPOINTS_DIR, RECON_CONFIG_FILE, KG_FILE

MAX_CHECKPOINTS = 12  # se conservan los últimos N; los más viejos se podan


def _ensure_dir():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


def create(label: str = "") -> str:
    """Crea un checkpoint del estado mutable. Devuelve su id (carpeta)."""
    _ensure_dir()
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S_%f")
    cp_id = f"{stamp}__{label}" if label else stamp
    cp_dir = os.path.join(CHECKPOINTS_DIR, cp_id)
    os.makedirs(cp_dir, exist_ok=True)

    for src, name in ((RECON_CONFIG_FILE, "config.json"), (KG_FILE, "knowledge_graph.json")):
        if os.path.exists(src):
            try:
                shutil.copy2(src, os.path.join(cp_dir, name))
            except Exception:
                pass

    with open(os.path.join(cp_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"id": cp_id, "label": label, "at": datetime.utcnow().isoformat()},
                  f, ensure_ascii=False, indent=2)

    _prune_old()
    return cp_id


def restore(cp_id: str, kg=None) -> bool:
    """Restaura un checkpoint (rollback trivial). Si se pasa kg, lo recarga en vivo."""
    cp_dir = os.path.join(CHECKPOINTS_DIR, cp_id)
    if not os.path.isdir(cp_dir):
        return False

    cfg_bak = os.path.join(cp_dir, "config.json")
    kg_bak = os.path.join(cp_dir, "knowledge_graph.json")
    if os.path.exists(cfg_bak):
        shutil.copy2(cfg_bak, RECON_CONFIG_FILE)
    if os.path.exists(kg_bak):
        shutil.copy2(kg_bak, KG_FILE)

    # Recargar en memoria para que la reversión sea efectiva inmediatamente.
    import core.recon_config as recon_config
    recon_config.reset_cache_for_tests()
    recon_config.load()
    if kg is not None:
        try:
            kg._load()
        except Exception:
            pass
    return True


def latest() -> str | None:
    """Id del checkpoint más reciente (el último estado bueno conocido)."""
    cps = list_all()
    return cps[0]["id"] if cps else None


def list_all() -> list[dict]:
    """Checkpoints existentes, del más nuevo al más viejo."""
    _ensure_dir()
    out = []
    for name in os.listdir(CHECKPOINTS_DIR):
        cp_dir = os.path.join(CHECKPOINTS_DIR, name)
        if not os.path.isdir(cp_dir):
            continue
        meta = {"id": name, "label": "", "at": None}
        mp = os.path.join(cp_dir, "meta.json")
        if os.path.exists(mp):
            try:
                with open(mp, encoding="utf-8") as f:
                    meta.update(json.load(f))
            except Exception:
                pass
        out.append(meta)
    out.sort(key=lambda m: m["id"], reverse=True)
    return out


def _prune_old():
    cps = list_all()
    for old in cps[MAX_CHECKPOINTS:]:
        try:
            shutil.rmtree(os.path.join(CHECKPOINTS_DIR, old["id"]))
        except Exception:
            pass
