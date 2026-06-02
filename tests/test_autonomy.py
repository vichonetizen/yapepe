"""
Prueba de la Capa 4 — Autonomía / auto-mantenimiento.

Valida que la consolidación periódica:
  1. Fusiona hechos de memoria duplicados (sumando frecuencias).
  2. Poda hechos de bajo valor y antiguos.
  3. Deduplica aprendizajes PAC casi idénticos.
  4. Crea índices sin error y produce un informe.

Determinista, sin LLM. Ejecutar:  python tests/test_autonomy.py
"""
import asyncio
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
import core.semantic_memory as sm
import core.autonomy as auto

TEST_DB = "data/_test_autonomy.db"


def _ok(cond, label):
    print(("  ✅ " if cond else "  ❌ ") + label)
    if not cond:
        raise AssertionError(label)


async def main():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    sm._reset_engine_for_tests(TEST_DB)
    sm.embedder.set_fake_embedder(None)  # sin backend → el paso de indexación no-opera

    engine = sm._get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE memory_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, content TEXT,
                confidence REAL DEFAULT 1.0, frequency INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)
        """))
        await conn.execute(text("""
            CREATE TABLE conversation_learnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, content TEXT,
                topics TEXT, source TEXT, doc_refs TEXT, confidence REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP)
        """))
        # Hechos: 2 idénticos (se fusionan) + 1 único + 1 ruido antiguo (se poda)
        await conn.execute(text(
            "INSERT INTO memory_facts (category,content,confidence,frequency,updated_at) VALUES "
            "('pref','Prefiero Python',0.8,3,datetime('now')),"
            "('pref','Prefiero Python',0.6,2,datetime('now')),"
            "('tech','Usa FastAPI',0.9,1,datetime('now')),"
            "('noise','dato irrelevante de una vez',0.3,1,datetime('now','-60 days'))"
        ))
        # Aprendizajes: 2 con mismo prefijo (dedup) + 1 distinto
        await conn.execute(text(
            "INSERT INTO conversation_learnings (session_id,content,source,confidence) VALUES "
            "('s','La autopoiesis es la organización de lo vivo que se produce a sí misma.','ia',0.8),"
            "('s','La autopoiesis es la organización de lo vivo que se produce a sí misma.','ia',0.8),"
            "('s','El conocimiento es un fenómeno biológico según Maturana.','ia',0.8)"
        ))

    print("\n[1] ensure_indexes no falla aunque falten tablas")
    await auto.ensure_indexes()
    _ok(True, "índices creados sin excepción")

    print("\n[2] Consolidación")
    report = await auto.consolidate_memory(kg=None)
    print("     informe:", report)
    _ok(report["facts_merged"] == 1, "fusionó 1 hecho duplicado")
    _ok(report["facts_pruned"] == 1, "podó 1 hecho ruidoso/antiguo")
    _ok(report["learnings_deduped"] == 1, "dedupó 1 aprendizaje")

    print("\n[3] Estado final coherente")
    async with engine.connect() as conn:
        facts = (await conn.execute(text("SELECT COUNT(*) FROM memory_facts"))).scalar()
        merged = (await conn.execute(text(
            "SELECT frequency FROM memory_facts WHERE content='Prefiero Python'"))).scalar()
        learns = (await conn.execute(text("SELECT COUNT(*) FROM conversation_learnings"))).scalar()
    _ok(facts == 2, f"quedan 2 hechos (fusionado + único; ruido podado) — hay {facts}")
    _ok(merged == 5, f"la frecuencia se sumó (3+2=5) — es {merged}")
    _ok(learns == 2, f"quedan 2 aprendizajes — hay {learns}")

    print("\n[4] status reporta el último ciclo")
    st = auto.status()
    _ok(st["last_run"] is not None and st["last_report"] is not None, "status registra el ciclo")

    print("\n🎉 CAPA 4 (AUTONOMÍA) VALIDADA\n")

    await sm._get_engine().dispose()
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except OSError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
