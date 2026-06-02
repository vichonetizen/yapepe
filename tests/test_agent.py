"""
Prueba de la Capa 6 — Bucle agente (ejecutar tareas).

Verifica el bucle PLAN→ACT→OBSERVE→REFLECT→STOP y su disciplina de seguridad,
usando un modelo GUIONIZADO (determinista, sin proveedor externo):

  1. Herramientas seguras: escritura confinada al workspace, path traversal y
     secretos bloqueados, comandos destructivos en denylist.
  2. Camino feliz: el agente usa una herramienta, observa, y entrega respuesta.
  3. Aprende: al terminar, captura el enfoque como RECETA reutilizable.
  4. Presupuesto: si nunca termina, se detiene al agotar los pasos.
  5. Confirmación: un comando no trivial PAUSA el bucle (awaiting_approval) y se
     reanuda con la decisión del usuario.
  6. Robustez: una respuesta no-JSON no rompe el bucle.

Ejecutar:  python tests/test_agent.py
"""
import asyncio
import json
import os
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
import core.semantic_memory as sm
import core.agent_tools as agent_tools
import core.agent as agent
from core.recipe_store import init_recipes_db
from core.document_store import init_documents_db

TMP = os.path.join("data", "_test_agent")
TEST_DB = os.path.join(TMP, "agent.db")


def _ok(cond, label):
    print(("  ✅ " if cond else "  ❌ ") + label)
    if not cond:
        raise AssertionError(label)


def scripted(responses):
    """Modelo falso: devuelve respuestas JSON predefinidas, una por paso."""
    box = {"i": 0}
    async def _llm(system_prompt, messages):
        i = box["i"]; box["i"] += 1
        return responses[min(i, len(responses) - 1)]
    return _llm


def J(**kw):
    return json.dumps(kw, ensure_ascii=False)


async def main():
    if os.path.isdir(TMP):
        shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(os.path.join(TMP, "ws"), exist_ok=True)
    os.makedirs(os.path.join(TMP, "root"), exist_ok=True)

    # Aislar workspace y raíz de lectura al directorio temporal.
    agent_tools.AGENT_WORKSPACE = os.path.join(TMP, "ws")
    agent_tools.AGENT_READ_ROOT = os.path.join(TMP, "root")

    sm._reset_engine_for_tests(TEST_DB)
    sm.embedder.set_fake_embedder(None)
    await init_documents_db()
    await init_recipes_db()
    engine = sm._get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(
            "CREATE TABLE IF NOT EXISTS memory_facts (id INTEGER PRIMARY KEY, category TEXT, content TEXT)"))

    # ── [1] Seguridad de las herramientas ────────────────────────────────────────
    print("\n[1] Herramientas seguras (frontera datos/control, sandbox)")
    r = await agent_tools.t_write_file("nota.txt", "hola mundo")
    _ok("workspace" in r and os.path.exists(os.path.join(TMP, "ws", "nota.txt")),
        "write_file escribe dentro del workspace")
    r = await agent_tools.t_write_file("../../escape.txt", "x")
    _ok("DENEGADO" in r, "write_file bloquea path traversal")
    # secreto en la raíz de lectura
    open(os.path.join(TMP, "root", ".env"), "w").write("API_KEY=secreto")
    r = await agent_tools.t_read_file(".env")
    _ok("DENEGADO" in r, "read_file bloquea leer secretos (.env)")
    open(os.path.join(TMP, "root", "ok.txt"), "w").write("contenido visible")
    r = await agent_tools.t_read_file("ok.txt")
    _ok("contenido visible" in r, "read_file lee dentro de la raíz permitida")
    _ok(agent_tools.command_is_denied("rm -rf /"), "denylist atrapa 'rm -rf /'")
    _ok(agent_tools.command_is_safe("echo hola"), "allowlist reconoce 'echo' como seguro")
    _ok(not agent_tools.command_is_safe("pip install requests"), "'pip install' NO es auto-seguro")

    # ── [2-3] Camino feliz + captura de receta ───────────────────────────────────
    print("\n[2] Camino feliz: usa herramienta → observa → responde")
    llm = scripted([
        J(thought="busco en el corpus", action="search_documents",
          action_input={"query": "autopoiesis"}),
        J(thought="ya tengo lo necesario", final_answer="La autopoiesis es la organización de lo vivo."),
    ])
    res = await agent.run("Explica autopoiesis", llm=llm, session_id="t", max_steps=6)
    _ok(res["status"] == "done", f"el agente termina (status={res['status']})")
    _ok(any(t["action"] == "search_documents" for t in res["transcript"]),
        "el transcript registra el uso de la herramienta")
    _ok("autopoiesis" in res["answer"].lower(), "entrega una respuesta final")

    print("\n[3] Aprende: captura el enfoque como receta")
    _ok(res.get("recipe_id") is not None, f"capturó receta (id={res.get('recipe_id')})")
    async with engine.connect() as conn:
        n = (await conn.execute(text("SELECT COUNT(*) FROM task_recipes"))).scalar()
    _ok(n == 1, f"la receta quedó persistida ({n})")

    # ── [4] Presupuesto ──────────────────────────────────────────────────────────
    print("\n[4] Presupuesto de pasos")
    llm_loop = scripted([J(thought="sigo buscando", action="search_memory",
                           action_input={"query": "algo"})])
    res = await agent.run("tarea sin fin", llm=llm_loop, session_id="t", max_steps=3)
    _ok(res["status"] == "budget_exhausted", "se detiene al agotar el presupuesto")
    _ok(res["steps"] == 3, f"respetó el límite de pasos ({res['steps']})")

    # ── [5] Confirmación de comando ──────────────────────────────────────────────
    print("\n[5] Confirmación humana para comandos no triviales")
    llm_cmd = scripted([
        J(thought="instalo algo", action="run_command", action_input={"command": "pip install requests"}),
        J(thought="rechazado, finalizo", final_answer="No se instaló nada."),
    ])
    res = await agent.run("instala requests", llm=llm_cmd, session_id="t",
                          max_steps=6, allow_commands=True)
    _ok(res["status"] == "awaiting_approval", "un comando no trivial PAUSA el bucle")
    _ok(res["pending_action"]["command"] == "pip install requests", "expone el comando a confirmar")
    res2 = await agent.continue_run(res["run_id"], approve=False)
    _ok(res2["status"] == "done", "tras rechazar, el bucle se reanuda y termina")
    _ok(any("RECHAZÓ" in t["observation"] for t in res2["transcript"]),
        "registra el rechazo como observación (dato, no orden)")

    # ── [6] Robustez ante respuesta no-JSON ──────────────────────────────────────
    print("\n[6] Robustez ante salida inválida")
    llm_bad = scripted([
        "esto no es json en absoluto",
        J(thought="ahora sí", final_answer="ok"),
    ])
    res = await agent.run("algo", llm=llm_bad, session_id="t", max_steps=5)
    _ok(res["status"] == "done", "se recupera de una respuesta no-JSON y termina")

    print("\n🎉 CAPA 6 (BUCLE AGENTE) VALIDADA\n")
    await sm._get_engine().dispose()
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
