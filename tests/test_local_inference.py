"""
Prueba de la Capa 2 — Inferencia local (Ollama, GPU 2 GB).

Valida que:
  1. El orquestador, en modo local, produce el prompt COMPRIMIDO (V1) y no el
     Maestro completo de 14 módulos (que satura a un modelo de 1-3B).
  2. Un modelo local real de Ollama genera una respuesta no vacía y sin error.
  3. El selector de modelo de chat ignora los modelos de embeddings.

Requiere Ollama corriendo con al menos un modelo de chat (llama3.2 / gemma2 / qwen).
Ejecutar:  python tests/test_local_inference.py
"""
import asyncio
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_client import AIClient
from core.knowledge_graph import KnowledgeGraph
from core.meta_controller import MetaController
from core.memory import init_db, AsyncSessionFactory
from app import _pick_chat_model, _check_ollama_sync


def _ok(cond, label):
    print(("  ✅ " if cond else "  ❌ ") + label)
    if not cond:
        raise AssertionError(label)


async def main():
    print("\n[0] Selector de modelo de chat ignora embeddings")
    picked = _pick_chat_model(["nomic-embed-text:latest", "llama3.2:3b", "gemma2:2b"])
    _ok("embed" not in picked, f"eligió un modelo de chat, no de embeddings ({picked})")

    info = _check_ollama_sync()
    if not info["running"]:
        print("\n⚠️  Ollama no está corriendo — se omite la prueba de generación local.")
        return
    model = _pick_chat_model(info["models"])
    print(f"\n   Ollama activo. Modelo de chat elegido: {model}")

    await init_db()
    kg = KnowledgeGraph()
    meta = MetaController(kg)
    ai = AIClient()
    ai.register_custom("ollama", "ollama", info["base_url"], model)

    print("\n[1] El orquestador en modo local usa el prompt COMPRIMIDO")
    async with AsyncSessionFactory() as db:
        orch = await meta.orchestrate(
            "¿Qué es la autopoiesis en una frase?", "test-local-session", db,
            local_mode=True,
        )
    sp = orch["system_prompt"]
    _ok("asistente experto" in sp.lower(), "el prompt local arranca con la identidad mínima")
    _ok("MÓDULO DIAGNÓSTICO" not in sp, "NO contiene los párrafos largos del prompt Maestro")
    _ok(len(sp) < 4000, f"el prompt es compacto (len={len(sp)} chars)")

    print("\n[2] Generación local real con Ollama")
    text = ""
    async for chunk in ai.stream(
        "ollama", model, sp, orch["messages"],
    ):
        text += chunk
    _ok(len(text.strip()) > 0, "el modelo local devolvió texto")
    _ok(not text.strip().startswith("⚠️"), "la respuesta no es un error")
    print("\n   --- Respuesta del modelo local ---")
    print("   " + text.strip()[:400].replace("\n", "\n   "))

    print("\n🎉 CAPA 2 (INFERENCIA LOCAL) VALIDADA\n")


if __name__ == "__main__":
    asyncio.run(main())
