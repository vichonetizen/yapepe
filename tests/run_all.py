"""
LAS 7 PRUEBAS — verificación integral del sistema Pentamodal.

Una sola orden que comprueba que todo "fluye": arranque, las 4 capas cognitivas,
el estado real de la base de conocimiento y una consulta semántica real en español.

    python tests/run_all.py

Las pruebas de capa corren en subprocesos aislados (cada una usa su propia DB
temporal). Las pruebas 1, 6 y 7 corren en este proceso contra la base real.
"""
import asyncio
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _run_file(rel_path: str) -> tuple[bool, str]:
    p = os.path.join(ROOT, rel_path)
    try:
        r = subprocess.run(
            [sys.executable, p], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600, cwd=ROOT,
        )
        ok = r.returncode == 0
        tail = (r.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else (r.stderr or "").strip()[-120:]
        return ok, detail
    except Exception as e:
        return False, str(e)[:120]


async def _check_import() -> tuple[bool, str]:
    try:
        import app  # noqa: F401
        n_routes = len(app.app.routes)
        return True, f"app importa OK ({n_routes} rutas registradas)"
    except Exception as e:
        return False, f"fallo al importar: {str(e)[:100]}"


async def _check_system_state() -> tuple[bool, str]:
    import core.semantic_memory as sm
    sm.embedder.configure({})
    st = await sm.status()
    ok = st["available"] and st["indexed_chunks"] > 0 and st["pending"] == 0
    return ok, (f"backend={st['model']} · {st['indexed_chunks']}/{st['total_chunks']} "
                f"chunks indexados · pendientes={st['pending']}")


async def _check_real_query() -> tuple[bool, str]:
    import core.semantic_memory as sm
    sm.embedder.configure({})
    await sm.embedder.resolve()
    if not sm.embedder.available():
        return False, "sin backend de embeddings (instala Ollama + bge-m3)"
    hits = await sm.semantic_search_documents("sanar el cuerpo con luz y colores", top_k=1) or []
    if not hits:
        return False, "sin resultados"
    fn = hits[0]["filename"]
    ok = "COLOR" in fn.upper()
    return ok, f"top = [{fn[:45]}] (score {hits[0]['score']:.2f})"


CHECKS = [
    ("1. Arranque (import de la app)",            "import"),
    ("2. Capa 1 — Memoria semántica",             "tests/test_semantic_memory.py"),
    ("3. Capa 2 — Inferencia local (Ollama)",     "tests/test_local_inference.py"),
    ("4. Capa 3 — Recetas de tareas",             "tests/test_recipes.py"),
    ("5. Capa 4 — Autonomía",                     "tests/test_autonomy.py"),
    ("6. Estado de la base de conocimiento",      "state"),
    ("7. Consulta semántica real (español)",      "query"),
]


async def main():
    print("=" * 64)
    print("  PENTAMODAL — LAS 7 PRUEBAS")
    print("=" * 64)
    passed = 0
    for label, kind in CHECKS:
        if kind == "import":
            ok, detail = await _check_import()
        elif kind == "state":
            ok, detail = await _check_system_state()
        elif kind == "query":
            ok, detail = await _check_real_query()
        else:
            ok, detail = _run_file(kind)
        mark = "✅ PASA" if ok else "❌ FALLA"
        passed += 1 if ok else 0
        print(f"\n{mark}  {label}")
        print(f"        {detail}")

    print("\n" + "=" * 64)
    print(f"  RESULTADO: {passed}/{len(CHECKS)} pruebas pasaron")
    print("=" * 64)
    sys.exit(0 if passed == len(CHECKS) else 1)


if __name__ == "__main__":
    asyncio.run(main())
