"""
Prueba de la Capa 3 — Recetas de tareas.

Valida que el sistema:
  1. Captura una receta cuando una tarea con sustancia fue bien calificada (y NO
     cuando la calificación es baja o la tarea es trivial).
  2. Recupera la receta correcta ante un pedido semánticamente similar (aunque use
     otras palabras), descartando recetas no relacionadas.
  3. Degrada a búsqueda por palabras clave si no hay embeddings.

Usa un embebedor determinista. Ejecutar:  python tests/test_recipes.py
"""
import asyncio
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.semantic_memory as sm
import core.recipe_store as rs

TEST_DB = "data/_test_recipes.db"

_AXES = [
    ["arquitectura", "microservicios", "servicios", "ecommerce", "tienda", "online", "dominios", "estructuro"],
    ["cocina", "pan", "receta", "harina", "horno", "casero"],
]


def fake_embed(texts):
    # dim 3: eje 0 = arquitectura, eje 1 = cocina, eje 2 = "neutral/otro".
    # Lo no relacionado cae al eje neutral → coseno ~0 con las recetas reales.
    out = []
    for t in texts:
        tl = t.lower()
        v = [0.0, 0.0, 0.0]
        for i, lex in enumerate(_AXES):
            for kw in lex:
                if kw in tl:
                    v[i] += 1.0
        if v[0] == 0.0 and v[1] == 0.0:
            v[2] = 1.0
        out.append(v)
    return out


def _ok(cond, label):
    print(("  ✅ " if cond else "  ❌ ") + label)
    if not cond:
        raise AssertionError(label)


async def main():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    sm._reset_engine_for_tests(TEST_DB)
    await rs.init_recipes_db()
    sm.embedder.set_fake_embedder(fake_embed)

    print("\n[1] Captura condicional (umbrales)")
    r_low = await rs.maybe_capture_recipe("s1", "haz una tarea", "respuesta", "compleja", 2.0)
    _ok(r_low is None, "NO captura con calificación baja (2.0)")
    r_simple = await rs.maybe_capture_recipe("s1", "hola", "hola!", "simple", 5.0)
    _ok(r_simple is None, "NO captura tareas triviales (simple)")
    rid = await rs.maybe_capture_recipe(
        "s1",
        "diseña una arquitectura de microservicios para un ecommerce",
        "1) Identifica dominios de negocio. 2) Un API gateway al frente. "
        "3) Base de datos por servicio. 4) Comunicación asíncrona por eventos.",
        "compleja", 5.0,
    )
    _ok(isinstance(rid, int), "SÍ captura tarea compleja bien calificada")

    # receta distractora (otro tema)
    await rs.maybe_capture_recipe(
        "s1", "dame una receta de pan casero",
        "Mezcla harina, agua y levadura; amasa; deja reposar; hornea.",
        "media", 5.0,
    )

    print("\n[2] Recuperación semántica por significado")
    hits = await rs.search_recipes("cómo estructuro los servicios de una tienda online", top_k=2)
    _ok(len(hits) >= 1, "recupera al menos una receta")
    _ok("microservicios" in hits[0]["trigger"], "recupera la receta de arquitectura, no la de pan")
    ctx = rs.build_recipe_context(hits[:1])
    _ok("Enfoque que funcionó" in ctx, "el contexto inyectable está bien formado")

    print("\n[3] No alucina recetas para temas no relacionados")
    none_hits = await rs.search_recipes("cuál es la capital de Francia", top_k=2)
    _ok(all("microservicios" not in h["trigger"] and "pan" not in h["trigger"] for h in none_hits)
        or len(none_hits) == 0,
        "no fuerza recetas irrelevantes (floor de similitud)")

    print("\n[4] Reutilización contada")
    await rs.mark_reused([rid])
    stats = await rs.get_recipe_stats()
    _ok(stats["total"] == 2 and stats["total_reused"] >= 1, "stats: 2 recetas, ≥1 reutilización")

    print("\n[5] Degradación a palabras clave sin embeddings")
    sm.embedder.set_fake_embedder(None)
    kw = await rs.search_recipes("ecommerce", top_k=2)
    _ok(any("ecommerce" in h["trigger"] for h in kw), "fallback por palabra clave encuentra la receta")

    print("\n🎉 CAPA 3 (RECETAS DE TAREAS) VALIDADA\n")

    await sm._get_engine().dispose()
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except OSError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
