"""
Prueba de la Capa 1 — Memoria semántica.

Valida el MECANISMO (almacenamiento de vectores, coseno, ranking, mezcla híbrida
y degradación a BM25) usando un embebedor determinista inyectado. No mide la
calidad de un modelo real de embeddings — eso depende de Ollama / la nube — sino
que la tubería relaciona ideas por SIGNIFICADO y no por palabras compartidas.

Ejecutar:  python tests/test_semantic_memory.py
"""
import asyncio
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.document_store as ds
import core.semantic_memory as sm

TEST_DB = "data/_test_semantic.db"


# ── Embebedor determinista: 3 ejes conceptuales con sinónimos ─────────────────────
# La gracia: palabras DISTINTAS que comparten significado caen en el mismo eje.
_AXES = [
    # eje 0 — autoorganización (Maturana/Varela)
    ["autopoiesis", "autoorganiza", "autoconstruye", "autorreferencia", "vivo"],
    # eje 1 — color / terapia
    ["color", "cromoterapia", "luz", "aura", "pigmento"],
    # eje 2 — cuerpo / práctica contemplativa
    ["yoga", "meditacion", "respiracion", "chakra", "postura"],
]


def fake_embed(texts):
    vecs = []
    for t in texts:
        tl = t.lower()
        v = [0.0] * len(_AXES)
        for i, lex in enumerate(_AXES):
            for kw in lex:
                if kw in tl:
                    v[i] += 1.0
        if not any(v):
            v[0] = 1e-6  # evitar vector nulo
        vecs.append(v)
    return vecs


def _ok(cond, label):
    print(("  ✅ " if cond else "  ❌ ") + label)
    if not cond:
        raise AssertionError(label)


async def main():
    # Redirigir AMBOS módulos a una DB temporal limpia
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    ds._DB_PATH = TEST_DB
    ds._engine = None
    ds._has_chunks = None
    sm._reset_engine_for_tests(TEST_DB)

    await ds.init_documents_db()
    await sm.init_semantic_db()

    # Tres documentos, cada uno dominado por un eje conceptual distinto
    await ds.add_document("maturana.txt",
        "La autopoiesis describe lo vivo como un sistema que se mantiene a sí mismo.".encode())
    await ds.add_document("colores.txt",
        "La cromoterapia usa la luz y el pigmento del aura para sanar.".encode())
    await ds.add_document("yoga.txt",
        "El yoga integra la respiracion y la postura en la meditacion.".encode())

    print("\n[1] Backend no disponible → degradación a BM25")
    sm.embedder.set_fake_embedder(None)
    _ok(await sm.semantic_search_documents("autopoiesis") is None,
        "semantic_search devuelve None sin backend")
    hyb = await sm.hybrid_search_documents("autopoiesis", top_k=3)
    _ok(len(hyb) >= 1 and hyb[0]["filename"] == "maturana.txt",
        "hybrid cae a BM25 y encuentra el doc por palabra exacta")

    print("\n[2] Indexación con embebedor determinista")
    sm.embedder.set_fake_embedder(fake_embed)
    n = await sm.index_document_chunks()
    _ok(n == 3, f"se indexaron los 3 chunks (n={n})")
    st = await sm.status()
    _ok(st["indexed_chunks"] == 3, "status reporta 3 chunks indexados")

    print("\n[3] Relación por SIGNIFICADO sin palabras compartidas (el corazón de Capa 1)")
    # Estas palabras NO aparecen en ningún documento, pero comparten eje conceptual
    # con 'autopoiesis'/'vivo'. Así BM25 no tiene de dónde agarrarse y solo el
    # significado puede recuperar el doc correcto.
    q = "autoorganiza autoconstruye autorreferencia"
    sem = await sm.semantic_search_documents(q, top_k=3)
    _ok(sem is not None and sem[0]["filename"] == "maturana.txt",
        "semántico relaciona 'autoorganiza' → 'autopoiesis' (0 palabras en común)")
    # Verificar que BM25 puro NO lo lograría (sin solapamiento léxico)
    bm = await ds.search_documents(q, top_k=3)
    bm_top = bm[0]["filename"] if bm else "∅"
    print(f"     (comparación: BM25 puro top = {bm_top})")
    _ok(bm_top != "maturana.txt",
        "BM25 puro NO encuentra el doc → el aporte semántico es real")

    print("\n[4] Búsqueda híbrida combina ambas señales")
    hyb = await sm.hybrid_search_documents(q, top_k=3)
    _ok(hyb and hyb[0]["filename"] == "maturana.txt",
        "híbrido prioriza el doc semánticamente correcto")
    _ok("semantic" in hyb[0] and "bm25" in hyb[0],
        "el resultado expone sus componentes bm25/semantic (auditable)")

    print("\n[5] Otro eje: 'cuerpo y aliento contemplativo' → yoga.txt")
    sem2 = await sm.semantic_search_documents("practica de respiracion y postura", top_k=3)
    _ok(sem2 and sem2[0]["filename"] == "yoga.txt", "eje 2 recupera el doc correcto")

    print("\n🎉 TODAS LAS PRUEBAS DE MEMORIA SEMÁNTICA PASARON\n")

    # Limpieza
    await sm._get_engine().dispose()
    await ds._get_engine().dispose()
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except OSError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
