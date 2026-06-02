# Capa 1 — Memoria semántica

## Qué resuelve
Antes, la app relacionaba ideas **por palabras compartidas** (BM25 + grafo léxico).
No conectaba "autopoiesis" con "sistema que se auto-construye" si no coincidían los
términos. La Capa 1 añade **embeddings**: relaciona ideas **por significado**, que es
lo que hace que el asistente *parezca pensar*.

## Cómo funciona
- `core/semantic_memory.py` — motor de embeddings *pluggable* + almacén de vectores
  (SQLite, BLOB float32) + similitud coseno (numpy) + búsqueda híbrida.
- `meta_controller.orchestrate()` ahora usa `hybrid_search_documents()`:
  combina **BM25 (léxico)** + **embeddings (significado)**. Si no hay backend de
  embeddings, cae automáticamente a BM25 — la app nunca se rompe.

## Backends (en orden de preferencia)
| Prioridad | Backend | Modelo | Notas |
|---|---|---|---|
| 1 | **Ollama (local)** | `nomic-embed-text` | **Recomendado.** Gratis, ilimitado, sin internet. Ideal para GPU de 2 GB. |
| 2 | OpenAI | `text-embedding-3-small` | Nube, requiere API key. |
| 3 | Google | `gemini-embedding-001` | Nube. ⚠️ El *free tier* limita tokens/petición y por día. |
| 4 | (ninguno) | — | Degrada a BM25 puro. |

## Activar el camino LOCAL (recomendado, 2 GB GPU)
```bat
:: 1. Instala Ollama → https://ollama.com/download
:: 2. Descarga el modelo de embeddings (~270 MB, corre en CPU o GPU pequeña)
ollama pull nomic-embed-text
:: 3. (opcional) un modelo de chat pequeño para 2 GB de VRAM
ollama pull qwen2.5:3b
```
Al reiniciar la app, detecta Ollama, lo prefiere como backend y **auto-indexa**
todo el corpus en segundo plano (gratis, sin límite de cuota).

## Endpoints nuevos
- `GET  /semantic/status`  — backend activo, chunks indexados / pendientes.
- `POST /semantic/reindex` — indexa los chunks que falten (necesario tras configurar
  un backend de nube; con Ollama el backfill es automático al arrancar).

## Decisiones de diseño importantes
- **El backfill automático al arrancar solo corre con Ollama** (local, gratis).
  Con backends de nube hay que disparar `/semantic/reindex` manualmente, para no
  consumir cuota sin que lo pidas.
- La indexación es **idempotente y reanudable**: si un lote falla (p. ej. 429 de
  cuota), guarda el progreso parcial y continúa en la siguiente llamada.
- Los vectores se versionan por modelo (`backend:modelo`). Si cambias de backend,
  reindexa para mezclar vectores comparables.

## Pruebas
`python tests/test_semantic_memory.py` — valida el mecanismo con un embebedor
determinista: almacenamiento, coseno, ranking, mezcla híbrida y degradación a BM25.
Demuestra que relaciona ideas **sin palabras compartidas**.

## Próximas capas (no incluidas todavía)
- Capa 2 — inferencia local en GPU 2 GB (Ollama `qwen2.5:3b` / `llama3.2:3b`).
- Capa 3 — recetas de tareas (replicar tareas pasadas).
- Capa 4 — autonomía / consolidación periódica de memoria.
