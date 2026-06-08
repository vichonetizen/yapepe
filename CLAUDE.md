# Pentamodal 3.0 — Guía para Claude Code

Asistente de IA personal (FastAPI) **multi-proveedor** con RAG sobre un corpus propio, memoria semántica, inferencia local y módulos de autonomía. Backend Python; la UI se sirve como HTML en `/`. Todo en **español**.

## Arrancar / instalar / probar

- **Instalar:** `setup.bat` → `pip install -r requirements.txt` y crea `.env` desde `.env.example`.
- **Configurar `.env`:** al menos `ANTHROPIC_API_KEY`. Opcionales: `GOOGLE_API_KEY`, `GROQ_API_KEY`. `CLAUDE_MODEL` (def. `claude-sonnet-4-6`), `HOST`/`PORT` (def. `127.0.0.1:8000`).
- **Arrancar:** `run.bat` o `start.ps1` (ambos = `python app.py`, que lanza uvicorn en HOST:PORT). UI en http://127.0.0.1:8000.
- **Probar:** `python tests/run_all.py` (o sueltos: `test_semantic_memory.py`, `test_local_inference.py`, `test_recipes.py`, `test_autonomy.py`, `test_agent.py`, `test_self_reconstruction.py`).
- **Local (opcional, recomendado):** instalar **Ollama** → `ollama pull nomic-embed-text` (embeddings) + un chat pequeño (`qwen2.5:3b` / `llama3.2:3b`) para GPU de 2 GB.

## Arquitectura

Entrada: `app.py` (FastAPI, todos los endpoints) → lógica en `core/`.

- **Ruteo / orquestación:** `ai_client.py` (`get_route()` decide local vs nube por complejidad), `meta_controller.py` (`orchestrate()`), `claude_client.py` (Anthropic), `provider_store.py` (multi-proveedor).
- **Prompt "Maestro":** `modules.py` — 14 módulos; `build_local_system_prompt()` arma la versión comprimida para modelos locales pequeños.
- **Capa 1 · Memoria semántica:** `semantic_memory.py` (embeddings + vectores SQLite + coseno numpy + búsqueda híbrida BM25+semántica; degrada a BM25 si no hay backend). Ver `docs/CAPA1_memoria_semantica.md`.
- **Capa 2 · Inferencia local:** Ollama + prompt comprimido. Ver `docs/CAPA2_inferencia_local.md`.
- **Capa 3 · Recetas:** `recipe_store.py` (replicar tareas pasadas).
- **Capa 4 · Autonomía:** `autonomy.py` (consolidación de memoria); `self_reconstruction.py` + `recon_config.py` + `checkpoints.py` (ciclos de reconstrucción con propuestas que se aprueban/rechazan).
- **Conocimiento:** `document_store.py` (RAG sobre PDFs/DOCX de `documents/`), `knowledge_graph.py` + `associations.py` + `anchor.py` (grafo), `learning_engine.py` ("learnings"), `memory.py`.
- **Agente:** `agent.py` + `agent_tools.py` (loop con herramientas).
- **Infra:** `config.py`, `local_auth.py` (token en `data/local.token`), `device.py` (FS/clipboard/sysinfo del equipo), `api/index.py` (entrada serverless).

Almacenes: SQLite (vectores/memoria), `documents/` (corpus), `data/`, `logs/`.

## Endpoints clave (`app.py`)

`POST /chat` (núcleo, soporta `maestro_mode`) · `/documents/*` (RAG) · `/semantic/status|reindex` · `/autonomy/*` · `/recon/*` (auto-reconstrucción) · `/agent/run|continue` · `/recipes` · `/providers` + `/provider-configs` · `/ollama/*` · `/learnings/*` · `/stats` + `/stats/neural` · `/device/*` (FS/clipboard locales) · `/api/health`.

## Convenciones

- **Idioma:** todo en español (código, comentarios, UI, respuestas).
- **Windows + PowerShell.** Servidor en `127.0.0.1:8000`.
- **Degradación elegante:** si falta una clave o servicio, cae al siguiente backend. La app **nunca debe romperse** por falta de un backend.
- **Indexación idempotente y reanudable:** no re-embeber lo ya hecho; respetar cuotas de nube (el backfill automático solo corre con Ollama).
- **Secretos en `.env`** (no commitear; ya está en `.gitignore`).

## Política de autonomía (IMPORTANTE)

Modo de trabajo: **edición de archivos libre**, pero **pedir confirmación explícita antes de** acciones irreversibles o con efecto externo:
- `git push`, `git reset --hard`, `git rebase`, borrar ramas.
- Borrar archivos (`Remove-Item`).
- Enviar correos, pagos/compras, enviar formularios o publicar en el navegador.
- `pip install` de paquetes nuevos.

Antes de una sesión de cambios grande, dejar git commiteado (red de seguridad). Para arrancar el servidor usar `start.ps1` (o `run.bat`).
