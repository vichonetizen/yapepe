import json
import uuid
import asyncio
import math
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete as sql_delete
import uvicorn
from pathlib import Path

from core.local_auth import get_token, verify_token
from core.device import (
    get_allowed_roots, list_path, read_file,
    get_sysinfo, clipboard_read, clipboard_write,
)

from config import PORT, HOST
from core.memory import (
    init_db, get_db, AsyncSessionFactory,
    SessionRecord, Message, save_message, save_memory_fact,
    get_all_facts, update_message_rating, update_module_performance,
    get_stats,
)
from core.knowledge_graph import KnowledgeGraph
from core.meta_controller import MetaController
from core.ai_client import AIClient, ROUTING, PROVIDER_INFO, PROVIDER_DEFAULTS
from core.document_store import (
    init_documents_db, add_document, list_documents,
    delete_document, get_doc_count, extract_text,
)
from core.provider_store import (
    init_provider_store, get_all_providers, get_provider_full,
    upsert_provider, delete_provider, load_keys_into_client,
)
from core.learning_engine import (
    init_learnings_db, extract_and_store_learning,
    get_all_learnings, delete_learning, get_learning_stats,
)
from core.semantic_memory import (
    init_semantic_db, embedder, index_document_chunks, index_doc,
    maybe_autoindex_on_start, status as semantic_status,
)
from core.recipe_store import (
    init_recipes_db, maybe_capture_recipe, mark_reused,
    get_all_recipes, delete_recipe, get_recipe_stats,
)
from core.autonomy import (
    ensure_indexes, consolidate_memory, status as autonomy_status,
)
import core.self_reconstruction as recon
import core.checkpoints as checkpoints
import core.associations as associations
import core.agent as agent
from config import (
    AUTONOMY_INTERVAL_HOURS, IS_VERCEL, DEVICE_ACCESS_ENABLED, es_cliente_local,
    RECON_ENABLED, RECON_MODE, RECON_MAX_CYCLES,
)

kg        = KnowledgeGraph()
meta_ctrl = MetaController(kg)
ai_client = AIClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await init_documents_db()
    await init_provider_store()
    await load_keys_into_client(ai_client)
    await init_learnings_db()
    await init_semantic_db()
    await init_recipes_db()
    await ensure_indexes()

    # Sync .env keys → DB (only if DB slot is empty)
    from config import GOOGLE_API_KEY, GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
    env_map = {
        "google": GOOGLE_API_KEY, "groq": GROQ_API_KEY,
        "openai": OPENAI_API_KEY, "anthropic": ANTHROPIC_API_KEY,
    }
    for slug, key in env_map.items():
        if key and key.strip():
            existing = await get_provider_full(slug)
            if existing and not existing.get("api_key"):
                await upsert_provider(
                    slug, existing["name"], existing["description"],
                    key, existing["base_url"], existing["default_model"], existing["icon"],
                )
                ai_client.update_key(slug, key)

    # Auto-detect Ollama: si está corriendo, registrarlo sin intervención del usuario
    try:
        ollama_info = await asyncio.get_event_loop().run_in_executor(None, _check_ollama_sync)
        if ollama_info["running"]:
            existing_ollama = await get_provider_full("ollama")
            if not existing_ollama or not (existing_ollama or {}).get("api_key"):
                model = _pick_chat_model(ollama_info["models"]) if ollama_info["models"] else "llama3.2:3b"
                await upsert_provider(
                    slug="ollama", name="Ollama (Local)",
                    description="Detectado automáticamente al iniciar. Sin API key ni internet.",
                    api_key="ollama", base_url=ollama_info["base_url"],
                    default_model=model, icon="🏠",
                )
                ai_client.update_key("ollama", "ollama")
                ai_client.register_custom("ollama", "ollama", ollama_info["base_url"], model)
    except Exception:
        pass

    # Memoria semántica: configurar backend de embeddings (Ollama local > nube > BM25)
    # y reindexar en segundo plano los documentos ya cargados, sin bloquear el arranque.
    try:
        ollama_url = ai_client.custom_configs.get("ollama", {}).get("base_url", "")
        embedder.configure(ai_client.keys, ollama_url=ollama_url)
        # Solo auto-indexa si el backend es local (Ollama). Con nube, el usuario
        # dispara la indexación manualmente vía POST /semantic/reindex.
        asyncio.create_task(maybe_autoindex_on_start())
    except Exception:
        pass

    # Capa 4 — Autonomía: bucle periódico de consolidación de memoria.
    # En Vercel (serverless) no hay proceso persistente: se usa Vercel Cron contra
    # POST /autonomy/consolidate en su lugar.
    if AUTONOMY_INTERVAL_HOURS > 0 and not IS_VERCEL:
        asyncio.create_task(_autonomy_loop())

    yield


async def _autonomy_loop():
    """Cada AUTONOMY_INTERVAL_HOURS, en segundo plano y sin que el usuario lo pida:
      1. consolida la memoria (Capa 4), y
      2. ejecuta un ciclo de auto-reconstrucción (Capa 5): teje asociaciones
         neuronales y reconfigura parámetros seguros contra el oráculo fijo.
    Es el "bucle maestro autónomo" del plan operativo §3."""
    interval = AUTONOMY_INTERVAL_HOURS * 3600
    cycles = 0
    while True:
        await asyncio.sleep(interval)
        try:
            await consolidate_memory(kg)
        except Exception:
            pass
        if RECON_ENABLED and (RECON_MAX_CYCLES == 0 or cycles < RECON_MAX_CYCLES):
            try:
                await recon.run_cycle(kg)
                cycles += 1
            except Exception:
                pass


app = FastAPI(title="Pentamodal 3.0 — Multi-Provider", lifespan=lifespan)

# Restrict CORS to the local origin only — blocks cross-origin requests from other sites
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*", "X-Local-Token"],
    allow_credentials=True,
)


# ── EstudIA — motor de estudio (paquete core/study) ──────────────────────────────
# Aditivo: expone /study/* sin tocar los endpoints existentes. La regla dura del
# proyecto (degradación elegante) aplica: si el módulo no carga, la app sigue viva.
try:
    from core.study.api import router as estudia_router
    app.include_router(estudia_router)
except Exception as _estudia_err:  # noqa: BLE001
    import logging
    logging.getLogger("pentamodal").warning("EstudIA no disponible: %s", _estudia_err)


def _peticion_local(request: Request) -> bool:
    """Confianza por petición: el CLIENTE conecta desde loopback (no el bind del servidor)."""
    return es_cliente_local(request.client.host if request.client else None)


class _TokenMiddleware(BaseHTTPMiddleware):
    """Require X-Local-Token header on every request except the HTML page and CORS preflights."""
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Capacidades de dispositivo: solo para clientes en loopback (y si no se
        # apagaron por DISABLE_DEVICE_ACCESS). Además exigen token más abajo.
        if path.startswith("/device/") and not (DEVICE_ACCESS_ENABLED and _peticion_local(request)):
            return JSONResponse(
                {"detail": "Acceso a dispositivo deshabilitado en este entorno."},
                status_code=404,
            )
        if (request.method == "OPTIONS"
                or (request.method == "GET" and path in ("/", "/api/health"))):
            return await call_next(request)
        token = request.headers.get("X-Local-Token", "")
        if not verify_token(token):
            return JSONResponse({"detail": "No autorizado. Token local requerido."}, status_code=403)
        return await call_next(request)


app.add_middleware(_TokenMiddleware)


# ── Request models ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:           str
    session_id:        str
    parallel_mode:     bool = False
    pac_mode:          bool = False
    rag_only:          bool = False   # Modo Autónomo: solo documentos, sin API key externa
    maestro_mode:      bool = False   # Instrucción Maestra v2.0: RL + PAT + Autocrítica
    override_provider: str  = ""
    override_model:    str  = ""


class FeedbackRequest(BaseModel):
    message_id: int
    rating:     float
    complexity: str       = ""
    modules:    list[str] = []
    provider:   str       = ""


class ClipboardWriteRequest(BaseModel):
    text: str


class SessionCreateRequest(BaseModel):
    title: str = "Nueva sesión"


class SessionRenameRequest(BaseModel):
    title: str


class ApiKeyRequest(BaseModel):
    provider: str
    api_key:  str


class ProviderConfigRequest(BaseModel):
    slug:          str
    name:          str
    description:   str   = ""
    api_key:       str   = ""
    base_url:      str   = ""
    default_model: str   = ""
    icon:          str   = "🔑"


# ── Static ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health(request: Request):
    """Diagnóstico sin token. Fuera de un cliente local solo confirma que la app
    vive: la ruta de datos (con nombre de usuario) y la lista de proveedores
    configurados no se exponen a peticiones remotas/anónimas."""
    if not _peticion_local(request):
        return {"ok": True}
    return {
        "ok": True,
        "is_vercel": IS_VERCEL,
        "providers": ai_client.available_providers(),
        "data_dir": str(getattr(__import__("config"), "DATA_DIR", "")),
    }


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    try:
        html = (Path(__file__).parent / "frontend" / "index.html").read_text(encoding="utf-8")
    except Exception as e:
        return HTMLResponse(
            "<html><body style='font-family:sans-serif;padding:2rem'>"
            "<h1>Pentamodal</h1>"
            "<p>✅ La aplicación está corriendo, pero no se pudo cargar la interfaz "
            f"(<code>frontend/index.html</code>).</p><p>Detalle: {e}</p>"
            "<p>Prueba <a href='/api/health'>/api/health</a> para verificar el estado.</p>"
            "</body></html>",
            status_code=200,
        )
    # El token solo se embebe si la PETICIÓN viene de loopback. A clientes remotos
    # o serverless NO se les inyecta, para que un visitante anónimo no pueda leerlo
    # del HTML y usar la API. (Un despliegue público requeriría autenticación real.)
    if _peticion_local(request):
        token = get_token()
        # Patch window.fetch to auto-include the local auth token on every API call
        inject = (
            "<script>(function(){"
            "var T='" + token + "';"
            "var _f=window.fetch.bind(window);"
            "window.fetch=function(u,o){"
            "o=o||{};"
            "var loc=typeof u==='string'&&(u.startsWith('/')||u.startsWith('http://127.0.0.1')||u.startsWith('http://localhost'));"
            "if(loc){o.headers=Object.assign({'X-Local-Token':T},o.headers||{});}"
            "return _f(u,o);};"
            "})();</script>"
        )
        html = html.replace("<head>", "<head>" + inject, 1)
    return HTMLResponse(html)


# ── Sessions ───────────────────────────────────────────────────────────────────

@app.post("/session")
async def create_session(req: SessionCreateRequest, db: AsyncSession = Depends(get_db)):
    sid = str(uuid.uuid4())
    db.add(SessionRecord(id=sid, title=req.title))
    await db.commit()
    return {"session_id": sid}


@app.get("/sessions/{sid}/export")
async def export_session(sid: str, db: AsyncSession = Depends(get_db)):
    from fastapi.responses import Response
    result = await db.execute(
        select(SessionRecord).where(SessionRecord.id == sid)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Sesión no encontrada")

    msgs_result = await db.execute(
        select(Message).where(Message.session_id == sid).order_by(Message.created_at.asc())
    )
    messages = msgs_result.scalars().all()

    lines = [
        f"# {session.title}",
        f"\n> Exportado desde Pentamodal 3.0 · {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"> {len(messages)} mensajes\n",
        "---\n",
    ]
    for msg in messages:
        if msg.role == "user":
            lines.append(f"**Tú:**\n\n{msg.content}\n")
        else:
            badge = f" · *{msg.provider} / {msg.model_used}*" if msg.provider else ""
            lines.append(f"**Asistente{badge}:**\n\n{msg.content}\n")
        lines.append("---\n")

    content = "\n".join(lines)
    filename = f"pentamodal-{session.title[:30].replace(' ','_')}.md"
    return Response(
        content=content.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.patch("/sessions/{sid}")
async def rename_session(sid: str, req: SessionRenameRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(SessionRecord).where(SessionRecord.id == sid))
    rec = result.scalar_one_or_none()
    if not rec:
        raise HTTPException(404, "Sesión no encontrada")
    rec.title = req.title[:100]
    await db.commit()
    return {"status": "ok", "title": rec.title}


@app.get("/sessions")
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(SessionRecord).order_by(SessionRecord.created_at.desc()).limit(20)
    )
    return [
        {"id": s.id, "title": s.title, "message_count": s.message_count,
         "created_at": s.created_at.isoformat()}
        for s in result.scalars().all()
    ]


# ── Chat (streaming) ───────────────────────────────────────────────────────────

@app.delete("/sessions/{sid}")
async def delete_session(sid: str, db: AsyncSession = Depends(get_db)):
    await db.execute(sql_delete(Message).where(Message.session_id == sid))
    result = await db.execute(sql_delete(SessionRecord).where(SessionRecord.id == sid))
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(404, "Sesión no encontrada")
    return {"status": "deleted", "id": sid}


async def _auto_summarize_doc(filename: str, file_bytes: bytes):
    """Genera un resumen IA del documento recién subido y lo guarda en PAC."""
    await asyncio.sleep(1.0)
    try:
        route = ai_client.get_route("simple")
        if not route:
            return
        text = await extract_text(filename, file_bytes)
        excerpt = text[:2000].strip()
        if len(excerpt) < 80:
            return
        provider, model = route[0]
        prompt = (
            f"Resume en 4-5 oraciones el contenido principal del documento «{filename}».\n"
            f"Incluye: tema central, conceptos clave, y conclusión o propósito principal.\n\n"
            f"Inicio del documento:\n{excerpt}"
        )
        summary = ""
        async for chunk in ai_client.stream(
            provider, model,
            "Eres un especialista en resumir documentos académicos y técnicos. "
            "Responde únicamente con el resumen, sin prefijos ni encabezados.",
            [{"role": "user", "content": prompt}],
        ):
            summary += chunk
        if summary.strip():
            await extract_and_store_learning(
                session_id="__docs__",
                user_msg=f"[Resumen automático: {filename}]",
                assistant_msg=summary.strip(),
                doc_hits=[{"filename": filename, "content": excerpt[:200], "score": 1.0}],
            )
    except Exception:
        pass


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):

    async def generate():
        async with AsyncSessionFactory() as db:
            try:
                # Ensure session exists
                r = await db.execute(select(SessionRecord).where(SessionRecord.id == req.session_id))
                if not r.scalar_one_or_none():
                    db.add(SessionRecord(id=req.session_id))
                    await db.commit()

                # Save user message
                await save_message(req.session_id, "user", req.message, [], "simple", db)

                # Orchestrate modules / system prompt
                orch = await meta_ctrl.orchestrate(
                    req.message, req.session_id, db,
                    pac_mode=req.pac_mode,
                    rag_only=req.rag_only,
                    maestro_mode=req.maestro_mode,
                )
                system_prompt = orch["system_prompt"]
                api_messages  = orch["messages"]
                complexity    = orch["complexity"]
                modules       = orch["modules"]

                # Capa 3: si se aplicaron recetas previas, cuéntalas como reutilizadas
                if orch.get("recipe_ids"):
                    asyncio.create_task(mark_reused(orch["recipe_ids"]))

                # Determine provider route
                if req.override_provider and req.override_provider in ai_client.keys:
                    route = [(req.override_provider,
                              req.override_model or PROVIDER_DEFAULTS.get(req.override_provider, ""))]
                elif req.rag_only:
                    # Modo Autónomo: prefer local Ollama, else any available provider
                    if "ollama" in ai_client.keys:
                        ollama_cfg = ai_client.custom_configs.get("ollama", {})
                        route = [("ollama", ollama_cfg.get("default_model", "llama3.2:3b"))]
                    else:
                        route = ai_client.get_route(complexity)
                else:
                    route = ai_client.get_route(complexity)

                if not route:
                    yield f"data: {json.dumps({'type':'error','content':'No hay proveedores configurados. Agrega al menos una API key en ⚙ Config.'})}\n\n"
                    return

                # Capa 2 — Inferencia local: si la ruta resultó ser Ollama (modelo
                # pequeño en GPU 2 GB), reconstruye el prompt en su versión comprimida
                # (V1) y recorta el historial para no saturar su ventana de contexto.
                if not req.rag_only and len(route) == 1 and route[0][0] == "ollama":
                    from core.modules import build_local_system_prompt
                    system_prompt = build_local_system_prompt(
                        modules, complexity,
                        orch.get("memory_context", ""), orch.get("kg_context", ""),
                        orch.get("doc_context", ""), orch.get("pac_context", ""),
                        req.pac_mode, recipe_context=orch.get("recipe_context", ""),
                    )
                    api_messages = api_messages[-9:]

                is_parallel = (req.parallel_mode or len(route) > 1) and len(route) > 1

                # ── Parallel mode ──────────────────────────────────────────────
                if is_parallel:
                    providers_info = [
                        {"provider": p, "model": m, **PROVIDER_INFO.get(p, {})}
                        for p, m in route
                    ]
                    yield f"data: {json.dumps({'type':'start','mode':'parallel','providers_info':providers_info,'modules':modules,'complexity':complexity})}\n\n"

                    full_texts: dict[str, str] = {p: "" for p, _ in route}

                    async for item in ai_client.stream_parallel(route, system_prompt, api_messages):
                        prov = item.get("provider", "")
                        if item["type"] == "text":
                            full_texts[prov] = full_texts.get(prov, "") + item["content"]
                            yield f"data: {json.dumps(item)}\n\n"
                        elif item["type"] == "done_provider":
                            yield f"data: {json.dumps(item)}\n\n"

                    # Save one message per provider
                    msg_ids: dict[str, int] = {}
                    for prov, mod in route:
                        mid = await save_message(
                            req.session_id, "assistant", full_texts.get(prov, ""),
                            modules, complexity, db,
                            provider=prov, model_used=mod,
                        )
                        msg_ids[prov] = mid
                        facts = meta_ctrl.extract_memory_facts(req.message, full_texts.get(prov, ""))
                        for fact in facts:
                            await save_memory_fact(fact["category"], fact["content"], db)
                        kg.update_from_conversation(req.message, full_texts.get(prov, ""))

                    yield f"data: {json.dumps({'type':'done','message_ids':msg_ids,'complexity':complexity,'modules':modules})}\n\n"

                # ── Single provider ────────────────────────────────────────────
                else:
                    provider, model = route[0]
                    pinfo = PROVIDER_INFO.get(provider, {})
                    yield f"data: {json.dumps({'type':'start','mode':'single','provider':provider,'model':model,'provider_info':pinfo,'modules':modules,'complexity':complexity})}\n\n"

                    full_text = ""
                    async for chunk in ai_client.stream(provider, model, system_prompt, api_messages):
                        if await request.is_disconnected():
                            return
                        full_text += chunk
                        yield f"data: {json.dumps({'type':'text','provider':provider,'content':chunk})}\n\n"

                    mid = await save_message(
                        req.session_id, "assistant", full_text,
                        modules, complexity, db,
                        provider=provider, model_used=model,
                    )
                    facts = meta_ctrl.extract_memory_facts(req.message, full_text)
                    for fact in facts:
                        await save_memory_fact(fact["category"], fact["content"], db)
                    kg.update_from_conversation(req.message, full_text)

                    # PAC: async learning extraction (fire-and-forget)
                    asyncio.create_task(extract_and_store_learning(
                        req.session_id, req.message, full_text,
                        doc_hits=orch.get("doc_hits"),
                    ))

                    n_learnings = (await get_learning_stats())["total"]
                    serialized_hits = [
                        {"filename": h["filename"], "excerpt": h["content"][:250], "score": h["score"]}
                        for h in (orch.get("doc_hits") or [])
                    ]
                    yield f"data: {json.dumps({'type':'done','message_id':mid,'complexity':complexity,'modules':modules,'provider':provider,'model':model,'learnings_total':n_learnings,'pac_mode':req.pac_mode,'rag_only':req.rag_only,'maestro_mode':req.maestro_mode,'doc_hits':serialized_hits})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type':'error','content':str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Feedback ───────────────────────────────────────────────────────────────────

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    await update_message_rating(req.message_id, req.rating, db)
    if req.complexity and req.modules:
        await update_module_performance(req.complexity, req.modules, req.rating, db, req.provider)

    # Capa 3: si la respuesta fue bien calificada en una tarea con sustancia,
    # captura una receta (pedido + enfoque que funcionó) para replicarla luego.
    if req.rating >= 4.0 and req.complexity in ("media", "compleja", "critica"):
        amsg = (await db.execute(
            select(Message).where(Message.id == req.message_id)
        )).scalar_one_or_none()
        if amsg and amsg.role == "assistant":
            umsg = (await db.execute(
                select(Message)
                .where(Message.session_id == amsg.session_id,
                       Message.role == "user",
                       Message.created_at <= amsg.created_at)
                .order_by(Message.created_at.desc())
                .limit(1)
            )).scalar_one_or_none()
            if umsg:
                asyncio.create_task(maybe_capture_recipe(
                    amsg.session_id, umsg.content, amsg.content,
                    req.complexity, req.rating,
                ))
    return {"status": "ok"}


# ── Data endpoints ─────────────────────────────────────────────────────────────

class AnchorRequest(BaseModel):
    content: str
    category: str = "session_insight"


@app.post("/memory/anchor")
async def anchor_memory(req: AnchorRequest, db: AsyncSession = Depends(get_db)):
    if not req.content.strip():
        raise HTTPException(400, "Contenido vacío")
    await save_memory_fact(req.category, req.content[:500], db, confidence=1.0)
    return {"status": "anchored"}


@app.get("/memory")
async def get_memory(db: AsyncSession = Depends(get_db)):
    return {"facts": await get_all_facts(db)}


@app.get("/kg")
async def get_knowledge_graph():
    return kg.to_vis_data()


@app.get("/stats")
async def get_statistics(db: AsyncSession = Depends(get_db)):
    stats = await get_stats(db)
    stats["kg_nodes"]   = kg.graph.number_of_nodes()
    stats["kg_edges"]   = kg.graph.number_of_edges()
    stats["providers"]  = ai_client.status()
    stats["routing"]    = {
        cx: [{"provider": p, "model": m, **PROVIDER_INFO.get(p, {})} for p, m in pairs]
        for cx, pairs in ROUTING.items()
    }
    return stats


@app.get("/providers")
async def get_providers():
    return {
        "available": ai_client.available_providers(),
        "status":    ai_client.status(),
        "routing":   {
            cx: [{"provider": p, "model": m, **PROVIDER_INFO.get(p, {})} for p, m in pairs]
            for cx, pairs in ROUTING.items()
        },
    }


# ── Documents ──────────────────────────────────────────────────────────────────

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".txt", ".md", ".pdf", ".docx", ".csv"}
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Formato no soportado: {ext}. Válidos: {', '.join(allowed)}")
    try:
        content = await file.read()
        result = await add_document(file.filename, content)
        # Resumen IA en background para enriquecer el contexto RAG
        asyncio.create_task(_auto_summarize_doc(file.filename, content))
        # Indexado semántico (embeddings) del nuevo documento, sin bloquear la respuesta
        if result.get("id"):
            asyncio.create_task(index_doc(result["id"]))
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/documents")
async def get_documents():
    return await list_documents()


@app.delete("/documents/{doc_id}")
async def remove_document(doc_id: int):
    ok = await delete_document(doc_id)
    if not ok:
        raise HTTPException(404, "Documento no encontrado")
    return {"status": "deleted", "id": doc_id}


@app.get("/documents/stats")
async def documents_stats():
    return await get_doc_count()


# ── Memoria semántica (embeddings) ──────────────────────────────────────────────

@app.get("/semantic/status")
async def semantic_status_endpoint():
    """Estado del backend de embeddings y progreso de indexación."""
    return await semantic_status()


@app.get("/autonomy/status")
async def autonomy_status_endpoint():
    """Estado del último ciclo de consolidación autónoma (Capa 4)."""
    return {"interval_hours": AUTONOMY_INTERVAL_HOURS, **autonomy_status()}


@app.post("/autonomy/consolidate")
async def autonomy_consolidate():
    """Dispara una consolidación de memoria a mano (fusiona, poda, indexa pendientes)."""
    report = await consolidate_memory(kg)
    return {"status": "ok", "report": report}


# ── Capa 5 — Auto-reconstrucción (base estructural de los 5 documentos) ──────────
@app.get("/recon/status")
async def recon_status():
    """Estado del bucle maestro: línea base del oráculo, perfil y último ciclo."""
    return {"enabled": RECON_ENABLED, "mode": RECON_MODE, **recon.status()}


@app.post("/recon/cycle")
async def recon_cycle():
    """Ejecuta un ciclo de auto-reconstrucción a mano (perfilar → proponer sobre
    copia → evaluar contra oráculo fijo → promover/descartar → vigilar)."""
    report = await recon.run_cycle(kg)
    return {"status": "ok", "report": report}


@app.get("/recon/associations")
async def recon_associations():
    """Densidad asociativa actual del grafo (asociaciones neuronales tejidas)."""
    return associations.stats(kg.graph)


@app.get("/recon/proposals")
async def recon_proposals():
    """Propuestas pendientes de aprobación (cambios que afectan a las respuestas)."""
    return recon.list_proposals()


@app.post("/recon/proposals/{pid}/approve")
async def recon_proposal_approve(pid: int):
    res = recon.resolve_proposal(pid, approve=True, kg=kg)
    if not res.get("ok"):
        raise HTTPException(400, res.get("error", "no se pudo aplicar"))
    return res


@app.post("/recon/proposals/{pid}/reject")
async def recon_proposal_reject(pid: int):
    res = recon.resolve_proposal(pid, approve=False, kg=kg)
    if not res.get("ok"):
        raise HTTPException(404, res.get("error", "no encontrada"))
    return res


@app.get("/recon/checkpoints")
async def recon_checkpoints():
    """Checkpoints reversibles del estado mutable (config + grafo)."""
    return {"latest": checkpoints.latest(), "all": checkpoints.list_all()}


# ── Capa 6 — Bucle agente (ejecutar tareas) ──────────────────────────────────────
class AgentRunRequest(BaseModel):
    task:           str
    session_id:     str  = "agent"
    max_steps:      int  = 0          # 0 = usar el default del config
    allow_write:    bool = True       # escribir en el workspace aislado
    allow_commands: bool = False      # ejecutar comandos (con confirmación)


class AgentContinueRequest(BaseModel):
    run_id:  str
    approve: bool


@app.post("/agent/run")
async def agent_run(req: AgentRunRequest):
    """Lanza el bucle agente sobre una tarea. Devuelve el resultado o, si un
    comando requiere confirmación, status 'awaiting_approval' con un run_id."""
    return await agent.run(
        req.task, ai_client=ai_client, session_id=req.session_id,
        max_steps=(req.max_steps or None), allow_write=req.allow_write,
        allow_commands=req.allow_commands, kg=kg,
    )


@app.post("/agent/continue")
async def agent_continue(req: AgentContinueRequest):
    """Reanuda una corrida pausada, aprobando o rechazando el comando pendiente."""
    res = await agent.continue_run(req.run_id, approve=req.approve)
    if res.get("status") == "error":
        raise HTTPException(404, res.get("error", "sesión no encontrada"))
    return res


@app.get("/agent/tools")
async def agent_tools_list():
    """Herramientas disponibles para el agente y su nivel de riesgo."""
    import core.agent_tools as at
    return {name: {"risk": s["risk"], "args": s["args"], "desc": s["desc"]}
            for name, s in at.TOOLS.items()}


@app.post("/recon/rollback")
async def recon_rollback(cp_id: str = ""):
    """Revierte al checkpoint indicado (o al último bueno si no se especifica)."""
    target = cp_id or checkpoints.latest()
    if not target:
        raise HTTPException(404, "no hay checkpoints")
    ok = checkpoints.restore(target, kg=kg)
    if not ok:
        raise HTTPException(404, "checkpoint no encontrado")
    return {"status": "rolled_back", "checkpoint": target}


@app.get("/recipes")
async def list_recipes():
    """Recetas de tareas aprendidas (Capa 3)."""
    return await get_all_recipes()


@app.get("/recipes/stats")
async def recipes_stats():
    return await get_recipe_stats()


@app.delete("/recipes/{rid}")
async def remove_recipe(rid: int):
    ok = await delete_recipe(rid)
    if not ok:
        raise HTTPException(404, "Receta no encontrada")
    return {"status": "deleted", "id": rid}


@app.post("/semantic/reindex")
async def semantic_reindex():
    """Indexa (o re-indexa) los chunks de documentos que aún no tienen embedding.
    Útil tras instalar Ollama o configurar una API key con embeddings."""
    ollama_url = ai_client.custom_configs.get("ollama", {}).get("base_url", "")
    embedder.configure(ai_client.keys, ollama_url=ollama_url)
    indexed = await index_document_chunks()
    return {"indexed": indexed, **(await semantic_status())}


# ── Provider configs (full CRUD) ───────────────────────────────────────────────

@app.get("/provider-configs")
async def list_provider_configs():
    return await get_all_providers()


@app.get("/provider-configs/{slug}")
async def get_provider_config(slug: str, request: Request):
    p = await get_provider_full(slug)
    if not p:
        raise HTTPException(404, "Proveedor no encontrado")
    # A clientes no-loopback, no devolver la API key en texto plano.
    if not _peticion_local(request):
        key = p.get("api_key") or ""
        p["api_key"] = ("•" * min(12, max(0, len(key) - 4)) + key[-4:]) if len(key) > 4 else ""
    return p


@app.post("/provider-configs")
async def save_provider_config(req: ProviderConfigRequest):
    result = await upsert_provider(
        req.slug, req.name, req.description,
        req.api_key, req.base_url, req.default_model, req.icon,
    )
    # Sync to in-memory client immediately
    ai_client.update_key(req.slug, req.api_key)
    if req.base_url or req.slug not in {"google", "groq", "openai", "anthropic"}:
        ai_client.register_custom(req.slug, req.api_key, req.base_url, req.default_model)
    return {**result, "providers": ai_client.available_providers()}


@app.delete("/provider-configs/{slug}")
async def remove_provider_config(slug: str):
    result = await delete_provider(slug)
    ai_client.update_key(slug, "")           # clear from in-memory client
    ai_client.custom_configs.pop(slug, None)
    return result


# ── Config ─────────────────────────────────────────────────────────────────────

@app.post("/config/apikey")
async def set_api_key(req: ApiKeyRequest):
    ai_client.update_key(req.provider, req.api_key)
    return {"status": "ok", "providers": ai_client.available_providers()}


# ── Ollama (local / sin API key) ───────────────────────────────────────────────

def _pick_chat_model(models: list[str]) -> str:
    """Elige un modelo de CHAT de la lista de Ollama, excluyendo modelos de
    embeddings (p. ej. nomic-embed-text) que no sirven para conversar.
    Prefiere modelos que caben bien en una GPU de 2 GB."""
    chat = [m for m in models if "embed" not in m.lower()]
    # Orden pensado para GPU de 2 GB: modelos ~1B primero (caben y no cuelgan el
    # runner). Los de 2-3B se dejan como última opción (requieren más VRAM).
    for pref in ("llama3.2:1b", "qwen2.5:1.5b", "qwen2.5:0.5b", "gemma2:2b",
                 "phi3.5", "llama3.2:3b", "qwen2.5:3b"):
        for m in chat:
            if m.startswith(pref):
                return m
    if chat:
        return chat[0]
    return models[0] if models else "llama3.2:3b"


def _check_ollama_sync() -> dict:
    import urllib.request as _ur
    import json as _json
    try:
        with _ur.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            data = _json.loads(r.read())
            models = [m["name"] for m in data.get("models", [])]
            return {"running": True, "models": models, "base_url": "http://localhost:11434/v1"}
    except Exception:
        return {"running": False, "models": [], "base_url": "http://localhost:11434/v1"}


@app.get("/ollama/status")
async def ollama_status():
    import asyncio
    info = await asyncio.get_event_loop().run_in_executor(None, _check_ollama_sync)
    return info


@app.post("/ollama/pull")
async def ollama_pull(body: dict):
    """Stream ollama pull <model> output as SSE."""
    model = str((body or {}).get("model", "llama3.2:3b")).strip()
    # Sanitize model name — only allow alphanumeric, colon, dot, dash
    import re as _re
    if not model or not _re.match(r'^[a-zA-Z0-9:.\-_/]+$', model):
        raise HTTPException(400, "Nombre de modelo inválido")

    async def _stream():
        try:
            proc = await asyncio.create_subprocess_exec(
                "ollama", "pull", model,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                yield f"data: {json.dumps({'type':'progress','text':text})}\n\n"

            rc = await proc.wait()
            if rc == 0:
                yield f"data: {json.dumps({'type':'done','model':model})}\n\n"
            else:
                yield f"data: {json.dumps({'type':'error','text':f'Falló con código {rc}'})}\n\n"

        except FileNotFoundError:
            yield f"data: {json.dumps({'type':'error','text':'Ollama no está instalado. Descárgalo en: ollama.com/download'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','text':str(e)})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/ollama/register")
async def ollama_register(body: dict = None):
    """Register or refresh Ollama as a local provider."""
    info = await asyncio.get_event_loop().run_in_executor(None, _check_ollama_sync)
    if not info["running"]:
        raise HTTPException(503, "Ollama no está corriendo. Inicia Ollama primero.")
    model = (body or {}).get("model") or _pick_chat_model(info["models"])
    result = await upsert_provider(
        slug="ollama",
        name="Ollama (Local)",
        description="Modelo local sin API key. Responde desde documentos sin internet.",
        api_key="ollama",
        base_url=info["base_url"],
        default_model=model,
        icon="🏠",
    )
    ai_client.update_key("ollama", "ollama")
    ai_client.register_custom("ollama", "ollama", info["base_url"], model)
    return {**result, "models": info["models"], "selected_model": model}


# ── Learnings (PAC) ────────────────────────────────────────────────────────────

@app.get("/learnings")
async def list_learnings():
    return await get_all_learnings()

@app.delete("/learnings/all")
async def remove_all_learnings():
    from core.learning_engine import delete_all_learnings
    count = await delete_all_learnings()
    return {"status": "deleted", "count": count}


@app.delete("/learnings/{lid}")
async def remove_learning(lid: int):
    ok = await delete_learning(lid)
    if not ok:
        raise HTTPException(404, "Aprendizaje no encontrado")
    return {"status": "deleted", "id": lid}

@app.get("/learnings/stats")
async def learnings_stats():
    return await get_learning_stats()


@app.get("/stats/neural")
async def neural_stats(db: AsyncSession = Depends(get_db)):
    """Compute neural evolution data: complexity trajectory, module diversity, neural score."""
    from sqlalchemy import select as sa_select
    from core.memory import Message as Msg, SessionRecord as SR

    # All assistant messages with metadata
    result = await db.execute(
        sa_select(Msg).where(Msg.role == "assistant").order_by(Msg.created_at.asc())
    )
    msgs = result.scalars().all()

    if not msgs:
        return {
            "neural_score": 0.0,
            "level": "Inicial",
            "dimensions": {},
            "complexity_trajectory": [],
            "projection": {"sessions_needed": 10, "next_level": "Intermedio"},
        }

    # Module usage counts → dimensions
    CX_WEIGHT = {"simple": 1, "media": 2, "compleja": 3, "critica": 4}
    module_usage: dict[str, int] = {}
    complexity_counts: dict[str, int] = {}
    total_weight = 0
    rated_weights: list[float] = []

    # Group by session for trajectory
    session_data: dict[str, dict] = {}
    for msg in msgs:
        mods = json.loads(msg.modules_used or "[]")
        for m in mods:
            module_usage[m] = module_usage.get(m, 0) + 1
        cx = msg.complexity or "simple"
        complexity_counts[cx] = complexity_counts.get(cx, 0) + 1
        w = CX_WEIGHT.get(cx, 1)
        total_weight += w
        if msg.rating:
            rated_weights.append(msg.rating * w)

        sid = msg.session_id
        if sid not in session_data:
            session_data[sid] = {"cx_sum": 0, "count": 0, "ts": msg.created_at.isoformat() if msg.created_at else ""}
        session_data[sid]["cx_sum"] += w
        session_data[sid]["count"] += 1

    # Neural score: avg complexity weight × module diversity × rating boost
    n = len(msgs)
    avg_cx = total_weight / max(n, 1)
    mod_diversity = min(len(module_usage) / 11.0, 1.0)  # 11 total modules
    avg_rating_boost = (sum(rated_weights) / max(sum(CX_WEIGHT.get(m.complexity or "simple", 1) for m in msgs if m.rating), 1)) if rated_weights else 3.0
    neural_score = round(avg_cx * 1.5 * mod_diversity * (avg_rating_boost / 3.0) * 10, 1)
    neural_score = min(neural_score, 10.0)

    # Level classification
    if neural_score < 2: level = "Inicial"
    elif neural_score < 4: level = "Emergente"
    elif neural_score < 6: level = "Intermedio"
    elif neural_score < 8: level = "Avanzado"
    else: level = "Maestro"

    # Complexity trajectory (avg per session)
    trajectory = []
    for sid, d in list(session_data.items())[:20]:
        trajectory.append({
            "session": sid[:8],
            "avg_complexity": round(d["cx_sum"] / max(d["count"], 1), 2),
            "messages": d["count"],
            "timestamp": d["ts"],
        })

    # Projection: sessions needed to reach next level
    if level == "Maestro":
        sessions_needed = 0
        next_level = "Maestro"
    else:
        target_scores = {"Inicial": 2, "Emergente": 4, "Intermedio": 6, "Avanzado": 8, "Maestro": 10}
        next_levels = {"Inicial": "Emergente", "Emergente": "Intermedio", "Intermedio": "Avanzado", "Avanzado": "Maestro"}
        next_level = next_levels.get(level, "Maestro")
        target = target_scores.get(next_level, 10)
        sessions_per_point = max(len(session_data) / max(neural_score, 0.1), 1)
        sessions_needed = math.ceil((target - neural_score) * sessions_per_point)

    # Radar dimensions (normalized 0-100)
    all_mods = ["D", "P", "G", "R", "TdM", "PL", "EV", "MEM", "KG", "OPT", "THINK", "RL", "PAT", "AUTODEV"]
    max_usage = max(module_usage.values()) if module_usage else 1
    dimensions = {m: round(module_usage.get(m, 0) / max_usage * 100) for m in all_mods}

    return {
        "neural_score": neural_score,
        "level": level,
        "dimensions": dimensions,
        "complexity_distribution": complexity_counts,
        "complexity_trajectory": trajectory,
        "projection": {
            "sessions_needed": sessions_needed,
            "next_level": next_level,
            "current_level": level,
        },
        "total_messages": n,
        "module_diversity": round(mod_diversity * 100),
    }


# ── Device Access (local only, token-protected) ────────────────────────────────

@app.get("/device/roots")
async def device_roots():
    return {"roots": get_allowed_roots()}


@app.get("/device/sysinfo")
async def device_sysinfo():
    return get_sysinfo()


@app.get("/device/fs")
async def device_fs(path: str):
    try:
        return list_path(path)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/device/fs/read")
async def device_fs_read(path: str):
    try:
        content = read_file(path)
        return {"path": path, "content": content}
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))


@app.get("/device/clipboard")
async def device_clipboard_get():
    return {"text": clipboard_read()}


@app.post("/device/clipboard")
async def device_clipboard_set(req: ClipboardWriteRequest):
    clipboard_write(req.text)
    return {"ok": True}


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  PENTAMODAL 3.0 — Multi-Provider")
    print(f"  URL: http://{HOST}:{PORT}")
    configured = ai_client.available_providers()
    print(f"  Proveedores: {', '.join(configured) if configured else 'Ninguno (configura API keys)'}")
    print(f"{'='*55}\n")
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
