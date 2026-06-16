"""
llm.py — Adaptador opcional de LLM para EstudIA (pista integrada).

Cablea el motor de estudio al ai_client multi-proveedor del proyecto SIN romper la
ruta offline: si no hay proveedor disponible (sin API key ni Ollama), las funciones
devuelven vacío/None y generate/grade caen a sus plantillas deterministas.

La regla dura se mantiene fuera del LLM: aquí solo se PROPONEN preguntas/notas;
parse_questions_json descarta toda evidencia que no esté en el conjunto de chunks
entregado (anti-alucinación de citas), y la validación final la hace grounding.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from core.ai_client import AIClient

_client = AIClient()
_ollama_checked = False


def _pick_chat_model(models: list[str]) -> str:
    """Elige un modelo de CHAT de Ollama apto para GPU de 2 GB (igual que app.py)."""
    chat = [m for m in models if "embed" not in m.lower() and "bge" not in m.lower()
            and "nomic" not in m.lower()]
    for pref in ("llama3.2:1b", "qwen2.5:1.5b", "qwen2.5:0.5b", "gemma2:2b",
                 "phi3.5", "llama3.2:3b", "qwen2.5:3b"):
        for m in chat:
            if m.startswith(pref):
                return m
    return chat[0] if chat else (models[0] if models else "llama3.2:3b")


def _ensure_ollama() -> None:
    """Auto-detecta y registra Ollama local (sin API key) una vez por proceso."""
    global _ollama_checked
    if _ollama_checked or "ollama" in _client.keys:
        _ollama_checked = True
        return
    _ollama_checked = True
    try:
        import json as _json
        import urllib.request as _ur
        with _ur.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            models = [m["name"] for m in _json.loads(r.read()).get("models", [])]
        if not models:
            return
        model = _pick_chat_model(models)
        _client.update_key("ollama", "ollama")
        _client.register_custom("ollama", "ollama", "http://localhost:11434/v1", model)
    except Exception:
        pass


def current_provider() -> Optional[tuple[str, str]]:
    """Devuelve (proveedor, modelo) que se usaría, o None si no hay ninguno."""
    _ensure_ollama()
    route = _client.get_route("media")
    return route[0] if route else None


def provider_available() -> bool:
    """True si hay un proveedor utilizable. Controlable con ESTUDIA_LLM:
      - "off"/"0"/"false": nunca usa LLM (solo plantillas offline; cero latencia).
      - "auto" (def.): usa LLM si hay proveedor (nube u Ollama local).
    En GPU pequeña el modelo local puede ser débil/lento: poner "off" evita el
    intento inútil y va directo a la ruta determinista."""
    if os.getenv("ESTUDIA_LLM", "auto").strip().lower() in ("0", "off", "false", "no"):
        return False
    _ensure_ollama()
    try:
        return bool(_client.get_route("media"))
    except Exception:
        return False


async def _complete(system_prompt: str, user_msg: str, complexity: str = "media") -> str:
    route = _client.get_route(complexity)
    if not route:
        return ""
    provider, model = route[0]
    parts: list[str] = []
    async for chunk in _client.stream(provider, model, system_prompt,
                                      [{"role": "user", "content": user_msg}]):
        parts.append(chunk)
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Generación de preguntas
# --------------------------------------------------------------------------- #
_GEN_SYSTEM = (
    "Eres un generador de preguntas de estudio universitario en español. "
    "Respondes EXCLUSIVAMENTE con un array JSON. Cada pregunta DEBE citar su "
    "evidencia con 'evidence': [chunk_id] usando SOLO los ids entre corchetes que "
    "se te entregan. Prohibido inventar ids o contenido fuera de los fragmentos."
)


def build_gen_prompt(req, chunks, n: int = 5) -> tuple[str, str]:
    block = "\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks[:12])
    user = (
        f"Fragmentos fuente (cada uno con su id entre corchetes):\n{block}\n\n"
        f"Genera {n} preguntas (mezcla open|mcq|tf|cloze) sobre estos fragmentos. "
        f"Dificultad objetivo: {getattr(req, 'difficulty', 3)}/5.\n"
        "Devuelve SOLO un array JSON de objetos con campos: "
        "type, prompt, answer_key, rationale, evidence (lista de chunk_id de los de "
        "arriba), concept_ids (opcional), options (solo mcq), difficulty (1-5)."
    )
    return _GEN_SYSTEM, user


def parse_questions_json(text: str, allowed_ids: set[str]) -> list[dict]:
    """Extrae el array JSON y descarta preguntas sin evidencia válida del corpus."""
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except (json.JSONDecodeError, TypeError):
        return []
    out: list[dict] = []
    for d in arr:
        if not isinstance(d, dict):
            continue
        ev = [str(e) for e in (d.get("evidence") or []) if str(e) in allowed_ids]
        if not ev:
            continue  # sin evidencia válida -> se descarta (anti-alucinación)
        d["evidence"] = ev
        out.append(d)
    return out


async def agenerate_questions(req, chunks, n: int = 5) -> list[dict]:
    if not provider_available():
        return []
    allowed = {c.chunk_id for c in chunks}
    system, user = build_gen_prompt(req, chunks, n)
    try:
        text = await _complete(system, user, complexity="media")
    except Exception:
        return []
    return parse_questions_json(text, allowed)


# --------------------------------------------------------------------------- #
# Calificación de respuestas abiertas
# --------------------------------------------------------------------------- #
_GRADE_SYSTEM = (
    "Eres un calificador de respuestas de estudio en español. Respondes SOLO con un "
    "objeto JSON {\"score\": number 0..1, \"feedback\": string}. El feedback debe "
    "basarse en la respuesta esperada; no inventes datos fuera de ella."
)


async def agrade_open(question, user_answer: str) -> Optional[dict]:
    if not provider_available():
        return None
    user = (
        f"Pregunta: {question.prompt}\n"
        f"Respuesta esperada: {question.answer_key}\n"
        f"Respuesta del alumno: {user_answer}\n"
        "Califica de 0 a 1 y da feedback breve."
    )
    try:
        text = await _complete(_GRADE_SYSTEM, user, complexity="simple")
    except Exception:
        return None
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {"score": max(0.0, min(1.0, float(d.get("score", 0.0)))),
                "feedback": str(d.get("feedback", ""))}
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
