"""
Unified multi-provider AI client.
Routing: simple/media → Groq  |  compleja → Google  |  critica → ambos en paralelo
Auto-fallback: Google 429 → Groq
"""
import asyncio
from typing import AsyncGenerator
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ANTHROPIC_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, OPENAI_API_KEY


class _RateLimitError(Exception):
    """Señal interna: proveedor devolvió 429; activa fallback automático."""
    pass

# ── Routing table ──────────────────────────────────────────────────────────────
ROUTING: dict[str, list[tuple[str, str]]] = {
    "simple":   [("groq",      "llama-3.1-8b-instant")],
    "media":    [("groq",      "llama-3.3-70b-versatile")],
    "compleja": [("google",    "gemini-2.0-flash")],
    "critica":  [("google",    "gemini-2.0-flash"),
                 ("groq",      "llama-3.3-70b-versatile")],
}

PROVIDER_DEFAULTS: dict[str, str] = {
    "google":    "gemini-2.0-flash",
    "groq":      "llama-3.3-70b-versatile",
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o-mini",
}

PROVIDER_INFO: dict[str, dict] = {
    "google":    {"name": "Google Gemini",    "icon": "🌐", "color": "#4285f4"},
    "groq":      {"name": "Groq",             "icon": "⚡", "color": "#f97316"},
    "anthropic": {"name": "Claude Anthropic", "icon": "🎭", "color": "#6366f1"},
    "openai":    {"name": "OpenAI",           "icon": "🟢", "color": "#10b981"},
}


class AIClient:
    def __init__(self):
        self.keys: dict[str, str] = {}
        # custom_configs: slug → {api_key, base_url, default_model} for non-built-in or OpenAI-compat
        self.custom_configs: dict[str, dict] = {}
        self._load_env()

    def _load_env(self):
        if ANTHROPIC_API_KEY: self.keys["anthropic"] = ANTHROPIC_API_KEY
        if GOOGLE_API_KEY:    self.keys["google"]    = GOOGLE_API_KEY
        if GROQ_API_KEY:      self.keys["groq"]      = GROQ_API_KEY
        if OPENAI_API_KEY:    self.keys["openai"]    = OPENAI_API_KEY

    def update_key(self, provider: str, key: str):
        if key.strip():
            self.keys[provider] = key.strip()
        elif provider in self.keys:
            del self.keys[provider]

    def register_custom(self, slug: str, api_key: str, base_url: str, default_model: str):
        """Register a custom or OpenAI-compatible provider."""
        if api_key.strip():
            self.custom_configs[slug] = {
                "api_key": api_key.strip(),
                "base_url": base_url.strip(),
                "default_model": default_model.strip(),
            }
            self.keys[slug] = api_key.strip()
        else:
            self.custom_configs.pop(slug, None)
            self.keys.pop(slug, None)

    def available_providers(self) -> list[str]:
        return list(self.keys.keys())

    def status(self) -> dict:
        built_in = ["google", "groq", "openai", "anthropic"]
        result = {
            p: {"configured": p in self.keys, **PROVIDER_INFO.get(p, {})}
            for p in built_in
        }
        for slug, cfg in self.custom_configs.items():
            if slug not in result:
                result[slug] = {
                    "configured": True,
                    "name": slug,
                    "icon": "🔌",
                    "color": "#a78bfa",
                }
        return result

    def get_route(self, complexity: str) -> list[tuple[str, str]]:
        candidates = ROUTING.get(complexity, ROUTING["media"])
        available  = [(p, m) for p, m in candidates if p in self.keys]
        if available:
            return available
        # Local-first: Ollama cubre cualquier complejidad sin API key
        if "ollama" in self.keys:
            cfg   = self.custom_configs.get("ollama", {})
            model = cfg.get("default_model", "llama3.2:3b")
            return [("ollama", model)]
        # Cloud fallback: prefer faster/cheaper providers first
        for prov in ["groq", "openai", "google", "anthropic"]:
            if prov in self.keys:
                return [(prov, PROVIDER_DEFAULTS.get(prov, prov))]
        # Last resort: any other custom provider
        for slug, cfg in self.custom_configs.items():
            if slug != "ollama":
                return [(slug, cfg["default_model"])]
        return []

    # ── Public streaming API ───────────────────────────────────────────────────

    async def stream(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        messages: list[dict],
    ) -> AsyncGenerator[str, None]:
        dispatch = {
            "anthropic": self._stream_anthropic,
            "google":    self._stream_google,
            "groq":      self._stream_groq,
            "openai":    self._stream_openai,
        }
        fn = dispatch.get(provider)

        # Custom / OpenAI-compatible provider
        if fn is None:
            cfg = self.custom_configs.get(provider)
            if cfg:
                async def _custom_gen(m=model, sp=system_prompt, msgs=messages, c=cfg):
                    async for chunk in self._stream_openai_compat(m, sp, msgs, c["api_key"], c.get("base_url")):
                        yield chunk
                fn = lambda m, sp, msgs: _custom_gen()
            else:
                yield f"⚠️ Proveedor desconocido: {provider}"
                return

        try:
            async for chunk in fn(model, system_prompt, messages):
                yield chunk
        except _RateLimitError:
            # Auto-fallback when quota exhausted: try Groq → OpenAI
            fallback_order = ["groq", "openai", "anthropic"]
            for fb in fallback_order:
                if fb != provider and fb in self.keys:
                    fb_model = PROVIDER_DEFAULTS[fb]
                    fb_fn = dispatch[fb]
                    icon = PROVIDER_INFO[fb]["icon"]
                    yield (
                        f"\n\n> {icon} *{provider.title()} cuota agotada — "
                        f"fallback automático a {fb.title()} `{fb_model}`*\n\n"
                    )
                    async for chunk in fb_fn(fb_model, system_prompt, messages):
                        yield chunk
                    return
            yield f"\n\n⚠️ Cuota de {provider} agotada y no hay proveedor alternativo disponible.\n"

    async def stream_parallel(
        self,
        providers_models: list[tuple[str, str]],
        system_prompt: str,
        messages: list[dict],
    ) -> AsyncGenerator[dict, None]:
        """Run all providers concurrently; yield tagged chunks as they arrive."""
        queue: asyncio.Queue[dict] = asyncio.Queue()
        tasks = []

        for provider, model in providers_models:
            async def _run(prov=provider, mod=model):
                try:
                    async for chunk in self.stream(prov, mod, system_prompt, messages):
                        await queue.put({"type": "text", "provider": prov, "model": mod, "content": chunk})
                except Exception as e:
                    await queue.put({"type": "text", "provider": prov, "model": mod,
                                     "content": f"⚠️ Error ({prov}): {e}"})
                finally:
                    await queue.put({"type": "done_provider", "provider": prov, "model": mod})

            tasks.append(asyncio.create_task(_run()))

        completed: set[str] = set()
        total = len(providers_models)

        while len(completed) < total:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                break
            yield item
            if item["type"] == "done_provider":
                completed.add(item["provider"])

        await asyncio.gather(*tasks, return_exceptions=True)

    # ── Provider implementations ───────────────────────────────────────────────

    async def _stream_anthropic(
        self, model: str, system_prompt: str, messages: list[dict]
    ) -> AsyncGenerator[str, None]:
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=self.keys["anthropic"])
            async with client.messages.stream(
                model=model, max_tokens=4096,
                system=system_prompt, messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            yield f"⚠️ Error Anthropic: {e}"

    async def _stream_google(
        self, model: str, system_prompt: str, messages: list[dict]
    ) -> AsyncGenerator[str, None]:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.keys["google"])

            # Build contents with role "user"/"model" (Gemini convention)
            contents = []
            for msg in messages[:-1]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(
                    types.Content(role=role, parts=[types.Part(text=msg["content"])])
                )
            contents.append(
                types.Content(role="user", parts=[types.Part(text=messages[-1]["content"])])
            )

            cfg = types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=4096,
            )
            async for chunk in await client.aio.models.generate_content_stream(
                model=model, contents=contents, config=cfg
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                raise _RateLimitError(err)
            yield f"⚠️ Error Google AI: {e}"

    async def _stream_groq(
        self, model: str, system_prompt: str, messages: list[dict]
    ) -> AsyncGenerator[str, None]:
        try:
            from groq import AsyncGroq
            client = AsyncGroq(api_key=self.keys["groq"])
            groq_msgs = [{"role": "system", "content": system_prompt}]
            groq_msgs += [{"role": m["role"], "content": m["content"]} for m in messages]
            stream = await client.chat.completions.create(
                messages=groq_msgs, model=model, stream=True, max_tokens=4096,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            yield f"⚠️ Error Groq: {e}"

    async def _stream_openai(
        self, model: str, system_prompt: str, messages: list[dict]
    ) -> AsyncGenerator[str, None]:
        cfg = self.custom_configs.get("openai")
        api_key = cfg["api_key"] if cfg else self.keys.get("openai", "")
        base_url = cfg.get("base_url", "") if cfg else ""
        async for chunk in self._stream_openai_compat(model, system_prompt, messages, api_key, base_url):
            yield chunk

    async def _stream_openai_compat(
        self, model: str, system_prompt: str, messages: list[dict],
        api_key: str, base_url: str = "",
    ) -> AsyncGenerator[str, None]:
        """OpenAI-compatible streaming — works with OpenAI, OpenRouter, Together, Ollama, etc."""
        try:
            from openai import AsyncOpenAI
            kwargs: dict = {"api_key": api_key or "no-key"}
            if base_url:
                kwargs["base_url"] = base_url
            client = AsyncOpenAI(**kwargs)
            oai_msgs = [{"role": "system", "content": system_prompt}]
            oai_msgs += [{"role": m["role"], "content": m["content"]} for m in messages]
            stream = await client.chat.completions.create(
                messages=oai_msgs, model=model, stream=True, max_tokens=4096,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate_limit" in err.lower():
                raise _RateLimitError(err)
            yield f"⚠️ Error ({model}): {e}"
