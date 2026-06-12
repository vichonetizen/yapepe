"""Prueba del ruteo por complejidad de AIClient (core/ai_client.py).

Valida get_route(): que cada complejidad ('simple'/'media'/'compleja'/'critica')
devuelva el proveedor/modelo esperado segun las claves disponibles, y el
comportamiento de fallback (orden de nube, Ollama local-first, complejidad
desconocida, custom y sin claves). No llama a ninguna API externa: solo inyecta
claves con update_key/register_custom y consulta get_route.

Ejecutar:  python tests/test_routing.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_client import (
    AIClient,
    ROUTING,
    PROVIDER_DEFAULTS,
)


def _ok(cond, label):
    print(("  OK  " if cond else "  XX  ") + label)
    if not cond:
        raise AssertionError(label)


def _fresh_client():
    """AIClient sin ninguna clave: aislamos del entorno reseteando keys y configs."""
    c = AIClient()
    c.keys = {}
    c.custom_configs = {}
    return c


def main():
    print("\n[0] Sanidad de la tabla ROUTING (lo que el codigo realmente define)")
    _ok(ROUTING["simple"] == [("groq", "llama-3.1-8b-instant")],
        "ROUTING['simple'] -> groq llama-3.1-8b-instant")
    _ok(ROUTING["media"] == [("groq", "llama-3.3-70b-versatile")],
        "ROUTING['media'] -> groq llama-3.3-70b-versatile")
    _ok(ROUTING["compleja"] == [("google", "gemini-2.0-flash")],
        "ROUTING['compleja'] -> google gemini-2.0-flash")
    _ok(ROUTING["critica"] == [("google", "gemini-2.0-flash"),
                               ("groq", "llama-3.3-70b-versatile")],
        "ROUTING['critica'] -> google + groq (en paralelo)")

    print("\n[1] Todas las claves presentes: cada complejidad usa su ruta directa")
    c = _fresh_client()
    c.update_key("groq", "gk-test")
    c.update_key("google", "gg-test")
    # update_key debe guardar la clave (recortada) tal cual
    _ok(c.keys.get("groq") == "gk-test", "update_key inyecta la clave de groq")
    _ok("groq" in c.available_providers() and "google" in c.available_providers(),
        "available_providers refleja las claves inyectadas")

    _ok(c.get_route("simple") == [("groq", "llama-3.1-8b-instant")],
        "simple -> groq llama-3.1-8b-instant")
    _ok(c.get_route("media") == [("groq", "llama-3.3-70b-versatile")],
        "media -> groq llama-3.3-70b-versatile")
    _ok(c.get_route("compleja") == [("google", "gemini-2.0-flash")],
        "compleja -> google gemini-2.0-flash")
    _ok(c.get_route("critica") == [("google", "gemini-2.0-flash"),
                                   ("groq", "llama-3.3-70b-versatile")],
        "critica -> google y groq, ambos en paralelo")

    print("\n[2] Filtrado por clave faltante dentro de la ruta 'critica'")
    # Solo google: critica conserva unicamente las parejas cuyo proveedor tiene clave
    c = _fresh_client()
    c.update_key("google", "gg-test")
    _ok(c.get_route("critica") == [("google", "gemini-2.0-flash")],
        "critica con solo google -> deja unicamente la pareja de google")
    # 'simple' apunta a groq (sin clave) y google NO esta en esa ruta:
    # no hay candidatos disponibles -> fallback de nube al unico proveedor con clave
    _ok(c.get_route("simple") == [("google", PROVIDER_DEFAULTS["google"])],
        "simple sin groq -> fallback de nube a google con su modelo por defecto")

    print("\n[3] Fallback de nube: orden de preferencia groq>openai>google>anthropic")
    # 'compleja' pide google; sin google pero con varias claves de nube,
    # gana el primero del orden de preferencia que tenga clave.
    c = _fresh_client()
    c.update_key("anthropic", "an-test")
    c.update_key("openai", "oa-test")
    # compleja -> google no disponible; entre openai y anthropic gana openai
    _ok(c.get_route("compleja") == [("openai", PROVIDER_DEFAULTS["openai"])],
        "compleja sin google -> fallback prefiere openai sobre anthropic")
    # Si ademas hay groq, groq encabeza la preferencia
    c.update_key("groq", "gk-test")
    _ok(c.get_route("compleja") == [("groq", PROVIDER_DEFAULTS["groq"])],
        "con groq presente el fallback de nube elige groq primero")

    print("\n[4] Ollama es local-first cuando la ruta directa no tiene candidatos")
    c = _fresh_client()
    # Ollama registrado como custom + clave de nube (anthropic) presente:
    # get_route prioriza ollama por encima del fallback de nube.
    c.register_custom("ollama", "ollama", "http://localhost:11434/v1", "qwen2.5:7b")
    c.update_key("anthropic", "an-test")
    _ok(c.get_route("compleja") == [("ollama", "qwen2.5:7b")],
        "compleja sin google -> ollama local-first con su default_model")
    _ok(c.get_route("simple") == [("ollama", "qwen2.5:7b")],
        "simple -> ollama local-first (gana a la nube)")

    print("\n[5] Ollama en keys sin default_model configurado -> modelo por defecto")
    c = _fresh_client()
    # Forzamos 'ollama' en keys sin entrada en custom_configs:
    # get_route usa el default cableado 'llama3.2:3b'
    c.keys["ollama"] = "ollama"
    _ok(c.get_route("media") == [("ollama", "llama3.2:3b")],
        "ollama sin config -> usa el modelo por defecto llama3.2:3b")

    print("\n[6] Complejidad desconocida cae a la ruta 'media'")
    c = _fresh_client()
    c.update_key("groq", "gk-test")
    _ok(c.get_route("desconocida") == ROUTING["media"],
        "complejidad invalida -> usa ROUTING['media'] (groq 70b)")

    print("\n[7] Solo proveedor custom (no-ollama) -> ultimo recurso")
    c = _fresh_client()
    c.register_custom("openrouter", "or-test",
                      "https://openrouter.ai/api/v1", "meta-llama/llama-3.1-70b")
    _ok(c.get_route("simple") == [("openrouter", "meta-llama/llama-3.1-70b")],
        "sin claves estandar -> usa el custom con su default_model")

    print("\n[8] Sin ninguna clave -> lista vacia")
    c = _fresh_client()
    _ok(c.get_route("simple") == [], "sin claves get_route devuelve []")
    _ok(c.get_route("critica") == [], "sin claves critica tambien devuelve []")

    print("\n[9] update_key con cadena vacia elimina la clave")
    c = _fresh_client()
    c.update_key("groq", "gk-test")
    _ok("groq" in c.keys, "groq presente tras inyectar")
    c.update_key("groq", "   ")
    _ok("groq" not in c.keys, "update_key('') elimina la clave de groq")
    _ok(c.get_route("simple") == [], "tras eliminar la unica clave -> ruta vacia")

    print("\nEXITO: ROUTING VALIDADO\n")


if __name__ == "__main__":
    main()
