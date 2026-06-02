from typing import AsyncGenerator
import anthropic
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ANTHROPIC_API_KEY, MODEL


class ClaudeClient:
    def __init__(self):
        self._api_key = ANTHROPIC_API_KEY
        self._model = MODEL

    def update_key(self, key: str):
        self._api_key = key

    async def stream(
        self,
        system_prompt: str,
        messages: list[dict],
    ) -> AsyncGenerator[str, None]:
        if not self._api_key:
            yield "⚠️ API key no configurada. Ingresa tu ANTHROPIC_API_KEY en el archivo .env o en la configuración de la aplicación."
            return

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        try:
            async with client.messages.stream(
                model=self._model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.AuthenticationError:
            yield "⚠️ API key inválida. Verifica tu ANTHROPIC_API_KEY."
        except anthropic.RateLimitError:
            yield "⚠️ Límite de velocidad alcanzado. Intenta de nuevo en unos segundos."
        except Exception as e:
            yield f"⚠️ Error al conectar con Claude: {str(e)}"
