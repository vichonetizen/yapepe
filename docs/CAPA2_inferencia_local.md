# Capa 2 — Inferencia local (GPU 2 GB)

## Qué resuelve
Permite que el asistente **razone sin internet ni API keys**, usando modelos
pequeños vía Ollama que caben en una GPU de 2 GB. El prompt Maestro completo
(14 módulos × párrafo) satura a un modelo de 1-3B, así que la Capa 2 añade un
**prompt comprimido (V1)** que se activa automáticamente cuando responde un
modelo local.

## Cómo funciona
- `core/modules.py → build_local_system_prompt()` — identidad mínima + reglas
  esenciales + foco por complejidad en UNA línea + los contextos recuperados
  (documentos / memoria / grafo) recortados. Mucho más liviano que el Maestro.
- `meta_controller.orchestrate(local_mode=True)` — usa ese prompt y recorta el
  historial a los últimos 8 turnos (la ventana de contexto de un modelo pequeño
  es limitada).
- `app.py` — cuando la ruta elegida resulta ser **Ollama**, reconstruye el prompt
  en su versión comprimida y recorta los mensajes antes de generar.
- `app.py → _pick_chat_model()` — al detectar Ollama, elige un modelo de **chat**
  e **ignora los de embeddings** (p. ej. `nomic-embed-text`), evitando que el chat
  intente "conversar" con un modelo que solo vectoriza.

## Ruteo híbrido (lo que pasa hoy)
`ai_client.get_route()` decide por complejidad y disponibilidad:
- Si solo tienes **Google** configurado: consultas `simple`/`media` caen a **Ollama
  local** (no hay Groq) y `compleja`/`critica` van a **Google (nube)**.
- Sin ninguna clave de nube: **todo es local** (Ollama).
- Modo Autónomo (RAG-only) y override manual fuerzan local.

Resultado: lo fácil se resuelve gratis y local; lo difícil usa la nube. Para forzar
todo local, quita las claves de nube o usa el Modo Autónomo.

## Modelos recomendados para 2 GB de VRAM
| Modelo | Tamaño | Uso |
|---|---|---|
| `llama3.2:3b` | ~2.0 GB | Mejor calidad que cabe (puede descargar algo a CPU). |
| `gemma2:2b` | ~1.6 GB | Equilibrado, multilingüe. |
| `llama3.2:1b` | ~1.3 GB | El más rápido; respuestas más simples. |
| `nomic-embed-text` | ~0.3 GB | Embeddings (Capa 1), no chat. |

## Pruebas
`python tests/test_local_inference.py` — valida que en modo local se usa el prompt
comprimido (no el Maestro), que el selector ignora embeddings, y que un modelo
local genera una respuesta real no vacía.
