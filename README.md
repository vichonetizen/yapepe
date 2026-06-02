# Pentamodal — Asistente de inteligencia operativa local

Asistente de IA con arquitectura **pentamodal adaptativa** que corre en tu máquina,
aprende de tus documentos y conversaciones, relaciona ideas por significado y puede
razonar **localmente en una GPU de 2 GB** (sin internet ni API keys).

No es un chat pasivo: diagnostica, anticipa, diseña, se adapta y modela intenciones
antes de responder, y **mejora con el uso**.

---

## Las 4 capas cognitivas

| Capa | Qué hace | Módulo |
|---|---|---|
| **1 · Memoria semántica** | Relaciona ideas por **significado** (embeddings), no solo por palabras. Búsqueda híbrida BM25 + vectorial. | `core/semantic_memory.py` |
| **2 · Inferencia local 2 GB** | Razona con modelos pequeños vía Ollama. Prompt comprimido para modelos 1-3B + fallback si no caben en la GPU. | `core/modules.py`, `core/ai_client.py` |
| **3 · Recetas de tareas** | Aprende **qué enfoque funciona para qué pedido** y lo **replica** ante tareas similares. | `core/recipe_store.py` |
| **4 · Autonomía** | Consolida la memoria sola cada N horas: fusiona, poda, deduplica, indexa pendientes. | `core/autonomy.py` |

Sobre ellas operan los módulos pentamodales (Diagnóstico, Predictivo, Generativo,
Reactivo, Teoría de la Mente, Planificación, Evaluación, Memoria, Grafo, Optimización,
Razonamiento, Refuerzo, Patrones, Autoprogramación) que se activan según la
complejidad detectada (`core/meta_controller.py`).

---

## Instalación

```bat
setup.bat            :: instala dependencias Python y crea .env
```

Requisitos: **Python 3.11+** (probado en 3.14) y, para el modo local, **Ollama**.

### Camino LOCAL recomendado (GPU 2 GB, gratis, sin internet)
```bat
:: Instala Ollama → https://ollama.com/download
ollama pull bge-m3            :: embeddings multilingües (Capa 1)
ollama pull llama3.2:1b       :: chat que cabe en 2 GB de VRAM (Capa 2)
```
> En 2 GB de VRAM corre `llama3.2:1b`. Modelos de 2-3B (`gemma2:2b`, `llama3.2:3b`)
> requieren más memoria y pueden no caber; la app cae automáticamente a uno menor.

### Camino NUBE (opcional)
Pon tus API keys en `.env` o en ⚙ Config (Google / Groq / OpenAI / Anthropic).
El ruteo es híbrido: lo simple se resuelve local y lo complejo en la nube.

---

## Uso

```bat
run.bat              :: arranca en http://127.0.0.1:8000
```
Al iniciar detecta Ollama, indexa documentos en local y empieza a consolidar memoria.

**Modos en la interfaz:**
- **Autónomo (RAG-only):** responde solo desde tus documentos, sin nube.
- **PAC:** protocolo de aprendizaje continuo (cita fuentes, registra aprendizajes).
- **Maestro:** Instrucción Maestra v2.0 (refuerzo + patrones + autocrítica).
- **Paralelo:** varias IAs responden a la vez para comparar.

Sube documentos (.pdf .docx .txt .md .csv) en la pestaña 📄 Documentos; se indexan
y quedan disponibles para búsqueda semántica. Califica las respuestas: con nota alta
en tareas con sustancia, el sistema **guarda una receta** para replicar el enfoque.

---

## Endpoints clave
- `GET /semantic/status` · `POST /semantic/reindex` — estado/indexación de embeddings.
- `GET /recipes` · `GET /recipes/stats` — recetas aprendidas.
- `POST /autonomy/consolidate` · `GET /autonomy/status` — consolidación de memoria.
- `GET /stats` · `GET /stats/neural` — métricas y evolución del sistema.

---

## Pruebas

```bat
python tests/run_all.py        :: las 7 pruebas integrales
```
Cubren: arranque, las 4 capas, el estado de la base de conocimiento y una consulta
semántica real en español. Cada prueba de capa también se ejecuta por separado
(`tests/test_*.py`).

---

## Privacidad y datos
- Todo es **local**. La interfaz exige un **token local** (`data/local.token`).
- `.env`, la base de datos (`data/`) y los documentos (`documents/`) están en
  `.gitignore`: nunca se suben al repositorio.

---

## Configuración (.env)
| Variable | Por defecto | Descripción |
|---|---|---|
| `PORT` / `HOST` | `8000` / `127.0.0.1` | Dónde escucha el servidor. |
| `AUTONOMY_INTERVAL_HOURS` | `6` | Cada cuántas horas consolida la memoria (0 = off). |
| `GOOGLE_API_KEY` … | — | Claves de nube opcionales. |

Detalle por capa en `docs/CAPA1_*`, `docs/CAPA2_*`.
