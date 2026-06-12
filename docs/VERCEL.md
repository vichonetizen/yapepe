# Desplegar en Vercel

La app detecta automáticamente Vercel (`VERCEL=1`) y se adapta. **La versión local
no se ve afectada** — sigues usando `run.bat` como siempre.

## 🔒 Importante: el chat web NO funciona en Vercel (decisión deliberada)

Desde el endurecimiento de seguridad (commit `f0bb454`), la página servida en `/`
**solo embebe el token de acceso cuando la petición viene de loopback** (tu propio
equipo). En Vercel ninguna petición es local, así que la interfaz carga pero **toda
llamada a la API responde 403**: el chat web no funciona allí. Es una decisión
*fail-closed* consciente — la alternativa sería entregar el token a cualquier
visitante anónimo, dándole acceso completo a la API.

Lo único que sobrevive en Vercel es el **uso programático con `LOCAL_TOKEN`**
(p. ej. Vercel Cron enviando el header `X-Local-Token`), con un **requisito** de
longitud mínima de 32 caracteres (ver tabla de variables). Si algún día quieres
chat web público, hará falta autenticación real (login), no este token.

## ⚠️ Qué cambia en Vercel (léelo antes)
Vercel ejecuta funciones serverless (sin GPU, sin proceso persistente, disco efímero).
Por eso en Vercel:

| Funciona | No funciona / cambia |
|---|---|
| ✅ API vía `X-Local-Token` (cron/scripts) | ❌ **Chat web** (la página no recibe token → API responde 403) |
| ✅ Módulos pentamodales y prompts | ❌ Ollama / inferencia local (no hay GPU) → usa API de nube |
| ✅ Las capas a nivel de código | ❌ Acceso a archivos/portapapeles del equipo (device) |
| ✅ Proveedores de nube (Google/Groq/OpenAI) | ⚠️ Procesos en segundo plano (autonomía) → usar Vercel Cron |
| | ⚠️ **Memoria/RAG no persisten** entre invocaciones (FS efímero) |

> Para que la memoria, los documentos y los embeddings **persistan**, necesitas una
> base de datos externa (ver más abajo). Sin ella, cada arranque en frío empieza
> con la base vacía.

## Pasos para desplegar
1. En [vercel.com](https://vercel.com) → **Add New → Project** → importa
   `github.com/vichonetizen/yapepe`.
2. Framework Preset: **Other** (la app usa `vercel.json` + `api/index.py`).
3. En **Settings → Environment Variables**, añade:
   | Variable | Valor | Obligatoria |
   |---|---|---|
   | `GOOGLE_API_KEY` | tu clave de Google AI Studio | sí (o GROQ/OPENAI) |
   | `LOCAL_TOKEN` | cadena secreta de **mínimo 32 caracteres** (requisito, no sugerencia: si es más corta se **descarta** con un warning y cada instancia genera la suya → todas tus llamadas reciben 403) | sí |
   | `AUTONOMY_INTERVAL_HOURS` | `0` | recomendado (no hay bucle en serverless) |
   | `DATABASE_URL` | URL de DB externa (opcional, ver persistencia) | no |
4. **Deploy**. Abre la URL: verás la interfaz, pero el chat web responderá 403
   (ver arriba). Para usar la API, envía el header `X-Local-Token: <LOCAL_TOKEN>`
   desde tu cron o tus scripts. Comprobación rápida: `GET /api/health` → `{"ok": true}`.

## Persistencia (memoria + RAG en Vercel)
El SQL del proyecto es de estilo SQLite. La opción compatible más simple es
**[Turso](https://turso.tech) (libSQL)** o cualquier SQLite remoto:
1. Crea una base en Turso y obtén su URL.
2. Ponla en `DATABASE_URL` (formato `sqlite+aiosqlite`/libSQL compatible).
3. Indexa el corpus una vez (los embeddings se guardan en esa DB) llamando a
   `POST /semantic/reindex` tras subir documentos.
> Con `DATABASE_URL` apuntando a una DB externa, la memoria, los aprendizajes,
> las recetas y los embeddings persisten entre invocaciones.

## Autonomía vía Cron (opcional)
Como no hay proceso persistente, programa la consolidación con Vercel Cron
llamando periódicamente a `POST /autonomy/consolidate` con el header
`X-Local-Token: <LOCAL_TOKEN>`. Esto **solo funciona si `LOCAL_TOKEN` tiene ≥32
caracteres** (si no, se descarta y el cron recibirá 403). Configúralo en el
dashboard de Vercel o con una función cron dedicada que incluya ese header.

## Límites a tener en cuenta
- **Duración:** las respuestas en streaming están limitadas por `maxDuration`
  (60 s en `vercel.json`; el plan Hobby puede recortar a ~10 s). Respuestas largas
  podrían cortarse — el plan Pro las permite completas.
- **Tamaño de función:** numpy + los SDKs de IA son pesados. Si el build supera el
  límite de Vercel, quita de `requirements.txt` los proveedores que no uses.
- **Costo:** en Vercel todo el razonamiento usa API de nube (no es local ni gratis).

## Resumen
- **Local (run.bat):** experiencia completa — local, gratis, con GPU y memoria persistente.
- **Vercel:** solo uso programático de la API (cron/scripts con `X-Local-Token` y
  `LOCAL_TOKEN` ≥32 chars); el chat web está deshabilitado por diseño (fail-closed).
  Para memoria persistente conecta una DB externa.
