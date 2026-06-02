# Desplegar en Vercel

La app detecta automáticamente Vercel (`VERCEL=1`) y se adapta. **La versión local
no se ve afectada** — sigues usando `run.bat` como siempre.

## ⚠️ Qué cambia en Vercel (léelo antes)
Vercel ejecuta funciones serverless (sin GPU, sin proceso persistente, disco efímero).
Por eso en Vercel:

| Funciona | No funciona / cambia |
|---|---|
| ✅ Interfaz web y chat | ❌ Ollama / inferencia local (no hay GPU) → usa API de nube |
| ✅ Módulos pentamodales y prompts | ❌ Acceso a archivos/portapapeles del equipo (device) |
| ✅ Las 4 capas a nivel de código | ⚠️ Procesos en segundo plano (autonomía) → usar Vercel Cron |
| ✅ Cloud chat (Google/Groq/OpenAI) | ⚠️ **Memoria/RAG no persisten** entre invocaciones (FS efímero) |

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
   | `LOCAL_TOKEN` | una cadena secreta cualquiera (p. ej. 32+ chars) | sí |
   | `AUTONOMY_INTERVAL_HOURS` | `0` | recomendado (no hay bucle en serverless) |
   | `DATABASE_URL` | URL de DB externa (opcional, ver persistencia) | no |
4. **Deploy**. Abre la URL: verás la interfaz y podrás chatear (vía nube).

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
llamando periódicamente a `POST /autonomy/consolidate` (requiere enviar el header
`X-Local-Token`). Configúralo en el dashboard de Vercel o con una función cron
dedicada que incluya ese header.

## Límites a tener en cuenta
- **Duración:** las respuestas en streaming están limitadas por `maxDuration`
  (60 s en `vercel.json`; el plan Hobby puede recortar a ~10 s). Respuestas largas
  podrían cortarse — el plan Pro las permite completas.
- **Tamaño de función:** numpy + los SDKs de IA son pesados. Si el build supera el
  límite de Vercel, quita de `requirements.txt` los proveedores que no uses.
- **Costo:** en Vercel todo el razonamiento usa API de nube (no es local ni gratis).

## Resumen
- **Local (run.bat):** experiencia completa — local, gratis, con GPU y memoria persistente.
- **Vercel:** acceso web desde cualquier lugar, en modo nube; para memoria persistente
  conecta una DB externa.
