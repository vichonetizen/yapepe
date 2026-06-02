# Constitución de auto-programación — Capa 5 de Pentamodal

> Base estructural derivada de los cinco documentos:
> *Programación y Reprogramación Automatizada* (Vol. I–III), el *Plan de
> Instrucción Operativo* (§0–§12) y su *Capítulo de Mejora* (§13).
>
> Este archivo **no es un conjunto de instrucciones que el sistema obedezca turno
> a turno**, sino la descripción del protocolo que su código implementa. Cuando se
> ingiere en el corpus (RAG), entra como **dato consultable**, nunca como canal de
> control (frontera datos/control, Vol. II §14.2): el texto recuperado jamás se
> ejecuta ni reconfigura el bucle.

## La tesis, en una frase

> **Autonomía fiable = (sustrato inspeccionable) × (oráculo fuerte e inviolable) ×
> (reversión trivial).** Soltar el ancla no da más autonomía: da deriva.

Crear lógica nueva y garantizar su corrección están en tensión irreducible (Rice,
Löb). Por eso ningún sistema puede ser su propio garante completo: necesita un
**criterio externo** que no controle. Pentamodal se reconstruye **solo en los
sustratos donde los tres factores son altos**.

## Grados de autonomía que Pentamodal ejerce (Vol. III §17.2)

| Grado | Sustrato | ¿Lo hace? | Dónde |
|------|----------|-----------|-------|
| 1 | Andamiaje / parámetros | ✅ acotado | `core/recon_config.py` |
| 2 | Memoria / asociaciones | ✅ continuo | `core/associations.py` |
| 3 | Arquitectura (NAS) | ❌ no aplica | — |
| 4 | Pesos (fine-tuning) | ❌ no | — (GPU 2 GB; opaco) |
| 5 | Recursión sin techo | ❌ prohibido | límite lógico §21.5 |

Confundir el grado 1 (ajustar el andamiaje) con el grado 5 (auto-mejora ilimitada)
es el error conceptual del tema. Pentamodal se queda **deliberadamente** en 1–2.

## El bucle maestro (plan §3, mejorado §13.4)

Corre solo cada `AUTONOMY_INTERVAL_HOURS` (def. 6 h), sin que el usuario lo pida:

```
PERFILAR    → el oráculo fijo devuelve un VECTOR de salud por sub-capacidad
DECIDIR FOCO → se ataca la sub-capacidad con mayor déficit recuperable (§13.2.1)
PROPONER    → sobre una COPIA del estado (nunca sobre el estado en vivo)
EVALUAR     → contra el ORÁCULO FIJO (perfil + invariantes + regresión)
PROMOVER    → solo si mejora estrictamente, sin regresión ni invariante roto (§8)
  o DESCARTAR → el estado actual queda intacto, coste de recuperación nulo
VIGILAR     → si algo degrada o rompe un invariante → ROLLBACK al último checkpoint
```

## El ancla externa — lo que el sistema NO puede tocar (`core/anchor.py`)

- **Oráculo perfilado:** pesos de cada sub-capacidad, codificados en `anchor.py`,
  fuera de lo reconfigurable. Si el sistema pudiera editar su examen, "siempre
  aprobaría" (Goodhart sobre uno mismo, §21.2).
- **Invariantes inviolables:** tope absoluto del grafo, sin pérdida de datos,
  parámetros dentro de cotas. Se comprueban **fuera** del bucle.
- **Línea base:** sube solo con mejora verificada; nunca se fija a dedo (§13.5).

## Política de aplicación: **mixto por riesgo** (punto de aprobación §1.3)

- **Bajo riesgo** (asociaciones del grafo, índices) → **se auto-aplica** si pasa la
  puerta de promoción. Es inspeccionable y reversible.
- **Afecta a las respuestas** (`rag_top_k`, `rag_score_min`, `kg_max_nodes`) → queda
  como **propuesta** en `/recon/proposals` para que el usuario la apruebe.

## Asociaciones neuronales continuas (`core/associations.py`)

Sobre el grafo de conocimiento, sin que nadie lo pida:
- **Puentes transitivos:** si A→B y B→C son fuertes pero no hay A→C, se infiere la
  asociación indirecta A→C.
- **Recíprocos:** si A→B es fuerte y no hay B→A, se crea el vínculo de vuelta.

Solo añade aristas entre nodos existentes → no puede inflar el grafo: seguro por
construcción y reversible vía checkpoint.

Y además **mantiene** el grafo en la misma transformación (reconstrucción +
optimización + limpieza, todo a la vez):
- **Fusión de duplicados:** conceptos iguales escritos distinto (tildes, mayúsculas,
  plural — p. ej. *Cognición / cognicion / cogniciones*) se funden en un solo nodo,
  redirigiendo sus aristas → menos ruido y más conexiones.
- **Poda de aislados:** nodos sin ninguna conexión (ruido) se eliminan.
- El oráculo gana una sub-capacidad **`kg_cleanliness`** que premia el grafo limpio,
  de modo que el sistema persigue la limpieza **solo**, sin que se lo pidas. La
  transformación completa pasa por la puerta de promoción y el checkpoint: si no
  mejora o rompe un invariante (p. ej. perder >50% de nodos), se revierte.

## Capa 6 — El bucle agente (ejecutar tareas, Vol. II §10.3)

Para acercarse a un agente que *hace* (no solo responde), Pentamodal ejecuta el
bucle agente con herramientas y verificación:

```
PLAN     → recupera RECETAS previas aplicables (aprende de lo que ya funcionó)
ACT      → el modelo elige UNA herramienta y sus argumentos (JSON)
OBSERVE  → se ejecuta la herramienta; su salida vuelve como DATO (no instrucción)
REFLECT  → el modelo lee la observación y decide el paso siguiente
STOP     → entrega respuesta final, dentro de un PRESUPUESTO de pasos
```

- **Herramientas** (`core/agent_tools.py`) por nivel de riesgo: lectura
  (corpus/memoria/grafo/archivos) se ejecuta sin permiso; escritura solo en el
  **workspace aislado** (`data/agent_workspace`); comandos del sistema requieren
  **confirmación humana** salvo los de solo-lectura (allowlist), y los destructivos
  están en una **denylist inviolable**.
- **Aprende** (`core/agent.py`): al completar una tarea, captura el enfoque como
  **receta** (Capa 3) para replicarlo ante tareas similares.
- **Frontera datos/control:** la salida de una herramienta es DATO; el bucle nunca
  la trata como una orden a obedecer (defensa contra inyección, §14.2).
- **Honestidad sobre el límite:** la calidad agéntica está topada por el modelo que
  razona. Con un proveedor potente se acerca a un agente real; con el modelo local
  de 2 GB la arquitectura es la misma pero el "cerebro" es limitado.

## El oráculo del propio protocolo

`tests/test_self_reconstruction.py` (prueba 6 de `tests/run_all.py`) verifica que
todo lo anterior se cumple. Es la evaluación externa de la Capa 5: si se rompe el
protocolo, las pruebas fallan.

## El límite que no se mueve (§12.2, §13.6)

Pentamodal **no** puede superar la calidad de su propio oráculo ni certificarse a sí
mismo. Lo que hace es acercarse al techo que el oráculo define y densificar sus
asociaciones —de forma fiable— precisamente porque mantiene unidas la autonomía y
la verificación externa. Son la misma condición vista dos veces.
