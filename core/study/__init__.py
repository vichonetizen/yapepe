"""
EstudIA — motor de estudio (pista INTEGRADA sobre pentamodal-app).

Monta EstudIA SOBRE la infraestructura existente de pentamodal-app (RAG híbrido,
grafo de conocimiento, memoria, ai_client) en vez de duplicarla. La regla dura del
proyecto —toda Question y todo GradeResult debe llevar `evidence` (chunk_ids
resolubles a una cita con doc_id/página)— se concentra en un único punto:
`grounding.py`. Nada se emite sin pasar por esa compuerta.

Olas de construcción (ver diseño dual-design):
  0. Cimientos de cita y persistencia   -> models.py, store.py, grounding.py  ← ESTA
  1. Comprensión (conceptos/autores/grafo tipado)
  2. Motor de estudio v1 (focus/generate/grade/session)
  3. Estructuras pedagógicas (mapa/jerarquía/comparativa/debate)
  4. Memoria FSRS (cards/plan/panel/examen)
  5. Aprendizaje adaptativo (mastery/clustering/reordenamiento)

Convenciones heredadas de pentamodal-app: todo en español, SQLite única
(data/pentamodal.db), degradación elegante, local-first.
"""

__version__ = "0.1.0"
