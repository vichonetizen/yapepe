import json
import re
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.memory import (
    get_session_history, get_relevant_facts, ModulePerformance, Message
)
from core.modules import (
    ACTIVATION_MATRIX, MAESTRO_ACTIVATION_MATRIX,
    assess_complexity, build_system_prompt, build_maestro_system_prompt,
    build_rag_only_system_prompt, build_local_system_prompt, MODULE_NAMES
)
from core.knowledge_graph import KnowledgeGraph
from core.document_store import search_documents
from core.semantic_memory import hybrid_search_documents
from core.learning_engine import search_learnings, build_pac_context
from core.recipe_store import search_recipes, build_recipe_context
import core.recon_config as recon_config


class MetaController:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def _build_rl_context(self, history: list) -> str:
        """Derive style preferences from conversation patterns."""
        if len(history) < 4:
            return ""
        lines = []
        # Check for reformulations (user repeated similar questions)
        user_msgs = [m.content for m in history if m.role == "user"]
        if len(user_msgs) >= 2:
            # Detect if last user msg is significantly shorter → user wants brevity
            avg_len = sum(len(m) for m in user_msgs[:-1]) / max(len(user_msgs) - 1, 1)
            last_len = len(user_msgs[-1])
            if last_len < avg_len * 0.4 and avg_len > 150:
                lines.append("• El usuario tiende a preferir respuestas concisas (últimas preguntas son cortas).")
        # Check rated messages for pattern
        rated = [m for m in history if hasattr(m, "rating") and m.rating and m.role == "assistant"]
        if rated:
            avg_r = sum(m.rating for m in rated) / len(rated)
            if avg_r >= 4.0:
                lines.append(f"• Las respuestas anteriores tuvieron calificación alta (promedio {avg_r:.1f}/5) — mantén el estilo actual.")
            elif avg_r <= 2.5:
                lines.append(f"• Las respuestas anteriores tuvieron calificación baja (promedio {avg_r:.1f}/5) — modifica el enfoque.")
        return "\n".join(lines) if lines else ""

    def _detect_patterns(self, history: list) -> str:
        """Surface recurring topics and contradictions in the conversation."""
        if len(history) < 6:
            return ""
        user_texts = " ".join(m.content.lower() for m in history if m.role == "user")
        topic_counts: dict[str, int] = {}
        # Simple frequency-based topic detection for common technical domains
        topic_keywords = {
            "autopoiesis": ["autopoiesis", "autopoiética", "maturana", "varela"],
            "cognición": ["cognición", "cognitivo", "conocimiento", "aprendizaje"],
            "sistemas": ["sistema", "arquitectura", "diseño", "componente"],
            "código": ["función", "clase", "módulo", "implementar", "código", "python"],
            "documentos": ["documento", "pdf", "fragmento", "texto", "archivo"],
            "IA/ML": ["modelo", "llm", "embeddings", "vector", "transformer", "neural"],
        }
        for topic, kws in topic_keywords.items():
            count = sum(user_texts.count(kw) for kw in kws)
            if count >= 2:
                topic_counts[topic] = count
        if not topic_counts:
            return ""
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        lines = ["Temas recurrentes detectados en esta conversación (úsalos como contexto implícito):"]
        for topic, cnt in sorted_topics:
            lines.append(f"• **{topic}** (mencionado {cnt} veces)")
        return "\n".join(lines)

    async def _get_adjusted_modules(
        self,
        complexity: str,
        db: AsyncSession,
        maestro_mode: bool = False,
    ) -> list[str]:
        matrix = MAESTRO_ACTIVATION_MATRIX if maestro_mode else ACTIVATION_MATRIX
        base = matrix[complexity].copy()

        result = await db.execute(
            select(ModulePerformance)
            .where(ModulePerformance.complexity == complexity)
            .where(ModulePerformance.sample_count >= 3)
            .order_by(ModulePerformance.avg_rating.desc())
        )
        top_perfs = result.scalars().all()

        if not top_perfs:
            return base

        best_combo_str = top_perfs[0].modules_combo
        best_combo = set(best_combo_str.split(","))
        base_set = set(base)

        extra = best_combo - base_set
        for mod in extra:
            if mod in MODULE_NAMES and top_perfs[0].avg_rating > 3.5:
                base.append(mod)

        return base

    async def orchestrate(
        self,
        user_input: str,
        session_id: str,
        db: AsyncSession,
        pac_mode: bool = False,
        rag_only: bool = False,
        maestro_mode: bool = False,
        local_mode: bool = False,
    ) -> dict:
        history = await get_session_history(session_id, db, limit=12)
        complexity = assess_complexity(user_input, len(history))
        active_modules = await self._get_adjusted_modules(complexity, db, maestro_mode=maestro_mode)

        memory_context = ""
        if "MEM" in active_modules:
            facts = await get_relevant_facts(user_input, db, limit=5)
            if facts:
                memory_context = "\n".join(f"• {f}" for f in facts)

        kg_context = ""
        if "KG" in active_modules:
            kg_context = self.kg.get_context(user_input)

        # RAG contextual: enriquecer la query con el historial reciente
        doc_context = ""
        doc_hits: list[dict] = []
        try:
            search_query = user_input
            if history and len(history) >= 2:
                # Últimos 3 mensajes del usuario como contexto de búsqueda
                recent_user = " ".join(
                    m.content[:120] for m in history[-6:] if m.role == "user"
                )
                search_query = f"{user_input} {recent_user}"
            # Búsqueda híbrida: BM25 (léxico) + embeddings (significado).
            # Si no hay backend de embeddings, cae automáticamente a BM25 puro.
            # top_k y umbral los puede auto-ajustar la Capa 5 (recon_config), con
            # los valores de fábrica de siempre como defecto.
            top_k = int(recon_config.get("rag_top_k"))
            score_min = float(recon_config.get("rag_score_min"))
            raw_hits = await hybrid_search_documents(search_query, top_k=top_k)
            # Filtrar fragmentos con relevancia casi nula
            doc_hits = [h for h in raw_hits if h["score"] > score_min]
            if doc_hits:
                chunks = []
                for hit in doc_hits:
                    chunks.append(f"[{hit['filename']}]\n{hit['content'][:600]}")
                doc_context = "\n\n---\n".join(chunks)
        except Exception:
            pass

        # PAC: retrieve relevant learnings from previous conversations
        pac_context = ""
        if pac_mode:
            try:
                past_learnings = await search_learnings(user_input, top_k=5)
                pac_context = build_pac_context(past_learnings)
            except Exception:
                pass

        # Capa 3: recetas de tareas previas aplicables a esta consulta
        recipe_context = ""
        recipe_ids: list[int] = []
        try:
            recipes = await search_recipes(user_input, top_k=2)
            if recipes:
                recipe_context = build_recipe_context(recipes)
                recipe_ids = [r["id"] for r in recipes]
        except Exception:
            pass

        # Maestro mode: RL context + pattern detection
        rl_context = ""
        pattern_context = ""
        if maestro_mode:
            rl_context = self._build_rl_context(history)
            pattern_context = self._detect_patterns(history)

        # RAG-only mode: answer exclusively from documents
        if rag_only:
            # Provide conversation history as plain text context
            history_lines = []
            for m in history[-6:]:
                role = "Usuario" if m.role == "user" else "Asistente"
                history_lines.append(f"{role}: {m.content[:300]}")
            history_ctx = "\n".join(history_lines) if history_lines else ""
            system_prompt = build_rag_only_system_prompt(
                doc_context, pac_context, history_ctx
            )
            # In RAG-only mode, send only the current message (no history in messages list,
            # since history is embedded in the system prompt to keep context tight)
            api_messages = [{"role": "user", "content": user_input}]
        elif local_mode:
            # Inferencia local (Ollama 1-3B): prompt comprimido (V1) + historial corto
            # para no saturar la ventana de contexto del modelo pequeño.
            system_prompt = build_local_system_prompt(
                active_modules, complexity, memory_context, kg_context,
                doc_context, pac_context, pac_mode, recipe_context=recipe_context,
            )
            api_messages = [
                {"role": m.role, "content": m.content}
                for m in history[-8:]
            ]
            api_messages.append({"role": "user", "content": user_input})
        elif maestro_mode:
            system_prompt = build_maestro_system_prompt(
                active_modules, complexity, memory_context, kg_context,
                doc_context, pac_context, pac_mode,
                rl_context=rl_context, pattern_context=pattern_context,
                recipe_context=recipe_context,
            )
            api_messages = [
                {"role": m.role, "content": m.content}
                for m in history[-12:]
            ]
            api_messages.append({"role": "user", "content": user_input})
        else:
            system_prompt = build_system_prompt(
                active_modules, complexity, memory_context, kg_context,
                doc_context, pac_context, pac_mode, recipe_context=recipe_context,
            )
            api_messages = [
                {"role": m.role, "content": m.content}
                for m in history[-12:]
            ]
            api_messages.append({"role": "user", "content": user_input})

        return {
            "complexity": complexity,
            "modules": active_modules,
            "system_prompt": system_prompt,
            "messages": api_messages,
            "doc_hits": doc_hits,
            "pac_mode": pac_mode,
            "rag_only": rag_only,
            "maestro_mode": maestro_mode,
            "local_mode": local_mode,
            # Contextos crudos: permiten reconstruir un prompt compacto si la ruta
            # finalmente resulta ser un modelo local (Ollama).
            "memory_context": memory_context,
            "kg_context": kg_context,
            "doc_context": doc_context,
            "pac_context": pac_context,
            "recipe_context": recipe_context,
            "recipe_ids": recipe_ids,
        }

    def extract_memory_facts(self, user_text: str, assistant_text: str) -> list[dict]:
        facts = []
        user_sentences = [s.strip() for s in re.split(r'[.!?\n]', user_text) if 15 < len(s.strip()) < 250]

        preference_signals = [
            "prefiero", "me gusta", "siempre uso", "mi forma de", "no me gusta",
            "nunca uso", "trabajo con", "uso siempre", "odio", "me encanta",
            "I prefer", "I like", "I always use", "I never",
        ]
        for sentence in user_sentences:
            sl = sentence.lower()
            for signal in preference_signals:
                if signal in sl:
                    facts.append({
                        "category": "user_preference",
                        "content": sentence[:250],
                    })
                    break

        tech_signals = [
            "framework", "librería", "stack", "base de datos", "database", "lenguaje",
            "arquitectura", "proyecto", "empresa", "equipo", "python", "javascript",
            "typescript", "react", "fastapi", "django", "postgres", "mysql", "mongodb",
            "docker", "kubernetes", "aws", "gcp", "azure", "estoy construyendo",
            "mi proyecto", "mi empresa", "trabajo en",
        ]
        for sentence in user_sentences:
            sl = sentence.lower()
            matched = [s for s in tech_signals if s in sl]
            if matched:
                facts.append({
                    "category": "technical_context",
                    "content": sentence[:250],
                })
                break

        goal_signals = [
            "quiero lograr", "mi objetivo", "necesito que", "el problema es",
            "estoy intentando", "trato de", "necesito saber", "quiero entender",
        ]
        for sentence in user_sentences:
            sl = sentence.lower()
            for signal in goal_signals:
                if signal in sl:
                    facts.append({
                        "category": "user_goal",
                        "content": sentence[:250],
                    })
                    break

        return facts[:4]
