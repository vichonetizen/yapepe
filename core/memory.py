from datetime import datetime
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, Float, Integer, DateTime, select, func, text
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class SessionRecord(Base):
    __tablename__ = "sessions"
    id:            Mapped[str]      = mapped_column(String(50), primary_key=True)
    created_at:    Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    title:         Mapped[str]      = mapped_column(String(200), default="Nueva sesión")
    message_count: Mapped[int]      = mapped_column(Integer, default=0)


class Message(Base):
    __tablename__ = "messages"
    id:           Mapped[int]            = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id:   Mapped[str]            = mapped_column(String(50))
    role:         Mapped[str]            = mapped_column(String(20))
    content:      Mapped[str]            = mapped_column(Text)
    modules_used: Mapped[str]            = mapped_column(Text, default="[]")
    complexity:   Mapped[str]            = mapped_column(String(20), default="simple")
    provider:     Mapped[Optional[str]]  = mapped_column(String(50), nullable=True)
    model_used:   Mapped[Optional[str]]  = mapped_column(String(100), nullable=True)
    created_at:   Mapped[datetime]       = mapped_column(DateTime, default=datetime.utcnow)
    rating:       Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class MemoryFact(Base):
    __tablename__ = "memory_facts"
    id:         Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    category:   Mapped[str]      = mapped_column(String(50))
    content:    Mapped[str]      = mapped_column(Text)
    confidence: Mapped[float]    = mapped_column(Float, default=1.0)
    frequency:  Mapped[int]      = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModulePerformance(Base):
    __tablename__ = "module_performance"
    id:           Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    complexity:   Mapped[str]   = mapped_column(String(20))
    modules_combo: Mapped[str]  = mapped_column(String(200))
    provider:     Mapped[str]   = mapped_column(String(50), default="")
    avg_rating:   Mapped[float] = mapped_column(Float, default=3.0)
    sample_count: Mapped[int]   = mapped_column(Integer, default=0)


engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionFactory = async_sessionmaker(engine, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Safe schema migration: add new columns when upgrading from v1
        for col in ["provider VARCHAR(50)", "model_used VARCHAR(100)"]:
            try:
                await conn.execute(text(f"ALTER TABLE messages ADD COLUMN {col}"))
            except Exception:
                pass
        try:
            await conn.execute(text("ALTER TABLE module_performance ADD COLUMN provider VARCHAR(50) DEFAULT ''"))
        except Exception:
            pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionFactory() as session:
        yield session


async def get_session_history(session_id: str, db: AsyncSession, limit: int = 20) -> list:
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    return list(reversed(result.scalars().all()))


async def save_message(
    session_id: str, role: str, content: str,
    modules: list, complexity: str, db: AsyncSession,
    provider: str = "", model_used: str = "",
) -> int:
    msg = Message(
        session_id=session_id, role=role, content=content,
        modules_used=json.dumps(modules), complexity=complexity,
        provider=provider, model_used=model_used,
    )
    db.add(msg)
    await db.flush()
    await db.commit()

    result = await db.execute(select(SessionRecord).where(SessionRecord.id == session_id))
    rec = result.scalar_one_or_none()
    if rec:
        rec.message_count += 1
        await db.commit()
    return msg.id


async def save_memory_fact(category: str, content: str, db: AsyncSession, confidence: float = 1.0):
    result = await db.execute(
        select(MemoryFact).where(MemoryFact.category == category, MemoryFact.content == content)
    )
    existing = result.scalar_one_or_none()
    if existing:
        existing.frequency  += 1
        existing.confidence  = min(1.0, existing.confidence + 0.05)
        existing.updated_at  = datetime.utcnow()
    else:
        db.add(MemoryFact(category=category, content=content, confidence=confidence))
    await db.commit()


async def get_relevant_facts(query: str, db: AsyncSession, limit: int = 5) -> list[str]:
    result = await db.execute(
        select(MemoryFact)
        .order_by(MemoryFact.frequency.desc(), MemoryFact.confidence.desc())
        .limit(limit * 3)
    )
    query_lower = query.lower()
    scored = []
    for fact in result.scalars().all():
        # Anchored facts (confidence ≥ 0.99) get triple weight so siempre suben al tope
        anchor_boost = 3.0 if fact.confidence >= 0.99 else 1.0
        score = fact.frequency * (fact.confidence ** 2) * anchor_boost
        for word in query_lower.split():
            if len(word) > 3 and word in fact.content.lower():
                score += 3
        scored.append((score, fact.content))
    scored.sort(reverse=True)
    return [c for _, c in scored[:limit]]


async def get_all_facts(db: AsyncSession) -> list[dict]:
    result = await db.execute(select(MemoryFact).order_by(MemoryFact.updated_at.desc()))
    return [
        {"id": f.id, "category": f.category, "content": f.content,
         "confidence": round(f.confidence, 2), "frequency": f.frequency,
         "updated_at": f.updated_at.isoformat()}
        for f in result.scalars().all()
    ]


async def update_message_rating(message_id: int, rating: float, db: AsyncSession):
    result = await db.execute(select(Message).where(Message.id == message_id))
    msg = result.scalar_one_or_none()
    if msg:
        msg.rating = rating
        await db.commit()
    return msg


async def update_module_performance(
    complexity: str, modules: list, rating: float, db: AsyncSession, provider: str = ""
):
    combo = ",".join(sorted(modules))
    result = await db.execute(
        select(ModulePerformance).where(
            ModulePerformance.complexity == complexity,
            ModulePerformance.modules_combo == combo,
        )
    )
    perf = result.scalar_one_or_none()
    if perf:
        alpha = 0.15
        perf.avg_rating   = perf.avg_rating * (1 - alpha) + rating * alpha
        perf.sample_count += 1
        if provider:
            perf.provider = provider
    else:
        db.add(ModulePerformance(
            complexity=complexity, modules_combo=combo,
            avg_rating=rating, sample_count=1, provider=provider,
        ))
    await db.commit()


async def get_stats(db: AsyncSession) -> dict:
    msg_count  = (await db.execute(select(func.count(Message.id)))).scalar() or 0
    fact_count = (await db.execute(select(func.count(MemoryFact.id)))).scalar() or 0
    avg_rating = (await db.execute(
        select(func.avg(Message.rating)).where(Message.rating.is_not(None))
    )).scalar() or 0

    perf_rows  = (await db.execute(select(ModulePerformance))).scalars().all()
    all_msgs   = (await db.execute(
        select(Message).where(Message.role == "assistant")
    )).scalars().all()

    module_usage: dict[str, int] = {}
    provider_usage: dict[str, int] = {}

    for msg in all_msgs:
        for mod in json.loads(msg.modules_used or "[]"):
            module_usage[mod] = module_usage.get(mod, 0) + 1
        if msg.provider:
            provider_usage[msg.provider] = provider_usage.get(msg.provider, 0) + 1

    return {
        "total_messages": msg_count,
        "total_facts":    fact_count,
        "avg_rating":     round(avg_rating, 2),
        "module_usage":   module_usage,
        "provider_usage": provider_usage,
        "module_performance": [
            {"complexity": p.complexity, "modules": p.modules_combo,
             "provider": p.provider, "avg_rating": round(p.avg_rating, 2),
             "samples": p.sample_count}
            for p in perf_rows
        ],
    }
