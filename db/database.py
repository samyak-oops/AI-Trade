"""
db/database.py – Async SQLite engine, session factory, and init helper.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.config import get_settings
from db.models import Base

_engine = None
_SessionFactory: async_sessionmaker[AsyncSession] | None = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = f"sqlite+aiosqlite:///{settings.db_path}"
        _engine = create_async_engine(db_url, echo=False, future=True)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _SessionFactory
    if _SessionFactory is None:
        engine = _get_engine()
        _SessionFactory = async_sessionmaker(engine, expire_on_commit=False)
    return _SessionFactory


async def init_db() -> None:
    """Create all tables (idempotent)."""
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager yielding a DB session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
