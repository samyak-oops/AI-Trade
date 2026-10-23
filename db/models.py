"""
db/models.py – SQLAlchemy ORM models for all persistent data.
"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── Signal direction ──────────────────────────────────────────────────────────
class Direction(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


class SignalOutcome(str, enum.Enum):
    PENDING = "PENDING"
    TARGET1_HIT = "TARGET1_HIT"
    TARGET2_HIT = "TARGET2_HIT"
    STOPPED_OUT = "STOPPED_OUT"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


# ── Chat sessions ─────────────────────────────────────────────────────────────
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(256), default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("chat_sessions.session_id"), nullable=False)
    role = Column(String(16), nullable=False)          # "user" | "assistant" | "system"
    content = Column(Text, nullable=False)
    image_path = Column(String(512), nullable=True)    # uploaded image, if any
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


# ── Trading signals ───────────────────────────────────────────────────────────
class TradeSignal(Base):
    __tablename__ = "trade_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=True)
    symbol = Column(String(32), nullable=False, index=True)
    exchange = Column(String(8), default="NSE")
    direction = Column(Enum(Direction), nullable=False)
    timeframe = Column(String(16))                     # "intraday" | "swing" | "positional"

    entry_price = Column(Float)
    stop_loss = Column(Float)
    target1 = Column(Float)
    target2 = Column(Float)
    risk_reward = Column(Float)
    position_size = Column(Float)

    confidence_score = Column(Float, nullable=False)
    confidence_breakdown = Column(JSON)                # dict of sub-scores
    strategies_triggered = Column(JSON)                # list of strategy names
    is_calibrated = Column(Boolean, default=False)
    calibrated_score = Column(Float, nullable=True)

    outcome = Column(Enum(SignalOutcome), default=SignalOutcome.PENDING)
    outcome_price = Column(Float, nullable=True)
    outcome_at = Column(DateTime, nullable=True)
    pnl_pct = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)


# ── Watchlist ─────────────────────────────────────────────────────────────────
class WatchlistItem(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=True)
    symbol = Column(String(32), nullable=False)
    exchange = Column(String(8), default="NSE")
    alert_threshold = Column(Float, default=90.0)     # alert when score crosses this
    last_score = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    alerted_at = Column(DateTime, nullable=True)

    __table_args__ = (UniqueConstraint("session_id", "symbol", "exchange"),)


# ── User profile ──────────────────────────────────────────────────────────────
class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False)
    risk_appetite = Column(String(16), default="moderate")   # conservative/moderate/aggressive
    capital = Column(Float, default=100_000.0)
    horizon = Column(String(16), default="swing")            # intraday/swing/positional
    tax_slab_pct = Column(Float, default=30.0)
    preferred_sectors = Column(JSON, default=list)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
