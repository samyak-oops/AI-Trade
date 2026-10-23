"""
agents/graph.py – LangGraph pipeline definition.
Flow: Supervisor → [Technical + Fundamental + News] → Backtest →
      Confidence Engine → GATE → Risk Manager → Synthesizer
                                 └─ (fail) → No Trade Node
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.nodes import (
    backtest_node,
    chat_node,
    confidence_node,
    fd_agent_node,
    fundamental_agent_node,
    mf_agent_node,
    news_agent_node,
    no_trade_node,
    risk_manager_node,
    supervisor_node,
    synthesizer_node,
    technical_agent_node,
    vision_node,
)
from agents.state import AgentState

logger = logging.getLogger(__name__)


def _route_after_supervisor(state: AgentState) -> str:
    """Routing after supervisor: pick which specialist to call."""
    route = state.get("route", "analyze")
    if route == "mf":
        return "mf_agent"
    if route == "fd":
        return "fd_agent"
    if route in ("chat", "news", "watchlist"):
        return "chat"
    return "vision"    # default: equity analysis pipeline


def _route_after_confidence(state: AgentState) -> str:
    """HARD GATE: if confidence passes, go to risk manager; else no_trade."""
    report = state.get("confidence_report")
    if report is None:
        return "no_trade"
    if report.passed_gate():
        return "risk_manager"
    return "no_trade"


def build_graph() -> Any:
    """
    Assemble and compile the LangGraph StateGraph.
    Returns a compiled graph (app) ready for ainvoke / astream.
    """
    builder = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("vision", vision_node)
    builder.add_node("technical_agent", technical_agent_node)
    builder.add_node("fundamental_agent", fundamental_agent_node)
    builder.add_node("news_agent", news_agent_node)
    builder.add_node("mf_agent", mf_agent_node)
    builder.add_node("fd_agent", fd_agent_node)
    builder.add_node("backtest", backtest_node)
    builder.add_node("confidence_engine", confidence_node)
    builder.add_node("risk_manager", risk_manager_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("no_trade", no_trade_node)
    builder.add_node("chat", chat_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", _route_after_supervisor, {
        "vision":    "vision",
        "mf_agent":  "mf_agent",
        "fd_agent":  "fd_agent",
        "chat":      "chat",
    })

    # Vision → parallel technical + fundamental + news
    builder.add_edge("vision", "technical_agent")
    builder.add_edge("vision", "fundamental_agent")
    builder.add_edge("vision", "news_agent")

    # After all three complete → backtest
    builder.add_edge("technical_agent", "backtest")
    builder.add_edge("fundamental_agent", "backtest")
    builder.add_edge("news_agent", "backtest")

    # Backtest → Confidence Engine
    builder.add_edge("backtest", "confidence_engine")

    # HARD GATE
    builder.add_conditional_edges("confidence_engine", _route_after_confidence, {
        "risk_manager": "risk_manager",
        "no_trade":     "no_trade",
    })

    # Risk manager → Synthesizer → END
    builder.add_edge("risk_manager", "synthesizer")
    builder.add_edge("synthesizer", END)
    builder.add_edge("no_trade", END)

    # MF / FD / Chat → Synthesizer → END
    builder.add_edge("mf_agent", "synthesizer")
    builder.add_edge("fd_agent", "synthesizer")
    builder.add_edge("chat", END)

    # ── Compile with in-memory checkpointer ───────────────────────────────────
    memory = MemorySaver()
    app = builder.compile(checkpointer=memory)
    return app


# Singleton graph instance
_graph = None


def get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
