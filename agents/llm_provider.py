"""
agents/llm_provider.py – Groq LLM client (vision + text).
"""
from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq

from core.config import get_settings


@lru_cache(maxsize=1)
def get_llm(streaming: bool = True) -> ChatGroq:
    """Return a cached ChatGroq instance."""
    cfg = get_settings()
    return ChatGroq(
        model=cfg.llm_model,
        api_key=cfg.groq_api_key,
        temperature=0.1,
        max_tokens=4096,
        streaming=streaming,
    )
