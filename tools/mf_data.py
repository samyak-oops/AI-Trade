"""
tools/mf_data.py – Mutual fund data via AMFI NAV file + mfapi.in.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from cachetools import TTLCache

logger = logging.getLogger(__name__)

_nav_cache: TTLCache = TTLCache(maxsize=512, ttl=86400)  # 24h
_scheme_cache: TTLCache = TTLCache(maxsize=128, ttl=3600)  # 1h


MFAPI_BASE = "https://api.mfapi.in/mf"


def search_scheme(name: str) -> List[Dict[str, Any]]:
    """
    Search mutual fund schemes by name substring.
    Returns list of {schemeCode, schemeName}.
    """
    try:
        resp = httpx.get(f"{MFAPI_BASE}/search?q={name}", timeout=10)
        resp.raise_for_status()
        return resp.json()[:20]
    except Exception as exc:
        logger.error("MF search failed for %s: %s", name, exc)
        return []


def get_nav(scheme_code: int | str) -> Dict[str, Any]:
    """
    Fetch latest NAV + basic metadata for a scheme.
    """
    key = f"nav:{scheme_code}"
    if key in _nav_cache:
        return _nav_cache[key]
    try:
        resp = httpx.get(f"{MFAPI_BASE}/{scheme_code}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        meta = data.get("meta", {})
        navs = data.get("data", [])  # list of {date, nav}
        latest = navs[0] if navs else {}

        result: Dict[str, Any] = {
            "scheme_code": scheme_code,
            "scheme_name": meta.get("scheme_name"),
            "fund_house": meta.get("fund_house"),
            "scheme_type": meta.get("scheme_type"),
            "scheme_category": meta.get("scheme_category"),
            "nav": float(latest.get("nav", 0)),
            "nav_date": latest.get("date"),
            "source": "mfapi.in",
            "timestamp": datetime.utcnow().isoformat(),
        }
        _nav_cache[key] = result
        return result
    except Exception as exc:
        logger.error("NAV fetch failed for %s: %s", scheme_code, exc)
        return {"scheme_code": scheme_code, "error": str(exc),
                "source": "mfapi.in", "timestamp": datetime.utcnow().isoformat()}


def get_scheme_history(scheme_code: int | str) -> List[Dict[str, str]]:
    """Return full NAV history [{date, nav}] sorted newest first."""
    key = f"hist:{scheme_code}"
    if key in _scheme_cache:
        return _scheme_cache[key]
    try:
        resp = httpx.get(f"{MFAPI_BASE}/{scheme_code}", timeout=10)
        resp.raise_for_status()
        navs = resp.json().get("data", [])
        _scheme_cache[key] = navs
        return navs
    except Exception as exc:
        logger.error("Scheme history failed for %s: %s", scheme_code, exc)
        return []


def compute_returns(scheme_code: int | str) -> Dict[str, Optional[float]]:
    """
    Compute 1Y, 3Y, 5Y, and inception CAGR returns from NAV history.
    """
    import numpy as np

    history = get_scheme_history(scheme_code)
    if not history:
        return {"1y": None, "3y": None, "5y": None}

    def nav_on_date(target_date: datetime) -> Optional[float]:
        for h in history:
            try:
                dt = datetime.strptime(h["date"], "%d-%m-%Y")
                if dt <= target_date:
                    return float(h["nav"])
            except Exception:
                continue
        return None

    now = datetime.utcnow()
    current_nav = float(history[0]["nav"])

    def cagr(past_nav: Optional[float], years: float) -> Optional[float]:
        if past_nav and past_nav > 0:
            return round((current_nav / past_nav) ** (1 / years) - 1, 4)
        return None

    return {
        "1y": cagr(nav_on_date(now - __import__("dateutil.relativedelta", fromlist=["relativedelta"]).relativedelta(years=1)), 1),
        "3y": cagr(nav_on_date(now - __import__("dateutil.relativedelta", fromlist=["relativedelta"]).relativedelta(years=3)), 3),
        "5y": cagr(nav_on_date(now - __import__("dateutil.relativedelta", fromlist=["relativedelta"]).relativedelta(years=5)), 5),
        "current_nav": current_nav,
        "scheme_code": scheme_code,
        "source": "mfapi.in",
        "timestamp": datetime.utcnow().isoformat(),
    }


def sip_calculator(
    monthly_amount: float,
    annual_return_pct: float,
    years: int,
) -> Dict[str, float]:
    """
    Compute SIP maturity value.
    """
    r = annual_return_pct / 100 / 12
    n = years * 12
    if r == 0:
        maturity = monthly_amount * n
    else:
        maturity = monthly_amount * (((1 + r) ** n - 1) / r) * (1 + r)
    invested = monthly_amount * n
    return {
        "monthly_sip": monthly_amount,
        "annual_return_pct": annual_return_pct,
        "years": years,
        "total_invested": round(invested, 2),
        "maturity_value": round(maturity, 2),
        "wealth_gain": round(maturity - invested, 2),
    }
