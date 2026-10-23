"""
tools/news_data.py – News + sentiment via RSS feeds + optional web search.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

import feedparser
from cachetools import TTLCache

logger = logging.getLogger(__name__)

_news_cache: TTLCache = TTLCache(maxsize=64, ttl=3600)  # 1h

RSS_FEEDS: Dict[str, str] = {
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol": "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "Business Standard": "https://www.business-standard.com/rss/markets-106.rss",
    "LiveMint": "https://www.livemint.com/rss/markets",
    "NDTV Profit": "https://feeds.feedburner.com/ndtvprofit-latest",
}


def _clean_text(text: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", text).strip()


def fetch_news(symbol: str | None = None, max_items: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch latest market news from RSS feeds.
    If symbol is given, filter items that mention the symbol.
    """
    key = f"news:{symbol}:{max_items}"
    if key in _news_cache:
        return _news_cache[key]

    all_items: List[Dict[str, Any]] = []
    for source_name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:
                title = _clean_text(entry.get("title", ""))
                summary = _clean_text(entry.get("summary", ""))
                link = entry.get("link", "")
                published = entry.get("published", "")
                if symbol and symbol.upper() not in (title + summary).upper():
                    continue
                all_items.append({
                    "title": title,
                    "summary": summary[:400],
                    "link": link,
                    "published": published,
                    "source": source_name,
                })
        except Exception as exc:
            logger.warning("RSS fetch failed for %s: %s", source_name, exc)

    # Sort by recency if published is parseable
    all_items = all_items[:max_items]
    result = {
        "symbol": symbol,
        "items": all_items,
        "count": len(all_items),
        "timestamp": datetime.utcnow().isoformat(),
        "sources": list(RSS_FEEDS.keys()),
    }
    _news_cache[key] = all_items
    return all_items


def get_fii_dii_data() -> Dict[str, Any]:
    """
    Approximate FII/DII flow data.
    In production, integrate with NSE/BSE API or nsepython.
    Returns a stub that signals 'data unavailable' if not connected.
    """
    try:
        import nsepython as nse  # type: ignore
        # nsepython may have limited coverage – wrap safely
        data = nse.fii_dii_data()
        return {
            "data": data,
            "source": "nsepython/NSE",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        logger.warning("FII/DII data unavailable: %s", exc)
        return {
            "data": None,
            "source": "NSE",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(exc),
            "note": "FII/DII data unavailable. Confidence score will be capped.",
        }


def simple_sentiment_score(texts: List[str]) -> Dict[str, Any]:
    """
    Very lightweight keyword-based sentiment scorer.
    Returns score in [-1, +1] and label.
    """
    positive_kw = [
        "rally", "surge", "gain", "outperform", "buy", "upgrade", "strong",
        "beat", "growth", "profit", "record", "high", "bullish", "positive",
    ]
    negative_kw = [
        "fall", "drop", "loss", "underperform", "sell", "downgrade", "weak",
        "miss", "decline", "warning", "fraud", "investigation", "bearish",
        "negative", "default", "ban",
    ]
    pos = neg = 0
    for text in texts:
        t = text.lower()
        pos += sum(1 for kw in positive_kw if kw in t)
        neg += sum(1 for kw in negative_kw if kw in t)
    total = pos + neg
    if total == 0:
        score = 0.0
    else:
        score = (pos - neg) / total
    label = "bullish" if score > 0.2 else "bearish" if score < -0.2 else "neutral"
    return {"score": round(score, 3), "label": label, "pos_count": pos, "neg_count": neg}
