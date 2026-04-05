"""
News aggregator — fetches headlines from RSS feeds and economic calendar.
Used by the AI news analyzer to decide whether to trade.
"""

import time
import requests
import feedparser
from datetime import datetime, timezone, timedelta
from config import settings


RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.forexlive.com/feed/news",
    "https://www.fxstreet.com/rss/news",
    "https://www.marketwatch.com/rss/topstories",
]

# Forex Factory calendar endpoint (unofficial but stable)
FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


def fetch_headlines(max_age_minutes: int = 15) -> list[str]:
    """
    Fetch recent headlines from RSS feeds.
    Returns list of headline strings from the last max_age_minutes.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
    headlines = []

    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                # Parse published time
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

                if published is None or published >= cutoff:
                    title = entry.get("title", "").strip()
                    if title and _is_relevant(title):
                        headlines.append(title)
        except Exception:
            continue

    return list(set(headlines))  # deduplicate


def fetch_calendar_events(lookahead_minutes: int = 60) -> list[dict]:
    """
    Fetch upcoming high-impact economic events from Forex Factory calendar.
    Returns list of dicts with: title, impact, time, currency.
    """
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(minutes=lookahead_minutes)

    try:
        resp = requests.get(FF_CALENDAR_URL, timeout=8)
        resp.raise_for_status()
        events = resp.json()
    except Exception:
        return []

    upcoming = []
    for event in events:
        try:
            event_time = datetime.fromisoformat(event.get("date", "").replace("Z", "+00:00"))
            impact = event.get("impact", "").lower()
            currency = event.get("country", "")
            title = event.get("title", "")

            if event_time >= now and event_time <= cutoff:
                if impact in ("high", "medium") and currency in ("USD", "US"):
                    upcoming.append({
                        "title": title,
                        "impact": impact,
                        "time_utc": event_time.strftime("%H:%M UTC"),
                        "currency": currency,
                    })
        except Exception:
            continue

    return upcoming


def is_hardcoded_avoid(headlines: list[str], events: list[dict]) -> bool:
    """
    Returns True if any hardcoded high-impact event is detected.
    These always trigger AVOID regardless of AI decision.
    """
    keywords = settings.HIGH_IMPACT_KEYWORDS
    all_text = " ".join(headlines + [e["title"] for e in events]).lower()
    return any(kw in all_text for kw in keywords)


def _is_relevant(headline: str) -> bool:
    """Filter headlines to gold/USD/macro relevant ones only."""
    keywords = [
        "gold", "xau", "fed", "federal reserve", "inflation", "dollar", "usd",
        "interest rate", "cpi", "nfp", "payroll", "gdp", "treasury", "yield",
        "recession", "powell", "fomc", "rate cut", "rate hike", "geopolit",
        "war", "oil", "commodit", "risk", "safe haven",
    ]
    h = headline.lower()
    return any(kw in h for kw in keywords)
