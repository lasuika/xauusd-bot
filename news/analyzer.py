"""
AI news analyzer using Claude Haiku.
Classifies current news impact on XAUUSD as TRADE_NORMAL / REDUCE_RISK / AVOID.
"""

import json
import time
import anthropic
from config import settings
from news.fetcher import fetch_headlines, fetch_calendar_events, is_hardcoded_avoid


DECISIONS = ("TRADE_NORMAL", "REDUCE_RISK", "AVOID")

SYSTEM_PROMPT = """You are a XAUUSD market analyst. Your job is to assess whether current
news and economic events warrant halting or reducing gold trading in the next 30 minutes.
Be conservative — when in doubt, choose REDUCE_RISK over TRADE_NORMAL.
Respond with valid JSON only. No explanation outside the JSON."""

USER_PROMPT_TEMPLATE = """Given these inputs, classify the impact on XAUUSD spot price
in the next 30 minutes.

Recent headlines (last 15 minutes):
{headlines}

Upcoming economic events (next 60 minutes):
{calendar_events}

Respond with this exact JSON format:
{{"decision": "TRADE_NORMAL|REDUCE_RISK|AVOID", "reason": "<one concise sentence>"}}

Decision criteria:
- TRADE_NORMAL: no significant market-moving event expected
- REDUCE_RISK: moderate uncertainty — cut position size 50%
- AVOID: high-impact event imminent — halt all new entries"""


class NewsAnalyzer:
    def __init__(self):
        self._client = None
        self._last_result = {"decision": "TRADE_NORMAL", "reason": "No analysis yet"}
        self._last_fetch_time = 0

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set in .env")
            self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        return self._client

    def analyze(self, force: bool = False) -> dict:
        """
        Returns the current trading decision.
        Caches result for NEWS_POLL_INTERVAL_SECONDS — only re-fetches when stale.
        Set force=True to bypass cache.
        """
        now = time.time()
        if not force and (now - self._last_fetch_time) < settings.NEWS_POLL_INTERVAL_SECONDS:
            return self._last_result

        try:
            headlines = fetch_headlines(max_age_minutes=15)
            events = fetch_calendar_events(lookahead_minutes=60)

            # Hardcoded AVOID check bypasses AI entirely
            if is_hardcoded_avoid(headlines, events):
                result = {
                    "decision": "AVOID",
                    "reason": "Hardcoded high-impact event detected (NFP/FOMC/CPI/etc.)"
                }
                self._last_result = result
                self._last_fetch_time = now
                return result

            # No relevant news — skip API call to save cost
            if not headlines and not events:
                result = {"decision": "TRADE_NORMAL", "reason": "No relevant news or events"}
                self._last_result = result
                self._last_fetch_time = now
                return result

            headlines_text = "\n".join(f"- {h}" for h in headlines[:10]) or "None"
            events_text = "\n".join(
                f"- {e['time_utc']} [{e['impact'].upper()}] {e['title']}" for e in events
            ) or "None"

            prompt = USER_PROMPT_TEMPLATE.format(
                headlines=headlines_text,
                calendar_events=events_text,
            )

            client = self._get_client()
            message = client.messages.create(
                model=settings.NEWS_MODEL,
                max_tokens=128,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = message.content[0].text.strip()
            result = json.loads(raw)

            # Validate response
            if result.get("decision") not in DECISIONS:
                result["decision"] = "REDUCE_RISK"

            self._last_result = result
            self._last_fetch_time = now

        except Exception as e:
            # On any error — default to REDUCE_RISK (safe fallback)
            self._last_result = {
                "decision": "REDUCE_RISK",
                "reason": f"News analysis error: {str(e)[:80]}"
            }

        return self._last_result

    def get_risk_multiplier(self) -> float:
        """
        Returns a multiplier to apply to risk_pct based on current news decision.
        TRADE_NORMAL → 1.0 (full risk)
        REDUCE_RISK  → 0.5 (half risk)
        AVOID        → 0.0 (no new trades)
        """
        decision = self._last_result.get("decision", "TRADE_NORMAL")
        return {"TRADE_NORMAL": 1.0, "REDUCE_RISK": 0.5, "AVOID": 0.0}.get(decision, 0.5)
