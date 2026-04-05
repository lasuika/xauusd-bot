"""
Dynamic risk manager.
Determines risk % per trade based on daily P&L state and enforces circuit breakers.
Designed for backtesting simulation — Phase 2 will extend this for live use.
"""

import numpy as np
import pandas as pd
from config import settings


class RiskManager:
    """
    Tracks account state and returns appropriate risk % per trade.
    Call update_trade() after each closed trade.
    Call get_risk_pct() before sizing a new position.
    """

    def __init__(self, initial_equity: float):
        self.equity = initial_equity
        self.initial_equity = initial_equity

        # Daily / weekly tracking
        self._day_start_equity = initial_equity
        self._week_start_equity = initial_equity
        self._current_day = None
        self._current_week = None

        # Historical performance (rolling)
        self._daily_pnl_history: list[float] = []   # one entry per completed day
        self._weekly_pnl_history: list[float] = []  # one entry per completed week

        # State flags
        self.daily_stopped = False
        self.weekly_stopped = False

    def start_candle(self, timestamp_ms: int) -> None:
        """Call at the start of each candle to handle day/week rollovers."""
        dt = pd.Timestamp(timestamp_ms, unit="ms", tz="UTC")
        day_key = (dt.year, dt.month, dt.day)
        week_key = (dt.year, dt.isocalendar()[1])

        # New day
        if self._current_day is None:
            self._current_day = day_key
            self._day_start_equity = self.equity
            self.daily_stopped = False

        elif day_key != self._current_day:
            # Record completed day
            day_pnl_pct = (self.equity - self._day_start_equity) / self._day_start_equity
            self._daily_pnl_history.append(day_pnl_pct)
            if len(self._daily_pnl_history) > settings.CIRCUIT_DAILY_HISTORY_DAYS:
                self._daily_pnl_history.pop(0)

            self._current_day = day_key
            self._day_start_equity = self.equity
            self.daily_stopped = False  # reset for new day

        # New week
        if self._current_week is None:
            self._current_week = week_key
            self._week_start_equity = self.equity
            self.weekly_stopped = False

        elif week_key != self._current_week:
            week_pnl_pct = (self.equity - self._week_start_equity) / self._week_start_equity
            self._weekly_pnl_history.append(week_pnl_pct)
            if len(self._weekly_pnl_history) > settings.CIRCUIT_WEEKLY_HISTORY_WEEKS:
                self._weekly_pnl_history.pop(0)

            self._current_week = week_key
            self._week_start_equity = self.equity
            self.weekly_stopped = False

    def update_equity(self, new_equity: float) -> None:
        """Update equity after a trade closes."""
        self.equity = new_equity

    def get_daily_pnl_pct(self) -> float:
        """Current day's P&L as a fraction of day-start equity."""
        if self._day_start_equity == 0:
            return 0.0
        return (self.equity - self._day_start_equity) / self._day_start_equity

    def get_weekly_pnl_pct(self) -> float:
        if self._week_start_equity == 0:
            return 0.0
        return (self.equity - self._week_start_equity) / self._week_start_equity

    def _daily_stop_threshold(self) -> float:
        """Dynamic daily stop: 2× avg winning day, fallback 5%."""
        winning_days = [p for p in self._daily_pnl_history if p > 0]
        if len(winning_days) >= 5:
            avg_win = np.mean(winning_days)
            return -(avg_win * settings.CIRCUIT_DAILY_MULTIPLIER)
        return -settings.CIRCUIT_DAILY_FALLBACK

    def _weekly_stop_threshold(self) -> float:
        """Dynamic weekly stop: 3× avg winning week, fallback 10%."""
        winning_weeks = [p for p in self._weekly_pnl_history if p > 0]
        if len(winning_weeks) >= 2:
            avg_win = np.mean(winning_weeks)
            return -(avg_win * settings.CIRCUIT_WEEKLY_MULTIPLIER)
        return -settings.CIRCUIT_WEEKLY_FALLBACK

    def can_trade(self) -> tuple[bool, str]:
        """
        Returns (True, '') if trading is allowed, or (False, reason) if stopped.
        """
        if self.weekly_stopped:
            return False, "weekly_circuit_breaker"

        if self.daily_stopped:
            return False, "daily_circuit_breaker"

        daily_pnl = self.get_daily_pnl_pct()
        if daily_pnl <= settings.RISK_DAILY_STOP:
            self.daily_stopped = True
            return False, f"daily_loss_stop ({daily_pnl*100:.1f}%)"

        weekly_pnl = self.get_weekly_pnl_pct()
        if weekly_pnl <= self._weekly_stop_threshold():
            self.weekly_stopped = True
            return False, f"weekly_loss_stop ({weekly_pnl*100:.1f}%)"

        return True, ""

    def get_risk_pct(self) -> float:
        """
        Returns the appropriate risk % per trade based on current daily P&L.
        """
        daily_pnl = self.get_daily_pnl_pct()

        if daily_pnl > settings.RISK_DAILY_UP_THRESHOLD:
            return settings.RISK_HIGH_PCT   # up >2% → press with 3%
        elif daily_pnl < -settings.RISK_DAILY_DOWN_SOFT:
            return settings.RISK_LOW_PCT    # down >1% → protect with 1%
        else:
            return settings.RISK_BASE_PCT   # flat → baseline 2%

    def position_size(self, sl_distance: float) -> float:
        """
        Calculate position size in units given a SL distance (in price terms).
        risk_amount = equity * risk_pct
        size = risk_amount / sl_distance
        """
        if sl_distance <= 0:
            return 0.0
        risk_amount = self.equity * self.get_risk_pct()
        return risk_amount / sl_distance
