import os
from dotenv import load_dotenv

load_dotenv()

# ── Bybit connection ──────────────────────────────────────────────────────────
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "true").lower() == "true"

# ── Symbol ────────────────────────────────────────────────────────────────────
SYMBOL = "XAUUSDT"
INTERVAL = "1"          # 1-minute candles
LEVERAGE = 20           # default leverage (10 / 20 / 40)

# ── Strategy parameters (tuned by optimizer) ─────────────────────────────────
BB_PERIOD = 20
BB_STD = 2.0

RSI_PERIOD = 14
RSI_OVERSOLD = 28       # long entry threshold
RSI_OVERBOUGHT = 72     # short entry threshold

VOLUME_MULTIPLIER = 1.5         # current vol must be > X * rolling avg
VOLUME_ROLLING_PERIOD = 20

ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.0         # SL = entry ± (ATR * this)
ATR_TP_MULTIPLIER = 0.8         # TP = entry ± (ATR * this)  → 0.8:1 RR

# ── Trend filter (EMA-based — only trade WITH the trend) ─────────────────────
TREND_EMA_FAST = 50             # fast EMA period
TREND_EMA_SLOW = 200            # slow EMA period
# Long only when close > EMA50 > EMA200 (uptrend)
# Short only when close < EMA50 < EMA200 (downtrend)
# Both directions when EMAs are mixed / choppy

# ── Session filter ────────────────────────────────────────────────────────────
SESSION_ATR_WINDOW_DAYS = 7     # rolling window for ATR percentile
SESSION_ATR_PERCENTILE = 40     # only trade when ATR > this percentile

# ── Smart exit thresholds ─────────────────────────────────────────────────────
SMART_EXIT_RSI_MIDLINE = 50     # RSI cross through this = momentum lost
SMART_EXIT_SL_RETRACE = 0.50    # price retraced X% toward SL = warning
SMART_EXIT_STALL_CANDLES = 3    # candles with no new progress = stall

# ── Risk management ───────────────────────────────────────────────────────────
RISK_BASE_PCT = 0.02            # 2% baseline risk per trade
RISK_HIGH_PCT = 0.03            # 3% when day is up > 2%
RISK_LOW_PCT = 0.01             # 1% when day is down 1–4%
RISK_DAILY_UP_THRESHOLD = 0.02  # day P&L above this → use RISK_HIGH_PCT
RISK_DAILY_DOWN_SOFT = 0.01     # day P&L below this → use RISK_LOW_PCT
RISK_DAILY_STOP = 0.04          # day P&L below this → stop for the day

# Dynamic circuit breaker (rolling history)
CIRCUIT_DAILY_HISTORY_DAYS = 14
CIRCUIT_DAILY_MULTIPLIER = 2.0  # stop if loss > X * avg winning day
CIRCUIT_WEEKLY_HISTORY_WEEKS = 4
CIRCUIT_WEEKLY_MULTIPLIER = 3.0 # stop if loss > X * avg winning week

# Fallback thresholds (no history yet)
CIRCUIT_DAILY_FALLBACK = 0.05
CIRCUIT_WEEKLY_FALLBACK = 0.10

MAX_CONCURRENT_POSITIONS = 2

# ── Backtesting ───────────────────────────────────────────────────────────────
BACKTEST_YEARS = 2
SPREAD_PIPS = 0.20              # simulated spread cost ($0.20/oz — realistic for Bybit XAUUSD)
SLIPPAGE_PIPS = 0.10            # simulated slippage ($0.10/oz — tight scalp)

# Walk-forward split (months)
WALKFORWARD_TRAIN_MONTHS = 18
WALKFORWARD_TEST_MONTHS = 6

# ── Parameter optimization grid ───────────────────────────────────────────────
OPT_BB_PERIODS = [15, 17, 20, 23, 25]
OPT_RSI_OVERSOLD = [25, 27, 28, 30]
OPT_RSI_OVERBOUGHT = [70, 72, 73, 75]
OPT_ATR_SL_MULT = [0.8, 1.0, 1.2, 1.5]
OPT_VOLUME_MULT = [1.2, 1.5, 1.8, 2.0]

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_CACHE_DIR = "data/cache"
BACKTEST_OUTPUT_DIR = "backtest/output"
CACHE_FILE = f"{DATA_CACHE_DIR}/{SYMBOL}_1m.parquet"

# ── Claude API (news filter) ──────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
NEWS_MODEL = "claude-haiku-4-5-20251001"
NEWS_POLL_INTERVAL_SECONDS = 300    # 5 minutes

# High-impact events — always AVOID regardless of AI decision
HIGH_IMPACT_KEYWORDS = [
    "fomc", "federal reserve", "fed rate", "interest rate decision",
    "non-farm payroll", "nfp", "cpi", "consumer price index",
    "ppi", "producer price", "fed chair", "powell", "gdp",
    "unemployment rate", "retail sales"
]
