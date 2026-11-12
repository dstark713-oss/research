# риск_модели/гарч/config.py
# ============================================================
# Константы GARCH risk sizing.
# ============================================================
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
STOCKS_DB = ROOT / "dataset" / "sql" / "stocks.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"
SPLIT_JSON = ROOT / "backtests" / "split_info.json"
ML_EQUITY_CSV = ROOT / "backtests" / "portfolio_tests" / "output" / "rs_is_wf_equity.csv"
ML_TRADES_CSV = ROOT / "backtests" / "portfolio_tests" / "output" / "rs_is_wf_trades.csv"

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 20
REBAL_STEP = 20
TRADING_DAYS = 252

# --- GARCH ---
GARCH_P = 1
GARCH_Q = 1
GARCH_DIST = "t"
GARCH_MEAN = "Constant"
GARCH_VOL = "Garch"
GARCH_POWER = 2.0

# --- Sizing ---
TARGET_VOL_ANNUAL = 0.15   # предварительное значение (не подбирать на IS)
MAX_EXPOSURE = 1.0
MIN_EXPOSURE = 0.0
VOL_FLOOR = 0.01           # защита от деления на ~0

# --- Smoke ---
SMOKE_TRAIN_FRAC = 0.6     # 60% train, 40% "val" (без WF)