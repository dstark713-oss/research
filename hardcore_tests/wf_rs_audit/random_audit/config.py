# random_audit/config.py
# ============================================================
# Все константы эксперимента. НЕ импортируем ничего из rs_model —
# независимая реализация.
# ============================================================
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
STOCKS_DB = ROOT / "dataset" / "sql" / "stocks.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"
SPLIT_JSON = ROOT / "backtests" / "split_info.json"
ML_EQUITY_CSV = ROOT / "backtests" / "portfolio_tests" / "output" / "rs_is_wf_equity.csv"
ML_TRADES_CSV = ROOT / "backtests" / "portfolio_tests" / "output" / "rs_is_wf_trades.csv"

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- параметры исходного WF (фиксированы, не менять) ---
HORIZON = 20
PURGE = 21
N_FOLDS = 6
TOP_N = 10
REBAL_STEP = 20
TRADING_DAYS = 252
SPLIT_DATE = None  # загружается из split_info.json

# --- Monte Carlo ---
N_SIMULATIONS = 10000
BASE_SEED = 20240101

# --- Scenario B: persistence levels ---
# Доля портфеля, которая СОХРАНЯЕТСЯ от предыдущего ребаланса
PERSISTENCE_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

# --- Bootstrap ---
BLOCK_SIZE_MAIN = 20              # основной блок = горизонт target
BLOCK_SIZES_SENSITIVITY = [10, 20, 40, 60]
N_BOOTSTRAP = 2000

# --- Допуск совпадения независимой реконструкции ML с оригиналом ---
ML_RECON_TOL = 0.01