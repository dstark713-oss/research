# hardcore_tests/stress_on_is/config.py
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
FREEZE_DIR = ROOT / "freeze_model" / "reports"
BENCH_DB = ROOT / "benchmark" / "sql" / "BENCHMARKS.sqlite"
RGBITR_DB = ROOT / "benchmark" / "sql" / "RGBITR.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"
STOCKS_DB = ROOT / "dataset" / "sql" / "stocks.sqlite"

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_DIR = REPORTS_DIR / "visual"
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

# Sber "Самостоятельный"
SBER_COMMISSION_BPS = 5.0
SLIPPAGE_BPS = 2.5

# Survivorship bias
SURVIVORSHIP_SCENARIOS = {
    "sp500_base": -0.0202,
    "sp500_stress": -0.0404,
    "nifty_base": -0.0494,
    "nifty_stress": -0.0988,
}

# Monte Carlo
MC_SIMULATIONS = 1000
MC_SEED = 42
MC_BLOCK_SIZE = 20

# Общий таймаут
TOTAL_TIMEOUT_SEC = 7200  # 2 часа