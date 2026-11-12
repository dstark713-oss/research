# random_audit/bootstrap/config.py
# ============================================================
# Константы для bootstrap-прогона. НЕ пересчитываем Monte Carlo.
# ============================================================
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent
ML_EQUITY_CSV = ROOT / "backtests" / "portfolio_tests" / "output" / "rs_is_wf_equity.csv"
RANDOM_RESULTS_CSV = ROOT / "hardcore_tests" / "wf_rs_audit" / "random_audit" / "reports" / "random_audit_results.csv"

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 20
TOP_N = 10
REBAL_STEP = 20
TRADING_DAYS = 252

# --- bootstrap ---
N_RANDOM_PORTFOLIOS = 100        # сколько random портфелей сравниваем
BLOCK_SIZES = [10, 20, 40, 60]   # чувствительность к длине блока
N_BOOTSTRAP = 2000               # итераций на один портфель × блок
BASE_SEED = 20240101