# regime_model/target_vol/config.py
# ============================================================
# Константы для тупой Target Volatility модели.
# ============================================================
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252

# --- Sizing ---
TARGET_VOL_ANNUAL = 0.15
MAX_EXPOSURE = 1.0
MIN_EXPOSURE = 0.0
VOL_FLOOR = 0.01

# --- Realized vol window ---
REALIZED_VOL_WINDOW = 20   # прошлые 20 дней