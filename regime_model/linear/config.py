# regime_model/linear/config.py
# ============================================================
# Константы линейной модели сайзинга.
# Схема аналогична GARCH: target vol 15%, exposure = min(1, target/forecast).
# ============================================================
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 20
REBAL_STEP = 20
TRADING_DAYS = 252

# --- Sizing (те же, что в GARCH) ---
TARGET_VOL_ANNUAL = 0.15
MAX_EXPOSURE = 1.0
MIN_EXPOSURE = 0.0
VOL_FLOOR = 0.01

# --- Linear model ---
RIDGE_ALPHA = 1.0           # L2 регуляризация
MIN_TRAIN_DAYS = 100        # минимум train для fit (для стабильности)

# --- Smoke ---
SMOKE_TRAIN_FRAC = 0.6