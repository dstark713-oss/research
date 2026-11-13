# backtests/garch_tuning/config.py
# ============================================================
# Константы для тюнинга GARCH (daily vs horizon).
# ============================================================
from pathlib import Path

OUT_DIR = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- refit_freq для daily GARCH ---
# 1 = fit каждый день (дорого, ~30-60 мин)
# 5 = fit раз в 5 дней (~5-10 мин)
REFIT_FREQ = 5