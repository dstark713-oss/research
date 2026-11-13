# backtests/rm_tuning/config.py
# ============================================================
# Константы для тюнинга ребаланса RS-модели.
# Модель/target/фичи — НЕ здесь (импортируются из rs_is_wf_v2).
# ============================================================
from pathlib import Path

OUT_DIR = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- горизонты ребаланса ---
HORIZONS = [1, 5, 10, 20, 30, 60]