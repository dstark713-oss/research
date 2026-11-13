# freeze_model/config.py
# ============================================================
# Замороженные параметры модели (не менять после freeze).
# ============================================================
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- RS-модель (заморожена) ---
RS_PARAMS = {
    "depth": 3,
    "learning_rate": 0.03,
    "iterations": 300,
    "l2_leaf_reg": 50,
    "random_strength": 3,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "border_count": 64,
    "rsm": 0.8,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}

RS_HORIZON = 20       # target горизонт (фикс)
RS_REBALANCE = 20     # горизонт ребаланса (фикс, оптимум)
RS_TOP_N = 10
RS_PURGE = 21
RS_N_FOLDS = 6

# --- GARCH daily (заморожена) ---
GARCH_TARGET_VOL = 0.15     # целевая годовая волатильность
GARCH_MIN_TRAIN = 150       # минимум дней для fit
GARCH_MAX_EXPOSURE = 1.0
GARCH_MIN_EXPOSURE = 0.0

# --- split ---
SPLIT_DATE = None  # загружается из backtests/split_info.json