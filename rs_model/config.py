# rs_model/config.py
# ============================================================
# Центральные константы. Всё, что «магическое» — здесь.
# ============================================================
from pathlib import Path

ROOT = Path(__file__).parent.parent
SPLIT_JSON = ROOT / "backtests" / "split_info.json"
STOCKS_DB = ROOT / "dataset" / "sql" / "stocks.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"

HORIZON = 20
PURGE = 21           # 20 + 1 (безопаснее)
N_GRADES = 5         # для YetiRank

# 11 фич — ровно как в ТЗ
FEATURES = [
    "upside_semivariance_20",
    "downside_semivariance_20",
    "distance_to_low_60",
    "range_volatility_ratio_20",
    "distance_to_high_60",
    "range_position_change_5",
    "negative_return_fraction_20",
    "up_down_day_ratio_20",
    "dist_from_high_intraday_252",
    "intraday_return_skew_20",
    "return_autocorr_2_20",
]

# Параметры базовой модели (CatBoostRegressor)
BASE_PARAMS = {
    "depth": 3,
    "learning_rate": 0.03,
    "iterations": 1500,
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

# Параметры Ranker (challenger)
RANKER_PARAMS = {
    "depth": 3,
    "learning_rate": 0.03,
    "iterations": 1500,
    "l2_leaf_reg": 50,
    "random_strength": 3,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "border_count": 64,
    "rsm": 0.8,
    "loss_function": "YetiRank",
    "eval_metric": "NDCG:top=10",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}

# Ансамбль
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]
USE_ENSEMBLE = True

# Grid search (не более 8 конфигураций)
GRID = [
    {"depth": 3, "l2_leaf_reg": 30, "random_strength": 3, "learning_rate": 0.03},
    {"depth": 3, "l2_leaf_reg": 50, "random_strength": 3, "learning_rate": 0.03},
    {"depth": 3, "l2_leaf_reg": 100, "random_strength": 3, "learning_rate": 0.03},
    {"depth": 3, "l2_leaf_reg": 50, "random_strength": 1, "learning_rate": 0.03},
    {"depth": 4, "l2_leaf_reg": 50, "random_strength": 3, "learning_rate": 0.03},
    {"depth": 4, "l2_leaf_reg": 100, "random_strength": 3, "learning_rate": 0.03},
    {"depth": 3, "l2_leaf_reg": 50, "random_strength": 3, "learning_rate": 0.05},
    {"depth": 5, "l2_leaf_reg": 100, "random_strength": 3, "learning_rate": 0.03},
]