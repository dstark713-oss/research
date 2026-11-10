# rs_model/grid_search.py
# ============================================================
# Ограниченный grid search: 8 конфигураций.
# Оценивается на внутренних WF validation folds.
# Финальный fold (стрессовый) НЕ используется для выбора.
# ============================================================
import numpy as np
import pandas as pd

from .config import GRID, FEATURES
from .base_model import create_base_model, fit_base_model
from .data import to_long
from .metrics import mean_ic


def _prepare_xy(df_long: pd.DataFrame, dates):
    df = df_long[df_long["date"].isin(dates)]
    X = df[FEATURES].values
    y = df["target"].values
    return X, y, df


def run_grid(features_rank, target_gauss, folds, exclude_last=True):
    """
    Прогоняет GRID на folds[:-1] (если exclude_last), возвращает
    DataFrame с mean IC по фолдам для каждой конфигурации.
    """
    folds_use = folds[:-1] if exclude_last else folds
    results = []
    for cfg_idx, cfg in enumerate(GRID):
        fold_ics = []
        for fold_idx, (train_dates, val_dates) in enumerate(folds_use):
            # train
            train_long = to_long(features_rank, target_gauss)
            train_long = train_long[train_long["date"].isin(train_dates)]
            X_tr = train_long[FEATURES].values
            y_tr = train_long["target"].values

            val_long = to_long(features_rank, target_gauss)
            val_long = val_long[val_long["date"].isin(val_dates)]
            X_va = val_long[FEATURES].values
            y_va = val_long["target"].values

            model = create_base_model(cfg, seed=42)
            fit_base_model(model, X_tr, y_tr, X_va, y_va)

            # предсказания в wide
            pred = val_long.copy()
            pred["pred"] = model.predict(X_va)
            pred_wide = pred.pivot(index="date", columns="ticker", values="pred")
            target_wide = pred.pivot(index="date", columns="ticker", values="target")
            ic = mean_ic(pred_wide, target_wide)
            fold_ics.append(ic)
        results.append({
            "config_idx": cfg_idx,
            "params": cfg,
            "mean_ic": np.nanmean(fold_ics),
            "fold_ics": fold_ics,
        })
    return pd.DataFrame(results)