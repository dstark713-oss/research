# random_audit/bootstrap/turnover_fix.py
# ============================================================
# Фикс turnover ML: загружаем turnover из rs_is_wf_equity.csv
# (колонка turnover) и считаем annualized правильно.
# ============================================================
import numpy as np
import pandas as pd

from .config import TRADING_DAYS, REBAL_STEP


def compute_turnover_annual(turnover_series: pd.Series) -> dict:
    """
    Annualized one-way turnover.
    turnover_annual = mean(turnover) * (TRADING_DAYS / REBAL_STEP).
    """
    to = turnover_series.dropna()
    if len(to) == 0:
        return {"mean_turnover": np.nan, "turnover_annual": np.nan,
                "n_rebalances": 0}
    mean_to = float(to.mean())
    rebalances_per_year = TRADING_DAYS / REBAL_STEP
    annual = mean_to * rebalances_per_year
    return {
        "mean_turnover": mean_to,
        "turnover_annual": annual,
        "rebalances_per_year": rebalances_per_year,
        "n_rebalances": len(to),
    }


def recompute_ml_turnover(ml_equity_df: pd.DataFrame) -> dict:
    """Загружает turnover из ML equity и считает annualized."""
    if "turnover" not in ml_equity_df.columns:
        return {"error": "no turnover column in ML equity"}
    return compute_turnover_annual(ml_equity_df["turnover"])