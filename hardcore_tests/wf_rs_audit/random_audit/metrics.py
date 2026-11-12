# random_audit/metrics.py
# ============================================================
# Независимый пересчёт метрик. НЕ импортируем rs_model.metrics.
# ============================================================
import numpy as np
import pandas as pd

from .config import TRADING_DAYS, REBAL_STEP


def compute_metrics(equity: pd.Series, turnover: pd.Series = None,
                    rf_annual: float = 0.0) -> dict:
    """
    Метрики на дневной сетке (reindex + ffill).
    CAGR, MDD, VOL, Sharpe, Sortino, Calmar, Turnover.
    """
    eq = equity.dropna().sort_index()
    if len(eq) < 2:
        return {}
    full_idx = pd.date_range(eq.index.min(), eq.index.max(), freq="B")
    eq_d = eq.reindex(full_idx, method="ffill").dropna()
    rets = eq_d.pct_change().dropna()
    n = len(rets)
    years = n / TRADING_DAYS
    total = eq_d.iloc[-1] / eq_d.iloc[0]
    cagr = total ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std(ddof=1) * np.sqrt(TRADING_DAYS)
    cummax = eq_d.cummax()
    dd = eq_d / cummax - 1
    mdd = dd.min()
    rf_d = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = rets - rf_d
    sharpe = (excess.mean() / excess.std(ddof=1)) * np.sqrt(TRADING_DAYS) \
        if excess.std(ddof=1) > 0 else np.nan
    downside_sq = np.minimum(excess, 0) ** 2
    downside_dev = np.sqrt(downside_sq.mean())
    sortino = (excess.mean() / downside_dev) * np.sqrt(TRADING_DAYS) \
        if downside_dev > 0 else np.nan
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan
    if turnover is not None and len(turnover) > 0:
        mean_to = turnover.mean()
        turnover_annual = mean_to * (TRADING_DAYS / REBAL_STEP)
    else:
        turnover_annual = 0.0
    return {
        "CAGR": cagr, "MDD": mdd, "VOL": vol,
        "Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar,
        "Turnover": turnover_annual,
        "TotalReturn": total - 1,
        "Years": years, "N_days": n,
    }


def compute_metrics_from_dict(result: dict) -> dict:
    """Из результата portfolio_engine.run_random_portfolio."""
    dates = result["dates"]
    equity = result["equity"]
    turnover = result["turnover"]
    if len(dates) == 0:
        return {}
    eq_s = pd.Series(equity, index=pd.DatetimeIndex(dates))
    to_s = pd.Series(turnover, index=pd.DatetimeIndex(dates))
    return compute_metrics(eq_s, to_s)