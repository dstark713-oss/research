# random_audit/bootstrap/metrics_local.py
# ============================================================
# Локальный пересчёт метрик из port_rs (20-дневные доходности).
# Не импортируем rs_model.metrics.
# ============================================================
import numpy as np
import pandas as pd

from .config import TRADING_DAYS, REBAL_STEP


def compute_metrics_from_port_rs(port_rs: pd.Series,
                                 turnover: pd.Series = None) -> dict:
    """
    port_rs: 20-дневные доходности на датах ребаланса.
    Equity = cumprod(1 + port_rs).
    Метрики — на equity, приведённой к дневной сетке (reindex + ffill).
    """
    rs = port_rs.dropna().sort_index()
    if len(rs) < 2:
        return {}
    equity = (1 + rs).cumprod()
    # дневная сетка
    full_idx = pd.date_range(equity.index.min(), equity.index.max(), freq="B")
    eq_d = equity.reindex(full_idx, method="ffill").dropna()
    rets = eq_d.pct_change().dropna()
    n = len(rets)
    years = n / TRADING_DAYS
    total = eq_d.iloc[-1] / eq_d.iloc[0]
    cagr = total ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std(ddof=1) * np.sqrt(TRADING_DAYS)
    cummax = eq_d.cummax()
    dd = eq_d / cummax - 1
    mdd = float(dd.min())
    sharpe = (rets.mean() / rets.std(ddof=1)) * np.sqrt(TRADING_DAYS) \
        if rets.std(ddof=1) > 0 else np.nan
    downside = rets[rets < 0]
    down_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = (rets.mean() / down_std) * np.sqrt(TRADING_DAYS) \
        if down_std and down_std > 0 else np.nan
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan

    if turnover is not None and len(turnover) > 0:
        to = turnover.dropna()
        mean_to = float(to.mean()) if len(to) > 0 else np.nan
        turnover_annual = mean_to * (TRADING_DAYS / REBAL_STEP) if np.isfinite(mean_to) else np.nan
    else:
        turnover_annual = np.nan

    return {
        "CAGR": cagr, "MDD": mdd, "VOL": vol,
        "Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar,
        "Turnover": turnover_annual,
        "TotalReturn": total - 1,
        "Years": years, "N_days": n,
    }