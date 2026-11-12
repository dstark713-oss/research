# backtests/rm_iswf/metrics.py
# ============================================================
# Метрики для базового портфеля и GARCH-версии.
# Включает exposure stats.
# ============================================================
import numpy as np
import pandas as pd

TRADING_DAYS = 252
REBAL_STEP = 20


def compute_metrics_from_daily(daily_ret: pd.Series,
                               turnover: pd.Series = None) -> dict:
    """Метрики из дневных доходностей (не из equity)."""
    r = daily_ret.dropna().sort_index()
    if len(r) < 2:
        return {}
    equity = (1 + r).cumprod()
    n = len(r)
    years = n / TRADING_DAYS
    total = equity.iloc[-1]
    cagr = total ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    cummax = equity.cummax()
    dd = equity / cummax - 1
    mdd = float(dd.min())
    sharpe = (r.mean() / r.std(ddof=1)) * np.sqrt(TRADING_DAYS) \
        if r.std(ddof=1) > 0 else np.nan
    # Sortino: RMS(min(r, 0)) по всем наблюдениям
    downside_sq = np.minimum(r, 0) ** 2
    downside_dev = np.sqrt(downside_sq.mean())
    sortino = (r.mean() / downside_dev) * np.sqrt(TRADING_DAYS) \
        if downside_dev > 0 else np.nan
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan
    if turnover is not None and len(turnover) > 0:
        mean_to = turnover.mean()
        turnover_annual = mean_to * (TRADING_DAYS / REBAL_STEP)
    else:
        turnover_annual = np.nan
    return {
        "CAGR": cagr, "MDD": mdd, "VOL": vol,
        "Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar,
        "Turnover": turnover_annual,
        "TotalReturn": total - 1,
        "Years": years, "N_days": n,
    }


def compute_exposure_stats(exposure: pd.Series) -> dict:
    """Статистика экспозиции."""
    e = exposure.dropna()
    if len(e) == 0:
        return {}
    return {
        "exposure_mean": float(e.mean()),
        "exposure_median": float(e.median()),
        "exposure_min": float(e.min()),
        "exposure_max": float(e.max()),
        "exposure_std": float(e.std(ddof=1)),
        "frac_below_1": float((e < 0.999).mean()),
        "frac_full_exposure": float((e >= 0.999).mean()),
    }