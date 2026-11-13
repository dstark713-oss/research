# hardcore_tests/stress_on_is/common/metrics.py
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def compute_metrics(daily_ret: pd.Series, turnover: float = 0.0) -> dict:
    r = daily_ret.dropna().sort_index()
    if len(r) < 2:
        return {}
    eq = (1 + r).cumprod()
    n = len(r); years = n / TRADING_DAYS
    total = eq.iloc[-1]
    cagr = total ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    cummax = eq.cummax()
    dd = eq / cummax - 1
    mdd = float(dd.min())
    sharpe = (r.mean() / r.std(ddof=1)) * np.sqrt(TRADING_DAYS) if r.std(ddof=1) > 0 else np.nan
    downside = np.minimum(r, 0) ** 2
    down_dev = np.sqrt(downside.mean())
    sortino = (r.mean() / down_dev) * np.sqrt(TRADING_DAYS) if down_dev > 0 else np.nan
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan
    return {
        "CAGR": cagr, "MDD": mdd, "VOL": vol,
        "Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar,
        "TotalReturn": total - 1, "Years": years, "N_days": n,
        "Turnover": turnover,
    }

def drawdown_series(daily_ret: pd.Series) -> pd.Series:
    eq = (1 + daily_ret.dropna()).cumprod()
    return eq / eq.cummax() - 1

def recovery_time(drawdown: pd.Series) -> pd.Series:
    """Для каждой даты — сколько дней до восстановления (0 если не в просадке)."""
    dd = drawdown.fillna(0).values
    n = len(dd)
    out = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if dd[i] >= -1e-9:
            out[i] = 0
        else:
            j = i + 1
            while j < n and dd[j] < -1e-9:
                j += 1
            out[i] = j - i if j < n else np.nan
    return pd.Series(out, index=drawdown.index)