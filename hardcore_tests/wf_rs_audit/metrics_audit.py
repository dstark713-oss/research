# hardcore_tests/wf_rs_audit/metrics_audit.py
# ============================================================
# Блок 9: независимый пересчёт метрик.
# ============================================================
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import REBAL_STEP, TRADING_DAYS, verdict


def _independent_metrics(eq_series: pd.Series) -> dict:
    eq = eq_series.dropna().sort_index()
    if len(eq) < 2:
        return {}
    # дневная сетка
    full_idx = pd.date_range(eq.index.min(), eq.index.max(), freq="B")
    eq_daily = eq.reindex(full_idx, method="ffill").dropna()
    rets = eq_daily.pct_change().dropna()
    n = len(rets)
    years = n / TRADING_DAYS

    total = eq_daily.iloc[-1] / eq_daily.iloc[0]
    cagr = total ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std(ddof=1) * np.sqrt(TRADING_DAYS)
    cummax = eq_daily.cummax()
    dd = eq_daily / cummax - 1
    mdd = dd.min()
    sharpe = (rets.mean() / rets.std(ddof=1)) * np.sqrt(TRADING_DAYS) \
        if rets.std(ddof=1) > 0 else np.nan
    downside = rets[rets < 0]
    down_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = (rets.mean() / down_std) * np.sqrt(TRADING_DAYS) \
        if down_std and down_std > 0 else np.nan
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan
    return {"CAGR": cagr, "MDD": mdd, "VOL": vol, "Sharpe": sharpe,
            "Sortino": sortino, "Calmar": calmar,
            "TotalReturn": total - 1, "Years": years, "N_days": n}


def run() -> list:
    csv = Path(__file__).parent.parent.parent / "backtests" / "portfolio_tests" / "output" / "rs_is_wf_equity.csv"
    if not csv.exists():
        return [verdict("9.1 Независимый пересчёт метрик",
                        "NOT_VERIFIED", "MEDIUM",
                        evidence=f"Нет {csv}")]

    port = pd.read_csv(csv, parse_dates=["date"]).sort_values("date")
    eq = port.set_index("date")["equity"]
    metrics = _independent_metrics(eq)

    # сравнение с заявленными (из отчёта)
    declared = {
        "CAGR": 0.2197, "MDD": -0.1532, "VOL": 0.1710,
        "Sharpe": 1.2434, "Sortino": 0.5424, "Calmar": 1.4344,
    }
    diffs = {}
    for k, v_decl in declared.items():
        v_calc = metrics.get(k, np.nan)
        if np.isfinite(v_calc) and np.isfinite(v_decl):
            diffs[k] = abs(v_calc - v_decl)
        else:
            diffs[k] = np.nan

    max_diff = max([d for d in diffs.values() if np.isfinite(d)] or [0])
    if max_diff > 0.02:
        return [verdict(
            "9.1 Независимый пересчёт метрик",
            "FAIL", "HIGH",
            evidence=f"max diff = {max_diff:.4f}. "
                     f"Заявлено: {declared}, пересчитано: {metrics}. "
                     f"Расхождение >2%.",
        )]

    # turnover: проверяем annualization
    if "turnover" in port.columns:
        to = port["turnover"].dropna()
        mean_to = to.mean()
        annual_to = mean_to * (TRADING_DAYS / REBAL_STEP)
        declared_to = 9.7606
        if abs(annual_to - declared_to) > 0.5:
            return [
                verdict("9.1 Независимый пересчёт метрик", "PASS",
                        evidence=f"max diff = {max_diff:.6f}. Метрики совпадают."),
                verdict("9.2 Turnover annualization", "FAIL", "MEDIUM",
                        evidence=f"Пересчитано: {annual_to:.4f}, "
                                 f"заявлено: {declared_to:.4f}"),
            ]

    return [verdict(
        "9.1 Независимый пересчёт метрик",
        "PASS",
        evidence=f"max diff = {max_diff:.6f}. Метрики совпадают с заявленными.",
    )]