# test_14_cash_carry.py
import time
import numpy as np
import pandas as pd
from ..common.data_loader import load_frozen_equity, load_rgbitr, load_rgbi
from ..common.metrics import compute_metrics
from ..common.reporting import save_report

def run(verbose=True) -> dict:
    t0 = time.time()
    frozen = load_frozen_equity()
    base = pd.Series(frozen["sized_ret"].values, index=pd.DatetimeIndex(frozen["date"]))
    exposure = pd.Series(frozen["exposure"].values, index=pd.DatetimeIndex(frozen["date"]))
    cash_share = 1 - exposure
    rgb = load_rgbitr()
    rgbi = load_rgbi()
    # годовая carry: RGBITR - RGBI за предыдущий год
    common = base.index.intersection(rgb.index).intersection(rgbi.index)
    rgb_y = rgb.loc[common].groupby(rgb.loc[common].index.year).last()
    rgbi_y = rgbi.loc[common].groupby(rgbi.loc[common].index.year).last()
    carry_y = rgb_y.pct_change() - rgbi_y.pct_change()
    carry_y = carry_y.shift(1)  # без look-ahead
    carry_daily = carry_y / 252
    # применяем carry на cash_share
    carry_series = pd.Series(0.0, index=base.index)
    for d in base.index:
        if d.year in carry_daily.index:
            carry_series.loc[d] = carry_daily.loc[d.year]
    adj = base + cash_share * carry_series
    m_base = compute_metrics(base)
    m_adj = compute_metrics(adj)
    lines = ["# Test 14: Cash Carry",
             "\n## Методика",
             "Доходность кэша: (RGBITR - RGBI) за предыдущий год, без look-ahead.",
             "Применяется на долю cash (1 - exposure).",
             "\n## Результаты",
             "| Сценарий | CAGR | MDD | Sharpe |",
             "|----------|------|-----|--------|",
             f"| baseline | {m_base['CAGR']*100:+.2f}% | {m_base['MDD']*100:+.2f}% | {m_base['Sharpe']:.3f} |",
             f"| + cash carry | {m_adj['CAGR']*100:+.2f}% | {m_adj['MDD']*100:+.2f}% | {m_adj['Sharpe']:.3f} |",
             f"\nВремя: {time.time()-t0:.1f}s"]
    save_report("test_14_cash_carry", lines)

    from ..common.reporting import plot_equity_curves
    plot_equity_curves(
        {"baseline": base, "cash_carry": adj},
        "Test 14: Cash Carry — equity",
        "test_14_cash_carry",
    )

    return {"baseline": m_base, "cash_carry": m_adj}