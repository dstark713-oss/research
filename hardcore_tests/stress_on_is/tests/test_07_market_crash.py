# test_07_market_crash.py
import time
import numpy as np
import pandas as pd
from ..common.data_loader import load_frozen_equity
from ..common.metrics import compute_metrics
from ..common.reporting import save_report

def run(verbose=True) -> dict:
    t0 = time.time()
    frozen = load_frozen_equity()
    base = pd.Series(frozen["sized_ret"].values, index=pd.DatetimeIndex(frozen["date"]))
    # однодневный шок: -50% для топ-50, -25% для остальных
    # proxy: -40% в один случайный день
    shock_date = base.index[len(base) // 2]  # середина
    shock_ret = -0.40
    adj = base.copy()
    adj.loc[shock_date] = shock_ret
    m_adj = compute_metrics(adj)
    m_base = compute_metrics(base)
    lines = ["# Test 07: Market Crash",
             "\n## Методика",
             "Однодневный шок: -40% (proxy для -50%/-25%).",
             f"Дата шока: {shock_date.date()}",
             "\n## Результаты",
             "| Сценарий | CAGR | MDD | Sharpe |",
             "|----------|------|-----|--------|",
             f"| baseline | {m_base['CAGR']*100:+.2f}% | {m_base['MDD']*100:+.2f}% | {m_base['Sharpe']:.3f} |",
             f"| crash | {m_adj['CAGR']*100:+.2f}% | {m_adj['MDD']*100:+.2f}% | {m_adj['Sharpe']:.3f} |",
             f"\nВремя: {time.time()-t0:.1f}s"]
    save_report("test_07_market_crash", lines)

    from ..common.reporting import plot_equity_curves
    plot_equity_curves(
        {"baseline": base, "crash": adj},
        "Test 07: Market Crash — equity",
        "test_07_crash_equity",
    )

    return {"baseline": m_base, "crash": m_adj}