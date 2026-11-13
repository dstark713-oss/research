# test_02_slippage.py
import time
import numpy as np
import pandas as pd

from ..common.data_loader import load_frozen_equity, load_frozen_holdings
from ..common.metrics import compute_metrics
from ..common.portfolio import apply_costs_at_rebal
from ..common.reporting import save_report
from ..config import SBER_COMMISSION_BPS

LEVELS = [1, 2, 5, 10, 15, 20, 25, 50]
TOP_N = 10


def run(verbose=True) -> dict:
    t0 = time.time()
    frozen = load_frozen_equity()
    holdings = load_frozen_holdings()
    base = pd.Series(frozen["sized_ret"].values, index=pd.DatetimeIndex(frozen["date"]))
    rebal_dates = sorted(holdings.keys())

    results = {"baseline": compute_metrics(base)}

    for slip in LEVELS:
        cost_bps = 2 * (SBER_COMMISSION_BPS + slip)
        net = apply_costs_at_rebal(base, rebal_dates, holdings, cost_bps, TOP_N)
        results[f"slip_{slip}bps"] = compute_metrics(net)

    lines = ["# Test 02: Slippage",
             "\n## Методика",
             "Издержки применяются ТОЛЬКО в даты ребаланса, на фактический turnover.",
             f"Комиссия Сбера: {SBER_COMMISSION_BPS:.1f} б.п. на сторону.",
             f"Slippage на сторону: {LEVELS} б.п.",
             "\n## Результаты",
             "| Сценарий | CAGR | MDD | VOL | Sharpe | Sortino | Calmar |",
             "|----------|------|-----|-----|--------|---------|--------|"]
    for k, m in results.items():
        if not m: continue
        lines.append(f"| {k} | {m['CAGR']*100:+.2f}% | {m['MDD']*100:+.2f}% | "
                     f"{m['VOL']*100:+.2f}% | {m['Sharpe']:.3f} | "
                     f"{m['Sortino']:.3f} | {m['Calmar']:.3f} |")
    lines.append(f"\nВремя: {time.time()-t0:.1f}s")
    save_report("test_02_slippage", lines)

    from ..common.reporting import plot_equity_curves
    curves = {"baseline": base}
    for slip in [5, 10, 20]:
        cost = 2 * (SBER_COMMISSION_BPS + slip)
        curves[f"slip_{slip}bps"] = apply_costs_at_rebal(base, rebal_dates, holdings, cost, TOP_N)
    plot_equity_curves(curves, "Test 02: Slippage — equity", "test_02_equity")

    return results