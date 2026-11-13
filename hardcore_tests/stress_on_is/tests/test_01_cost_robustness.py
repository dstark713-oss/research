# test_01_cost_robustness.py
import time
import numpy as np
import pandas as pd

from ..common.data_loader import load_frozen_equity, load_frozen_holdings
from ..common.metrics import compute_metrics
from ..common.portfolio import apply_costs_at_rebal
from ..common.reporting import save_report
from ..config import SBER_COMMISSION_BPS, SLIPPAGE_BPS

LEVELS = [5, 10, 15, 20, 25, 30]
TOP_N = 10


def run(verbose=True) -> dict:
    t0 = time.time()
    frozen = load_frozen_equity()
    holdings = load_frozen_holdings()
    base = pd.Series(frozen["sized_ret"].values, index=pd.DatetimeIndex(frozen["date"]))
    rebal_dates = sorted(holdings.keys())

    m_base = compute_metrics(base)
    results = {"baseline": m_base}

    # базовые издержки (Сбер + slippage)
    base_cost_bps = 2 * (SBER_COMMISSION_BPS + SLIPPAGE_BPS)  # 15 б.п. round-trip

    for extra in LEVELS:
        cost_bps = base_cost_bps + extra
        net = apply_costs_at_rebal(base, rebal_dates, holdings, cost_bps, TOP_N)
        results[f"extra_{extra}bps"] = compute_metrics(net)

    lines = ["# Test 01: Cost Robustness",
             "\n## Методика",
             "Издержки применяются ТОЛЬКО в даты ребаланса, на фактический turnover.",
             f"Базовая нагрузка (Сбер 5 б.п. + slippage 2.5 б.п. на сторону): "
             f"{base_cost_bps:.1f} б.п. round-trip.",
             f"Дополнительная нагрузка: {LEVELS} б.п. round-trip.",
             "\n## Результаты",
             "| Сценарий | CAGR | MDD | VOL | Sharpe | Sortino | Calmar |",
             "|----------|------|-----|-----|--------|---------|--------|"]
    for k, m in results.items():
        if not m: continue
        lines.append(f"| {k} | {m['CAGR']*100:+.2f}% | {m['MDD']*100:+.2f}% | "
                     f"{m['VOL']*100:+.2f}% | {m['Sharpe']:.3f} | "
                     f"{m['Sortino']:.3f} | {m['Calmar']:.3f} |")
    lines.append(f"\nВремя: {time.time()-t0:.1f}s")
    save_report("test_01_cost_robustness", lines)

    # визуализация
    from ..common.reporting import plot_equity_curves
    curves = {"baseline": base}
    for extra in [10, 20, 30]:
        cost = base_cost_bps + extra
        curves[f"extra_{extra}bps"] = apply_costs_at_rebal(base, rebal_dates, holdings, cost, TOP_N)
    plot_equity_curves(curves, "Test 01: Cost Robustness — equity", "test_01_equity")

    return results

