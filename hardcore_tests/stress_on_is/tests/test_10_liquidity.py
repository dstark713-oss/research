# test_10_liquidity.py
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
    m_base = compute_metrics(base)
    # proxy: увеличение slippage (лишняя ликвидность)
    results = {"baseline": m_base}
    for extra_slip in [5, 10, 20, 50]:
        cost = 2 * (5.0 + extra_slip) / 10000.0
        adj = base - cost
        results[f"adv_slip_{extra_slip}bps"] = compute_metrics(adj)
    lines = ["# Test 10: Liquidity (proxy)",
             "\n## Методика",
             "Упрощённый прокси: увеличение slippage как proxy для ликвидностного риска.",
             "\n## Результаты",
             "| Сценарий | CAGR | MDD | Sharpe |",
             "|----------|------|-----|--------|"]
    for k, m in results.items():
        if not m: continue
        lines.append(f"| {k} | {m['CAGR']*100:+.2f}% | {m['MDD']*100:+.2f}% | {m['Sharpe']:.3f} |")
    lines.append(f"\nВремя: {time.time()-t0:.1f}s")
    save_report("test_10_liquidity", lines)
    return results