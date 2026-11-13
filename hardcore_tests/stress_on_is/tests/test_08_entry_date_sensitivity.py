# test_08_entry_date_sensitivity.py
import time
import numpy as np
import pandas as pd
from ..common.data_loader import load_frozen_equity
from ..common.metrics import compute_metrics
from ..common.reporting import save_report

SHIFTS = [1, 5, 10, 20]

def run(verbose=True) -> dict:
    t0 = time.time()
    frozen = load_frozen_equity()
    base = pd.Series(frozen["sized_ret"].values, index=pd.DatetimeIndex(frozen["date"]))
    results = {"baseline": compute_metrics(base)}
    for shift in SHIFTS:
        adj = base.iloc[shift:]
        results[f"shift_{shift}"] = compute_metrics(adj)
    lines = ["# Test 08: Entry Date Sensitivity",
             "\n## Методика",
             f"Сдвиг первой даты входа на {SHIFTS} дней.",
             "\n## Результаты",
             "| Сценарий | CAGR | MDD | Sharpe |",
             "|----------|------|-----|--------|"]
    for k, m in results.items():
        if not m: continue
        lines.append(f"| {k} | {m['CAGR']*100:+.2f}% | {m['MDD']*100:+.2f}% | {m['Sharpe']:.3f} |")
    lines.append(f"\nВремя: {time.time()-t0:.1f}s")
    save_report("test_08_entry_date_sensitivity", lines)

    from ..common.reporting import plot_equity_curves
    curves = {"baseline": base}
    for shift in SHIFTS:
        curves[f"shift_{shift}"] = base.iloc[shift:]
    plot_equity_curves(curves, "Test 08: Entry Date Sensitivity", "test_08_equity")

    return results