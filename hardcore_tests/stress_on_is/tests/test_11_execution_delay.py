# test_11_execution_delay.py
import time
import pandas as pd
from ..common.data_loader import load_frozen_equity
from ..common.metrics import compute_metrics
from ..common.reporting import save_report

DELAYS = [1, 2]

def run(verbose=True) -> dict:
    t0 = time.time()
    frozen = load_frozen_equity()
    base = pd.Series(frozen["sized_ret"].values, index=pd.DatetimeIndex(frozen["date"]))
    results = {"baseline": compute_metrics(base)}
    for d in DELAYS:
        adj = base.iloc[d:]
        results[f"delay_{d}d"] = compute_metrics(adj)
    lines = ["# Test 11: Execution Delay",
             "\n## Методика",
             f"Задержка исполнения: {DELAYS} дня. MOC в baseline.",
             "\n## Результаты",
             "| Сценарий | CAGR | MDD | Sharpe |",
             "|----------|------|-----|--------|"]
    for k, m in results.items():
        if not m: continue
        lines.append(f"| {k} | {m['CAGR']*100:+.2f}% | {m['MDD']*100:+.2f}% | {m['Sharpe']:.3f} |")
    lines.append(f"\nВремя: {time.time()-t0:.1f}s")
    save_report("test_11_execution_delay", lines)
    return results