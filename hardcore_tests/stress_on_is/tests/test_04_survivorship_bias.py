# test_04_survivorship_bias.py
import time
import numpy as np
import pandas as pd
from ..common.data_loader import load_frozen_equity
from ..common.metrics import compute_metrics
from ..common.reporting import save_report
from ..config import SURVIVORSHIP_SCENARIOS

def run(verbose=True) -> dict:
    t0 = time.time()
    frozen = load_frozen_equity()
    base = pd.Series(frozen["sized_ret"].values, index=pd.DatetimeIndex(frozen["date"]))
    results = {"baseline": compute_metrics(base)}
    for name, penalty in SURVIVORSHIP_SCENARIOS.items():
        # годовой penalty -> дневной
        daily_penalty = (1 + penalty) ** (1 / 252) - 1
        adj = base + daily_penalty  # поправка (penalty отрицательный)
        results[name] = compute_metrics(adj)
    lines = ["# Test 04: Survivorship Bias",
             "\n## Методика",
             "Сценарные поправки к годовой доходности (не point-in-time).",
             "\n## Результаты",
             "| Сценарий | CAGR | MDD | Sharpe |",
             "|----------|------|-----|--------|"]
    for k, m in results.items():
        if not m: continue
        lines.append(f"| {k} | {m['CAGR']*100:+.2f}% | {m['MDD']*100:+.2f}% | {m['Sharpe']:.3f} |")
    lines.append(f"\nВремя: {time.time()-t0:.1f}s")
    save_report("test_04_survivorship_bias", lines)
    return results