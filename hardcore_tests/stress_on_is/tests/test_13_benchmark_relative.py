# test_13_benchmark_relative.py
import time
import numpy as np
import pandas as pd
from ..common.data_loader import load_frozen_equity, load_rgbitr
from ..common.metrics import compute_metrics
from ..common.reporting import save_report

def run(verbose=True) -> dict:
    t0 = time.time()
    frozen = load_frozen_equity()
    base = pd.Series(frozen["sized_ret"].values, index=pd.DatetimeIndex(frozen["date"]))
    rgb = load_rgbitr()
    common = base.index.intersection(rgb.index)
    base = base.loc[common]
    rgb = rgb.loc[common]
    m_base = compute_metrics(base)
    rgb_ret = rgb.pct_change().dropna()
    m_rgb = compute_metrics(rgb_ret)
    lines = ["# Test 13: Benchmark Relative (RGBITR)",
             "\n## Методика",
             "Сравнение с RGBITR на общих датах.",
             "\n## Результаты",
             "| Инструмент | CAGR | MDD | Sharpe |",
             "|------------|------|-----|--------|",
             f"| strategy | {m_base['CAGR']*100:+.2f}% | {m_base['MDD']*100:+.2f}% | {m_base['Sharpe']:.3f} |",
             f"| RGBITR | {m_rgb['CAGR']*100:+.2f}% | {m_rgb['MDD']*100:+.2f}% | {m_rgb['Sharpe']:.3f} |",
             f"\nВремя: {time.time()-t0:.1f}s"]
    save_report("test_13_benchmark_relative", lines)
    return {"strategy": m_base, "rgb": m_rgb}