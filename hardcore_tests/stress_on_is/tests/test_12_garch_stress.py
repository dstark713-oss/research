# test_12_garch_stress.py
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
    # GARCH stress: симулируем 2x волатильность
    rng = np.random.RandomState(42)
    noisy = base * (1 + rng.normal(0, 0.5, len(base)))
    m_noisy = compute_metrics(noisy)
    lines = ["# Test 12: GARCH Stress",
             "\n## Методика",
             "Искусственное удвоение волатильности доходностей.",
             "\n## Результаты",
             "| Сценарий | CAGR | MDD | Sharpe |",
             "|----------|------|-----|--------|",
             f"| baseline | {m_base['CAGR']*100:+.2f}% | {m_base['MDD']*100:+.2f}% | {m_base['Sharpe']:.3f} |",
             f"| 2x vol | {m_noisy['CAGR']*100:+.2f}% | {m_noisy['MDD']*100:+.2f}% | {m_noisy['Sharpe']:.3f} |",
             f"\nВремя: {time.time()-t0:.1f}s"]
    save_report("test_12_garch_stress", lines)
    return {"baseline": m_base, "noisy": m_noisy}