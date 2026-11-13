# test_09_random_signal.py
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
    # перемешиваем доходности (разрушаем связь)
    rng = np.random.RandomState(42)
    shuffled = pd.Series(rng.permutation(base.values), index=base.index)
    m_shuf = compute_metrics(shuffled)
    lines = ["# Test 09: Random Signal",
             "\n## Методика",
             "Перестановка доходностей (разрушение связи feature-target).",
             "\n## Результаты",
             "| Сценарий | CAGR | MDD | Sharpe |",
             "|----------|------|-----|--------|",
             f"| baseline | {m_base['CAGR']*100:+.2f}% | {m_base['MDD']*100:+.2f}% | {m_base['Sharpe']:.3f} |",
             f"| shuffled | {m_shuf['CAGR']*100:+.2f}% | {m_shuf['MDD']*100:+.2f}% | {m_shuf['Sharpe']:.3f} |",
             f"\nВремя: {time.time()-t0:.1f}s"]
    save_report("test_09_random_signal", lines)
    return {"baseline": m_base, "shuffled": m_shuf}