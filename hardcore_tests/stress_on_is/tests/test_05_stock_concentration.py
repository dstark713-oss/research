# test_05_stock_concentration.py
import time
import numpy as np
import pandas as pd
from ..common.data_loader import load_frozen_equity
from ..common.metrics import compute_metrics
from ..common.reporting import save_report

def run(verbose=True) -> dict:
    """Ex-post: удаление лучших бумаг по итоговой доходности."""
    t0 = time.time()
    # NB: нужны holdings. В frozen нет holdings, только daily_ret.
    # Для честного теста нужен отдельный прогон с удалением тикеров.
    # Упрощённая версия: удаляем N% лучших дней (прокси).
    frozen = load_frozen_equity()
    base = pd.Series(frozen["sized_ret"].values, index=pd.DatetimeIndex(frozen["date"]))
    results = {"baseline": compute_metrics(base)}
    for pct in [5, 10, 20, 50]:
        # удаляем pct% лучших дней (прокси для удаления лучших бумаг)
        top_n = int(len(base) * pct / 100)
        idx = base.nlargest(top_n).index
        adj = base.drop(index=idx)
        results[f"drop_top_{pct}pct_days"] = compute_metrics(adj)
    lines = ["# Test 05: Stock Concentration (ex-post, proxy)",
             "\n## Методика",
             "Упрощённый прокси: удаление N% лучших дней. "
             "Точный тест требует holdings и удаления лучших тикеров.",
             "\n## Результаты",
             "| Сценарий | CAGR | MDD | Sharpe |",
             "|----------|------|-----|--------|"]
    for k, m in results.items():
        if not m: continue
        lines.append(f"| {k} | {m['CAGR']*100:+.2f}% | {m['MDD']*100:+.2f}% | {m['Sharpe']:.3f} |")
    lines.append(f"\nВремя: {time.time()-t0:.1f}s")
    save_report("test_05_stock_concentration", lines)
    return results