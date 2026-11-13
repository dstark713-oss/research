# test_03_annual_robustness.py
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
    rows = []
    for year in sorted(base.index.year.unique()):
        seg = base[base.index.year == year]
        if len(seg) < 30: continue
        m = compute_metrics(seg)
        # rgb за год
        rgb_seg = rgb[(rgb.index.year == year)]
        if len(rgb_seg) > 2:
            rgb_ret = rgb_seg.iloc[-1] / rgb_seg.iloc[0] - 1
        else:
            rgb_ret = np.nan
        rows.append({"year": year, **m, "rgb_ret": rgb_ret})
    lines = ["# Test 03: Annual Robustness",
             "\n## Методика",
             "Метрики по каждому календарному году. Сравнение с RGBITR.",
             "\n## Результаты",
             "| Год | CAGR | MDD | Sharpe | RGBITR |",
             "|-----|------|-----|--------|--------|"]
    for r in rows:
        lines.append(f"| {r['year']} | {r.get('CAGR', np.nan)*100:+.2f}% | "
                     f"{r.get('MDD', np.nan)*100:+.2f}% | {r.get('Sharpe', np.nan):.3f} | "
                     f"{r.get('rgb_ret', np.nan)*100:+.2f}% |")
    lines.append(f"\nВремя: {time.time()-t0:.1f}s")
    save_report("test_03_annual_robustness", lines)

    from ..common.reporting import plot_hist, setup_dark_style, save_figure
    import matplotlib.pyplot as plt
    years = [r["year"] for r in rows]
    cagrs = [r["CAGR"]*100 for r in rows]
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(years, cagrs, color="#999999", edgecolor="#444444")
    ax.set_xlabel("Год")
    ax.set_ylabel("CAGR, %")
    ax.set_title("Test 03: Annual CAGR", color="#eeeeee")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="#555555", linewidth=0.5)
    plt.tight_layout()
    save_figure(fig, "test_03_annual_cagr")
    plt.close(fig)

    return {"rows": rows}