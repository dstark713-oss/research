# backtests/rm_iswf/plot_wf.py
# ============================================================
# График: GARCH vs baseline + vol/forecast/exposure.
# Тёмный фон, монохром (стиль проекта).
# ============================================================
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

MONO = ["#ffffff", "#cccccc", "#999999", "#666666", "#bbbbbb", "#777777"]


def setup_dark_style():
    plt.rcParams.update({
        "figure.facecolor": "#1a1a1a",
        "axes.facecolor": "#1a1a1a",
        "savefig.facecolor": "#1a1a1a",
        "axes.edgecolor": "#888888",
        "axes.labelcolor": "#dddddd",
        "axes.titlecolor": "#eeeeee",
        "xtick.color": "#aaaaaa",
        "ytick.color": "#aaaaaa",
        "text.color": "#dddddd",
        "grid.color": "#333333",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "legend.facecolor": "#2a2a2a",
        "legend.edgecolor": "#888888",
        "legend.labelcolor": "#dddddd",
        "font.size": 10,
    })


def plot_equity(baseline_ret: pd.Series, garch_ret: pd.Series,
                out_path: Path):
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    eq_b = (1 + baseline_ret.dropna()).cumprod()
    eq_g = (1 + garch_ret.dropna()).cumprod()

    ax.plot(eq_b.index, eq_b.values, color=MONO[1], linewidth=1.3,
            label="RS baseline (100% exposure)")
    ax.plot(eq_g.index, eq_g.values, color=MONO[0], linewidth=1.6,
            label="RS + GARCH sizing")

    ax.set_xlabel("Дата")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.set_title("GARCH risk sizing vs RS baseline (IS)",
                 color="#eeeeee", fontsize=13)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="#555555", linewidth=0.5, linestyle=":")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, facecolor="#1a1a1a")
    plt.close()
    print(f"[plot] {out_path}")


def plot_vol_and_exposure(vol_forecast: pd.Series,
                          exposure: pd.Series,
                          out_path: Path,
                          target_vol: float):
    setup_dark_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 1. vol forecast
    ax1 = axes[0]
    ax1.plot(vol_forecast.index, vol_forecast.values * 100,
             color=MONO[0], linewidth=1.3, label="GARCH forecast vol (annualized)")
    ax1.axhline(target_vol * 100, color=MONO[2], linewidth=1.0,
                linestyle="--", label=f"Target vol = {target_vol*100:.0f}%")
    ax1.set_ylabel("Volatility, %")
    ax1.set_title("GARCH forecast volatility (20d horizon)", color="#eeeeee")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. exposure
    ax2 = axes[1]
    ax2.plot(exposure.index, exposure.values,
             color=MONO[0], linewidth=1.3, label="Exposure")
    ax2.axhline(1.0, color=MONO[3], linewidth=0.8, linestyle=":",
                label="Max (100%)")
    ax2.axhline(0.0, color=MONO[3], linewidth=0.8, linestyle=":")
    ax2.set_ylabel("Exposure")
    ax2.set_xlabel("Дата")
    ax2.set_title("GARCH exposure over time", color="#eeeeee")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, facecolor="#1a1a1a")
    plt.close()
    print(f"[plot] {out_path}")