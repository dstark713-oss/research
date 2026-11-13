# hardcore_tests/stress_on_is/common/reporting.py
from datetime import datetime
from pathlib import Path
from ..config import REPORTS_DIR, VISUAL_DIR

def save_report(name: str, lines: list):
    path = REPORTS_DIR / f"{name}.md"
    header = [f"# {name}\n", f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n"]
    path.write_text("\n".join(header + lines), encoding="utf-8")
    print(f"[report] {path}")

def save_figure(fig, name: str):
    path = VISUAL_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, facecolor="#1a1a1a")
    print(f"[visual] {path}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def plot_equity_curves(curves: dict, title: str, name: str):
    """curves: dict {label: daily_ret Series}."""
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (label, ret) in enumerate(curves.items()):
        eq = (1 + ret.dropna()).cumprod()
        ax.plot(eq.index, eq.values, color=MONO[i % len(MONO)],
                linewidth=1.3, label=label)
    ax.set_xlabel("Дата")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.set_title(title, color="#eeeeee", fontsize=13)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="#555555", linewidth=0.5, linestyle=":")
    plt.tight_layout()
    save_figure(fig, name)
    plt.close(fig)


def plot_hist(values, title: str, name: str, xlabel: str):
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=50, color=MONO[1], edgecolor="#444444", alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Частота")
    ax.set_title(title, color="#eeeeee", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, name)
    plt.close(fig)