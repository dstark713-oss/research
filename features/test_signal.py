# research/features/test_signal.py
# ============================================================
# Тест сигнала: работают ли фичи из EDA для отбора топ-10?
# Только IS. Без ML, без оптимизации.
#
# Логика:
#   каждые 20 дней ранжируем акции по фиче,
#   берём топ-10 (или другой квантль),
#   держим 20 дней,
#   считаем средний RS топ-10.
#
# Сравниваем с:
#   - средним RS всех акций (случайный выбор)
#   - IMOEX (RS=0)
#   - равновзвешенным портфелем
#
# Запуск: python test_signal.py
# ============================================================
import json
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# КОНСТАНТЫ
# ============================================================
ROOT = Path(__file__).parent.parent
STOCKS_DB = ROOT / "dataset" / "sql" / "stocks.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"
SPLIT_JSON = ROOT / "backtests" / "split_info.json"

OUT_SQL = Path(__file__).parent / "sql" / "test_signal.sqlite"
OUT_SQL.parent.mkdir(exist_ok=True)
OUT_MD = Path(__file__).parent / "test_signal_report.md"
PLOTS_DIR = Path(__file__).parent / "plots_test"
PLOTS_DIR.mkdir(exist_ok=True)

HORIZON = 20
TOP_N = 10
MIN_BARS = 500

MONO = ["#ffffff", "#cccccc", "#999999", "#666666", "#444444",
        "#bbbbbb", "#777777", "#aaaaaa"]

# ============================================================
# ГРАФИКИ
# ============================================================
def setup_dark_style():
    plt.rcParams.update({
        "figure.facecolor": "#1a1a1a", "axes.facecolor": "#1a1a1a",
        "savefig.facecolor": "#1a1a1a", "axes.edgecolor": "#888888",
        "axes.labelcolor": "#dddddd", "axes.titlecolor": "#eeeeee",
        "xtick.color": "#aaaaaa", "ytick.color": "#aaaaaa",
        "text.color": "#dddddd", "grid.color": "#333333",
        "grid.linestyle": "--", "grid.linewidth": 0.5,
        "legend.facecolor": "#2a2a2a", "legend.edgecolor": "#888888",
        "legend.labelcolor": "#dddddd", "font.size": 10,
    })

def plot_equity(curves: dict, path: Path, title: str):
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (name, s) in enumerate(curves.items()):
        ax.plot(s.index, s.values, color=MONO[i % len(MONO)],
                linewidth=1.0, label=name)
    ax.set_xlabel("Дата"); ax.set_ylabel("Equity (нач. = 1.0)")
    ax.set_title(title, color="#eeeeee")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120, facecolor="#1a1a1a"); plt.close()
    print(f"  график: {path}")

def plot_bar(labels, values, path, title, ylabel, colors=None):
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    if colors is None:
        colors = [MONO[i % len(MONO)] for i in range(len(labels))]
    ax.bar(range(len(labels)), values, color=colors, edgecolor="#444444")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title, color="#eeeeee")
    ax.axhline(0, color="#555555", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120, facecolor="#1a1a1a"); plt.close()
    print(f"  график: {path}")

# ============================================================
# HELPERS
# ============================================================
def section(t): return f"\n{'='*70}\n{t}\n{'='*70}"

class Report:
    def __init__(self): self.lines=[]
    def emit(self, s=""): print(s); self.lines.append(str(s))
    def save(self, p): p.write_text("\n".join(self.lines), encoding="utf-8"); print(f"\nОтчёт: {p}")

def load_split():
    return json.loads(SPLIT_JSON.read_text())["split_date"]

def load_prices(split_date):
    conn = sqlite3.connect(STOCKS_DB)
    df = pd.read_sql("SELECT ticker, date, close, volume FROM prices WHERE date < ?",
                     conn, params=(split_date,))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot(index="date", columns="ticker", values="close").sort_index()
    volume = df.pivot(index="date", columns="ticker", values="volume").sort_index()
    return close, volume

def load_imoex(split_date):
    conn = sqlite3.connect(INDEXES_DB)
    df = pd.read_sql("SELECT date, close FROM indexes WHERE ticker='IMOEX' AND date < ?",
                     conn, params=(split_date,))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()

def compute_rs(prices, imoex, horizon):
    fwd_ret = prices.shift(-horizon) / prices - 1
    imoex_fwd = imoex.shift(-horizon) / imoex - 1
    imoex_fwd = imoex_fwd.reindex(prices.index)
    return fwd_ret.sub(imoex_fwd, axis=0)

# ============================================================
# ФИЧИ (только те, что работают по EDA)
# ============================================================
def build_features(prices, volume):
    ret1 = prices.pct_change(1)
    feats = {}

    # vol_60 (главная фича)
    feats["vol_60"] = ret1.rolling(60).std()

    # price_position (52w)
    roll_max = prices.rolling(252, min_periods=100).max()
    roll_min = prices.rolling(252, min_periods=100).min()
    feats["price_position"] = (prices - roll_min) / (roll_max - roll_min)

    # kurt_60, skew_60
    feats["kurt_60"] = ret1.rolling(60).kurt()
    feats["skew_60"] = ret1.rolling(60).skew()

    # relative volume (новое)
    vol_ma20 = volume.rolling(20).mean()
    feats["rel_volume"] = volume / vol_ma20

    # amihud illiquidity (новое)
    feats["amihud"] = (ret1.abs() / volume.replace(0, np.nan)).rolling(20).mean()

    # dist from 52w high/low (новое)
    feats["dist_from_high"] = prices / roll_max - 1
    feats["dist_from_low"] = prices / roll_min - 1

    # rolling sharpe 60 (новое)
    feats["rolling_sharpe_60"] = ret1.rolling(60).mean() / ret1.rolling(60).std()

    return feats

# ============================================================
# ТЕСТ: ранжирование по фиче, топ-10, держим 20 дней
# ============================================================
def test_ranking(feature_df, rs, feature_name, top_n=TOP_N, horizon=HORIZON,
                 ascending=True, rep=None):
    """
    Каждые `horizon` дней:
      - ранжируем акции по feature (ascending=True: меньше = лучше),
      - берём top_n,
      - считаем средний RS этих top_n за следующие horizon дней.
    Возвращает:
      - mean_rs (средний RS по всем периодам),
      - hit_rate (доля периодов с RS > 0),
      - series (RS по периодам)
    """
    dates = rs.index
    period_rs = []
    period_dates = []

    for i in range(0, len(dates) - horizon, horizon):
        d = dates[i]
        # фича на дату d
        f = feature_df.loc[d].dropna() if d in feature_df.index else None
        if f is None or len(f) < top_n * 2:
            continue
        # RS на дату d (это будущая доходность за [d, d+h])
        r = rs.loc[d].dropna()
        common = f.index.intersection(r.index)
        if len(common) < top_n * 2:
            continue
        f = f[common]; r = r[common]

        # ранжируем
        if ascending:
            selected = f.nsmallest(top_n).index
        else:
            selected = f.nlargest(top_n).index

        # средний RS выбранных
        rs_sel = r[selected].mean()
        period_rs.append(rs_sel)
        period_dates.append(d)

    if not period_rs:
        return None

    series = pd.Series(period_rs, index=period_dates)
    return {
        "feature": feature_name,
        "mean_rs": series.mean(),
        "median_rs": series.median(),
        "std_rs": series.std(),
        "hit_rate": (series > 0).mean(),
        "n_periods": len(series),
        "series": series,
    }

def test_all_stocks(rs, rep):
    """Средний RS всех акций (случайный выбор) — бенчмарк."""
    dates = rs.index
    period_rs = []
    period_dates = []
    for i in range(0, len(dates) - HORIZON, HORIZON):
        d = dates[i]
        r = rs.loc[d].dropna()
        if len(r) < 20: continue
        period_rs.append(r.mean())
        period_dates.append(d)
    series = pd.Series(period_rs, index=period_dates)
    return {
        "feature": "ALL_STOCKS",
        "mean_rs": series.mean(),
        "median_rs": series.median(),
        "std_rs": series.std(),
        "hit_rate": (series > 0).mean(),
        "n_periods": len(series),
        "series": series,
    }

# ============================================================
# MAIN
# ============================================================
def main():
    rep = Report()
    rep.emit("# Тест сигнала: работают ли фичи для отбора топ-10?\n")
    rep.emit(f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n")

    split = load_split()
    rep.emit(f"split_date = {split}")
    rep.emit(f"Горизонт: {HORIZON} дней, TOP_N = {TOP_N}")
    rep.emit(f"VAL не трогаем\n")

    prices, volume = load_prices(split)
    imoex = load_imoex(split)
    rep.emit(f"prices: {prices.shape[0]} дат × {prices.shape[1]} акций")
    rep.emit(f"IS: {prices.index.min().date()} → {prices.index.max().date()}\n")

    # --- RS ---
    rep.emit(section("ПОСТРОЕНИЕ RS И ФИЧ"))
    rs = compute_rs(prices, imoex, HORIZON)
    rep.emit(f"RS: {rs.dropna(how='all').shape[0]} дат с данными")

    feats = build_features(prices, volume)
    rep.emit(f"Построено фич: {len(feats)}")
    for name in feats:
        rep.emit(f"  {name}")

    # --- бенчмарк: все акции ---
    rep.emit(section("БЕНЧМАРК: СЛУЧАЙНЫЙ ВЫБОР (все акции)"))
    base = test_all_stocks(rs, rep)
    if base:
        rep.emit(f"  mean_rs   = {base['mean_rs']:+.6f}  (это 'случайный выбор')")
        rep.emit(f"  median_rs = {base['median_rs']:+.6f}")
        rep.emit(f"  hit_rate  = {base['hit_rate']:.2%}  (доля периодов с RS > 0)")
        rep.emit(f"  n_periods = {base['n_periods']}")

    # --- тест каждой фичи ---
    rep.emit(section("ТЕСТ ФИЧ (топ-10 по фиче, держим 20 дней)"))

    results = []
    # правила: для каждой фичи — направление (ascending = меньше лучше)
    rules = [
        ("vol_60", True, "низкая волатильность"),
        ("price_position", False, "высокое положение в диапазоне"),
        ("kurt_60", True, "низкий эксцесс"),
        ("skew_60", True, "низкая асимметрия"),
        ("rel_volume", False, "высокий относительный объём"),
        ("amihud", True, "низкая неликвидность"),
        ("dist_from_high", False, "близко к 52w high"),
        ("dist_from_low", False, "далеко от 52w low"),
        ("rolling_sharpe_60", False, "высокий rolling Sharpe"),
    ]

    for fname, ascending, desc in rules:
        if fname not in feats: continue
        r = test_ranking(feats[fname], rs, fname, ascending=ascending)
        if r is None: continue
        r["desc"] = desc
        r["ascending"] = ascending
        results.append(r)
        excess = r["mean_rs"] - base["mean_rs"] if base else np.nan
        rep.emit(f"\n  [{fname}] {desc}")
        rep.emit(f"    mean_rs   = {r['mean_rs']:+.6f}")
        rep.emit(f"    median_rs = {r['median_rs']:+.6f}")
        rep.emit(f"    hit_rate  = {r['hit_rate']:.2%}")
        rep.emit(f"    excess vs ALL = {excess:+.6f}")
        rep.emit(f"    n_periods = {r['n_periods']}")

    # --- сводная таблица ---
    rep.emit(section("СВОДКА: СИГНАЛ ЕСТЬ?"))
    rep.emit(f"\n  {'фича':<20} {'mean_rs':>10} {'hit_rate':>10} {'excess':>10} {'вердикт':>10}")
    rep.emit("  " + "-" * 65)
    for r in sorted(results, key=lambda x: -x["mean_rs"]):
        excess = r["mean_rs"] - base["mean_rs"] if base else 0
        if r["mean_rs"] > 0 and excess > 0.002 and r["hit_rate"] > 0.52:
            verdict = "СИГНАЛ"
        elif r["mean_rs"] > 0 and excess > 0:
            verdict = "слабый"
        else:
            verdict = "нет"
        rep.emit(f"  {r['feature']:<20} {r['mean_rs']:>+10.6f} "
                 f"{r['hit_rate']:>10.2%} {excess:>+10.6f} {verdict:>10}")

    # --- equity curves ---
    rep.emit(section("EQUITY CURVES"))
    curves = {}
    if base:
        curves["ALL_STOCKS"] = (1 + base["series"]).cumprod()
    for r in results:
        if r["mean_rs"] > 0 and (r["mean_rs"] - base["mean_rs"]) > 0.001:
            curves[r["feature"]] = (1 + r["series"]).cumprod()

    if curves:
        plot_equity(curves, PLOTS_DIR / "equity_curves.png",
                    "Equity: топ-10 по фичам vs все акции (IS)")
        # метрики
        rep.emit(f"\n  {'кривая':<20} {'total_ret':>12} {'n_periods':>10}")
        rep.emit("  " + "-" * 45)
        for name, s in curves.items():
            total = s.iloc[-1] - 1
            rep.emit(f"  {name:<20} {total:>+12.4f} {len(s):>10}")
    else:
        rep.emit("  нет фич с положительным excess")

    # --- сохраняем ---
    rep.emit(section("СОХРАНЕНИЕ"))
    rows = []
    if base:
        rows.append({
            "feature": "ALL_STOCKS", "desc": "случайный выбор",
            "mean_rs": base["mean_rs"], "median_rs": base["median_rs"],
            "std_rs": base["std_rs"], "hit_rate": base["hit_rate"],
            "n_periods": base["n_periods"],
            "excess_vs_all": 0.0,
        })
    for r in results:
        rows.append({
            "feature": r["feature"], "desc": r["desc"],
            "mean_rs": r["mean_rs"], "median_rs": r["median_rs"],
            "std_rs": r["std_rs"], "hit_rate": r["hit_rate"],
            "n_periods": r["n_periods"],
            "excess_vs_all": r["mean_rs"] - base["mean_rs"] if base else np.nan,
        })
    df_out = pd.DataFrame(rows)
    with sqlite3.connect(OUT_SQL) as c:
        df_out.to_sql("signal_summary", c, if_exists="replace", index=False)

    rep.emit(f"  SQL: {OUT_SQL}")
    rep.emit(f"  Report: {OUT_MD}")
    rep.emit(f"  Plots: {PLOTS_DIR}")

    rep.emit(section("ИТОГ"))
    rep.emit("Смотри таблицу выше. СИГНАЛ = mean_rs > 0, excess > 0.002, hit_rate > 0.52.")
    rep.emit("Если ни одна фича не даёт СИГНАЛ — отбор по одной фиче не работает,")
    rep.emit("нужна комбинация фич или другая логика.")

    rep.save(OUT_MD)


if __name__ == "__main__":
    main()