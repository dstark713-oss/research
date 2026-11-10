# backtests/plot_isolated_wf.py
# ============================================================
# ИЗОЛИРОВАННЫЙ прогон WF + отрисовка.
#
# 1. Загружает IS-данные, строит 11 фич (rs_pool).
# 2. WF: 6 фолдов, anchored expanding, purge=21.
# 3. Для каждого фолда: train CatBoostRegressor (iter=300),
#    на val — ребаланс раз в 20 дней:
#      - на дату rd выбираем топ-10 по prediction,
#      - портфельный RS = средний фактический RS этих 10 за [rd, rd+20],
#      - equity *= (1 + port_rs).
# 4. Строит единую equity-кривую по всем фолдам (только даты ребаланса).
# 5. Грузит бенчмарки (IMOEX, MCFTR, RGBITR, MIX) из SQL / fallback.
# 6. Нормирует всё от 1.0 на общей дате старта.
# 7. Рисует в тёмной монохромной стилистике.
#
# Запуск: python -m backtests.plot_isolated_wf
# ============================================================
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from rs_model.config import FEATURES, HORIZON, PURGE, BASE_PARAMS
from rs_model.data import build_dataset, to_long
from rs_model.wf import make_folds

ROOT = Path(__file__).parent.parent
BENCH_DB = ROOT / "benchmark" / "sql" / "BENCHMARKS.sqlite"
MCFTR_DB = ROOT / "benchmark" / "sql" / "MCFTR.sqlite"
RGBITR_DB = ROOT / "benchmark" / "sql" / "RGBITR.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"

OUT_PNG = Path(__file__).parent / "isolated_wf_vs_benchmark.png"
OUT_CSV = Path(__file__).parent / "isolated_wf_equity.csv"

MONO = ["#ffffff", "#cccccc", "#999999", "#666666", "#bbbbbb", "#777777"]

# параметры модели (фиксированные, по результатам сравнения)
MODEL_PARAMS = {
    "depth": 3,
    "learning_rate": 0.03,
    "iterations": 300,
    "l2_leaf_reg": 50,
    "random_strength": 3,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "border_count": 64,
    "rsm": 0.8,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}

REBAL_STEP = 20


# ============================================================
# ГРАФИК
# ============================================================
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


# ============================================================
# БЕНЧМАРКИ
# ============================================================
def load_benchmarks():
    out = {}

    # BENCHMARKS.sqlite equity_curves
    if BENCH_DB.exists():
        conn = sqlite3.connect(BENCH_DB)
        try:
            tabs = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            if "equity_curves" in tabs:
                eq = pd.read_sql("SELECT * FROM equity_curves", conn)
                date_col = next((c for c in ["TRADEDATE", "date"] if c in eq.columns), None)
                if date_col:
                    eq[date_col] = pd.to_datetime(eq[date_col])
                    eq = eq.set_index(date_col).sort_index()
                    for col in ["MCFTR_100", "RGBITR_100", "MIX_50_50"]:
                        if col in eq.columns:
                            s = eq[col].dropna()
                            if len(s) > 0:
                                out[col] = s
        except Exception as e:
            print(f"[bench] WARN: {e}")
        conn.close()

    # IMOEX
    if INDEXES_DB.exists():
        conn = sqlite3.connect(INDEXES_DB)
        try:
            imoex = pd.read_sql(
                "SELECT date, close FROM indexes WHERE ticker='IMOEX'", conn)
            imoex["date"] = pd.to_datetime(imoex["date"])
            s = imoex.set_index("date")["close"].sort_index()
            if len(s) > 0:
                out["IMOEX"] = s
        except Exception as e:
            print(f"[bench] WARN IMOEX: {e}")
        conn.close()

    # fallback MCFTR, RGBITR
    def _load_price_db(path, table):
        if not path.exists(): return None
        conn = sqlite3.connect(path)
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            cols = list(df.columns)
            dc = next((c for c in ["TRADEDATE", "date"] if c in cols), None)
            cc = next((c for c in ["CLOSE", "close"] if c in cols), None)
            if not dc or not cc: return None
            df[dc] = pd.to_datetime(df[dc])
            return df.set_index(dc)[cc].sort_index()
        except Exception:
            return None
        finally:
            conn.close()

    if "MCFTR_100" not in out:
        m = _load_price_db(MCFTR_DB, "MCFTR")
        if m is not None: out["MCFTR_100"] = m
    if "RGBITR_100" not in out:
        r = _load_price_db(RGBITR_DB, "RGBITR")
        if r is not None: out["RGBITR_100"] = r

    return out


# ============================================================
# WF + ребаланс раз в 20 дней
# ============================================================
def run_wf_isolated():
    split_date = json.loads(
        (Path(__file__).parent / "split_info.json").read_text()
    )["split_date"]
    print(f"[wf-iso] split_date = {split_date}")

    print("[wf-iso] загрузка данных и фич...")
    features_raw, features_rank, target_rs, target_gauss, target_grades = build_dataset(split_date)

    dates = target_rs.dropna(how="all").index
    folds = make_folds(dates, n_folds=6, purge=PURGE)
    print(f"[wf-iso] фолдов: {len(folds)}")

    # для каждого фолда — обучаем, на val делаем ребаланс раз в 20 дней
    records = []  # date, fold, port_rs

    for i, (tr_dates, va_dates) in enumerate(folds):
        print(f"\n[wf-iso] fold {i+1}: "
              f"train {tr_dates[0].date()} → {tr_dates[-1].date()} | "
              f"val {va_dates[0].date()} → {va_dates[-1].date()}")

        # --- train ---
        tr_long = to_long(features_rank, target_gauss)
        tr_long = tr_long[tr_long["date"].isin(tr_dates)].reset_index(drop=True)
        X_tr = tr_long[FEATURES].values
        y_tr = tr_long["target"].values

        # --- val: предсказания на каждую дату валидации ---
        va_long = to_long(features_rank, target_gauss)
        va_long = va_long[va_long["date"].isin(va_dates)].reset_index(drop=True)
        X_va = va_long[FEATURES].values

        model = CatBoostRegressor(**MODEL_PARAMS)
        model.fit(X_tr, y_tr, verbose=False)

        va_long["pred"] = model.predict(X_va)
        pred_wide = va_long.pivot(index="date", columns="ticker", values="pred")

        # --- ребаланс раз в 20 дней ---
        rebal_dates = va_dates[::REBAL_STEP]
        for rd in rebal_dates:
            if rd not in pred_wide.index:
                continue
            pred_row = pred_wide.loc[rd].dropna()
            if len(pred_row) < 10:
                continue
            top10 = pred_row.nlargest(10).index.tolist()

            # фактический RS этих 10 за [rd, rd+20]
            if rd not in target_rs.index:
                continue
            rs_row = target_rs.loc[rd, top10].dropna()
            if len(rs_row) < 10:
                continue
            port_rs = rs_row.mean()
            records.append({"date": rd, "fold": i + 1, "port_rs": port_rs})

        print(f"  ребалансов: {sum(1 for r in records if r['fold'] == i+1)}")

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df["equity"] = (1 + df["port_rs"]).cumprod()
    return df


# ============================================================
# MAIN
# ============================================================
def main():
    setup_dark_style()

    eq_df = run_wf_isolated()

    print(f"\n[wf-iso] ребалансов всего: {len(eq_df)}")
    print(f"[wf-iso] период: {eq_df['date'].min().date()} → {eq_df['date'].max().date()}")
    print(f"[wf-iso] equity start={eq_df['equity'].iloc[0]:.4f}, "
          f"end={eq_df['equity'].iloc[-1]:.4f}")
    print(f"[wf-iso] total return: {(eq_df['equity'].iloc[-1]-1)*100:.1f}%")
    # CAGR: (end/start)^(1/years) - 1
    years = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days / 365.25
    if years > 0:
        cagr = eq_df["equity"].iloc[-1] ** (1/years) - 1
        print(f"[wf-iso] CAGR: {cagr*100:.2f}%")

    # сохраняем CSV
    eq_df.to_csv(OUT_CSV, index=False)
    print(f"[wf-iso] equity CSV: {OUT_CSV}")

    # --- бенчмарки ---
    print("\n[bench] загрузка...")
    bench = load_benchmarks()
    print(f"[bench] loaded: {list(bench.keys())}")

    # --- график ---
    d_min = eq_df["date"].min()
    d_max = eq_df["date"].max()

    fig, ax = plt.subplots(figsize=(15, 8))

    # equity регрессора — ступеньки (ребаланс раз в 20 дней)
    ax.plot(eq_df["date"], eq_df["equity"], color=MONO[0],
            linewidth=2.0, label="Regressor top-10 (rebal 20d)",
            zorder=10, drawstyle="steps-post")

    # бенчмарки — нормируем от 1.0 на d_min
    colors = {
        "IMOEX": MONO[1],
        "MCFTR_100": MONO[2],
        "RGBITR_100": MONO[3],
        "MIX_50_50": MONO[4],
    }
    for name, s in bench.items():
        if name not in colors: continue
        s = s.sort_index()
        s_at_start = s[s.index >= d_min]
        if len(s_at_start) == 0:
            print(f"[bench] WARN: {name} нет данных после {d_min.date()}")
            continue
        base_val = s_at_start.iloc[0]
        s_norm = s / base_val
        s_norm = s_norm.loc[(s_norm.index >= d_min) & (s_norm.index <= d_max)]
        if len(s_norm) < 2:
            print(f"[bench] WARN: {name} мало точек")
            continue
        ax.plot(s_norm.index, s_norm.values, color=colors[name],
                linewidth=1.3, label=name, alpha=0.9)
        print(f"[bench] {name}: {len(s_norm)} точек, "
              f"start={s_norm.iloc[0]:.4f}, end={s_norm.iloc[-1]:.4f}")

    ax.set_xlabel("Дата")
    ax.set_ylabel("Equity (start = 1.0, log scale)")
    ax.set_title("Regressor top-10 (rebal 20d) vs бенчмарки — WF IS folds 1–6",
                 color="#eeeeee", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{y:.1f}" if y < 10 else f"{y:.0f}"
    ))
    ax.axhline(1.0, color="#555555", linewidth=0.5, linestyle=":")

    # разделители фолдов
    if "fold" in eq_df.columns:
        fold_dates = eq_df.groupby("fold")["date"].min()
        y_min, y_max = ax.get_ylim()
        for f, d in fold_dates.items():
            if f == 1: continue
            ax.axvline(d, color="#444444", linewidth=0.5, linestyle="--")
            ax.text(d, y_max * 0.95, f"F{int(f)}",
                    color="#888888", fontsize=8, ha="left", va="top")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=140, facecolor="#1a1a1a")
    plt.close()
    print(f"\n[wf-iso] график: {OUT_PNG}")


if __name__ == "__main__":
    main()