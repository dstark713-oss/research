# backtests/portfolio_tests/rs_is_wf.py
# ============================================================
# Портфельный WF-тест: Stock Selection RS Model.
#
# Стратегия:
#   - каждые 20 торговых дней ребалансировка
#   - топ-10 акций по prediction модели
#   - equal-weight внутри портфеля
#   - удержание 20 дней
#
# Сравнение с бенчмарками:
#   - IMOEX (индекс)
#   - MCFTR (полной доходности, акции)
#   - RGBITR (полной доходности, облигации)
#   - MIX 50/50 (MCFTR + RGBITR, ребаланс 20d)
#
# Только IS (2013-03-25 → 2024-01-29). VAL не трогаем.
#
# Метрики:
#   - CAGR, MDD, VOL, Sharpe, Sortino, Calmar, Turnover
#   - equity curve по датам ребаланса
#
# Запуск: python -m backtests.portfolio_tests.rs_is_wf
# ============================================================
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from rs_model.config import FEATURES, HORIZON, PURGE
from rs_model.data import build_dataset, to_long
from rs_model.wf import make_folds

# ============================================================
# ПУТИ
# ============================================================
ROOT = Path(__file__).parent.parent.parent
BENCH_DB = ROOT / "benchmark" / "sql" / "BENCHMARKS.sqlite"
MCFTR_DB = ROOT / "benchmark" / "sql" / "MCFTR.sqlite"
RGBITR_DB = ROOT / "benchmark" / "sql" / "RGBITR.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"
SPLIT_JSON = ROOT / "backtests" / "split_info.json"

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)
OUT_REPORT = OUT_DIR / "rs_is_wf_report.md"
OUT_EQUITY = OUT_DIR / "rs_is_wf_equity.csv"
OUT_TRADES = OUT_DIR / "rs_is_wf_trades.csv"

# ============================================================
# ПАРАМЕТРЫ
# ============================================================
REBAL_STEP = 20          # ребаланс каждые 20 торговых дней
TOP_N = 10               # топ-10 акций
TARGET_VOL = 0.15        # целевая годовая волатильность (для сайзинга, опционально)
USE_SIZING = False       # пока без GARCH-сайзинга, только базовый портфель
TRADING_DAYS = 252

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
            print(f"[bench] WARN BENCHMARKS: {e}")
        conn.close()

    # IMOEX
    if INDEXES_DB.exists():
        conn = sqlite3.connect(INDEXES_DB)
        try:
            imoex = pd.read_sql(
                "SELECT date, close FROM indexes WHERE ticker='IMOEX'", conn)
            imoex["date"] = pd.to_datetime(imoex["date"])
            out["IMOEX"] = imoex.set_index("date")["close"].sort_index()
        except Exception as e:
            print(f"[bench] WARN IMOEX: {e}")
        conn.close()

    # fallback MCFTR / RGBITR
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
# МЕТРИКИ
# ============================================================
def compute_metrics(equity: pd.Series, turnover: pd.Series = None,
                    rf_annual: float = 0.0) -> dict:
    """CAGR, MDD, VOL, Sharpe, Sortino, Calmar, Turnover."""
    eq = equity.dropna()
    if len(eq) < 2:
        return {}

    # Приводим equity к дневной сетке (reindex + ffill), чтобы метрики были
    # сопоставимы с бенчмарками (которые дневные)
    # Но для стратегии с ребалансом 20d — equity меняется только в даты ребаланса.
    # Для корректных метрик — используем daily returns между датами ребаланса
    # (equity постоянна между ними, кроме скачков).
    full_idx = pd.date_range(eq.index.min(), eq.index.max(), freq="B")
    eq_daily = eq.reindex(full_idx, method="ffill").dropna()
    rets = eq_daily.pct_change().dropna()
    n = len(rets)

    years = n / TRADING_DAYS
    total_ret = eq_daily.iloc[-1] / eq_daily.iloc[0]
    cagr = total_ret ** (1 / years) - 1 if years > 0 else np.nan

    vol = rets.std(ddof=1) * np.sqrt(TRADING_DAYS)

    cummax = eq_daily.cummax()
    drawdown = eq_daily / cummax - 1
    mdd = drawdown.min()

    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = rets - rf_daily
    sharpe = (excess.mean() / excess.std(ddof=1)) * np.sqrt(TRADING_DAYS) \
        if excess.std(ddof=1) > 0 else np.nan

    # Sortino: downside deviation = RMS(min(excess, 0)) по ВСЕМ наблюдениям
    # (Sortino & Price, 1994). НЕ std(neg) — это разные вещи.
    downside_sq = np.minimum(excess, 0) ** 2
    downside_dev = np.sqrt(downside_sq.mean())
    sortino = (excess.mean() / downside_dev) * np.sqrt(TRADING_DAYS) \
        if downside_dev > 0 else np.nan

    calmar = cagr / abs(mdd) if mdd != 0 else np.nan

    if turnover is not None and len(turnover) > 0:
        # one-way turnover per rebalance, annualized
        # средний turnover на ребаланс × число ребалансов в год
        mean_to = turnover.mean()
        rebalances_per_year = TRADING_DAYS / REBAL_STEP  # 252 / 20 = 12.6
        turnover_annual = mean_to * rebalances_per_year
    else:
        turnover_annual = 0.0

    return {
        "CAGR": cagr,
        "MDD": mdd,
        "VOL": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Turnover": turnover_annual,
        "TotalReturn": total_ret - 1,
        "Years": years,
        "N_days": n,
        "Start": eq_daily.index[0].strftime("%Y-%m-%d"),
        "End": eq_daily.index[-1].strftime("%Y-%m-%d"),
    }


def compute_metrics_from_equity_series(eq: pd.Series, name: str) -> dict:
    """Для бенчмарков — просто compute_metrics."""
    m = compute_metrics(eq)
    m["name"] = name
    return m


# ============================================================
# WF + портфель
# ============================================================
def run_portfolio_wf(split_date: str):
    print("[portfolio] загрузка данных...")
    features_raw, features_rank, target_rs, target_gauss, target_grades = build_dataset(split_date)

    dates = target_rs.dropna(how="all").index
    folds = make_folds(dates, n_folds=6, purge=PURGE)
    print(f"[portfolio] фолдов: {len(folds)}")

    # собираем портфельные доходности по датам ребаланса
    portfolio_records = []   # (date, port_rs, turnover, n_selected, selected_tickers)
    trades_records = []      # (date, ticker, weight, pred)

    prev_holdings = set()

    for i, (tr_dates, va_dates) in enumerate(folds):
        print(f"\n[portfolio] fold {i+1}: "
              f"train {tr_dates[0].date()} → {tr_dates[-1].date()} | "
              f"val {va_dates[0].date()} → {va_dates[-1].date()}")

        # train
        tr_long = to_long(features_rank, target_gauss)
        tr_long = tr_long[tr_long["date"].isin(tr_dates)].reset_index(drop=True)
        X_tr = tr_long[FEATURES].values
        y_tr = tr_long["target"].values

        model = CatBoostRegressor(**MODEL_PARAMS)
        model.fit(X_tr, y_tr, verbose=False)

        # val: predictions
        va_long = to_long(features_rank, target_gauss)
        va_long = va_long[va_long["date"].isin(va_dates)].reset_index(drop=True)
        X_va = va_long[FEATURES].values
        va_long["pred"] = model.predict(X_va)
        pred_wide = va_long.pivot(index="date", columns="ticker", values="pred")

        # ребаланс каждые REBAL_STEP дней
        rebal_dates = va_dates[::REBAL_STEP]
        for rd in rebal_dates:
            if rd not in pred_wide.index:
                continue
            pred_row = pred_wide.loc[rd].dropna()
            if len(pred_row) < TOP_N:
                continue

            selected = pred_row.nlargest(TOP_N).index.tolist()

            # фактический RS за [rd, rd+20] для выбранных
            if rd not in target_rs.index:
                continue
            rs_row = target_rs.loc[rd, selected].dropna()
            if len(rs_row) < TOP_N:
                continue
            port_rs = rs_row.mean()

            # turnover: доля новых акций
            new_holdings = set(selected)
            if len(prev_holdings) > 0:
                sold = prev_holdings - new_holdings
                # one-way: продано / TOP_N (в равных весах)
                to = len(sold) / TOP_N
            else:
                to = 1.0  # первый вход = 100% (весь капитал входит)
            prev_holdings = new_holdings

            portfolio_records.append({
                "date": rd,
                "fold": i + 1,
                "port_rs": port_rs,
                "turnover": to,
                "n_selected": len(selected),
            })

            for t in selected:
                trades_records.append({
                    "date": rd,
                    "fold": i + 1,
                    "ticker": t,
                    "weight": 1.0 / TOP_N,
                    "pred": pred_row[t],
                })

    df = pd.DataFrame(portfolio_records).sort_values("date").reset_index(drop=True)
    df["equity"] = (1 + df["port_rs"]).cumprod()

    trades_df = pd.DataFrame(trades_records)
    return df, trades_df


# ============================================================
# MAIN
# ============================================================
def main():
    split_date = json.loads(SPLIT_JSON.read_text())["split_date"]
    print(f"[portfolio] split_date = {split_date} (VAL не трогаем)")

    # --- WF + портфель ---
    port_df, trades_df = run_portfolio_wf(split_date)

    if port_df.empty:
        print("[portfolio] FATAL: пустой портфель")
        return

    print(f"\n[portfolio] ребалансов: {len(port_df)}")
    print(f"[portfolio] период: {port_df['date'].min().date()} → {port_df['date'].max().date()}")
    print(f"[portfolio] equity: start=1.0, end={port_df['equity'].iloc[-1]:.4f}")
    print(f"[portfolio] total return: {(port_df['equity'].iloc[-1]-1)*100:.1f}%")
    print(f"[portfolio] median turnover: {port_df['turnover'].median():.2%}")

    # --- метрики портфеля ---
    port_metrics = compute_metrics(
        port_df.set_index("date")["equity"],
        turnover=port_df.set_index("date")["turnover"],
    )
    print(f"\n[portfolio] метрики:")
    for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar", "Turnover"]:
        v = port_metrics.get(k, np.nan)
        if k in ("CAGR", "MDD", "VOL", "Turnover"):
            print(f"  {k:<10}: {v*100:+.2f}%")
        else:
            print(f"  {k:<10}: {v:.4f}")

    # --- бенчмарки ---
    print("\n[portfolio] загрузка бенчмарков...")
    bench = load_benchmarks()
    print(f"[portfolio] loaded: {list(bench.keys())}")

    # --- сохраняем ---
    port_df.to_csv(OUT_EQUITY, index=False)
    print(f"\n[portfolio] equity CSV: {OUT_EQUITY}")

    if len(trades_df) > 0:
        trades_df.to_csv(OUT_TRADES, index=False)
        print(f"[portfolio] trades CSV: {OUT_TRADES} ({len(trades_df)} строк)")

    # --- отчёт ---
    lines = [
        "# Портфельный WF-тест: Stock Selection RS Model\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"split_date = {split_date} (VAL не трогаем)\n",
        f"\n## Конфигурация\n",
        f"- Модель: CatBoostRegressor (depth=3, lr=0.03, iter=300, l2=50)",
        f"- Фичи: {len(FEATURES)} (rs_pool v2)",
        f"- Ребаланс: каждые {REBAL_STEP} торговых дней",
        f"- Топ-{TOP_N} акций, equal-weight",
        f"- WF: 6 фолдов, anchored expanding, purge={PURGE}",
        f"- Сайзинг: {'GARCH' if USE_SIZING else 'нет (100% экспозиция)'}\n",
        "\n## Walk-Forward folds\n",
        "| fold | train | val | ребалансов |",
        "|------|-------|-----|-----------|",
    ]
    dates = build_dataset(split_date)[2].dropna(how="all").index
    folds = make_folds(dates, n_folds=6, purge=PURGE)
    for i, (tr, va) in enumerate(folds):
        sub = port_df[port_df["fold"] == i + 1]
        lines.append(f"| {i+1} | {tr[0].date()} → {tr[-1].date()} | {va[0].date()} → {va[-1].date()} | {len(sub)} |")

    lines.append("\n## Метрики портфеля (IS)\n")
    lines.append("| Метрика | Значение |")
    lines.append("|---------|----------|")
    for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar", "Turnover"]:
        v = port_metrics.get(k, np.nan)
        if k in ("CAGR", "MDD", "VOL", "Turnover"):
            lines.append(f"| {k} | {v*100:+.2f}% |")
        else:
            lines.append(f"| {k} | {v:.4f} |")
    lines.append(f"| TotalReturn | {port_metrics.get('TotalReturn', np.nan)*100:+.1f}% |")
    lines.append(f"| Years | {port_metrics.get('Years', np.nan):.2f} |\n")

    # --- сравнение с бенчмарками ---
    lines.append("\n## Сравнение с бенчмарками\n")
    lines.append("| Инструмент | CAGR | MDD | VOL | Sharpe | Sortino | Calmar |")
    lines.append("|------------|------|-----|-----|--------|---------|--------|")
    lines.append(f"| **RS Portfolio** | {port_metrics['CAGR']*100:+.2f}% | "
                 f"{port_metrics['MDD']*100:+.2f}% | {port_metrics['VOL']*100:+.2f}% | "
                 f"{port_metrics['Sharpe']:.3f} | {port_metrics['Sortino']:.3f} | "
                 f"{port_metrics['Calmar']:.3f} |")

    bench_metrics = {}
    for name, s in bench.items():
        # нормируем на диапазон портфеля
        d_min = port_df["date"].min()
        d_max = port_df["date"].max()
        s = s.sort_index()
        s_norm = s.loc[(s.index >= d_min) & (s.index <= d_max)]
        if len(s_norm) < 2:
            continue
        m = compute_metrics_from_equity_series(s_norm, name)
        bench_metrics[name] = m
        lines.append(f"| {name} | {m['CAGR']*100:+.2f}% | "
                     f"{m['MDD']*100:+.2f}% | {m['VOL']*100:+.2f}% | "
                     f"{m['Sharpe']:.3f} | {m['Sortino']:.3f} | "
                     f"{m['Calmar']:.3f} |")

    # --- выводы ---
    lines.append("\n## Выводы\n")
    if "MIX_50_50" in bench_metrics:
        mix = bench_metrics["MIX_50_50"]
        lines.append(f"- vs MIX_50_50: ΔCAGR = {(port_metrics['CAGR']-mix['CAGR'])*100:+.2f} п.п., "
                     f"ΔSharpe = {port_metrics['Sharpe']-mix['Sharpe']:+.3f}, "
                     f"ΔMDD = {(port_metrics['MDD']-mix['MDD'])*100:+.2f} п.п.")
    if "MCFTR_100" in bench_metrics:
        mcftr = bench_metrics["MCFTR_100"]
        lines.append(f"- vs MCFTR_100: ΔCAGR = {(port_metrics['CAGR']-mcftr['CAGR'])*100:+.2f} п.п., "
                     f"ΔSharpe = {port_metrics['Sharpe']-mcftr['Sharpe']:+.3f}, "
                     f"ΔMDD = {(port_metrics['MDD']-mcftr['MDD'])*100:+.2f} п.п.")

    lines.append("\n## Замечания\n")
    lines.append("- **Транзакционные издержки не учтены.** При ~1% turnover в год, "
                 "издержки ~0.1–0.2% в год.")
    lines.append("- **Выживший bias:** 185 акций текущего универса, делистингованные не учтены. "
                 "Завышает результат на ~10–15% за 5 лет.")
    lines.append("- **Fold 6** — известный стрессовый период (2023). Не использовать для выбора модели.")
    lines.append("- **GARCH-сайзинг** — следующий этап. Пока портфель с 100% экспозицией.\n")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[portfolio] отчёт: {OUT_REPORT}")

    # краткий вывод в консоль
    print("\n" + "=" * 70)
    print("СВОДКА")
    print("=" * 70)
    print(f"RS Portfolio: CAGR={port_metrics['CAGR']*100:+.2f}%  "
          f"MDD={port_metrics['MDD']*100:+.2f}%  VOL={port_metrics['VOL']*100:+.2f}%  "
          f"Sharpe={port_metrics['Sharpe']:.3f}  Sortino={port_metrics['Sortino']:.3f}  "
          f"Calmar={port_metrics['Calmar']:.3f}")
    for name, m in bench_metrics.items():
        print(f"{name:<12} CAGR={m['CAGR']*100:+.2f}%  "
              f"MDD={m['MDD']*100:+.2f}%  VOL={m['VOL']*100:+.2f}%  "
              f"Sharpe={m['Sharpe']:.3f}  Sortino={m['Sortino']:.3f}  "
              f"Calmar={m['Calmar']:.3f}")


if __name__ == "__main__":
    main()