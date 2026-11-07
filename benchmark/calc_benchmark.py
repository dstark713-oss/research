# research/benchmark/calc_benchmark.py
# ============================================================
# Расчёт трёх бенчмарков + метрики риск-профиля.
#
#   MCFTR_100   — 100% MCFTR
#   RGBITR_100  — 100% RGBITR
#   MIX_50_50   — 50% MCFTR + 50% RGBITR, ребаланс раз в 20 торговых дней
#
# Метрики: CAGR, MDD, VOL, Sharpe, Sortino, Calmar, Turnover.
# Результат: sql/BENCHMARKS.sqlite (таблица metrics + equity curves)
# ============================================================
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "sql"
OUTPUT_DB = DATA_DIR / "BENCHMARKS.sqlite"

RF_ANNUAL = 0.0          # безрисковая ставка (годовая). Поставь свою, если надо.
TRADING_DAYS = 252
REBALANCE_EVERY = 20     # торговых дней для микса
WEIGHT_MCFTR = 0.5
WEIGHT_RGBITR = 0.5


# ============================================================
# LOAD
# ============================================================

def load_index(db_path: Path, table: str) -> pd.Series:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            f"SELECT TRADEDATE, CLOSE FROM {table} ORDER BY TRADEDATE",
            conn,
            parse_dates=["TRADEDATE"],
        )
    return df.set_index("TRADEDATE")["CLOSE"].astype(float)


# ============================================================
# EQUITY CURVES
# ============================================================

def build_single_asset_curve(price: pd.Series) -> pd.Series:
    """Equity curve для 100% актива (нормировано на 1.0)."""
    return (price / price.iloc[0]).rename("equity")


def build_mix_curve(mcftr: pd.Series, rgbitr: pd.Series,
                    every: int = REBALANCE_EVERY,
                    w_m: float = WEIGHT_MCFTR,
                    w_r: float = WEIGHT_RGBITR) -> pd.Series:
    """
    Микс с ребалансом раз в `every` торговых дней.

    Логика:
      - стартуем с весами w_m / w_r;
      - между ребалансами веса дрейфуют вместе с доходностями;
      - на ребалансе возвращаем к целевым весам (продаём/покупаем);
      - turnover считаем как сумму |Δweights| на каждом ребалансе.
    Возвращает equity Series и (опционально) turnover Series.
    """
    df = pd.DataFrame({"MCFTR": mcftr, "RGBITR": rgbitr}).dropna().sort_index()
    dates = df.index
    n = len(df)

    equity = np.empty(n)
    equity[0] = 1.0

    w = np.array([w_m, w_r], dtype=float)
    prices = df.values  # shape (n, 2)

    turnover = np.zeros(n)  # turnover на каждой дате (0 если не ребаланс)

    for i in range(1, n):
        # дрейф весов по доходностям
        ret = prices[i] / prices[i - 1]
        w_drift = w * ret
        port_ret = w_drift.sum() / w.sum()  # нормируем (сумма весов остаётся 1)
        w = w_drift / w_drift.sum()

        equity[i] = equity[i - 1] * port_ret

        # ребаланс каждые `every` дней (кроме последнего, чтобы не считать лишний turnover)
        if i % every == 0 and i != n - 1:
            target = np.array([w_m, w_r], dtype=float)
            turnover[i] = np.abs(target - w).sum() / 2.0  # односторонний turnover
            w = target

    eq = pd.Series(equity, index=dates, name="equity")
    to = pd.Series(turnover, index=dates, name="turnover")
    return eq, to


# ============================================================
# METRICS
# ============================================================

def compute_metrics(equity: pd.Series,
                    turnover: pd.Series | None = None,
                    rf_annual: float = RF_ANNUAL) -> dict:
    """
    Считает метрики по equity curve (дневной шаг).
    """
    eq = equity.dropna()
    if len(eq) < 2:
        return {}

    rets = eq.pct_change().dropna()
    n = len(rets)

    # --- CAGR ---
    years = n / TRADING_DAYS
    total_ret = eq.iloc[-1] / eq.iloc[0]
    cagr = total_ret ** (1 / years) - 1 if years > 0 else np.nan

    # --- VOL (annualized) ---
    vol = rets.std(ddof=1) * np.sqrt(TRADING_DAYS)

    # --- MDD ---
    cummax = eq.cummax()
    drawdown = eq / cummax - 1
    mdd = drawdown.min()  # отрицательное число

    # --- Sharpe ---
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = rets - rf_daily
    sharpe = (excess.mean() / excess.std(ddof=1)) * np.sqrt(TRADING_DAYS) \
        if excess.std(ddof=1) > 0 else np.nan

    # --- Sortino ---
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = (excess.mean() / downside_std) * np.sqrt(TRADING_DAYS) \
        if downside_std and downside_std > 0 else np.nan

    # --- Calmar ---
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan

    # --- Turnover (annualized) ---
    if turnover is not None:
        to = turnover.dropna()
        # сумма односторонних оборотов за период / годы
        total_turnover = to.sum()
        turnover_annual = total_turnover / years if years > 0 else np.nan
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
        "Start": eq.index[0].strftime("%Y-%m-%d"),
        "End": eq.index[-1].strftime("%Y-%m-%d"),
    }


# ============================================================
# SAVE
# ============================================================

def save_metrics(metrics: dict, db_path: Path, table: str = "metrics"):
    df = pd.DataFrame([metrics])
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)


def save_equity(equity: pd.Series, name: str, db_path: Path,
                table: str = "equity_curves"):
    df = equity.reset_index()
    df.columns = ["TRADEDATE", name]
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="append", index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("BENCHMARK CALCULATOR")
    print("=" * 70)

    mcftr = load_index(DATA_DIR / "MCFTR.sqlite", "MCFTR")
    rgbitr = load_index(DATA_DIR / "RGBITR.sqlite", "RGBITR")

    print(f"MCFTR:  {len(mcftr)} rows, {mcftr.index.min().date()} → {mcftr.index.max().date()}")
    print(f"RGBITR: {len(rgbitr)} rows, {rgbitr.index.min().date()} → {rgbitr.index.max().date()}")

    # ==================== 1. MCFTR 100% ====================
    eq_m = build_single_asset_curve(mcftr)
    m_m = compute_metrics(eq_m)
    print(f"\n[MCFTR_100] CAGR={m_m['CAGR']*100:.2f}%  MDD={m_m['MDD']*100:.2f}%  "
          f"Sharpe={m_m['Sharpe']:.2f}  Sortino={m_m['Sortino']:.2f}  Calmar={m_m['Calmar']:.2f}")

    # ==================== 2. RGBITR 100% ====================
    eq_r = build_single_asset_curve(rgbitr)
    m_r = compute_metrics(eq_r)
    print(f"[RGBITR_100] CAGR={m_r['CAGR']*100:.2f}%  MDD={m_r['MDD']*100:.2f}%  "
          f"Sharpe={m_r['Sharpe']:.2f}  Sortino={m_r['Sortino']:.2f}  Calmar={m_r['Calmar']:.2f}")

    # ==================== 3. MIX 50/50, ребаланс 20 дней ====================
    eq_mix, to_mix = build_mix_curve(mcftr, rgbitr)
    m_mix = compute_metrics(eq_mix, turnover=to_mix)
    print(f"[MIX_50_50]  CAGR={m_mix['CAGR']*100:.2f}%  MDD={m_mix['MDD']*100:.2f}%  "
          f"Sharpe={m_mix['Sharpe']:.2f}  Sortino={m_mix['Sortino']:.2f}  Calmar={m_mix['Calmar']:.2f}  "
          f"Turnover={m_mix['Turnover']:.2f}/yr")

    # ==================== SAVE ====================
    print("\n" + "-" * 70)
    print("Saving to SQLite...")

    # Удаляем старую таблицу equity перед append
    with sqlite3.connect(OUTPUT_DB) as conn:
        conn.execute("DROP TABLE IF EXISTS equity_curves")
        conn.execute("DROP TABLE IF EXISTS metrics")
        conn.commit()

    save_metrics({"MCFTR_100": m_m["CAGR"]}, OUTPUT_DB, "dummy")  # placeholder

    # Метрики всех трёх — в одну таблицу
    all_metrics = []
    for name, m in [("MCFTR_100", m_m), ("RGBITR_100", m_r), ("MIX_50_50", m_mix)]:
        row = {"benchmark": name}
        row.update(m)
        all_metrics.append(row)

    df_metrics = pd.DataFrame(all_metrics)
    with sqlite3.connect(OUTPUT_DB) as conn:
        df_metrics.to_sql("metrics", conn, if_exists="replace", index=False)

    # Equity curves — в одну таблицу (объединяем по датам)
    eq_df = pd.concat([eq_m.rename("MCFTR_100"),
                       eq_r.rename("RGBITR_100"),
                       eq_mix.rename("MIX_50_50")], axis=1)
    eq_df = eq_df.reset_index()
    eq_df.columns = ["TRADEDATE"] + list(eq_df.columns[1:])

    with sqlite3.connect(OUTPUT_DB) as conn:
        eq_df.to_sql("equity_curves", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eq_date ON equity_curves(TRADEDATE)")

    # Turnover микса — отдельно
    to_df = to_mix.reset_index()
    to_df.columns = ["TRADEDATE", "turnover"]
    with sqlite3.connect(OUTPUT_DB) as conn:
        to_df.to_sql("turnover_mix", conn, if_exists="replace", index=False)

    print(f"→ {OUTPUT_DB.name}")
    print(f"  metrics:       {len(df_metrics)} rows")
    print(f"  equity_curves: {len(eq_df)} rows")
    print(f"  turnover_mix:  {len(to_df)} rows")

    print("\n" + "=" * 70)
    print("METRICS TABLE")
    print("=" * 70)
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(df_metrics[["benchmark", "CAGR", "MDD", "VOL", "Sharpe",
                          "Sortino", "Calmar", "Turnover"]].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()