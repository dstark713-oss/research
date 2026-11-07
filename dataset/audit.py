# research/dataset/audit.py
# ============================================================
# Аудит данных: stocks.sqlite + бенчмарки.
# Проверяет:
#   - покрытие тикеров (кол-во, история, пропуски)
#   - дубли (ticker, date)
#   - аномалии OHLC (high < low, close вне [low, high] и т.п.)
#   - нулевые/отрицательные цены и объёмы
#   - экстремальные дневные доходности
#   - дыры в датах (пропущенные торговые дни)
# Вывод: консоль + audit_report.md
# ============================================================
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DB = Path(__file__).parent / "sql" / "stocks.sqlite"
BENCH_DB = Path(__file__).parent.parent / "benchmark" / "sql" / "BENCHMARKS.sqlite"
REPORT_PATH = Path(__file__).parent / "audit_report.md"

# порог «экстремальной» дневной доходности
EXTREME_RET = 0.30   # 30% за день
MIN_HISTORY = 250    # меньше — помечаем как «короткая история»


def section(title: str) -> str:
    line = "=" * 70
    return f"\n{line}\n{title}\n{line}"


def main():
    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# Аудит данных\n")
    emit(f"Дата аудита: {pd.Timestamp.now():%Y-%m-%d %H:%M}\n")

    # ==================== DATASET ====================
    emit(section("DATASET: stocks.sqlite"))

    if not DATASET_DB.exists():
        emit(f"FATAL: нет файла {DATASET_DB}")
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        return

    conn = sqlite3.connect(DATASET_DB)

    # --- общее ---
    n_tickers = pd.read_sql("SELECT COUNT(DISTINCT ticker) AS n FROM prices", conn)["n"][0]
    n_rows = pd.read_sql("SELECT COUNT(*) AS n FROM prices", conn)["n"][0]
    emit(f"Тикеров: {n_tickers}")
    emit(f"Строк:   {n_rows:,}")

    # --- по тикерам ---
    per_ticker = pd.read_sql("""
        SELECT ticker,
               COUNT(*)        AS n_bars,
               MIN(date)       AS first_date,
               MAX(date)       AS last_date,
               AVG(volume)     AS avg_volume
        FROM prices
        GROUP BY ticker
        ORDER BY n_bars DESC
    """, conn)

    emit("\n## Покрытие по тикерам")
    emit(f"Медиана баров: {per_ticker['n_bars'].median():.0f}")
    emit(f"Мин баров:     {per_ticker['n_bars'].min()} ({per_ticker.iloc[-1]['ticker']})")
    emit(f"Макс баров:    {per_ticker['n_bars'].max()} ({per_ticker.iloc[0]['ticker']})")

    short = per_ticker[per_ticker["n_bars"] < MIN_HISTORY]
    emit(f"\nТикеров с историей < {MIN_HISTORY} баров: {len(short)}")
    if len(short):
        emit(short[["ticker", "n_bars", "first_date", "last_date"]].to_string(index=False))

    emit("\nТоп-10 по длине истории:")
    emit(per_ticker.head(10)[["ticker", "n_bars", "first_date", "last_date"]].to_string(index=False))

    emit("\nХвост-10 (самые короткие):")
    emit(per_ticker.tail(10)[["ticker", "n_bars", "first_date", "last_date"]].to_string(index=False))

    # --- дубли ---
    dup = pd.read_sql("""
        SELECT ticker, date, COUNT(*) AS c
        FROM prices
        GROUP BY ticker, date
        HAVING c > 1
    """, conn)
    emit(f"\n## Дубли (ticker, date): {len(dup)}")
    if len(dup):
        emit(dup.head(20).to_string(index=False))

    # --- аномалии OHLC ---
    emit("\n## Аномалии OHLC")
    anomalies = pd.read_sql(f"""
        SELECT
            SUM(CASE WHEN high < low THEN 1 ELSE 0 END)              AS high_lt_low,
            SUM(CASE WHEN close > high OR close < low THEN 1 ELSE 0 END) AS close_outside,
            SUM(CASE WHEN open  > high OR open  < low THEN 1 ELSE 0 END) AS open_outside,
            SUM(CASE WHEN close <= 0 THEN 1 ELSE 0 END)              AS close_nonpos,
            SUM(CASE WHEN volume IS NULL OR volume <= 0 THEN 1 ELSE 0 END) AS vol_nonpos
        FROM prices
    """, conn)
    emit(anomalies.T.to_string(header=False))

    # --- экстремальные доходности ---
    emit(f"\n## Экстремальные дневные доходности (|ret| > {EXTREME_RET:.0%})")
    prices_df = pd.read_sql(
        "SELECT ticker, date, close FROM prices ORDER BY ticker, date", conn
    )
    prices_df["ret"] = prices_df.groupby("ticker")["close"].pct_change()
    extreme = prices_df[prices_df["ret"].abs() > EXTREME_RET].dropna()
    emit(f"Случаев: {len(extreme)}")
    if len(extreme):
        emit(extreme.sort_values("ret", key=abs, ascending=False).head(20).to_string(index=False))

    # --- дыры в датах ---
    emit("\n## Дыры в датах (пропущенные торговые дни)")
    # строим общий торговый календарь по всем тикерам
    all_dates = pd.to_datetime(prices_df["date"].unique())
    all_dates = pd.DatetimeIndex(sorted(all_dates))
    # для каждого тикера — сколько дат календаря отсутствует
    gaps = []
    for ticker, g in prices_df.groupby("ticker"):
        present = pd.DatetimeIndex(pd.to_datetime(g["date"].unique()))
        # ограничиваем календарь диапазоном жизни тикера
        cal = all_dates[(all_dates >= present.min()) & (all_dates <= present.max())]
        missing = len(cal) - len(present)
        gaps.append({"ticker": ticker, "calendar_days": len(cal),
                     "present": len(present), "missing": missing,
                     "missing_pct": missing / len(cal) if len(cal) else 0})
    gaps_df = pd.DataFrame(gaps).sort_values("missing_pct", ascending=False)
    emit(f"Тикеров с пропусками > 10%: {(gaps_df['missing_pct'] > 0.10).sum()}")
    emit("\nТоп-10 по доле пропусков:")
    emit(gaps_df.head(10).to_string(index=False))

    conn.close()

    # ==================== BENCHMARK ====================
    emit(section("BENCHMARK: BENCHMARKS.sqlite"))
    if BENCH_DB.exists():
        bconn = sqlite3.connect(BENCH_DB)
        metrics = pd.read_sql("SELECT * FROM metrics", bconn)
        emit("\nМетрики:")
        with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
            cols = ["benchmark", "CAGR", "MDD", "VOL", "Sharpe",
                    "Sortino", "Calmar", "Turnover"]
            emit(metrics[cols].to_string(index=False))

        eq = pd.read_sql("SELECT * FROM equity_curves", bconn)
        emit(f"\nEquity curves: {len(eq)} строк, "
             f"{eq['TRADEDATE'].min()} → {eq['TRADEDATE'].max()}")

        # проверка: нет ли NaN в equity
        for c in ["MCFTR_100", "RGBITR_100", "MIX_50_50"]:
            if c in eq.columns:
                emit(f"  {c}: NaN={eq[c].isna().sum()}, "
                     f"start={eq[c].iloc[0]:.4f}, end={eq[c].iloc[-1]:.4f}")
        bconn.close()
    else:
        emit(f"WARN: нет {BENCH_DB}")

    # ==================== ВЫВОД ====================
    emit(section("ИТОГ"))
    emit("Аудит завершён. Полный отчёт: audit_report.md")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nОтчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    main()