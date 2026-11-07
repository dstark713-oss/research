# research/dataset/clean_returns.py
# ============================================================
# Построение таблицы доходностей с обработкой экстремумов.
#
# СЫРЫЕ ЦЕНЫ (prices) НЕ ТРОГАЕМ.
# Результат — отдельная таблица returns в той же stocks.sqlite:
#
#   ticker    TEXT
#   date      TEXT
#   ret       REAL   -- дневная доходность; NaN если |ret| > 90%
#   is_extreme INTEGER -- 1 если 30% < |ret| <= 90% (флаг «подозрительно»)
#   PRIMARY KEY (ticker, date)
#
# Логика:
#   |ret| > 0.90          → ret = NaN, is_extreme = 0  (день не существует)
#   0.30 < |ret| <= 0.90  → ret остаётся, is_extreme = 1
#   |ret| <= 0.30         → ret остаётся, is_extreme = 0
#
# Запуск: python clean_returns.py
# ============================================================
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "sql" / "stocks.sqlite"
LOG_PATH = Path(__file__).parent / "clean_returns_log.md"

# пороги
RET_HARD = 0.90   # |ret| > 90%  → NaN (день не существует)
RET_SOFT = 0.30   # 30% < |ret| <= 90% → флаг is_extreme = 1


def section(title):
    line = "=" * 70
    return f"\n{line}\n{title}\n{line}"


def main():
    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# Обработка доходностей\n")
    emit(f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n")
    emit(f"Правила:")
    emit(f"  |ret| > {RET_HARD:.0%}         → ret = NaN (день не существует)")
    emit(f"  {RET_SOFT:.0%} < |ret| <= {RET_HARD:.0%}  → ret остаётся, is_extreme = 1")
    emit(f"  |ret| <= {RET_SOFT:.0%}        → ret остаётся, is_extreme = 0")
    emit(f"Сырые цены в prices НЕ модифицируются.\n")

    if not DB_PATH.exists():
        emit(f"FATAL: нет {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)

    # ==================== ЗАГРУЗКА ====================
    emit(section("ЗАГРУЗКА СЫРЫХ ЦЕН"))
    prices = pd.read_sql(
        "SELECT ticker, date, close FROM prices ORDER BY ticker, date",
        conn,
    )
    emit(f"Строк:   {len(prices):,}")
    emit(f"Тикеров: {prices['ticker'].nunique()}")

    # ==================== ДОХОДНОСТИ ====================
    emit(section("РАСЧЁТ ДОХОДНОСТЕЙ"))

    # ret внутри тикера; первый бар каждого тикера = NaN (нет предыдущего)
    prices["ret"] = prices.groupby("ticker")["close"].pct_change()

    n_total = len(prices)
    n_first = prices["ret"].isna().sum()
    emit(f"Всего строк:        {n_total:,}")
    emit(f"Первый бар (NaN):   {n_first:,}")

    # ==================== ОБРАБОТКА ====================
    emit(section("ОБРАБОТКА ЭКСТРЕМУМОВ"))

    abs_ret = prices["ret"].abs()

    # маска жёстких (> 90%) — заменяем на NaN
    mask_hard = abs_ret > RET_HARD
    # маска мягких (30% < |ret| <= 90%) — оставляем, но флаг
    mask_soft = (abs_ret > RET_SOFT) & (abs_ret <= RET_HARD)

    n_hard = int(mask_hard.sum())
    n_soft = int(mask_soft.sum())

    emit(f"Жёсткие (|ret| > {RET_HARD:.0%}):        {n_hard:,} → ret = NaN")
    emit(f"Мягкие ({RET_SOFT:.0%} < |ret| <= {RET_HARD:.0%}): {n_soft:,} → is_extreme = 1")

    # применяем
    prices["is_extreme"] = mask_soft.astype(int)
    prices.loc[mask_hard, "ret"] = np.nan
    # у жёстких флаг сбрасываем: день «не существует», флагу нечего означать
    prices.loc[mask_hard, "is_extreme"] = 0

    # ==================== ПРИМЕРЫ ====================
    emit("\nПримеры жёстких (первые 15, до замены на NaN):")
    hard_examples = prices[mask_hard].head(15)
    if len(hard_examples):
        emit(hard_examples[["ticker", "date", "close"]].to_string(index=False))

    emit(f"\nПримеры мягких (первые 15):")
    soft_examples = prices[mask_soft].head(15)
    if len(soft_examples):
        emit(soft_examples[["ticker", "date", "close", "ret"]].to_string(index=False))

    # ==================== ЗАПИСЬ В SQL ====================
    emit(section("ЗАПИСЬ В SQL"))

    out = prices[["ticker", "date", "ret", "is_extreme"]].copy()
    out = out.dropna(subset=["ret"])  # первый бар каждого тикера выкидываем
    emit(f"Строк к записи (без первого бара): {len(out):,}")

    conn.execute("DROP TABLE IF EXISTS returns")
    conn.execute("""
        CREATE TABLE returns (
            ticker     TEXT NOT NULL,
            date       TEXT NOT NULL,
            ret        REAL,
            is_extreme INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (ticker, date)
        )
    """)
    out.to_sql("returns", conn, if_exists="append", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_returns_ticker_date ON returns(ticker, date)")
    conn.commit()

    # ==================== ИТОГ ====================
    emit(section("ИТОГ"))
    n_ret = conn.execute("SELECT COUNT(*) FROM returns").fetchone()[0]
    n_nan = conn.execute("SELECT COUNT(*) FROM returns WHERE ret IS NULL").fetchone()[0]
    n_flag = conn.execute("SELECT COUNT(*) FROM returns WHERE is_extreme = 1").fetchone()[0]
    n_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM returns").fetchone()[0]

    emit(f"Тикеров:            {n_tickers}")
    emit(f"Строк в returns:    {n_ret:,}")
    emit(f"Из них ret IS NULL: {n_nan:,}  (жёсткие выбросы, «день не существует»)")
    emit(f"Из них is_extreme=1: {n_flag:,}  (мягкие выбросы, помечены)")
    emit(f"\nТаблица returns создана в {DB_PATH.name}")
    emit(f"Сырая таблица prices не тронута.")

    conn.close()
    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nЛог: {LOG_PATH}")


if __name__ == "__main__":
    main()