# research/backtests/split.py
# ============================================================
# Хронологический сплит данных на IS / VAL (80/20).
#
# Работает с:
#   - dataset/sql/stocks.sqlite :: prices   (сырые цены)
#   - dataset/sql/stocks.sqlite :: returns  (доходности, если есть)
#   - indexes/sql/indexes.sqlite :: indexes (индексы/макро)
#
# Логика:
#   - собираем ВСЕ уникальные даты из всех таблиц (общий календарь);
#   - сортируем;
#   - первые 80% дат → IS, последние 20% → VAL;
#   - дата отсечения (split_date) — первая дата VAL;
#   - сохраняем split_date и границы в backtests/split_info.json;
#   - НЕ режем сами данные — это делается на этапе чтения
#     (фильтр по date < split_date для IS, date >= split_date для VAL).
#
# Запуск: python split.py
# ============================================================
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

# пути
ROOT = Path(__file__).parent.parent
STOCKS_DB = ROOT / "dataset" / "sql" / "stocks.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"
OUTPUT_JSON = Path(__file__).parent / "split_info.json"

IS_RATIO = 0.80   # 80% на IS
VAL_RATIO = 0.20  # 20% на VAL


def section(title):
    line = "=" * 70
    return f"\n{line}\n{title}\n{line}"


def collect_dates() -> pd.DatetimeIndex:
    """
    Собирает все уникальные даты из stocks.prices, stocks.returns (если есть),
    indexes.indexes. Возвращает отсортированный DatetimeIndex.
    """
    all_dates = set()

    # --- stocks.prices ---
    if STOCKS_DB.exists():
        conn = sqlite3.connect(STOCKS_DB)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        if "prices" in tables:
            df = pd.read_sql("SELECT DISTINCT date FROM prices", conn)
            all_dates.update(pd.to_datetime(df["date"]).dt.normalize())
            print(f"  stocks.prices:    {len(df)} уникальных дат")

        if "returns" in tables:
            df = pd.read_sql("SELECT DISTINCT date FROM returns", conn)
            all_dates.update(pd.to_datetime(df["date"]).dt.normalize())
            print(f"  stocks.returns:   {len(df)} уникальных дат")

        conn.close()
    else:
        print(f"  WARN: нет {STOCKS_DB}")

    # --- indexes.indexes ---
    if INDEXES_DB.exists():
        conn = sqlite3.connect(INDEXES_DB)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        if "indexes" in tables:
            df = pd.read_sql("SELECT DISTINCT date FROM indexes", conn)
            all_dates.update(pd.to_datetime(df["date"]).dt.normalize())
            print(f"  indexes.indexes:  {len(df)} уникальных дат")

        conn.close()
    else:
        print(f"  WARN: нет {INDEXES_DB}")

    return pd.DatetimeIndex(sorted(all_dates))


def main():
    print("=" * 70)
    print("CHRONOLOGICAL SPLIT (IS / VAL)")
    print("=" * 70)
    print(f"IS ratio:  {IS_RATIO:.0%}")
    print(f"VAL ratio: {VAL_RATIO:.0%}")
    print("=" * 70)

    print("\nСбор уникальных дат:")
    dates = collect_dates()

    if len(dates) < 10:
        print(f"FATAL: слишком мало дат ({len(dates)}), нечего делить")
        return

    print(f"\nВсего уникальных дат: {len(dates)}")
    print(f"  min: {dates.min().date()}")
    print(f"  max: {dates.max().date()}")

    # --- граница ---
    n = len(dates)
    n_is = int(n * IS_RATIO)
    n_val = n - n_is

    is_dates = dates[:n_is]
    val_dates = dates[n_is:]

    split_date = val_dates[0]   # первая дата VAL — это и есть граница

    print(section("РЕЗУЛЬТАТ СПЛИТА"))
    print(f"IS:  {n_is:>5} дат  ({is_dates.min().date()} → {is_dates.max().date()})")
    print(f"VAL: {n_val:>5} дат  ({val_dates.min().date()} → {val_dates.max().date()})")
    print(f"\nsplit_date = {split_date.date()}")
    print(f"  IS:  date <  {split_date.date()}")
    print(f"  VAL: date >= {split_date.date()}")

    # --- сохраняем ---
    info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "is_ratio": IS_RATIO,
        "val_ratio": VAL_RATIO,
        "n_dates_total": n,
        "n_dates_is": n_is,
        "n_dates_val": n_val,
        "date_min": dates.min().strftime("%Y-%m-%d"),
        "date_max": dates.max().strftime("%Y-%m-%d"),
        "split_date": split_date.strftime("%Y-%m-%d"),
        "is_date_min": is_dates.min().strftime("%Y-%m-%d"),
        "is_date_max": is_dates.max().strftime("%Y-%m-%d"),
        "val_date_min": val_dates.min().strftime("%Y-%m-%d"),
        "val_date_max": val_dates.max().strftime("%Y-%m-%d"),
    }
    OUTPUT_JSON.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nСохранено: {OUTPUT_JSON}")

    # --- проверка: сколько строк попадёт в IS / VAL ---
    print(section("ПРОВЕРКА: объём данных в IS / VAL"))

    if STOCKS_DB.exists():
        conn = sqlite3.connect(STOCKS_DB)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        if "prices" in tables:
            n_is_rows = conn.execute(
                "SELECT COUNT(*) FROM prices WHERE date < ?", (info["split_date"],)
            ).fetchone()[0]
            n_val_rows = conn.execute(
                "SELECT COUNT(*) FROM prices WHERE date >= ?", (info["split_date"],)
            ).fetchone()[0]
            print(f"  stocks.prices:  IS={n_is_rows:,}  VAL={n_val_rows:,}")

        if "returns" in tables:
            n_is_rows = conn.execute(
                "SELECT COUNT(*) FROM returns WHERE date < ?", (info["split_date"],)
            ).fetchone()[0]
            n_val_rows = conn.execute(
                "SELECT COUNT(*) FROM returns WHERE date >= ?", (info["split_date"],)
            ).fetchone()[0]
            print(f"  stocks.returns: IS={n_is_rows:,}  VAL={n_val_rows:,}")

        conn.close()

    if INDEXES_DB.exists():
        conn = sqlite3.connect(INDEXES_DB)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "indexes" in tables:
            n_is_rows = conn.execute(
                "SELECT COUNT(*) FROM indexes WHERE date < ?", (info["split_date"],)
            ).fetchone()[0]
            n_val_rows = conn.execute(
                "SELECT COUNT(*) FROM indexes WHERE date >= ?", (info["split_date"],)
            ).fetchone()[0]
            print(f"  indexes:        IS={n_is_rows:,}  VAL={n_val_rows:,}")
        conn.close()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()