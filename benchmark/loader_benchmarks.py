# research/benchmark/loader_benchmarks.py
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests

ISS_BASE = "https://iss.moex.com/iss"
DATA_DIR = Path(__file__).parent / "sql"
DATA_DIR.mkdir(exist_ok=True)

BENCHMARKS = {
    "MCFTR": {"engine": "stock", "market": "index"},
    "RGBITR": {"engine": "stock", "market": "index"},
}

START_DATE = "2004-01-01"   # MCFTR начал считаться с 30.12.2004 [citation:1]
END_DATE   = "2026-10-31"   # до 31.10.2026, как договаривались


def fetch_index_history(secid: str, engine: str, market: str,
                        start: str, end: str) -> pd.DataFrame:
    """Скачивает дневную историю индекса с MOEX ISS с пагинацией."""
    url = f"{ISS_BASE}/history/engines/{engine}/markets/{market}/securities/{secid}.json"
    all_rows = []
    start_param = 0

    while True:
        params = {
            "from": start,
            "till": end,
            "iss.meta": "off",
            "start": start_param,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()

        hist = j.get("history", {})
        cols = hist.get("columns", [])
        rows = hist.get("data", [])

        if not rows:
            break

        for row in rows:
            all_rows.append(dict(zip(cols, row)))

        if len(rows) < 100:
            break

        start_param += len(rows)
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Оставляем только нужные колонки (для индексов CLOSE — главное)
    keep = ["TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    df = df[[c for c in keep if c in df.columns]].copy()

    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"]).dt.normalize()
    df = df.dropna(subset=["CLOSE"])
    df = df[df["CLOSE"] > 0]
    df = df.drop_duplicates(subset=["TRADEDATE"], keep="first")
    df = df.sort_values("TRADEDATE").reset_index(drop=True)

    return df


def save_to_sqlite(df: pd.DataFrame, table: str, db_path: Path):
    """Сохраняет DataFrame в SQLite (перезаписывает таблицу)."""
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)
        # Индекс по дате для быстрых запросов
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_date ON {table}(TRADEDATE)"
        )
    print(f"  → {db_path.name} :: {table} ({len(df)} rows)")


def main():
    print("=" * 60)
    print("BENCHMARK LOADER (MCFTR / RGBITR)")
    print("=" * 60)
    print(f"Period: {START_DATE} → {END_DATE}")
    print(f"Output: {DATA_DIR.absolute()}")
    print("=" * 60)

    for ticker, cfg in BENCHMARKS.items():
        print(f"\n[{ticker}]")
        try:
            df = fetch_index_history(
                secid=ticker,
                engine=cfg["engine"],
                market=cfg["market"],
                start=START_DATE,
                end=END_DATE,
            )

            if df.empty:
                print(f"  FAIL: пусто")
                continue

            db_path = DATA_DIR / f"{ticker}.sqlite"
            save_to_sqlite(df, table=ticker, db_path=db_path)

            print(f"  OK: {df['TRADEDATE'].min().date()} → {df['TRADEDATE'].max().date()}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()