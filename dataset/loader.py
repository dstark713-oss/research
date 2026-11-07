# research/dataset/loader.py
# ============================================================
# Загрузчик ВСЕХ обыкновенных акций MOEX (дневки, вся история).
#
# - Список бумаг: apimoex.get_board_securities по TQBR,
#   фильтр SECTYPE == '1' (обыкновенная акция), префы отбрасываем.
# - История: apimoex.get_board_history (дневки, TQBR).
# - Выход: dataset/sql/stocks.sqlite
#     таблица prices      (ticker, date, open, high, low, close, volume)
#     таблица securities  (ticker, shortname, isin, listlevel, ...)
#     индекс idx_prices_ticker_date
#
# Запуск: python loader.py
# Долго: ~200+ бумаг, последовательно. Можно прервать (Ctrl+C),
#        повторный запуск докачает только отсутствующие тикеры.
# ============================================================
import sqlite3
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

import apimoex

# ============================================================
# КОНФИГ
# ============================================================
DATA_DIR = Path(__file__).parent / "sql"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "stocks.sqlite"

BOARD = "TQBR"                 # основной режим торгов акциями
START_DATE = "2013-01-01"      # начало истории (можно "2000-01-01", если надо глубже)
END_DATE = "2026-10-31"        # до 31.10.2026, как договорились
SLEEP_BETWEEN = 0.25           # пауза между тикерами, чтобы не злить ISS

# ============================================================
# SQL SCHEMA
# ============================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker   TEXT NOT NULL,
    date     TEXT NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   REAL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
    ON prices(ticker, date);

CREATE TABLE IF NOT EXISTS securities (
    ticker     TEXT PRIMARY KEY,
    shortname  TEXT,
    isin       TEXT,
    listlevel  INTEGER,
    issuesize  REAL,
    loaded_at  TEXT
);
"""


def init_db(conn: sqlite3.Connection):
    conn.executescript(SCHEMA)
    conn.commit()


# ============================================================
# 1. СПИСОК ТИКЕРОВ (только обычка)
# ============================================================
def fetch_common_shares(session: requests.Session) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками SECID, SHORTNAME, ISIN, LISTLEVEL,
    SECTYPE и т.д. — только обыкновенные акции (SECTYPE == '1').
    """
    print(f"[1/2] Получаю список бумаг с {BOARD}...")
    data = apimoex.get_board_securities(
        session,
        board=BOARD,
        market="shares",
        engine="stock",
        columns=("SECID", "SHORTNAME", "ISIN", "LISTLEVEL",
                 "ISSUESIZE", "SECTYPE", "SECNAME"),
    )
    if not data:
        raise RuntimeError("пустой ответ от get_board_securities")

    df = pd.DataFrame(data)
    print(f"      всего бумаг на {BOARD}: {len(df)}")

    # SECTYPE: '1' = обыкновенная акция, '2' = привилегированная
    if "SECTYPE" in df.columns:
        before = len(df)
        df = df[df["SECTYPE"].astype(str) == "1"].copy()
        print(f"      после фильтра SECTYPE == '1': {len(df)} "
              f"(отброшено {before - len(df)})")
    else:
        # fallback: по названию — «ао» / «обыкн»
        mask = df["SECNAME"].str.contains("обыкн|ао", case=False, na=False)
        df = df[mask].copy()
        print(f"      SECTYPE нет, fallback по SECNAME: {len(df)}")

    # убираем возможные дубли SECID
    df = df.drop_duplicates(subset=["SECID"]).reset_index(drop=True)
    return df


# ============================================================
# 2. ИСТОРИЯ ПО ОДНОМУ ТИКЕРУ
# ============================================================
def fetch_history(session: requests.Session, ticker: str) -> pd.DataFrame:
    """
    Дневная история через get_board_history.
    Возвращает DataFrame с колонками:
      TRADEDATE, OPEN, HIGH, LOW, CLOSE, VOLUME
    (если каких-то колонок нет — заполняем NaN).
    """
    data = apimoex.get_board_history(
        session,
        security=ticker,
        board=BOARD,
        market="shares",
        engine="stock",
        start=START_DATE,
        end=END_DATE,
        columns=("TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"),
    )
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "TRADEDATE" not in df.columns or "CLOSE" not in df.columns:
        return pd.DataFrame()

    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["TRADEDATE", "CLOSE"])
    df = df[df["CLOSE"] > 0]
    df = df.drop_duplicates(subset=["TRADEDATE"], keep="first").sort_values("TRADEDATE")

    # гарантируем все колонки
    for c in ("OPEN", "HIGH", "LOW", "VOLUME"):
        if c not in df.columns:
            df[c] = None

    return df[["TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]


# ============================================================
# 3. ЗАПИСЬ В SQL
# ============================================================
def save_history(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame):
    """INSERT OR IGNORE — можно перезапускать без дублей."""
    rows = [
        (ticker, r.TRADEDATE, r.OPEN, r.HIGH, r.LOW, r.CLOSE, r.VOLUME)
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO prices "
        "(ticker, date, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def save_security(conn: sqlite3.Connection, row: pd.Series):
    conn.execute(
        "INSERT OR REPLACE INTO securities "
        "(ticker, shortname, isin, listlevel, issuesize, loaded_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            row.get("SECID"),
            row.get("SHORTNAME"),
            row.get("ISIN"),
            row.get("LISTLEVEL"),
            row.get("ISSUESIZE"),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def already_loaded(conn: sqlite3.Connection, ticker: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM prices WHERE ticker = ? LIMIT 1", (ticker,)
    )
    return cur.fetchone() is not None


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("DATASET LOADER — все обыкновенные акции MOEX (TQBR)")
    print("=" * 70)
    print(f"Board:      {BOARD}")
    print(f"Period:     {START_DATE} → {END_DATE}")
    print(f"SQLite:     {DB_PATH.absolute()}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    with requests.Session() as session:
        # --- 1. список ---
        try:
            shares = fetch_common_shares(session)
        except Exception as e:
            print(f"FATAL: не получил список бумаг: {e}")
            traceback.print_exc()
            conn.close()
            return

        if shares.empty:
            print("FATAL: список обыкновенных акций пуст")
            conn.close()
            return

        # --- 2. загрузка ---
        print(f"\n[2/2] Загрузка истории ({len(shares)} тикеров)...")
        success, skipped, failed = 0, 0, 0
        failed_list = []
        t0 = time.time()

        for i, (_, row) in enumerate(shares.iterrows(), 1):
            ticker = row["SECID"]

            if already_loaded(conn, ticker):
                print(f"  [{i:>3}/{len(shares)}] {ticker:<10} SKIP (уже в SQL)")
                save_security(conn, row)
                skipped += 1
                continue

            try:
                df = fetch_history(session, ticker)
                if df.empty:
                    print(f"  [{i:>3}/{len(shares)}] {ticker:<10} EMPTY")
                    failed += 1
                    failed_list.append(ticker)
                    continue

                save_history(conn, ticker, df)
                save_security(conn, row)
                print(f"  [{i:>3}/{len(shares)}] {ticker:<10} OK "
                      f"{len(df)} bars  "
                      f"({df['TRADEDATE'].iloc[0]} → {df['TRADEDATE'].iloc[-1]})")
                success += 1

            except Exception as e:
                print(f"  [{i:>3}/{len(shares)}] {ticker:<10} ERROR: {str(e)[:100]}")
                failed += 1
                failed_list.append(ticker)

            time.sleep(SLEEP_BETWEEN)

        elapsed = time.time() - t0

    # --- 3. итог ---
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Success: {success}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")
    if failed_list:
        print(f"Failed list: {', '.join(failed_list)}")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print(f"SQLite:  {DB_PATH.absolute()}")

    # быстрый чек
    cur = conn.execute("SELECT COUNT(DISTINCT ticker), COUNT(*) FROM prices")
    n_tickers, n_rows = cur.fetchone()
    print(f"В базе:  {n_tickers} тикеров, {n_rows} строк")
    conn.close()
    print("=" * 70)


if __name__ == "__main__":
    main()