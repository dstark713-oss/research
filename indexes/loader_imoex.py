# research/indexes/loader_imoex.py
# ============================================================
# Загрузчик индексов и макро-инструментов для фич.
#
# IMOEX      — индекс MOEX (engine=stock, market=index)
# RGBI       — индекс гособлигаций (engine=stock, market=index)
# GOLD       — GLDRUB_TOM (золото в рублях, engine=currency, market=selt)
# BRENT      — Yahoo Finance BZ=F (фьючерс Brent)
# CNYRUB_TOM — юань/рубль (engine=currency, market=selt)
#
# Выход: indexes/sql/indexes.sqlite
#   таблица indexes (ticker, date, open, high, low, close, volume)
#
# Запуск: python loader_imoex.py
# ============================================================
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests

import apimoex

DATA_DIR = Path(__file__).parent / "sql"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "indexes.sqlite"

START_DATE = "2013-01-01"
END_DATE = "2026-10-31"
SLEEP = 0.25

INSTRUMENTS = {
    "IMOEX":      {"source": "moex_index"},
    "RGBI":       {"source": "moex_index"},
    "GOLD":       {"source": "moex_currency", "ticker": "GLDRUB_TOM"},
    "BRENT":      {"source": "yahoo", "ticker": "BZ=F"},
    "CNYRUB_TOM": {"source": "moex_currency", "ticker": "CNYRUB_TOM"},
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS indexes (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL,
    volume REAL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_indexes_ticker_date ON indexes(ticker, date);
"""


def init_db(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def fetch_moex_index(session, secid: str) -> pd.DataFrame:
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/index/securities/{secid}.json"
    all_rows = []
    start = 0
    while True:
        r = session.get(url, params={
            "from": START_DATE, "till": END_DATE,
            "iss.meta": "off", "start": start,
        }, timeout=30)
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
        start += len(rows)
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    if "TRADEDATE" not in df.columns or "CLOSE" not in df.columns:
        return pd.DataFrame()

    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["CLOSE"])
    df = df[df["CLOSE"] > 0]
    df = df.drop_duplicates(subset=["TRADEDATE"], keep="first").sort_values("TRADEDATE")

    out = pd.DataFrame({
        "date": df["TRADEDATE"],
        "open": pd.to_numeric(df.get("OPEN"), errors="coerce") if "OPEN" in df else df["CLOSE"],
        "high": pd.to_numeric(df.get("HIGH"), errors="coerce") if "HIGH" in df else df["CLOSE"],
        "low": pd.to_numeric(df.get("LOW"), errors="coerce") if "LOW" in df else df["CLOSE"],
        "close": pd.to_numeric(df["CLOSE"], errors="coerce"),
        "volume": pd.to_numeric(df.get("VOLUME"), errors="coerce") if "VOLUME" in df else None,
    })
    return out


def fetch_moex_currency(session, ticker: str) -> pd.DataFrame:
    data = apimoex.get_board_history(
        session, security=ticker, board="CETS",
        engine="currency", market="selt",
        start=START_DATE, end=END_DATE,
        columns=("TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"),
    )
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "TRADEDATE" not in df.columns or "CLOSE" not in df.columns:
        return pd.DataFrame()
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["CLOSE"])
    df = df[df["CLOSE"] > 0]
    df = df.drop_duplicates(subset=["TRADEDATE"], keep="first").sort_values("TRADEDATE")
    return pd.DataFrame({
        "date": df["TRADEDATE"],
        "open": df.get("OPEN"), "high": df.get("HIGH"),
        "low": df.get("LOW"), "close": df["CLOSE"], "volume": df.get("VOLUME"),
    })


def fetch_yahoo(ticker: str) -> pd.DataFrame:
    """
    Загрузка с Yahoo Finance через yfinance.
    Возвращает DataFrame с колонками date, open, high, low, close, volume.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  WARN: yfinance не установлен (pip install yfinance)")
        return pd.DataFrame()

    try:
        # yfinance: end эксклюзивный, поэтому +1 день
        end_plus = (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(
            ticker,
            start=START_DATE,
            end=end_plus,
            progress=False,
            auto_adjust=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        # MultiIndex колонки (бывает в новых версиях yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        cmap = {str(c).lower(): c for c in df.columns}
        out = pd.DataFrame({
            "date": pd.to_datetime(df.index).strftime("%Y-%m-%d"),
            "open": df[cmap.get("open")] if "open" in cmap else None,
            "high": df[cmap.get("high")] if "high" in cmap else None,
            "low": df[cmap.get("low")] if "low" in cmap else None,
            "close": df[cmap.get("close")] if "close" in cmap else None,
            "volume": df[cmap.get("volume")] if "volume" in cmap else None,
        })
        out = out.dropna(subset=["close"])
        out = out[out["close"] > 0]
        out = out.drop_duplicates(subset=["date"], keep="first").sort_values("date")
        return out
    except Exception as e:
        print(f"  yahoo err {ticker}: {str(e)[:120]}")
        return pd.DataFrame()


def save(conn, ticker: str, df: pd.DataFrame):
    rows = [
        (ticker, r.date, r.open, r.high, r.low, r.close, r.volume)
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO indexes (ticker, date, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def main():
    print("=" * 70)
    print("INDEXES LOADER")
    print("=" * 70)
    print(f"Period: {START_DATE} → {END_DATE}")
    print(f"SQLite: {DB_PATH.absolute()}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0 (research_indexes/1.0)"})

        for ticker, cfg in INSTRUMENTS.items():
            print(f"\n[{ticker}] source={cfg['source']}")
            try:
                if cfg["source"] == "moex_index":
                    df = fetch_moex_index(session, ticker)
                elif cfg["source"] == "moex_currency":
                    df = fetch_moex_currency(session, cfg["ticker"])
                elif cfg["source"] == "yahoo":
                    df = fetch_yahoo(cfg["ticker"])
                else:
                    print(f"  unknown source")
                    continue

                if df.empty:
                    print(f"  EMPTY")
                    continue

                save(conn, ticker, df)
                print(f"  OK {len(df)} bars ({df['date'].iloc[0]} → {df['date'].iloc[-1]})")
            except Exception as e:
                print(f"  ERROR: {str(e)[:120]}")
            time.sleep(SLEEP)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    for ticker in INSTRUMENTS:
        n = conn.execute("SELECT COUNT(*) FROM indexes WHERE ticker=?", (ticker,)).fetchone()[0]
        if n:
            first = conn.execute("SELECT MIN(date) FROM indexes WHERE ticker=?", (ticker,)).fetchone()[0]
            last = conn.execute("SELECT MAX(date) FROM indexes WHERE ticker=?", (ticker,)).fetchone()[0]
            print(f"  {ticker:<14} {n:>6} bars  ({first} → {last})")
        else:
            print(f"  {ticker:<14} EMPTY")
    conn.close()


if __name__ == "__main__":
    main()