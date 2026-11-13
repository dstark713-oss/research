# hardcore_tests/stress_on_is/common/data_loader.py
import sqlite3
import numpy as np
import pandas as pd
from ..config import (
    FREEZE_DIR, RGBITR_DB, INDEXES_DB, STOCKS_DB, BENCH_DB,
)

def load_frozen_equity() -> pd.DataFrame:
    path = FREEZE_DIR / "frozen_equity.csv"
    if not path.exists():
        raise FileNotFoundError(f"Нет {path}")
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

def load_rgbitr() -> pd.Series:
    conn = sqlite3.connect(RGBITR_DB)
    df = pd.read_sql("SELECT TRADEDATE, CLOSE FROM RGBITR ORDER BY TRADEDATE", conn)
    conn.close()
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    return df.set_index("TRADEDATE")["CLOSE"].sort_index()

def load_rgbi() -> pd.Series:
    conn = sqlite3.connect(INDEXES_DB)
    df = pd.read_sql("SELECT date, close FROM indexes WHERE ticker='RGBI' ORDER BY date", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()

def load_prices() -> pd.DataFrame:
    """Wide: index=date, cols=ticker, values=close."""
    conn = sqlite3.connect(STOCKS_DB)
    df = pd.read_sql("SELECT ticker, date, close FROM prices", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="close").sort_index()

def load_volumes() -> pd.DataFrame:
    conn = sqlite3.connect(STOCKS_DB)
    df = pd.read_sql("SELECT ticker, date, volume FROM prices", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="volume").sort_index()

def load_benchmark(name: str = "MIX_50_50") -> pd.Series:
    conn = sqlite3.connect(BENCH_DB)
    eq = pd.read_sql("SELECT * FROM equity_curves", conn)
    conn.close()
    eq["TRADEDATE"] = pd.to_datetime(eq["TRADEDATE"])
    return eq.set_index("TRADEDATE")[name].dropna().sort_index()

import json

def load_frozen_holdings() -> dict:
    """Загружает holdings (rebal_date -> list[ticker])."""
    path = FREEZE_DIR / "frozen_holdings.json"
    if not path.exists():
        raise FileNotFoundError(f"Нет {path}. Запусти freeze_model.run.")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {pd.Timestamp(k): v for k, v in data.items()}