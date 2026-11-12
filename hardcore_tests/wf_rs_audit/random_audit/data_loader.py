# random_audit/data_loader.py
# ============================================================
# Загрузка сырых данных. НЕ импортируем rs_model, features.rs_pool.
# Только SQL → wide DataFrames.
# ============================================================
import json
import sqlite3

import pandas as pd

from .config import STOCKS_DB, INDEXES_DB, SPLIT_JSON


def load_split_date() -> str:
    return json.loads(SPLIT_JSON.read_text())["split_date"]


def load_ohlcv(split_date: str) -> dict:
    """Wide DataFrames: open/high/low/close/volume (index=date, cols=ticker)."""
    conn = sqlite3.connect(STOCKS_DB)
    df = pd.read_sql(
        "SELECT ticker, date, open, high, low, close, volume "
        "FROM prices WHERE date < ?",
        conn, params=(split_date,),
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = df.pivot(index="date", columns="ticker", values=col).sort_index()
    return out


def load_imoex(split_date: str) -> pd.Series:
    conn = sqlite3.connect(INDEXES_DB)
    df = pd.read_sql(
        "SELECT date, close FROM indexes WHERE ticker='IMOEX' AND date < ?",
        conn, params=(split_date,),
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()


def compute_rs(close: pd.DataFrame, imoex: pd.Series,
               horizon: int) -> pd.DataFrame:
    """
    RS_{i,t} = ret_i(t→t+h) - ret_IMOEX(t→t+h).
    Независимая реализация. Последние h дат = NaN.
    """
    fwd = close.shift(-horizon) / close - 1
    imoex_fwd = imoex.shift(-horizon) / imoex - 1
    imoex_fwd = imoex_fwd.reindex(close.index)
    return fwd.sub(imoex_fwd, axis=0)