# features/llm_pool/common.py
# ============================================================
# Общие хелперы для батчей фич.
# Все функции causality-preserving.
# ============================================================
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-9
HORIZON = 20
ROOT = Path(__file__).parent.parent.parent
STOCKS_DB = ROOT / "dataset" / "sql" / "stocks.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"
SPLIT_JSON = ROOT / "backtests" / "split_info.json"


def load_split_date() -> str:
    return json.loads(SPLIT_JSON.read_text())["split_date"]


def load_ohlcv(split_date: str) -> dict:
    """wide DataFrames: open, high, low, close, volume."""
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


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank по каждой дате."""
    return df.rank(axis=1, pct=True)


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    return df.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)


def safe_div(a, b):
    """a / (b + eps), с защитой от inf."""
    if isinstance(b, pd.DataFrame):
        return a / b.replace(0, np.nan).fillna(np.nan)
    return a / (b + EPS)