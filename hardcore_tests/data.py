# hardcore_tests/data.py
# ============================================================
# Общие загрузчики для hardcore-тестов. Только IS.
# ============================================================
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from features.rs_pool import build_rs_pool, RS_POOL_NAMES

HORIZON = 20
ROOT = Path(__file__).parent.parent
STOCKS_DB = ROOT / "dataset" / "sql" / "stocks.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"
SPLIT_JSON = ROOT / "backtests" / "split_info.json"

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def load_split_date() -> str:
    return json.loads(SPLIT_JSON.read_text())["split_date"]


def load_ohlcv(split_date: str) -> dict:
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


def build_target(raw: dict, imoex: pd.Series) -> pd.DataFrame:
    C = raw["close"]
    fwd = C.shift(-HORIZON) / C - 1
    imoex_fwd = imoex.shift(-HORIZON) / imoex - 1
    imoex_fwd = imoex_fwd.reindex(C.index)
    return fwd.sub(imoex_fwd, axis=0)


# hardcore_tests/data.py — заменить daily_ic на эту версию

def daily_ic(feature: pd.DataFrame, target: pd.DataFrame, min_stocks=20) -> pd.Series:
    """
    Ежедневный cross-sectional Spearman IC. ВЕКТОРИЗОВАН.
    Математически эквивалентен scipy.stats.spearmanr по каждой дате,
    но в 20-50 раз быстрее (pandas groupby вместо цикла).
    """
    common_dates = feature.index.intersection(target.index)
    common_tickers = feature.columns.intersection(target.columns)
    if len(common_tickers) < min_stocks:
        return pd.Series(dtype=float)

    f = feature.loc[common_dates, common_tickers]
    t = target.loc[common_dates, common_tickers]

    # длинный формат: (date, ticker, f, t)
    f_long = f.stack().rename("f")
    t_long = t.stack().rename("t")
    df = pd.concat([f_long, t_long], axis=1).dropna()
    if len(df) == 0:
        return pd.Series(dtype=float)

    # ранги внутри каждой даты
    df["fr"] = df.groupby(level=0)["f"].rank(method="average")
    df["tr"] = df.groupby(level=0)["t"].rank(method="average")

    # центрируем
    g = df.groupby(level=0)
    df["df"] = df["fr"] - g["fr"].transform("mean")
    df["dt"] = df["tr"] - g["tr"].transform("mean")
    df["dft"] = df["df"] * df["dt"]
    df["df2"] = df["df"] ** 2
    df["dt2"] = df["dt"] ** 2

    g2 = df.groupby(level=0)
    cov = g2["dft"].mean()
    var_f = g2["df2"].mean()
    var_t = g2["dt2"].mean()
    counts = g2.size()

    denom = np.sqrt(var_f * var_t)
    ic = cov / denom.replace(0, np.nan)
    ic = ic[(counts >= min_stocks) & np.isfinite(ic)]
    ic.index.name = "date"
    return ic.sort_index()

def load_all(split_date: str = None):
    if split_date is None:
        split_date = load_split_date()
    raw = load_ohlcv(split_date)
    imoex = load_imoex(split_date)
    features = build_rs_pool(raw)
    target = build_target(raw, imoex)
    return raw, imoex, features, target