# random_audit/bootstrap/data_loader.py
# ============================================================
# Загрузка ML equity (с port_rs и turnover) и результатов random.
# ============================================================
import json
import sqlite3

import pandas as pd

from .config import (
    ROOT, ML_EQUITY_CSV, RANDOM_RESULTS_CSV, HORIZON,
)

STOCKS_DB = ROOT / "dataset" / "sql" / "stocks.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"
SPLIT_JSON = ROOT / "backtests" / "split_info.json"


def load_split_date() -> str:
    return json.loads(SPLIT_JSON.read_text())["split_date"]


def load_ml_equity() -> pd.DataFrame:
    """
    ML equity с колонками date, port_rs, turnover, equity.
    В rs_is_wf_equity.csv есть port_rs и turnover.
    """
    if not ML_EQUITY_CSV.exists():
        raise FileNotFoundError(f"Нет {ML_EQUITY_CSV}")
    df = pd.read_csv(ML_EQUITY_CSV, parse_dates=["date"]).sort_values("date")
    required = {"date", "port_rs"}
    if not required.issubset(df.columns):
        raise ValueError(f"В {ML_EQUITY_CSV} нет колонок {required}. "
                         f"Есть: {list(df.columns)}")
    return df.reset_index(drop=True)


def load_random_results() -> pd.DataFrame:
    """Сохранённые результаты Monte Carlo (только метрики)."""
    if not RANDOM_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Нет {RANDOM_RESULTS_CSV}")
    return pd.read_csv(RANDOM_RESULTS_CSV)


def load_ohlcv(split_date: str) -> dict:
    conn = sqlite3.connect(STOCKS_DB)
    df = pd.read_sql(
        "SELECT ticker, date, close FROM prices WHERE date < ?",
        conn, params=(split_date,),
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot(index="date", columns="ticker", values="close").sort_index()
    return {"close": close}


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
               horizon: int = HORIZON) -> pd.DataFrame:
    """
    RS = ret_i(t→t+h) - ret_IMOEX(t→t+h).
    Независимая реализация. Последние h дат = NaN.
    """
    fwd = close.shift(-horizon) / close - 1
    imoex_fwd = imoex.shift(-horizon) / imoex - 1
    imoex_fwd = imoex_fwd.reindex(close.index)
    return fwd.sub(imoex_fwd, axis=0)