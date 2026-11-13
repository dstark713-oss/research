# риск_модели/гарч/data_loader.py
# ============================================================
# Загрузка:
#   - дневных цен акций (для восстановления доходностей портфеля)
#   - holdings базового RS-портфеля (из rs_is_wf_trades.csv)
#   - ML equity (для сравнения)
# ============================================================
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .config import STOCKS_DB, SPLIT_JSON, ML_EQUITY_CSV, ML_TRADES_CSV

from .config import STOCKS_DB, INDEXES_DB, SPLIT_JSON, ML_EQUITY_CSV, ML_TRADES_CSV


def load_split_date() -> str:
    return json.loads(SPLIT_JSON.read_text())["split_date"]


def load_stock_close(split_date: str) -> pd.DataFrame:
    """Wide DataFrame: index=date, cols=ticker, values=close."""
    conn = sqlite3.connect(STOCKS_DB)
    df = pd.read_sql(
        "SELECT ticker, date, close FROM prices WHERE date < ?",
        conn, params=(split_date,),
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="close").sort_index()


def load_ml_holdings() -> dict:
    """
    Возвращает dict: rebal_date -> list[ticker].
    Из rs_is_wf_trades.csv (колонки date, ticker, weight, pred).
    """
    if not ML_TRADES_CSV.exists():
        raise FileNotFoundError(
            f"Нет {ML_TRADES_CSV}. Нужны holdings базового портфеля."
        )
    df = pd.read_csv(ML_TRADES_CSV, parse_dates=["date"])
    holdings = df.groupby("date")["ticker"].apply(list).to_dict()
    return holdings


def load_ml_equity() -> pd.DataFrame:
    """ML equity с port_rs, turnover, equity."""
    if not ML_EQUITY_CSV.exists():
        raise FileNotFoundError(f"Нет {ML_EQUITY_CSV}")
    return pd.read_csv(ML_EQUITY_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

def load_imoex(split_date: str) -> pd.Series:
    """Wide Series: index=date, values=close. Только IS."""
    conn = sqlite3.connect(INDEXES_DB)
    df = pd.read_sql(
        "SELECT date, close FROM indexes WHERE ticker='IMOEX' AND date < ?",
        conn, params=(split_date,),
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()