# hardcore_tests/wf_rs_audit/survivorship.py
# ============================================================
# Блок 8: survivorship bias.
# Проверяем: сколько тикеров в каждой дате, есть ли делистингованные.
# ============================================================
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .common import STOCKS_DB, load_ohlcv, load_split_date, verdict


def run() -> list:
    split_date = load_split_date()
    raw = load_ohlcv(split_date)
    close = raw["close"]

    # --- 8.1: сколько тикеров на каждой дате ---
    n_tickers = close.notna().sum(axis=1)
    n_tickers = n_tickers[n_tickers > 0]

    # --- 8.2: проверяем, есть ли делистингованные ---
    # Делистингованный: торговался раньше, но пропал из данных к концу IS
    first_date = close.index[0]
    last_date = close.index[-1]
    has_early = close.loc[:first_date + pd.Timedelta(days=365)].notna().any()
    has_late = close.loc[last_date - pd.Timedelta(days=365):].notna().any()
    delisted = has_early & ~has_late
    n_delisted = int(delisted.sum())

    evidence = (
        f"Тикеров на дату: min={n_tickers.min()}, median={n_tickers.median():.0f}, "
        f"max={n_tickers.max()}. "
        f"Тикеров, торговавшихся в начале, но не в конце: {n_delisted}."
    )

    if n_delisted == 0:
        return [verdict(
            "8.1 Делистингованные тикеры в universe",
            "FAIL", "HIGH",
            evidence=evidence + " Нет ни одного делистингованного — "
                     "survivorship bias вероятен. Список тикеров, "
                     "вероятно, сформирован по текущему состоянию.",
            files=["dataset/loader.py"],
            recommendation="Проверить источник списка тикеров: если это "
                           "TQBR на текущую дату, то исторические "
                           "делистингованные отсутствуют.",
        )]
    return [verdict(
        "8.1 Делистингованные тикеры в universe",
        "PASS",
        evidence=evidence,
    )]