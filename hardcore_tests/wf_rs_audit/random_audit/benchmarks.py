# random_audit/benchmarks.py
# ============================================================
# Загрузка IMOEX как бенчмарка + независимая реконструкция ML-портфеля.
# ============================================================
import numpy as np
import pandas as pd

from .config import ML_EQUITY_CSV
from .data_loader import load_imoex, compute_rs


def load_imoex_series(split_date: str) -> pd.Series:
    return load_imoex(split_date)


def load_ml_equity_from_csv() -> pd.DataFrame:
    """
    Загружает ML equity из существующего CSV.
    Используется для контрольной проверки — независимая реконструкция
    должна совпасть с этим CSV.
    """
    if not ML_EQUITY_CSV.exists():
        raise FileNotFoundError(f"Нет {ML_EQUITY_CSV}")
    df = pd.read_csv(ML_EQUITY_CSV, parse_dates=["date"]).sort_values("date")
    return df.reset_index(drop=True)


def independently_reconstruct_ml(rebal_dates: pd.DatetimeIndex,
                                 target_rs: pd.DataFrame,
                                 ml_holdings: dict) -> dict:
    """
    Независимая реконструкция ML-портфеля по готовым holdings из CSV.
    holdings: dict date -> list of tickers (из ml_trades или ml_equity).
    """
    records = []
    prev = None
    for rd in rebal_dates:
        if rd not in ml_holdings:
            continue
        selected = ml_holdings[rd]
        if len(selected) < 10:
            continue
        if rd not in target_rs.index:
            continue
        rs_row = target_rs.loc[rd, selected].dropna()
        if len(rs_row) < 10:
            continue
        port_rs = float(rs_row.mean())
        to = 1.0 if prev is None else len(set(prev) - set(selected)) / 10
        records.append({"date": rd, "port_rs": port_rs, "turnover": to})
        prev = selected
    return records