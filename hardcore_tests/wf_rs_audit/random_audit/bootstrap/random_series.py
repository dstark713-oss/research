# random_audit/bootstrap/random_series.py
# ============================================================
# Пересчёт port_rs для 100 random портфелей.
# Это НЕ Monte Carlo — это 100 портфелей, выбранных seeded.
# Быстро: 100 × 130 дат.
# ============================================================
import numpy as np
import pandas as pd

from .config import TOP_N, BASE_SEED, N_RANDOM_PORTFOLIOS


def build_availability(close: pd.DataFrame, min_history: int = 60) -> pd.DataFrame:
    has_today = close.notna()
    has_prev = close.shift(1).notna()
    history = close.notna().cumsum()
    return has_today & has_prev & (history >= min_history)


def get_rebalance_dates(target: pd.DataFrame, rebal_step: int = 20) -> pd.DatetimeIndex:
    valid = target.notna().sum(axis=1) >= TOP_N
    dates = valid[valid].index
    return dates[::rebal_step]


def generate_random_portfolios(close: pd.DataFrame, target_rs: pd.DataFrame,
                               rebal_dates: pd.DatetimeIndex,
                               n_portfolios: int = N_RANDOM_PORTFOLIOS,
                               top_n: int = TOP_N,
                               base_seed: int = BASE_SEED) -> dict:
    """
    Возвращает dict: portfolio_id -> pd.Series(port_rs, index=rebal_dates).
    rebal_dates передаются ЯВНО — те же, что у ML.
    """
    availability = build_availability(close)

    out = {}
    for pid in range(n_portfolios):
        seed = base_seed + 300000 + pid
        rng = np.random.RandomState(seed)
        port_rs_list = []
        dates_list = []
        for rd in rebal_dates:
            if rd not in availability.index or rd not in target_rs.index:
                continue
            avail_row = availability.loc[rd]
            pool = avail_row[avail_row].index.tolist()
            if len(pool) < top_n:
                continue
            selected = rng.choice(pool, size=top_n, replace=False).tolist()
            rs_row = target_rs.loc[rd, selected].dropna()
            if len(rs_row) < top_n:
                continue
            port_rs_list.append(float(rs_row.mean()))
            dates_list.append(rd)
        if dates_list:
            out[pid] = pd.Series(port_rs_list, index=pd.DatetimeIndex(dates_list))
    return out