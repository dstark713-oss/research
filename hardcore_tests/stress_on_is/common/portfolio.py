# hardcore_tests/stress_on_is/common/portfolio.py
import numpy as np
import pandas as pd


def compute_turnover(prev: list, curr: list, top_n: int = 10) -> float:
    """One-way turnover: |prev - curr| / top_n."""
    if not prev:
        return 1.0
    sold = set(prev) - set(curr)
    return len(sold) / top_n


def apply_costs_at_rebal(daily_ret: pd.Series,
                          rebal_dates: list,
                          holdings_by_date: dict,
                          cost_bps_round_trip: float,
                          top_n: int = 10) -> pd.Series:
    """
    Применяет издержки ТОЛЬКО в даты ребаланса, на фактический turnover.
    """
    net = daily_ret.copy()
    prev = []
    for rd in sorted(rebal_dates):
        if rd not in holdings_by_date:
            continue
        curr = holdings_by_date[rd]
        to = compute_turnover(prev, curr, top_n)
        cost = to * cost_bps_round_trip / 10000.0
        if rd in net.index:
            net.loc[rd] -= cost
        prev = curr
    return net


def apply_exposure(daily_ret: pd.Series, exposure: pd.Series) -> pd.Series:
    return (daily_ret * exposure.reindex(daily_ret.index).fillna(1.0)).rename("scaled_ret")