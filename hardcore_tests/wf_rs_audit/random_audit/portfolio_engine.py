# random_audit/portfolio_engine.py
# ============================================================
# Независимый портфельный движок.
#
# Логика:
#   - на каждой rebalance date: сгенерировать портфель (list of top_n);
#   - port_rs(rd) = mean(RS[rd, selected]) — средний факт. RS за [rd, rd+20];
#   - equity = cumprod(1 + port_rs) по датам ребаланса;
#   - turnover: one-way = |new - prev| / top_n.
#
# НЕ импортируем существующий бэктест.
# ============================================================
import numpy as np
import pandas as pd

from .config import TOP_N, REBAL_STEP, HORIZON


def get_rebalance_dates(target: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Даты ребаланса:
      - только даты, где target не NaN хотя бы у TOP_N акций;
      - каждая REBAL_STEP-я дата.
    """
    valid = target.notna().sum(axis=1) >= TOP_N
    dates = valid[valid].index
    return dates[::REBAL_STEP]


def run_random_portfolio(rebal_dates: pd.DatetimeIndex,
                         target_rs: pd.DataFrame,
                         pool_getter,
                         selector_fn,
                         top_n: int = TOP_N) -> dict:
    """
    Прогон одного случайного портфеля.

    Аргументы:
      rebal_dates: даты ребаланса
      target_rs:   RS (index=date, cols=ticker)
      pool_getter: fn(rd) -> list допустимых тикеров
      selector_fn: fn(rng, pool, prev) -> list selected

    Возвращает:
      {
        "dates": [...],
        "port_rs": [...],
        "holdings": [list, ...],
        "turnover": [...],
        "equity": [...]
      }
    """
    equity = [1.0]
    dates_out = []
    port_rs_out = []
    holdings_out = []
    turnover_out = []
    prev = []

    for rd in rebal_dates:
        if rd not in target_rs.index:
            continue
        pool = pool_getter(rd)
        if len(pool) < top_n:
            continue

        selected = selector_fn(pool, prev)
        if len(selected) < top_n:
            continue

        rs_row = target_rs.loc[rd, selected].dropna()
        if len(rs_row) < top_n:
            continue

        port_rs = float(rs_row.mean())
        # turnover: one-way = |new - prev| / top_n
        if prev:
            sold = set(prev) - set(selected)
            to = len(sold) / top_n
        else:
            to = 1.0

        equity.append(equity[-1] * (1 + port_rs))
        dates_out.append(rd)
        port_rs_out.append(port_rs)
        holdings_out.append(selected)
        turnover_out.append(to)
        prev = selected

    return {
        "dates": dates_out,
        "port_rs": port_rs_out,
        "holdings": holdings_out,
        "turnover": turnover_out,
        "equity": equity[1:],
    }