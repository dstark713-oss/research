# hardcore_tests/stress_on_is/common/costs.py
import pandas as pd
from ..config import SBER_COMMISSION_BPS, SLIPPAGE_BPS

def round_trip_cost_bps(commission_bps=SBER_COMMISSION_BPS, slippage_bps=SLIPPAGE_BPS) -> float:
    return 2 * (commission_bps + slippage_bps)

def apply_cost(daily_ret: pd.Series, cost_bps_round_trip: float) -> pd.Series:
    cost = cost_bps_round_trip / 10000.0
    return (daily_ret - cost).rename("net_ret")

def extra_cost_bps(extra_bps_round_trip: float) -> float:
    return extra_bps_round_trip / 10000.0