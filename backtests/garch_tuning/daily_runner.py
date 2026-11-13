# backtests/garch_tuning/daily_runner.py
# ============================================================
# GARCH daily rebalance (быстрый, один fit + rolling update).
# ============================================================
import time
import numpy as np
import pandas as pd

from regime_model.garch.config import (
    HORIZON, TARGET_VOL_ANNUAL, MAX_EXPOSURE, MIN_EXPOSURE,
)
from regime_model.garch.garch_model import fit_garch
from regime_model.garch.sizing import vol_to_exposure


def run_garch_daily_fast(port_ret: pd.Series,
                         rebal_dates: list,
                         target_vol: float = TARGET_VOL_ANNUAL,
                         min_train_days: int = 250,
                         verbose: bool = True) -> dict:
    """
    GARCH daily: один fit на первом валидном train, затем rolling update дисперсии.
    """
    t0 = time.time()

    if len(rebal_dates) == 0:
        return {"exposure_daily": pd.Series(), "vol_forecast_daily": pd.Series(),
                "errors": ["no rebal_dates"], "elapsed": 0.0}

    first_rd = pd.Timestamp(min(rebal_dates))
    # индекс первой rebal_date (или ближайшей справа)
    idx_start = port_ret.index.get_indexer([first_rd], method="nearest")[0]
    idx_valid = idx_start + min_train_days

    if idx_valid >= len(port_ret):
        return {"exposure_daily": pd.Series(), "vol_forecast_daily": pd.Series(),
                "errors": [f"not enough days: idx_valid={idx_valid} >= {len(port_ret)}"],
                "elapsed": 0.0}

    first_valid_day = port_ret.index[idx_valid]
    train = port_ret.iloc[:idx_valid].dropna()

    if len(train) < min_train_days:
        return {"exposure_daily": pd.Series(), "vol_forecast_daily": pd.Series(),
                "errors": [f"train too short: {len(train)} < {min_train_days}"],
                "elapsed": 0.0}

    if verbose:
        print(f"  Fit GARCH on initial train ({len(train)} days, до {train.index[-1].date()})...")
    res = fit_garch(train)

    p = res.params
    omega = p.get("omega", 0.0) / (100 ** 2)
    alpha = p.get("alpha[1]", 0.0)
    beta = p.get("beta[1]", 0.0)
    mu = p.get("mu", 0.0) / 100

    # Начальная дисперсия (последняя из train)
    last_var = float(res.conditional_volatility.iloc[-1] ** 2) / (100 ** 2)
    last_ret = float(train.iloc[-1])

    # Дни, где считаем exposure
    all_days = port_ret.index[port_ret.index >= first_valid_day]

    exposure_records = {}
    vol_records = {}
    errors = []

    if verbose:
        print(f"  Rolling forecast on {len(all_days)} days from {first_valid_day.date()}...")

    for d in all_days:
        ret = port_ret.get(d, 0.0)
        if not np.isfinite(ret):
            ret = 0.0

        # Обновление GARCH
        eps = ret - mu
        var = omega + alpha * (eps ** 2) + beta * last_var
        var = max(var, 1e-10)
        last_var = var
        last_ret = ret

        # Annualized vol
        daily_vol = np.sqrt(var)
        vol_ann = daily_vol * np.sqrt(252)

        exp = vol_to_exposure(vol_ann, target_vol=target_vol,
                              max_exp=MAX_EXPOSURE, min_exp=MIN_EXPOSURE)
        exposure_records[d] = exp
        vol_records[d] = vol_ann

    elapsed = time.time() - t0
    if verbose:
        print(f"  Done in {elapsed:.2f}s, {len(exposure_records)} days")

    return {
        "exposure_daily": pd.Series(exposure_records).sort_index(),
        "vol_forecast_daily": pd.Series(vol_records).sort_index(),
        "errors": errors,
        "elapsed": elapsed,
    }


def apply_daily_exposure(port_ret: pd.Series,
                         exposure_daily: pd.Series) -> pd.Series:
    """scaled_ret_t = exposure_t * port_ret_t."""
    return (port_ret * exposure_daily.reindex(port_ret.index).fillna(0.0)).rename("scaled_ret")