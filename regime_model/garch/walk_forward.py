# regime_model/garch/walk_forward.py
# ============================================================
# Walk-Forward GARCH risk sizing.
#
# На каждой дате ребаланса rd:
#   1. train = дневные доходности базового портфеля ДО rd (без look-ahead).
#   2. GARCH fit на train.
#   3. forecast vol на 20 дней (annualized).
#   4. exposure = min(1, target_vol / forecast_vol).
#   5. Применяем exposure к портфелю на период (rd, next_rd].
#
# Возвращает:
#   - exposure_series: pd.Series (index=rebal_dates, values=exposure)
#   - vol_forecast_series: pd.Series (index=rebal_dates, values=forecast_vol)
#   - errors: list[dict] (если GARCH не сошёлся на какой-то дате)
# ============================================================
import time
import traceback

import numpy as np
import pandas as pd

from .config import (
    HORIZON, TARGET_VOL_ANNUAL, MAX_EXPOSURE, MIN_EXPOSURE,
)
from .garch_model import fit_garch, forecast_vol_annualized
from .sizing import vol_to_exposure


def run_garch_wf(port_ret: pd.Series,
                 rebal_dates: list,
                 target_vol: float = TARGET_VOL_ANNUAL,
                 min_train_days: int = 100,
                 verbose: bool = True) -> dict:
    """
    Walk-forward GARCH sizing на датах ребаланса.
    
    port_ret: дневные доходности базового портфеля (100% экспозиция).
    rebal_dates: даты ребаланса RS-модели.
    
    Возвращает:
      {
        "exposure": pd.Series (index=rebal_dates),
        "vol_forecast": pd.Series (index=rebal_dates),
        "errors": list of {"date": ..., "reason": ...},
        "fit_times": list of seconds per fit,
      }
    """
    t0 = time.time()
    exposure_records = {}
    vol_records = {}
    errors = []
    fit_times = []

    rebal_dates = sorted(rebal_dates)
    for i, rd in enumerate(rebal_dates):
        rd = pd.Timestamp(rd)
        train = port_ret.loc[port_ret.index < rd].dropna()

        if len(train) < min_train_days:
            errors.append({
                "date": rd,
                "reason": f"train too short ({len(train)} < {min_train_days})",
            })
            continue

        try:
            t_fit = time.time()
            res = fit_garch(train)
            fit_dt = time.time() - t_fit
            fit_times.append(fit_dt)

            vol_fc = forecast_vol_annualized(res, horizon=HORIZON)
            if not np.isfinite(vol_fc) or vol_fc <= 0:
                errors.append({
                    "date": rd,
                    "reason": f"invalid vol_forecast: {vol_fc}",
                })
                continue

            exp = vol_to_exposure(vol_fc, target_vol=target_vol,
                                  max_exp=MAX_EXPOSURE, min_exp=MIN_EXPOSURE)
            exposure_records[rd] = exp
            vol_records[rd] = vol_fc

            if verbose and (i % 20 == 0 or i == len(rebal_dates) - 1):
                print(f"  [{i+1}/{len(rebal_dates)}] {rd.date()}: "
                      f"train={len(train)}d, vol_fc={vol_fc*100:.2f}%, "
                      f"exp={exp:.3f} ({fit_dt:.2f}s)")

        except Exception as e:
            errors.append({
                "date": rd,
                "reason": f"{type(e).__name__}: {str(e)[:100]}",
            })
            continue

    elapsed = time.time() - t0
    if verbose:
        print(f"\n[garch-wf] done: {len(exposure_records)}/{len(rebal_dates)} "
              f"dates, {len(errors)} errors, {elapsed:.1f}s "
              f"(mean fit {np.mean(fit_times):.2f}s)" if fit_times else "")

    return {
        "exposure": pd.Series(exposure_records).sort_index(),
        "vol_forecast": pd.Series(vol_records).sort_index(),
        "errors": errors,
        "fit_times": fit_times,
        "elapsed": elapsed,
    }


def apply_exposure_to_daily(port_ret: pd.Series,
                            exposure: pd.Series,
                            rebal_dates: list) -> pd.Series:
    """
    Применяет exposure на период (rd, next_rd] к дневным доходностям.
    
    На каждом дне между rd и next_rd (включая next_rd, но не включая rd)
    доходность масштабируется на exposure[rd].
    
    ВАЖНО: на дате rd доходность = 0 (переключение портфеля по close).
    Первый день после rd — первый день с масштабированием.
    """
    rebal_sorted = sorted(rebal_dates)
    scaled = pd.Series(np.nan, index=port_ret.index, name="scaled_ret")

    for i, rd in enumerate(rebal_sorted):
        rd = pd.Timestamp(rd)
        if rd not in exposure.index:
            continue
        exp = exposure.loc[rd]
        if i + 1 < len(rebal_sorted):
            next_rd = pd.Timestamp(rebal_sorted[i + 1])
        else:
            next_rd = port_ret.index[-1]
        # дни: rd < t <= next_rd
        mask = (port_ret.index > rd) & (port_ret.index <= next_rd)
        scaled.loc[mask] = port_ret.loc[mask] * exp

    return scaled