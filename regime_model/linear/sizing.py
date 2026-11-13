# regime_model/linear/sizing.py
# ============================================================
# Преобразование forecast vol в exposure.
# Идентично GARCH — для честного сравнения.
# ============================================================
import numpy as np

from .config import (
    TARGET_VOL_ANNUAL, MAX_EXPOSURE, MIN_EXPOSURE, VOL_FLOOR,
)


def vol_to_exposure(vol_annual: float,
                    target_vol: float = TARGET_VOL_ANNUAL,
                    max_exp: float = MAX_EXPOSURE,
                    min_exp: float = MIN_EXPOSURE,
                    vol_floor: float = VOL_FLOOR) -> float:
    if not np.isfinite(vol_annual) or vol_annual <= 0:
        return min_exp
    v = max(vol_annual, vol_floor)
    exp = target_vol / v
    return float(np.clip(exp, min_exp, max_exp))


def apply_exposure_to_daily(port_ret, exposure, rebal_dates):
    """Применяет exposure на период (rd, next_rd]."""
    import pandas as pd
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
        mask = (port_ret.index > rd) & (port_ret.index <= next_rd)
        scaled.loc[mask] = port_ret.loc[mask] * exp
    return scaled