# риск_модели/гарч/sizing.py
# ============================================================
# Преобразование прогнозной волатильности в экспозицию.
#
# exposure = min(MAX_EXPOSURE, TARGET_VOL / max(vol_forecast, VOL_FLOOR))
# Без плеча: exposure <= 1.0
# ============================================================
import numpy as np
import pandas as pd

from .config import (
    TARGET_VOL_ANNUAL, MAX_EXPOSURE, MIN_EXPOSURE, VOL_FLOOR,
)


def vol_to_exposure(vol_annual: float,
                    target_vol: float = TARGET_VOL_ANNUAL,
                    max_exp: float = MAX_EXPOSURE,
                    min_exp: float = MIN_EXPOSURE,
                    vol_floor: float = VOL_FLOOR) -> float:
    """
    exposure = clip(target_vol / max(vol_annual, vol_floor), min_exp, max_exp).
    - высокий vol → низкая exposure;
    - низкий vol → высокая exposure (но <= max_exp).
    """
    if not np.isfinite(vol_annual) or vol_annual <= 0:
        return min_exp
    v = max(vol_annual, vol_floor)
    exp = target_vol / v
    return float(np.clip(exp, min_exp, max_exp))


def apply_exposure(daily_returns: pd.Series,
                   exposure: float) -> pd.Series:
    """
    Применяет постоянную экспозицию к дневным доходностям.
    scaled_ret = exposure * ret.
    """
    return (exposure * daily_returns).rename("scaled_ret")