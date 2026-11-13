# regime_model/target_vol/sizing.py
# ============================================================
# Тупая Target Volatility:
#   exposure = min(1, target_vol / realized_vol_20d)
#
# realized_vol_20d: std дневных доходностей за ПРОШЛЫЕ 20 дней,
# annualized. Без прогноза, без ML.
# ============================================================
import numpy as np
import pandas as pd

from .config import (
    TARGET_VOL_ANNUAL, MAX_EXPOSURE, MIN_EXPOSURE, VOL_FLOOR,
    REALIZED_VOL_WINDOW, TRADING_DAYS,
)


def realized_vol_annualized(port_ret: pd.Series,
                             window: int = REALIZED_VOL_WINDOW) -> pd.Series:
    """
    Реализованная волатильность (annualized) за прошлые `window` дней.
    На дату t: std(port_ret[t-window:t]) * sqrt(252).
    
    ВАЖНО: rolling с min_periods=window — causality-preserving
    (rolling по прошлым данным, без shift(-n)).
    """
    rv = port_ret.rolling(window, min_periods=window).std()
    return rv * np.sqrt(TRADING_DAYS)


def vol_to_exposure(vol_annual: float,
                    target_vol: float = TARGET_VOL_ANNUAL,
                    max_exp: float = MAX_EXPOSURE,
                    min_exp: float = MIN_EXPOSURE,
                    vol_floor: float = VOL_FLOOR) -> float:
    """exposure = clip(target / max(vol, floor), min, max)."""
    if not np.isfinite(vol_annual) or vol_annual <= 0:
        return min_exp
    v = max(vol_annual, vol_floor)
    exp = target_vol / v
    return float(np.clip(exp, min_exp, max_exp))


def compute_exposure_series(port_ret: pd.Series,
                             rebal_dates: list,
                             target_vol: float = TARGET_VOL_ANNUAL,
                             window: int = REALIZED_VOL_WINDOW) -> pd.Series:
    """
    Exposure на датах ребаланса.
    На каждой rd: rv = realized_vol до rd (не включая rd),
    exposure = min(1, target_vol / rv).
    
    Возвращает: pd.Series (index=rebal_dates, values=exposure).
    """
    rv = realized_vol_annualized(port_ret, window=window)
    out = {}
    for rd in rebal_dates:
        rd = pd.Timestamp(rd)
        # vol до rd (не включая rd — causality)
        rv_before = rv.loc[rv.index < rd].dropna()
        if len(rv_before) == 0:
            continue
        vol = float(rv_before.iloc[-1])
        out[rd] = vol_to_exposure(vol, target_vol=target_vol)
    return pd.Series(out).sort_index()


def apply_exposure_to_daily(port_ret: pd.Series,
                            exposure: pd.Series,
                            rebal_dates: list) -> pd.Series:
    """Применяет exposure на период (rd, next_rd]."""
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