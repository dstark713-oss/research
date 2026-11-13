# regime_model/regime_based/sizing.py
# ============================================================
# Exposure на основе режима (PC1, PC2, Hurst, cluster).
#
# Правило (на основе EDA):
#   - cluster 1 (кризисный): базовый множитель 0.5
#   - cluster 0 (спокойный): 1.0
#   - cluster 2 (трендовый): 1.0 (или 1.1, но max 1.0)
#   - cluster 3 (взрывной): 1.0
#   - PC1 высокий (кризис): дополнительно снижаем
#   - PC2 высокий (тренд): повышаем (в пределах max)
#   - Hurst растёт: снижаем (ранний сигнал)
#
# Итог: exposure = clip(base * f(PC1) * f(PC2) * f(Hurst), min, max)
# ============================================================
import numpy as np
import pandas as pd

TARGET_VOL_ANNUAL = 0.15
MAX_EXPOSURE = 1.0
MIN_EXPOSURE = 0.2
VOL_FLOOR = 0.01


def vol_to_exposure(vol_annual: float, target_vol: float = TARGET_VOL_ANNUAL) -> float:
    if not np.isfinite(vol_annual) or vol_annual <= 0:
        return MIN_EXPOSURE
    return float(np.clip(target_vol / max(vol_annual, VOL_FLOOR), MIN_EXPOSURE, MAX_EXPOSURE))


def regime_multiplier(pc1: float, pc2: float, hurst: float, cluster: int) -> float:
    """
    Множитель экспозиции по режиму.
    
    PC1 (кризисность): высокий → снижаем.
    PC2 (трендовость): высокий → повышаем (в пределах max).
    Hurst: высокий → снижаем (ранний сигнал кризиса).
    cluster 1 (кризис) → сильное снижение.
    """
    m = 1.0
    # кластер
    if cluster == 1:  # кризисный
        m *= 0.6
    elif cluster == 3:  # взрывной рост
        m *= 0.9  # чуть снижаем (взрывной рост → волатильность)
    # PC1: если > 1 (высокая кризисность) — снижаем
    if np.isfinite(pc1) and pc1 > 1.0:
        m *= max(0.5, 1.0 - 0.2 * (pc1 - 1.0))
    # PC2: если > 1 (сильный тренд) — повышаем
    if np.isfinite(pc2) and pc2 > 1.0:
        m *= min(1.2, 1.0 + 0.1 * (pc2 - 1.0))
    # Hurst: если > 0.65 — ранний сигнал, снижаем
    if np.isfinite(hurst) and hurst > 0.65:
        m *= max(0.6, 1.0 - 0.3 * (hurst - 0.65))
    return float(np.clip(m, 0.3, 1.2))


def compute_exposure_series(state: pd.DataFrame,
                             port_ret: pd.Series,
                             rebal_dates: list,
                             window_vol: int = 20) -> pd.Series:
    """
    Exposure на датах ребаланса:
      1. realized vol портфеля за window_vol дней (annualized).
      2. vol_to_exposure (target / vol).
      3. умножить на regime_multiplier(PC1, PC2, Hurst, cluster) 
         (state на ближайшую дату <= rd).
      4. clip [MIN, MAX].
    """
    rv = port_ret.rolling(window_vol, min_periods=window_vol).std() * np.sqrt(252)
    out = {}
    for rd in sorted(rebal_dates):
        rd = pd.Timestamp(rd)
        # vol до rd
        rv_before = rv.loc[rv.index < rd].dropna()
        if len(rv_before) == 0:
            continue
        vol = float(rv_before.iloc[-1])
        base_exp = vol_to_exposure(vol)
        # state на ближайшую дату <= rd
        state_before = state.loc[state.index <= rd]
        if len(state_before) == 0:
            out[rd] = base_exp
            continue
        st = state_before.iloc[-1]
        mult = regime_multiplier(st["PC1"], st["PC2"], st["hurst"], int(st["cluster"]))
        exp = float(np.clip(base_exp * mult, MIN_EXPOSURE, MAX_EXPOSURE))
        out[rd] = exp
    return pd.Series(out).sort_index()


def apply_exposure_to_daily(port_ret, exposure, rebal_dates):
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