# regime_model/linear/features.py
# ============================================================
# Признаки для линейной модели.
#
# Все признаки — causality-preserving: на дату t используют
# только данные <= t (rolling с min_periods, shift(1) где нужно).
#
# ВАЖНО: признаки строятся на ДНЕВНЫХ доходностях базового
# портфеля (100% экспозиция), не на масштабированных.
# ============================================================
import numpy as np
import pandas as pd


def build_features(port_ret: pd.Series) -> pd.DataFrame:
    """
    port_ret: дневные доходности базового портфеля.
    Возвращает DataFrame с признаками (index=date).
    """
    r = port_ret.dropna()
    abs_r = r.abs()

    feats = {}

    # --- rolling volatility ---
    feats["vol_5"] = r.rolling(5, min_periods=3).std()
    feats["vol_10"] = r.rolling(10, min_periods=5).std()
    feats["vol_20"] = r.rolling(20, min_periods=10).std()
    feats["vol_60"] = r.rolling(60, min_periods=20).std()

    # --- vol ratios ---
    feats["vol_ratio_5_20"] = feats["vol_5"] / (feats["vol_20"] + 1e-9)
    feats["vol_ratio_20_60"] = feats["vol_20"] / (feats["vol_60"] + 1e-9)

    # --- absolute returns ---
    feats["abs_ret_1"] = abs_r
    feats["abs_ret_5"] = abs_r.rolling(5, min_periods=3).mean()

    # --- returns (для направления) ---
    feats["ret_1"] = r
    feats["ret_5"] = r.rolling(5, min_periods=3).mean()

    # --- vol of vol ---
    feats["vol_of_vol_20"] = feats["vol_5"].rolling(20, min_periods=10).std()

    # --- downside vol ---
    neg = r.where(r < 0)
    feats["downside_vol_20"] = (neg ** 2).rolling(20, min_periods=10).mean().pow(0.5)

    # --- hurst proxy: наклон ACF(|ret|) на лагах 1..5 за 60 дней ---
    def _hurst_proxy(x):
        x = x[~np.isnan(x)]
        if len(x) < 30:
            return np.nan
        s = pd.Series(np.abs(x))
        try:
            acf = [s.autocorr(lag=k) for k in range(1, 6)]
            acf = [a for a in acf if np.isfinite(a)]
            if len(acf) < 3:
                return np.nan
            return float(np.mean(acf))
        except Exception:
            return np.nan

    feats["hurst_proxy_60"] = r.rolling(60, min_periods=30).apply(_hurst_proxy, raw=True)

    # --- skew, kurt (60d) ---
    feats["skew_60"] = r.rolling(60, min_periods=30).skew()
    feats["kurt_60"] = r.rolling(60, min_periods=30).kurt()

    df = pd.DataFrame(feats)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_target(port_ret: pd.Series, horizon: int = 20) -> pd.Series:
    """
    Target: будущая realized volatility на 20 дней вперёд (annualized).
    
    На дату t: std(дневных доходностей в (t, t+horizon]) * sqrt(252).
    
    ВАЖНО: target использует БУДУЩИЕ доходности — это НОРМАЛЬНО
    для target (мы предсказываем будущее). Но признаки — только прошлое.
    """
    r = port_ret.dropna()
    # rolling std вперёд: для каждого t — std в (t, t+h]
    fwd_vol = r.rolling(horizon, min_periods=horizon).std().shift(-horizon)
    # annualize
    return fwd_vol * np.sqrt(252)