# features/llm_pool/batch_001_050.py
# ============================================================
# Фичи 001–050.
# Candle anatomy, streaks, semivariance, tails, vol acceleration,
# range positions, drawdown, excursions, path efficiency.
#
# Все функции принимают dict raw (open/high/low/close/volume)
# и возвращают dict name -> wide DataFrame.
# Causality-preserving: только rolling/shift с положительными окнами.
# ============================================================
import numpy as np
import pandas as pd

from .common import EPS, safe_div


def build_batch_001_050(raw: dict) -> dict:
    O = raw["open"]; H = raw["high"]; L = raw["low"]; C = raw["close"]; V = raw["volume"]
    r = C.pct_change()
    eps = EPS

    out = {}

    # --- 001–010: candle anatomy ---
    body = (C - O).abs()
    rng = (H - L) + eps
    upper = (H - pd.concat([O, C], axis=0).groupby(level=0).max())  # H - max(O,C)
    # проще: max(O,C)
    mx_oc = O.where(O > C, C)
    mn_oc = O.where(O < C, C)
    upper_wick = H - mx_oc
    lower_wick = mn_oc - L

    out["body_range_ratio_5"] = (body / rng).rolling(5, min_periods=3).mean()
    out["body_range_ratio_20"] = (body / rng).rolling(20, min_periods=10).mean()
    out["upper_wick_ratio_5"] = (upper_wick / rng).rolling(5, min_periods=3).mean()
    out["lower_wick_ratio_5"] = (lower_wick / rng).rolling(5, min_periods=3).mean()
    out["upper_wick_imbalance_20"] = (
        (upper_wick / rng).rolling(20, min_periods=10).mean()
        - (lower_wick / rng).rolling(20, min_periods=10).mean()
    )

    clv = (C - L) / rng  # 0..1
    out["close_location_std_20"] = clv.rolling(20, min_periods=10).std()
    out["close_location_skew_20"] = clv.rolling(20, min_periods=10).skew()
    out["range_body_ratio_20"] = (
        (H - L).rolling(20, min_periods=10).mean()
        / (body.rolling(20, min_periods=10).mean() + eps)
    )
    out["range_expansion_5_20"] = (
        (H - L).rolling(5, min_periods=3).mean()
        / ((H - L).rolling(20, min_periods=10).mean() + eps)
    )
    out["close_to_high_frequency_20"] = (clv > 0.8).astype(float).rolling(20, min_periods=10).mean()

    # --- 011–020: streak / sign ---
    out["positive_return_fraction_20"] = (r > 0).astype(float).rolling(20, min_periods=10).mean()
    out["negative_return_fraction_20"] = (r < 0).astype(float).rolling(20, min_periods=10).mean()
    out["sign_imbalance_20"] = np.sign(r).rolling(20, min_periods=10).mean()

    def _entropy_sign(x):
        # энтропия знаков в окне (доля положительных vs отрицательных)
        p = (x > 0).mean()
        if p in (0.0, 1.0):
            return 0.0
        return -(p * np.log(p) + (1-p) * np.log(1-p))
    out["return_sign_entropy_20"] = r.rolling(20, min_periods=10).apply(_entropy_sign, raw=False)

    def _max_streak_pos(x):
        m = max_streak = 0
        for v in x:
            if v > 0:
                m += 1
                max_streak = max(max_streak, m)
            else:
                m = 0
        return max_streak
    def _max_streak_neg(x):
        m = max_streak = 0
        for v in x:
            if v < 0:
                m += 1
                max_streak = max(max_streak, m)
            else:
                m = 0
        return max_streak
    out["positive_streak_max_20"] = r.rolling(20, min_periods=10).apply(_max_streak_pos, raw=True)
    out["negative_streak_max_20"] = r.rolling(20, min_periods=10).apply(_max_streak_neg, raw=True)

    sign_r = np.sign(r)
    out["return_reversal_frequency_20"] = (sign_r != sign_r.shift(1)).astype(float).rolling(20, min_periods=10).mean()
    out["directional_consistency_10"] = np.sign(r).rolling(10, min_periods=5).mean().abs()
    out["up_down_day_ratio_20"] = (
        (r > 0).astype(float).rolling(20, min_periods=10).sum()
        / ((r < 0).astype(float).rolling(20, min_periods=10).sum() + eps)
    )
    out["alternation_ratio_20"] = (sign_r != sign_r.shift(1)).astype(float).rolling(20, min_periods=10).mean()

    # --- 021–030: semivariance / tails ---
    pos = r.where(r > 0)
    neg = r.where(r < 0)
    up_sv = (pos ** 2).rolling(20, min_periods=10).mean()
    dn_sv = (neg ** 2).rolling(20, min_periods=10).mean()
    out["upside_semivariance_20"] = up_sv
    out["downside_semivariance_20"] = dn_sv
    out["upside_downside_ratio_20"] = up_sv / (dn_sv + eps)
    out["downside_deviation_20"] = np.sqrt((neg.clip(upper=0) ** 2).rolling(20, min_periods=10).mean())

    def _top20_mean(x):
        x = x[~np.isnan(x)]
        if len(x) < 5: return np.nan
        k = max(1, int(len(x) * 0.2))
        return np.sort(x)[-k:].mean()
    def _bot20_mean(x):
        x = x[~np.isnan(x)]
        if len(x) < 5: return np.nan
        k = max(1, int(len(x) * 0.2))
        return np.sort(x)[:k].mean()
    out["positive_tail_mean_20"] = r.rolling(20, min_periods=10).apply(_top20_mean, raw=True)
    out["negative_tail_mean_20"] = r.rolling(20, min_periods=10).apply(_bot20_mean, raw=True)

    def _tail_asym(x):
        x = x[~np.isnan(x)]
        if len(x) < 10: return np.nan
        q95 = np.quantile(x, 0.95); q05 = np.quantile(x, 0.05)
        return abs(q95) / (abs(q05) + eps)
    out["tail_asymmetry_20"] = r.rolling(20, min_periods=10).apply(_tail_asym, raw=True)

    std20 = r.rolling(20, min_periods=10).std()
    out["large_move_frequency_20"] = (r.abs() > 2 * std20).astype(float).rolling(20, min_periods=10).mean()

    def _pos_large_ratio(x):
        x = x[~np.isnan(x)]
        if len(x) < 10: return np.nan
        s = x.std()
        if s == 0: return np.nan
        ext = x[np.abs(x) > 2 * s]
        if len(ext) == 0: return np.nan
        return (ext > 0).mean()
    out["positive_large_move_ratio_20"] = r.rolling(20, min_periods=10).apply(_pos_large_ratio, raw=True)
    out["negative_large_move_ratio_20"] = 1.0 - out["positive_large_move_ratio_20"]

    # --- 031–040: volatility structure ---
    std5 = r.rolling(5, min_periods=3).std()
    std60 = r.rolling(60, min_periods=20).std()
    out["volatility_acceleration_5_20"] = std5 / (std20 + eps)
    out["volatility_acceleration_20_60"] = std20 / (std60 + eps)
    out["volatility_of_volatility_20"] = std5.rolling(20, min_periods=10).std()
    out["range_volatility_ratio_20"] = (H - L).rolling(20, min_periods=10).std() / ((H - L).rolling(20, min_periods=10).mean() + eps)

    g = O / C.shift(1) - 1   # overnight
    d = C / O - 1            # intraday
    out["intraday_vol_share_20"] = d.rolling(20, min_periods=10).var() / (r.rolling(20, min_periods=10).var() + eps)
    out["overnight_vol_share_20"] = g.rolling(20, min_periods=10).var() / (r.rolling(20, min_periods=10).var() + eps)

    def _roll_corr(x, y, n):
        return x.rolling(n, min_periods=max(3, n//2)).corr(y)
    out["gap_intraday_correlation_20"] = _roll_corr(g, d, 20)

    def _vol_concentration(x):
        x = x[~np.isnan(x)]
        if len(x) < 10: return np.nan
        sq = x ** 2
        k = max(1, int(len(x) * 0.2))
        top = np.sort(sq)[-k:].sum()
        return top / (sq.sum() + eps)
    out["volatility_concentration_20"] = r.rolling(20, min_periods=10).apply(_vol_concentration, raw=True)

    def _cluster_length(x):
        x = x[~np.isnan(x)]
        if len(x) < 10: return np.nan
        s = x.std()
        if s == 0: return np.nan
        mask = np.abs(x) > s
        # средняя длина эпизодов True
        lengths = []
        cur = 0
        for v in mask:
            if v:
                cur += 1
            elif cur > 0:
                lengths.append(cur); cur = 0
        if cur > 0: lengths.append(cur)
        return np.mean(lengths) if lengths else 0.0
    out["volatility_cluster_length_20"] = r.rolling(20, min_periods=10).apply(_cluster_length, raw=True)
    out["range_volatility_divergence_20"] = (H - L).rolling(20, min_periods=10).std() / (std20 + eps)

    # --- 041–050: range positions / drawdown / excursions / path ---
    out["distance_to_high_20"] = C / (H.rolling(20, min_periods=10).max() + eps) - 1
    out["distance_to_low_20"] = C / (L.rolling(20, min_periods=10).min() + eps) - 1
    out["distance_to_high_60"] = C / (H.rolling(60, min_periods=20).max() + eps) - 1
    out["distance_to_low_60"] = C / (L.rolling(60, min_periods=20).min() + eps) - 1
    out["range_position_60"] = (
        (C - L.rolling(60, min_periods=20).min())
        / ((H.rolling(60, min_periods=20).max() - L.rolling(60, min_periods=20).min()) + eps)
    )
    out["range_position_change_5"] = out["range_position_60"] - out["range_position_60"].shift(5)

    # high/low touch frequency: доля close в верхних/нижних 10% локального диапазона
    loc_rng = (H - L).rolling(20, min_periods=10).mean() + eps
    out["high_touch_frequency_20"] = ((H - C) < 0.1 * loc_rng).astype(float).rolling(20, min_periods=10).mean()
    out["low_touch_frequency_20"] = ((C - L) < 0.1 * loc_rng).astype(float).rolling(20, min_periods=10).mean()

    # high rejection: H касается rolling max, но C ниже
    roll_max_20 = H.rolling(20, min_periods=10).max()
    out["high_rejection_ratio_20"] = ((H >= roll_max_20) & (C < H)).astype(float).rolling(20, min_periods=10).mean()
    roll_min_20 = L.rolling(20, min_periods=10).min()
    out["low_rejection_ratio_20"] = ((L <= roll_min_20) & (C > L)).astype(float).rolling(20, min_periods=10).mean()

    # drawdown
    roll_max_close_20 = C.rolling(20, min_periods=10).max()
    dd = C / roll_max_close_20 - 1
    out["drawdown_depth_20"] = dd
    # drawdown duration: сколько дней подряд C < rolling max
    under = (C < roll_max_close_20).astype(int)
    def _duration(x):
        # длина текущего эпизода 1s в конце окна
        l = 0
        for v in x[::-1]:
            if v == 1: l += 1
            else: break
        return l
    out["drawdown_duration_20"] = under.rolling(20, min_periods=10).apply(_duration, raw=True)

    # excursions (упрощённо)
    out["upside_excursion_20"] = (H - C.shift(1)).clip(lower=0).rolling(20, min_periods=10).mean()
    out["downside_excursion_20"] = (C.shift(1) - L).clip(lower=0).rolling(20, min_periods=10).mean()
    out["excursion_asymmetry_20"] = out["upside_excursion_20"] / (out["downside_excursion_20"] + eps)

    # path efficiency
    abs_sum_5 = r.abs().rolling(5, min_periods=3).sum()
    net_5 = (C / C.shift(5) - 1).abs()
    out["path_efficiency_5"] = net_5 / (abs_sum_5 + eps)

    return out