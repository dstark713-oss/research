# features/llm_pool/batch_051_100.py
# ============================================================
# Фичи 051–100.
# Drawdown, path efficiency, momentum shape, gap/intraday,
# volume, tails, beta asymmetries.
#
# Плюс 5 фич из 285+ (52w-high family) — потому что dist_from_high
# лучшая по EDA, а в 001–050 её нет.
# ============================================================
import numpy as np
import pandas as pd

from .common import EPS, safe_div


def build_batch_051_100(raw: dict, imoex: pd.Series = None) -> dict:
    O = raw["open"]; H = raw["high"]; L = raw["low"]; C = raw["close"]; V = raw["volume"]
    r = C.pct_change()
    eps = EPS

    out = {}

    # --- 051–060: drawdown / recovery / underwater ---
    roll_max_20 = C.rolling(20, min_periods=10).max()
    roll_max_60 = C.rolling(60, min_periods=20).max()
    dd_20 = C / roll_max_20 - 1
    dd_60 = C / roll_max_60 - 1

    out["drawdown_depth_20"] = dd_20
    out["drawdown_depth_60"] = dd_60
    out["underwater_fraction_60"] = (C < roll_max_60).astype(float).rolling(60, min_periods=20).mean()
    out["drawdown_frequency_60"] = (dd_60 < -0.05).astype(float).rolling(60, min_periods=20).mean()

    # recovery speed: mean recovery after local min
    def _recovery_speed(x):
        x = x[~np.isnan(x)]
        if len(x) < 10: return np.nan
        # расстояние от минимума до текущего значения, делённое на дни
        mn = x.min()
        if mn == 0: return np.nan
        return (x[-1] - mn) / mn / len(x)
    out["recovery_speed_60"] = C.rolling(60, min_periods=20).apply(_recovery_speed, raw=True)

    # max recovery: max rise from local min
    def _max_recovery(x):
        x = x[~np.isnan(x)]
        if len(x) < 10: return np.nan
        mn = x.min()
        if mn == 0: return np.nan
        return (x.max() - mn) / mn
    out["max_recovery_20"] = C.rolling(20, min_periods=10).apply(_max_recovery, raw=True)

    # upside/downside excursion (уже есть в 001–050, но здесь добавим 60d)
    out["upside_excursion_60"] = (H - C.shift(1)).clip(lower=0).rolling(60, min_periods=20).mean()
    out["downside_excursion_60"] = (C.shift(1) - L).clip(lower=0).rolling(60, min_periods=20).mean()
    out["excursion_asymmetry_60"] = out["upside_excursion_60"] / (out["downside_excursion_60"] + eps)

    # --- 061–070: path efficiency / trend quality ---
    abs_sum_20 = r.abs().rolling(20, min_periods=10).sum()
    abs_sum_60 = r.abs().rolling(60, min_periods=20).sum()
    net_20 = (C / C.shift(20) - 1).abs()
    net_60 = (C / C.shift(60) - 1).abs()

    out["path_efficiency_20"] = net_20 / (abs_sum_20 + eps)
    out["path_efficiency_60"] = net_60 / (abs_sum_60 + eps)
    out["efficiency_scale_ratio"] = out["path_efficiency_5"] if "path_efficiency_5" in out else net_20 / (abs_sum_20 + eps)  # fallback
    out["path_length_20"] = abs_sum_20
    out["net_gross_movement_20"] = (r.rolling(20, min_periods=10).sum().abs()) / (abs_sum_20 + eps)
    out["range_path_ratio_20"] = (C - C.shift(20)).abs() / ((H - L).rolling(20, min_periods=10).sum() + eps)

    # fractal dimension proxy: Hurst-like
    def _fractal_proxy(x):
        x = x[~np.isnan(x)]
        if len(x) < 10: return np.nan
        # простой proxy: std(diff)/std(x)
        return np.std(np.diff(x)) / (np.std(x) + eps)
    out["fractal_dimension_proxy_20"] = C.rolling(20, min_periods=10).apply(_fractal_proxy, raw=True)

    # path curvature: mean abs direction change
    out["path_curvature_20"] = r.diff().abs().rolling(20, min_periods=10).mean()

    # local trend consistency: sign consistency inside 5d blocks
    def _trend_consistency(x):
        x = x[~np.isnan(x)]
        if len(x) < 10: return np.nan
        blocks = [x[i:i+5] for i in range(0, len(x), 5) if len(x[i:i+5]) == 5]
        if not blocks: return np.nan
        signs = []
        for b in blocks:
            s = np.sign(b.sum())
            if s != 0: signs.append(s)
        if not signs: return np.nan
        return abs(sum(signs)) / len(signs)
    out["local_trend_consistency_20"] = r.rolling(20, min_periods=10).apply(_trend_consistency, raw=True)

    # --- 071–080: autocorrelation / mean reversion ---
    def _acf(x, lag):
        x = x[~np.isnan(x)]
        if len(x) < lag + 5: return np.nan
        return pd.Series(x).autocorr(lag=lag)
    def _acf_abs(x, lag):
        x = np.abs(x[~np.isnan(x)])
        if len(x) < lag + 5: return np.nan
        return pd.Series(x).autocorr(lag=lag)

    out["return_autocorr_2_20"] = r.rolling(20, min_periods=10).apply(lambda x: _acf(x, 2), raw=True)
    out["return_autocorr_5_60"] = r.rolling(60, min_periods=20).apply(lambda x: _acf(x, 5), raw=True)
    out["return_autocorr_10_60"] = r.rolling(60, min_periods=20).apply(lambda x: _acf(x, 10), raw=True)
    out["abs_return_autocorr_1_20"] = r.rolling(20, min_periods=10).apply(lambda x: _acf_abs(x, 1), raw=True)
    out["abs_return_autocorr_5_60"] = r.rolling(60, min_periods=20).apply(lambda x: _acf_abs(x, 5), raw=True)

    # variance ratio
    def _vr(x, q):
        x = x[~np.isnan(x)]
        if len(x) < q + 1: return np.nan
        v1 = np.var(x, ddof=1)
        xq = pd.Series(x).rolling(q).sum().dropna().values
        if len(xq) < 2: return np.nan
        vq = np.var(xq, ddof=1) / q
        return vq / v1 if v1 > 0 else np.nan
    out["variance_ratio_5_20"] = r.rolling(20, min_periods=10).apply(lambda x: _vr(x, 5), raw=True)
    out["variance_ratio_10_60"] = r.rolling(60, min_periods=20).apply(lambda x: _vr(x, 10), raw=True)

    # mean reversion strength: -corr(r_t, sum r_{t-1..t-5})
    out["mean_reversion_strength_20"] = -r.rolling(20, min_periods=10).corr(
        r.rolling(5, min_periods=3).sum().shift(1)
    )
    out["return_shock_reversal_20"] = -r.rolling(20, min_periods=10).corr(r.shift(1))
    out["price_deviation_reversion_20"] = -(C / C.rolling(20, min_periods=10).mean() - 1)

    # --- 081–100: gap / intraday / volume ---
    g = O / C.shift(1) - 1
    d = C / O - 1

    out["gap_mean_20"] = g.rolling(20, min_periods=10).mean()
    out["gap_skew_20"] = g.rolling(20, min_periods=10).skew()
    out["gap_reversal_20"] = g.rolling(20, min_periods=10).corr(d)
    out["gap_followthrough_20"] = g.rolling(20, min_periods=10).corr(r)
    out["intraday_return_skew_20"] = d.rolling(20, min_periods=10).skew()
    out["intraday_return_autocorr_1_20"] = d.rolling(20, min_periods=10).apply(lambda x: _acf(x, 1), raw=True)
    out["overnight_intraday_divergence_20"] = g.rolling(20, min_periods=10).mean() - d.rolling(20, min_periods=10).mean()
    out["gap_absorption_ratio_20"] = (np.sign(g) != np.sign(d)).astype(float).rolling(20, min_periods=10).mean()
    out["intraday_range_efficiency_20"] = (d.abs() / ((H - L) / C + eps)).rolling(20, min_periods=10).mean()
    out["close_open_direction_agreement_20"] = (np.sign(g) == np.sign(d)).astype(float).rolling(20, min_periods=10).mean()

    # volume
    med_v20 = V.rolling(20, min_periods=10).median()
    out["volume_surprise_20"] = np.log((V + 1) / (med_v20 + 1))

    def _vol_concentration(x):
        x = x[~np.isnan(x)]
        if len(x) < 10: return np.nan
        k = max(1, int(len(x) * 0.2))
        top = np.sort(x)[-k:].sum()
        return top / (x.sum() + eps)
    out["volume_concentration_20"] = V.rolling(20, min_periods=10).apply(_vol_concentration, raw=True)
    out["volume_return_correlation_20"] = V.rolling(20, min_periods=10).corr(r)
    out["volume_abs_return_correlation_20"] = V.rolling(20, min_periods=10).corr(r.abs())

    vol_up = V.where(r > 0).rolling(20, min_periods=10).sum()
    vol_dn = V.where(r < 0).rolling(20, min_periods=10).sum()
    out["volume_up_down_imbalance_20"] = vol_up / (vol_dn + eps)

    out["volume_price_efficiency_20"] = (r.rolling(20, min_periods=10).sum().abs()) / ((V * r.abs()).rolling(20, min_periods=10).sum() + eps)
    out["volume_range_correlation_20"] = V.rolling(20, min_periods=10).corr(H - L)

    med_v60 = V.rolling(60, min_periods=20).median()
    out["volume_shock_frequency_60"] = (V > 2 * med_v60).astype(float).rolling(60, min_periods=20).mean()

    # volume-weighted CLV
    clv = (C - L) / ((H - L) + eps)
    out["volume_weighted_close_location_20"] = (clv * V).rolling(20, min_periods=10).sum() / (V.rolling(20, min_periods=10).sum() + eps)
    out["volume_directional_persistence_20"] = (np.sign(r) * V).rolling(20, min_periods=10).apply(lambda x: _acf(x, 1), raw=True)

    # --- 285+ из расширенного пула (52w-high family) ---
    # dist_from_high_252 — лучшая по EDA, в 001-050 её не было
    roll_max_252 = C.rolling(252, min_periods=100).max()
    roll_min_252 = C.rolling(252, min_periods=100).min()
    out["dist_from_high_252"] = C / roll_max_252 - 1
    out["dist_from_low_252"] = C / roll_min_252 - 1
    out["price_position_252"] = (C - roll_min_252) / (roll_max_252 - roll_min_252 + eps)
    out["dist_from_high_120"] = C / C.rolling(120, min_periods=60).max() - 1
    out["dist_from_high_intraday_252"] = C / H.rolling(252, min_periods=100).max() - 1

    return out