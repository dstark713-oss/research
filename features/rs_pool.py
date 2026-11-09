# features/rs_pool.py
# ============================================================
# RS Pool v2: 11 фич после Feature Pool Stress Test v1.
#
# Убраны 5 фич:
#   - recovery_speed_60 (дубликат distance_to_low_60, corr 0.997)
#   - drawdown_depth_60 (дубликат distance_to_high_60, corr 0.923)
#   - volume_abs_return_correlation_20 (нестабильна + concentration 581%)
#   - max_recovery_20 (corr 0.78-0.84 с semivariance)
#   - path_curvature_20 (corr 0.71-0.75 с semivariance, нестабильна)
#
# Все фичи — causality-preserving (rolling/shift только с
# положительными окнами, min_periods <= окно).
#
# Использование:
#   from features.rs_pool import build_rs_pool, RS_POOL_NAMES
#   feats = build_rs_pool(raw)  # raw = dict(open/high/low/close/volume)
#
# Возвращает dict[name, pd.DataFrame] (index=date, columns=tickers).
# ============================================================
import numpy as np
import pandas as pd

EPS = 1e-9

# имена фич (11)
RS_POOL_NAMES = [
    "upside_semivariance_20",
    "downside_semivariance_20",
    "distance_to_low_60",
    "range_volatility_ratio_20",
    "distance_to_high_60",
    "range_position_change_5",
    "negative_return_fraction_20",
    "up_down_day_ratio_20",
    "dist_from_high_intraday_252",
    "intraday_return_skew_20",
    "return_autocorr_2_20",
]


def build_rs_pool(raw: dict) -> dict:
    """
    raw: dict с ключами 'open', 'high', 'low', 'close', 'volume'
         (wide DataFrame: index=date, columns=tickers).
    Возвращает dict[name, DataFrame] для 11 фич.
    """
    O = raw["open"]; H = raw["high"]; L = raw["low"]; C = raw["close"]; V = raw["volume"]
    r = C.pct_change()
    eps = EPS

    out = {}

    # --- 1. upside_semivariance_20 ---
    pos = r.where(r > 0)
    out["upside_semivariance_20"] = (pos ** 2).rolling(20, min_periods=10).mean()

    # --- 2. downside_semivariance_20 ---
    neg = r.where(r < 0)
    out["downside_semivariance_20"] = (neg ** 2).rolling(20, min_periods=10).mean()

    # --- 3. distance_to_low_60 ---
    out["distance_to_low_60"] = C / (L.rolling(60, min_periods=20).min() + eps) - 1

    # --- 4. range_volatility_ratio_20 ---
    out["range_volatility_ratio_20"] = (
        (H - L).rolling(20, min_periods=10).std()
        / ((H - L).rolling(20, min_periods=10).mean() + eps)
    )

    # --- 5. distance_to_high_60 ---
    out["distance_to_high_60"] = C / (H.rolling(60, min_periods=20).max() + eps) - 1

    # --- 6. range_position_change_5 ---
    roll_max_60 = H.rolling(60, min_periods=20).max()
    roll_min_60 = L.rolling(60, min_periods=20).min()
    range_pos_60 = (C - roll_min_60) / (roll_max_60 - roll_min_60 + eps)
    out["range_position_change_5"] = range_pos_60 - range_pos_60.shift(5)

    # --- 7. negative_return_fraction_20 ---
    out["negative_return_fraction_20"] = (
        (r < 0).astype(float).rolling(20, min_periods=10).mean()
    )

    # --- 8. up_down_day_ratio_20 ---
    out["up_down_day_ratio_20"] = (
        (r > 0).astype(float).rolling(20, min_periods=10).sum()
        / ((r < 0).astype(float).rolling(20, min_periods=10).sum() + eps)
    )

    # --- 9. dist_from_high_intraday_252 ---
    out["dist_from_high_intraday_252"] = (
        C / (H.rolling(252, min_periods=100).max() + eps) - 1
    )

    # --- 10. intraday_return_skew_20 ---
    d = C / O - 1
    out["intraday_return_skew_20"] = d.rolling(20, min_periods=10).skew()

    # --- 11. return_autocorr_2_20 ---
    def _acf(x, lag):
        x = x[~np.isnan(x)]
        if len(x) < lag + 5: return np.nan
        return pd.Series(x).autocorr(lag=lag)
    out["return_autocorr_2_20"] = r.rolling(20, min_periods=10).apply(
        lambda x: _acf(x, 2), raw=True
    )

    # проверка: все 11 на месте
    missing = [n for n in RS_POOL_NAMES if n not in out]
    if missing:
        raise RuntimeError(f"rs_pool: не построены фичи: {missing}")

    return out


if __name__ == "__main__":
    # быстрая проверка на синтетике
    rng = np.random.RandomState(42)
    n_dates, n_tickers = 500, 20
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    cols = [f"T{i}" for i in range(n_tickers)]
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.randn(n_dates, n_tickers) * 0.01, axis=0)),
        index=dates, columns=cols,
    )
    raw = {
        "open": close.shift(1).fillna(close),
        "high": close * (1 + rng.rand(n_dates, n_tickers) * 0.01),
        "low": close * (1 - rng.rand(n_dates, n_tickers) * 0.01),
        "close": close,
        "volume": pd.DataFrame(
            rng.randint(1000, 10000, (n_dates, n_tickers)),
            index=dates, columns=cols,
        ).astype(float),
    }
    feats = build_rs_pool(raw)
    print(f"построено {len(feats)} фич")
    for name in RS_POOL_NAMES:
        f = feats[name]
        n_valid = f.notna().sum().sum()
        print(f"  {name:<35} shape={f.shape}  valid={n_valid}")