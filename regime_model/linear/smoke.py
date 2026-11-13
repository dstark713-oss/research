# regime_model/linear/smoke.py
# ============================================================
# Smoke: fit Ridge на train, predict на val, exposure.
# ============================================================
import time

import numpy as np
import pandas as pd

from .config import HORIZON, TARGET_VOL_ANNUAL, SMOKE_TRAIN_FRAC, RIDGE_ALPHA
from regime_model.garch.data_loader import (
    load_split_date, load_stock_close, load_ml_holdings,
)
from regime_model.garch.portfolio_returns import build_daily_returns
from .features import build_features, build_target
from .linear_model import fit_ridge, predict_ridge, model_coefs
from .sizing import vol_to_exposure, apply_exposure_to_daily


def smoke(verbose: bool = True) -> dict:
    t0 = time.time()
    split_date = load_split_date()
    if verbose:
        print(f"[linear-smoke] split_date = {split_date}")

    # --- 1. данные ---
    if verbose:
        print("[linear-smoke] загрузка данных...")
    close = load_stock_close(split_date)
    holdings = load_ml_holdings()
    port_ret = build_daily_returns(close, holdings)
    if verbose:
        print(f"    port_ret: {len(port_ret)} дней, "
              f"{port_ret.index.min().date()} → {port_ret.index.max().date()}")

    # --- 2. признаки и target ---
    if verbose:
        print("[linear-smoke] построение признаков...")
    X = build_features(port_ret)
    y = build_target(port_ret, horizon=HORIZON)
    if verbose:
        print(f"    X: {X.shape}, y: non-NaN = {y.notna().sum()}")

    # --- 3. split train/val ---
    n = len(port_ret)
    train_end = int(n * SMOKE_TRAIN_FRAC)
    train_dates = port_ret.index[:train_end]
    val_dates = port_ret.index[train_end:]
    X_train = X.loc[X.index.isin(train_dates)]
    y_train = y.loc[y.index.isin(train_dates)]
    X_val = X.loc[X.index.isin(val_dates)]
    y_val = y.loc[y.index.isin(val_dates)]
    if verbose:
        print(f"[linear-smoke] train: {len(X_train)} дней, "
              f"val: {len(X_val)} дней")

    # --- 4. fit ---
    if verbose:
        print(f"[linear-smoke] fit Ridge(alpha={RIDGE_ALPHA})...")
    model = fit_ridge(X_train, y_train, alpha=RIDGE_ALPHA)
    coefs = model_coefs(model, X.columns.tolist())
    if verbose:
        print("    coefficients:")
        for name, coef in coefs.items():
            print(f"      {name:<20} = {coef:+.6f}")

    # --- 5. predict ---
    pred_val = predict_ridge(model, X_val)
    if verbose:
        print(f"[linear-smoke] pred_val: non-NaN = {pred_val.notna().sum()}, "
              f"mean = {pred_val.mean()*100:.2f}%, "
              f"min = {pred_val.min()*100:.2f}%, "
              f"max = {pred_val.max()*100:.2f}%")

    # --- 6. сравнение с target ---
    common = pred_val.dropna().index.intersection(y_val.dropna().index)
    if len(common) > 10:
        err = (pred_val.loc[common] - y_val.loc[common]).abs()
        if verbose:
            print(f"    vs actual vol: MAE = {err.mean()*100:.2f}%, "
                  f"median = {err.median()*100:.2f}%")
        # корреляция pred vs actual
        corr = pred_val.loc[common].corr(y_val.loc[common])
        if verbose:
            print(f"    corr(pred, actual) = {corr:+.4f}")

    # --- 7. последний прогноз → exposure ---
    last_pred = pred_val.dropna()
    if len(last_pred) == 0:
        raise RuntimeError("Нет прогнозов на val")
    last_vol = float(last_pred.iloc[-1])
    exposure = vol_to_exposure(last_vol)
    if verbose:
        print(f"[linear-smoke] last forecast vol = {last_vol*100:.2f}%, "
              f"exposure = {exposure:.4f}")

    # --- 8. проверки ---
    checks = {
        "exposure_in_0_1": 0.0 <= exposure <= 1.0,
        "vol_positive": last_vol > 0,
        "pred_not_all_nan": pred_val.notna().sum() > 0,
    }
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise RuntimeError(f"Smoke checks failed: {failed}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n[linear-smoke] все проверки OK за {elapsed:.1f} сек")

    return {
        "model": model,
        "coefs": coefs,
        "X": X,
        "y": y,
        "pred_val": pred_val,
        "vol_forecast": last_vol,
        "exposure": exposure,
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    smoke()