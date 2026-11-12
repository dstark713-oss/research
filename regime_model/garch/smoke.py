# риск_модели/гарч/smoke.py
# ============================================================
# Smoke: 
#   1. восстановить дневные доходности базового портфеля
#   2. fit GARCH на train (первые 60%)
#   3. forecast vol на 20 дней
#   4. exposure
#   5. применить к val (последние 40%) — только для проверки формы
# Цель — убедиться, что модель тянется и даёт вменяемые числа.
# ============================================================
import time

import numpy as np
import pandas as pd

from .config import HORIZON, TARGET_VOL_ANNUAL, SMOKE_TRAIN_FRAC
from .data_loader import load_split_date, load_stock_close, load_ml_holdings, load_ml_equity
from .portfolio_returns import build_daily_returns
from .garch_model import fit_garch, forecast_vol_annualized, garch_persistence, conditional_vol_daily
from .sizing import vol_to_exposure, apply_exposure


def smoke(verbose: bool = True) -> dict:
    t0 = time.time()
    split_date = load_split_date()
    if verbose:
        print(f"[garch-smoke] split_date = {split_date}")

    # --- 1. данные ---
    if verbose:
        print("[garch-smoke] загрузка close + holdings...")
    close = load_stock_close(split_date)
    holdings = load_ml_holdings()
    if verbose:
        print(f"    close: {close.shape}, holdings: {len(holdings)} rebalances, "
              f"{min(holdings.keys()).date()} → {max(holdings.keys()).date()}")

    # --- 2. дневные доходности базового портфеля ---
    if verbose:
        print("[garch-smoke] восстановление дневных доходностей портфеля...")
    port_ret = build_daily_returns(close, holdings)
    if verbose:
        print(f"    port_ret: {len(port_ret)} дней, "
              f"{port_ret.index.min().date()} → {port_ret.index.max().date()}")
        print(f"    mean daily = {port_ret.mean()*100:+.4f}%, "
              f"std daily = {port_ret.std()*100:.4f}%")

    # --- 3. split train/val ---
    n = len(port_ret)
    train_end = int(n * SMOKE_TRAIN_FRAC)
    train = port_ret.iloc[:train_end]
    val = port_ret.iloc[train_end:]
    if verbose:
        print(f"[garch-smoke] train: {len(train)} дней "
              f"({train.index.min().date()} → {train.index.max().date()})")
        print(f"              val:   {len(val)} дней "
              f"({val.index.min().date()} → {val.index.max().date()})")

    # --- 4. fit GARCH ---
    if verbose:
        print("[garch-smoke] fit GARCH(1,1, t)...")
    res = fit_garch(train)
    if verbose:
        print("    params:")
        for k, v in res.params.items():
            print(f"      {k:<15} = {v:+.6f}")
        print(f"    persistence (α+β) = {garch_persistence(res):.4f}")

    # --- 5. conditional vol in-sample ---
    cond_vol = conditional_vol_daily(res)
    if verbose:
        print(f"[garch-smoke] cond vol daily: "
              f"mean={cond_vol.mean()*100:.4f}%, "
              f"max={cond_vol.max()*100:.4f}%")
        ann = cond_vol.mean() * np.sqrt(252)
        print(f"    annualized mean = {ann*100:.2f}%")

    # --- 6. forecast на HORIZON ---
    if verbose:
        print(f"[garch-smoke] forecast vol на {HORIZON} дней...")
    vol_fc = forecast_vol_annualized(res, horizon=HORIZON)
    if verbose:
        print(f"    forecast annual vol = {vol_fc*100:.2f}%")

    # --- 7. exposure ---
    exposure = vol_to_exposure(vol_fc)
    if verbose:
        print(f"[garch-smoke] exposure = {exposure:.4f} "
              f"(target={TARGET_VOL_ANNUAL*100:.0f}%, forecast={vol_fc*100:.1f}%)")

    # --- 8. применить к val (только для проверки формы) ---
    val_scaled = apply_exposure(val, exposure)
    if verbose:
        print(f"[garch-smoke] val scaled: "
              f"mean={val_scaled.mean()*100:+.4f}%, "
              f"std={val_scaled.std()*100:.4f}% "
              f"(vs unscaled std={val.std()*100:.4f}%)")

    # --- 9. проверки корректности ---
    checks = {
        "exposure_in_0_1": 0.0 <= exposure <= 1.0,
        "vol_positive": vol_fc > 0,
        "cond_vol_positive": (cond_vol > 0).all(),
        "val_scaled_std_lower": val_scaled.std() <= val.std() + 1e-9,
    }
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise RuntimeError(f"Smoke checks failed: {failed}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n[garch-smoke] все проверки OK за {elapsed:.1f} сек")

    return {
        "res": res,
        "port_ret": port_ret,
        "cond_vol": cond_vol,
        "vol_forecast": vol_fc,
        "exposure": exposure,
        "checks": checks,
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    smoke()