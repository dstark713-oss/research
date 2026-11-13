# regime_model/target_vol/smoke.py
# ============================================================
# Smoke: проверка realized vol + exposure на реальных данных.
# ============================================================
import time

import numpy as np
import pandas as pd

from .config import TARGET_VOL_ANNUAL, REALIZED_VOL_WINDOW
from .sizing import realized_vol_annualized, vol_to_exposure, compute_exposure_series
from regime_model.garch.data_loader import (
    load_split_date, load_stock_close, load_ml_holdings, load_ml_equity,
)
from regime_model.garch.portfolio_returns import build_daily_returns


def smoke(verbose: bool = True) -> dict:
    t0 = time.time()
    split_date = load_split_date()
    if verbose:
        print(f"[tv-smoke] split_date = {split_date}")

    close = load_stock_close(split_date)
    holdings = load_ml_holdings()
    ml_eq = load_ml_equity()
    rebal_dates = sorted(ml_eq["date"].tolist())[1:]
    port_ret = build_daily_returns(close, holdings)
    # добавить rd с 0 (для consistency с GARCH/Linear)
    for rd in rebal_dates:
        if rd not in port_ret.index:
            port_ret.loc[rd] = 0.0
    port_ret = port_ret.sort_index()

    if verbose:
        print(f"    port_ret: {len(port_ret)} дней, "
              f"rebal_dates: {len(rebal_dates)}")

    # realized vol
    rv = realized_vol_annualized(port_ret, window=REALIZED_VOL_WINDOW)
    rv_valid = rv.dropna()
    if verbose:
        print(f"    realized vol 20d: n={len(rv_valid)}, "
              f"mean={rv_valid.mean()*100:.2f}%, "
              f"min={rv_valid.min()*100:.2f}%, "
              f"max={rv_valid.max()*100:.2f}%")

    # exposure series
    exposure = compute_exposure_series(port_ret, rebal_dates,
                                        target_vol=TARGET_VOL_ANNUAL,
                                        window=REALIZED_VOL_WINDOW)
    if verbose:
        print(f"    exposure: n={len(exposure)}, "
              f"mean={exposure.mean():.3f}, "
              f"min={exposure.min():.3f}, max={exposure.max():.3f}")

    # проверки
    checks = {
        "exposure_in_0_1": ((exposure >= 0) & (exposure <= 1)).all(),
        "realized_vol_positive": (rv_valid > 0).all(),
        "n_exposure_positive": len(exposure) > 50,
    }
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise RuntimeError(f"Smoke checks failed: {failed}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n[tv-smoke] все проверки OK за {elapsed:.1f} сек")

    return {
        "rv": rv,
        "exposure": exposure,
        "checks": checks,
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    smoke()