# regime_model/regime_based/smoke.py
# ============================================================
# Smoke: проверка state (PC1/PC2/Hurst/cluster на 185 акциях)
# + exposure на реальных данных.
# ============================================================
import time
import numpy as np
import pandas as pd

from .state_builder import build_state_matrix_from_prices, build_pc_and_cluster
from .sizing import compute_exposure_series
from regime_model.garch.data_loader import (
    load_split_date, load_stock_close, load_ml_holdings, load_ml_equity,
)
from regime_model.garch.portfolio_returns import build_daily_returns


def smoke(verbose: bool = True) -> dict:
    t0 = time.time()
    split_date = load_split_date()
    if verbose:
        print(f"[regime-smoke] split_date = {split_date}")

    # --- 1. загрузка ---
    if verbose:
        print("[regime-smoke] загрузка данных...")
    close = load_stock_close(split_date)
    holdings = load_ml_holdings()
    ml_eq = load_ml_equity()
    rebal_dates = sorted(ml_eq["date"].tolist())[1:]
    if verbose:
        print(f"    close: {close.shape}, rebal_dates: {len(rebal_dates)}")

    # --- 2. S_t на 185 акциях ---
    if verbose:
        print("[regime-smoke] построение S_t на 185 акциях...")
    states = build_state_matrix_from_prices(close)
    if verbose:
        print(f"    states: {len(states)} окон, {states.shape[1]} признаков")
    if len(states) < 10:
        raise RuntimeError(f"Слишком мало окон S_t: {len(states)} < 10")

    # --- 3. PCA + KMeans ---
    if verbose:
        print("[regime-smoke] PCA + KMeans...")
    state = build_pc_and_cluster(states, n_components=3, n_clusters=4)
    if verbose:
        print(f"    PC1/PC2/PC3 + cluster: {len(state)} окон")
        print(f"    PC1: mean={state['PC1'].mean():+.3f}, std={state['PC1'].std():.3f}")
        print(f"    PC2: mean={state['PC2'].mean():+.3f}, std={state['PC2'].std():.3f}")
        print(f"    cluster counts: {state['cluster'].value_counts().to_dict()}")
        print(f"    hurst: mean={state['hurst'].mean():.3f}, "
              f"min={state['hurst'].min():.3f}, max={state['hurst'].max():.3f}")

    # --- 4. доходности портфеля ---
    if verbose:
        print("[regime-smoke] восстановление доходностей портфеля...")
    port_ret = build_daily_returns(close, holdings)
    for rd in rebal_dates:
        if rd not in port_ret.index:
            port_ret.loc[rd] = 0.0
    port_ret = port_ret.sort_index()
    if verbose:
        print(f"    port_ret: {len(port_ret)} дней")

    # --- 5. exposure ---
    if verbose:
        print("[regime-smoke] расчёт exposure...")
    exposure = compute_exposure_series(state, port_ret, rebal_dates)
    if verbose:
        print(f"    exposure: n={len(exposure)}, mean={exposure.mean():.3f}, "
              f"min={exposure.min():.3f}, max={exposure.max():.3f}")

    # --- 6. проверки ---
    checks = {
        "exposure_in_0_1": bool(((exposure >= 0) & (exposure <= 1)).all()) if len(exposure) > 0 else False,
        "n_exposure_positive": len(exposure) > 50,
        "state_not_empty": len(state) > 10,
        "pc1_finite": np.isfinite(state["PC1"]).all(),
        "hurst_finite": np.isfinite(state["hurst"]).all(),
    }
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise RuntimeError(f"Smoke checks failed: {failed}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n[regime-smoke] все проверки OK за {elapsed:.1f} сек")

    return {
        "states": states,
        "state": state,
        "exposure": exposure,
        "checks": checks,
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    smoke()