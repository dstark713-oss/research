# freeze_model/pipeline.py
# ============================================================
# Замороженная модель: RS WF + GARCH daily WF.
# Возвращает daily_ret, exposure, holdings, rebal_dates.
# ============================================================
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from rs_model.config import FEATURES
from rs_model.data import build_dataset, to_long
from rs_model.wf import make_folds

from regime_model.garch.data_loader import load_stock_close
from regime_model.garch.portfolio_returns import simulate_buy_and_hold
from regime_model.garch.garch_model import fit_garch
from regime_model.garch.sizing import vol_to_exposure

from .config import (
    RS_PARAMS, RS_REBALANCE, RS_TOP_N, RS_PURGE, RS_N_FOLDS,
    GARCH_TARGET_VOL, GARCH_MIN_TRAIN,
    GARCH_MAX_EXPOSURE, GARCH_MIN_EXPOSURE,
)


def _run_rs_wf_collect(split_date: str, verbose: bool = True) -> dict:
    """
    RS WF. Возвращает:
      daily_df: DataFrame (date, ret, fold)
      holdings: dict (rebal_date -> list[ticker])
      rebal_dates: list
    """
    if verbose:
        print(f"[rs] Загрузка данных...")
    features_raw, features_rank, target_rs, target_gauss, _ = build_dataset(split_date)
    dates = target_rs.dropna(how="all").index
    folds = make_folds(dates, n_folds=RS_N_FOLDS, purge=RS_PURGE)

    close = load_stock_close(split_date)
    daily_ret_all = close.pct_change()

    records = []
    holdings_by_date = {}
    all_rebal_dates = []

    if verbose:
        print(f"[rs] WF: {len(folds)} фолдов...")

    for i, (tr_dates, va_dates) in enumerate(folds):
        tr_long = to_long(features_rank, target_gauss)
        tr_long = tr_long[tr_long["date"].isin(tr_dates)].reset_index(drop=True)
        X_tr = tr_long[FEATURES].values
        y_tr = tr_long["target"].values

        model = CatBoostRegressor(**RS_PARAMS)
        model.fit(X_tr, y_tr, verbose=False)

        va_long = to_long(features_rank, target_gauss)
        va_long = va_long[va_long["date"].isin(va_dates)].reset_index(drop=True)
        X_va = va_long[FEATURES].values
        va_long["pred"] = model.predict(X_va)
        pred_wide = va_long.pivot(index="date", columns="ticker", values="pred")

        rebal_dates = va_dates[::RS_REBALANCE]
        for j, rd in enumerate(rebal_dates):
            if rd not in pred_wide.index:
                continue
            pred_row = pred_wide.loc[rd].dropna()
            if len(pred_row) < RS_TOP_N:
                continue
            selected = pred_row.nlargest(RS_TOP_N).index.tolist()

            holdings_by_date[rd] = selected
            all_rebal_dates.append(rd)

            if j + 1 < len(rebal_dates):
                next_rd = rebal_dates[j + 1]
            else:
                next_rd = va_dates[-1]

            bh = simulate_buy_and_hold(daily_ret_all, selected, rd, next_rd)
            for d, r in bh.items():
                records.append({"date": d, "ret": r, "fold": i + 1})

    df = pd.DataFrame(records)
    df = df.drop_duplicates("date", keep="first").sort_values("date").reset_index(drop=True)

    def _date_to_fold(d):
        for idx, (_, va_dates) in enumerate(folds):
            if d in va_dates:
                return idx + 1
        return None

    df["fold"] = df["date"].apply(_date_to_fold)
    df = df.dropna(subset=["fold"]).reset_index(drop=True)
    df["fold"] = df["fold"].astype(int)

    if verbose:
        print(f"[rs] daily_df: {len(df)} дней, holdings: {len(holdings_by_date)}")

    return {
        "daily_df": df,
        "holdings": holdings_by_date,
        "rebal_dates": all_rebal_dates,
    }


def _garch_exposure_for_fold(train_ret: pd.Series,
                             val_ret: pd.Series,
                             target_vol: float = 0.15) -> pd.Series:
    if len(train_ret) < GARCH_MIN_TRAIN:
        raise ValueError(f"GARCH train too short: {len(train_ret)} < {GARCH_MIN_TRAIN}")

    res = fit_garch(train_ret)
    p = res.params
    omega = p.get("omega", 0.0) / (100 ** 2)
    alpha = p.get("alpha[1]", 0.0)
    beta = p.get("beta[1]", 0.0)
    mu = p.get("mu", 0.0) / 100

    last_var = float(res.conditional_volatility.iloc[-1] ** 2) / (100 ** 2)

    exposure = {}
    for d in val_ret.index:
        vol_ann = np.sqrt(last_var) * np.sqrt(252)
        exp = vol_to_exposure(vol_ann, target_vol=target_vol,
                              max_exp=GARCH_MAX_EXPOSURE,
                              min_exp=GARCH_MIN_EXPOSURE)
        exposure[d] = exp
        ret = val_ret.loc[d]
        if not np.isfinite(ret):
            ret = 0.0
        eps = ret - mu
        var = omega + alpha * (eps ** 2) + beta * last_var
        last_var = max(var, 1e-10)

    return pd.Series(exposure).sort_index()


def run_frozen_model(split_date: str, verbose: bool = True) -> dict:
    t0 = time.time()

    rs_result = _run_rs_wf_collect(split_date, verbose=verbose)
    daily_df = rs_result["daily_df"]
    holdings = rs_result["holdings"]
    rebal_dates = rs_result["rebal_dates"]

    daily_df["sized_ret"] = daily_df["ret"].copy()
    daily_df["exposure"] = 1.0

    unique_folds = sorted(daily_df["fold"].unique())

    if verbose:
        print(f"\n[garch] WF daily по фолдам: {unique_folds}")

    for fold_idx in unique_folds:
        if fold_idx == 1:
            if verbose:
                print(f"  fold 1: baseline (no GARCH)")
            continue

        train_mask = daily_df["fold"] < fold_idx
        val_mask = daily_df["fold"] == fold_idx

        train_ret = pd.Series(
            daily_df.loc[train_mask, "ret"].values,
            index=pd.DatetimeIndex(daily_df.loc[train_mask, "date"])
        )
        val_ret = pd.Series(
            daily_df.loc[val_mask, "ret"].values,
            index=pd.DatetimeIndex(daily_df.loc[val_mask, "date"])
        )

        if verbose:
            print(f"  fold {fold_idx}: train={len(train_ret)}, val={len(val_ret)}")

        if len(train_ret) < GARCH_MIN_TRAIN:
            if verbose:
                print(f"    SKIP: train too short ({len(train_ret)} < {GARCH_MIN_TRAIN})")
            continue

        try:
            exp = _garch_exposure_for_fold(train_ret, val_ret,
                                            target_vol=GARCH_TARGET_VOL)
            val_indices = daily_df.index[val_mask]
            daily_df.loc[val_indices, "exposure"] = exp.values
            daily_df.loc[val_indices, "sized_ret"] = (
                daily_df.loc[val_indices, "ret"].values * exp.values
            )
            if verbose:
                print(f"    OK: exposure mean={exp.mean():.3f}")
        except Exception as e:
            if verbose:
                print(f"    ERROR: {e}")

    daily_baseline = pd.Series(daily_df["ret"].values,
                               index=pd.DatetimeIndex(daily_df["date"])).sort_index()
    daily_sized = pd.Series(daily_df["sized_ret"].values,
                            index=pd.DatetimeIndex(daily_df["date"])).sort_index()
    exposure_series = pd.Series(daily_df["exposure"].values,
                                index=pd.DatetimeIndex(daily_df["date"])).sort_index()

    elapsed = time.time() - t0
    if verbose:
        print(f"\n[freeze] готово за {elapsed:.1f}s")
        print(f"  exposure: mean={exposure_series.mean():.3f}")

    return {
        "daily_ret_baseline": daily_baseline,
        "daily_ret_sized": daily_sized,
        "exposure": exposure_series,
        "holdings": holdings,
        "rebal_dates": rebal_dates,
        "split_date": split_date,
        "elapsed": elapsed,
    }