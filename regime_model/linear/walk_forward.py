# regime_model/linear/walk_forward.py
# ============================================================
# Walk-Forward линейной модели сайзинга.
#
# Логика:
#   - до появления достаточной истории базового портфеля (< 250 дней)
#     — fit на IMOEX (прокси рынка), predict на port_ret (или IMOEX,
#     если port_ret ещё нет).
#   - после 250 дней port_ret — fit на port_ret.
#
# Защита от look-ahead:
#   - train обрезается на HORIZON дней до rd (target не смотрит за rd).
# ============================================================
import time

import numpy as np
import pandas as pd

from .config import (
    HORIZON, TARGET_VOL_ANNUAL, MAX_EXPOSURE, MIN_EXPOSURE,
    RIDGE_ALPHA, MIN_TRAIN_DAYS,
)
from .features import build_features, build_target
from .linear_model import fit_ridge, predict_ridge, model_coefs
from .sizing import vol_to_exposure


def run_linear_wf(port_ret: pd.Series,
                  rebal_dates: list,
                  imoex_ret: pd.Series = None,
                  target_vol: float = TARGET_VOL_ANNUAL,
                  ridge_alpha: float = RIDGE_ALPHA,
                  min_train_days: int = MIN_TRAIN_DAYS,
                  verbose: bool = True) -> dict:
    """
    WF линейной модели.
    
    port_ret: дневные доходности базового портфеля (может быть пустым до 2018-09).
    rebal_dates: даты ребаланса.
    imoex_ret: дневные доходности IMOEX (fallback для ранних дат).
    
    Возвращает:
      exposure, vol_forecast, errors, coefs_history, source_history
    """
    t0 = time.time()
    exposure_records = {}
    vol_records = {}
    errors = []
    coefs_history = []
    source_history = []  # 'portfolio' или 'imoex' — на чём обучались

    # Признаки/target на port_ret (если есть)
    if port_ret is not None and len(port_ret) > 0:
        X_port = build_features(port_ret)
        y_port = build_target(port_ret, horizon=HORIZON)
    else:
        X_port = pd.DataFrame()
        y_port = pd.Series(dtype=float)

    # Признаки/target на IMOEX (fallback)
    if imoex_ret is not None and len(imoex_ret) > 0:
        X_im = build_features(imoex_ret)
        y_im = build_target(imoex_ret, horizon=HORIZON)
    else:
        X_im = pd.DataFrame()
        y_im = pd.Series(dtype=float)

    rebal_dates = sorted(rebal_dates)
    for i, rd in enumerate(rebal_dates):
        rd = pd.Timestamp(rd)

        # 1. Пытаемся fit на port_ret (если train >= min_train_days)
        source = None
        X_train = None
        y_train = None

        if len(X_port) > 0 and rd in X_port.index:
            try:
                idx_port = X_port.index.get_loc(rd)
            except KeyError:
                idx_port = -1
            if idx_port > HORIZON:
                train_end = idx_port - HORIZON
                train_dates = X_port.index[:train_end]
                Xt = X_port.loc[train_dates]
                yt = y_port.loc[train_dates]
                valid = yt.notna()
                Xt = Xt.loc[valid]
                yt = yt.loc[valid]
                if len(Xt) >= min_train_days:
                    X_train, y_train = Xt, yt
                    source = "portfolio"

        # 2. Fallback: fit на IMOEX
        if X_train is None and len(X_im) > 0:
            try:
                idx_im = X_im.index.get_loc(rd)
            except KeyError:
                # rd может быть не в IMOEX (выходной?) — берём ближайший до
                idx_im = X_im.index.searchsorted(rd) - 1
                if idx_im < 0:
                    errors.append({"date": rd, "reason": "rd not in IMOEX index"})
                    continue
            if idx_im > HORIZON:
                train_end = idx_im - HORIZON
                train_dates = X_im.index[:train_end]
                Xt = X_im.loc[train_dates]
                yt = y_im.loc[train_dates]
                valid = yt.notna()
                Xt = Xt.loc[valid]
                yt = yt.loc[valid]
                if len(Xt) >= min_train_days:
                    X_train, y_train = Xt, yt
                    source = "imoex"

        if X_train is None:
            errors.append({
                "date": rd,
                "reason": f"no train available (port<{min_train_days}, imoex<{min_train_days})",
            })
            continue

        # 3. Признаки на rd (для predict)
        # Приоритет: признаки port_ret (если есть rd), иначе IMOEX
        if len(X_port) > 0 and rd in X_port.index:
            X_rd = X_port.loc[[rd]]
        elif len(X_im) > 0:
            try:
                idx_im = X_im.index.get_loc(rd)
                X_rd = X_im.loc[[X_im.index[idx_im]]]
            except KeyError:
                idx_im = X_im.index.searchsorted(rd) - 1
                if idx_im < 0:
                    errors.append({"date": rd, "reason": "no features for rd"})
                    continue
                X_rd = X_im.loc[[X_im.index[idx_im]]]
        else:
            errors.append({"date": rd, "reason": "no features for rd"})
            continue

        try:
            model = fit_ridge(X_train, y_train, alpha=ridge_alpha)
            pred_rd = predict_ridge(model, X_rd).iloc[0]

            if not np.isfinite(pred_rd) or pred_rd <= 0:
                errors.append({
                    "date": rd,
                    "reason": f"invalid vol_pred: {pred_rd} (source={source})",
                })
                continue

            exp = vol_to_exposure(pred_rd, target_vol=target_vol,
                                  max_exp=MAX_EXPOSURE, min_exp=MIN_EXPOSURE)
            exposure_records[rd] = exp
            vol_records[rd] = pred_rd
            source_history.append({"date": rd, "source": source,
                                   "n_train": len(X_train)})

            coefs = model_coefs(model, X_train.columns.tolist())
            coefs_history.append({
                "date": rd,
                "source": source,
                "top3": coefs.head(3).to_dict(),
                "n_train": len(X_train),
            })

            if verbose and (i % 10 == 0 or i == len(rebal_dates) - 1):
                print(f"  [{i+1}/{len(rebal_dates)}] {rd.date()}: "
                      f"src={source:9s}, train={len(X_train):4d}d, "
                      f"pred_vol={pred_rd*100:.2f}%, exp={exp:.3f}")

        except Exception as e:
            errors.append({
                "date": rd,
                "reason": f"{type(e).__name__}: {str(e)[:100]}",
            })
            continue

    elapsed = time.time() - t0
    if verbose:
        n_port = sum(1 for s in source_history if s["source"] == "portfolio")
        n_im = sum(1 for s in source_history if s["source"] == "imoex")
        print(f"\n[linear-wf] done: {len(exposure_records)}/{len(rebal_dates)} "
              f"dates, {len(errors)} errors, {elapsed:.1f}s")
        print(f"  source: portfolio={n_port}, imoex={n_im}")

    return {
        "exposure": pd.Series(exposure_records).sort_index(),
        "vol_forecast": pd.Series(vol_records).sort_index(),
        "errors": errors,
        "coefs_history": coefs_history,
        "source_history": source_history,
        "elapsed": elapsed,
    }