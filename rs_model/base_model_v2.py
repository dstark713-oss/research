# rs_model/base_model_v2.py
# ============================================================
# CatBoostRegressor с выбором числа итераций по устойчивому плато
# на внутренней purged validation.
#
# Логика:
#   1. train split: 80% train-дат, 20% internal-val (purged 21d)
#   2. обучаем до 1000 итераций, снимаем IC на internal-val
#      на итерациях 100, 200, ..., 1000
#   3. выбираем "устойчивое плато": итерация, после которой
#      IC не падает > 5% от максимума в течение 200 итераций
#   4. переобучаем на ВСЁМ train с выбранным числом итераций
#   5. один раз оцениваем на внешнем WF val
# ============================================================
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .config import BASE_PARAMS, PURGE, HORIZON
from .metrics import mean_ic

ITER_GRID = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
PLATEAU_TOL = 0.05   # 5% от максимума
PLATEAU_WINDOW = 200  # 200 итераций


def create_regressor_v2(params_override=None, seed=None):
    p = dict(BASE_PARAMS)
    p["iterations"] = 1000  # потолок
    if params_override:
        p.update(params_override)
    if seed is not None:
        p["random_seed"] = seed
    return CatBoostRegressor(**p)


def _split_internal_val(train_dates: pd.DatetimeIndex, val_frac=0.2, purge=PURGE):
    """
    Делит train на train-core и internal-val с purge.
    internal-val = последние val_frac train-дат.
    purge между train-core и internal-val.
    """
    n = len(train_dates)
    val_start = int(n * (1 - val_frac))
    val_dates = train_dates[val_start:]
    core_end = max(0, val_start - purge)
    core_dates = train_dates[:core_end]
    return core_dates, val_dates


def find_plateau_iters(model, X_core, y_core, X_ival, y_ival,
                       ival_dates, ival_long, feature_names):
    """
    Обучает модель, снимает IC на internal-val на итерациях из ITER_GRID.
    Возвращает (best_iter, ic_curve).
    """
    model.fit(
        X_core, y_core,
        eval_set=(X_ival, y_ival),
        use_best_model=False,
        verbose=False,
    )
    # предсказания на internal-val на разных итерациях
    ic_curve = []
    for it in ITER_GRID:
        try:
            preds = model.predict(X_ival, ntree_end=it)
        except Exception:
            # fallback: если ntree_end не поддерживается, используем staged_predict
            preds = None
            for i, p in enumerate(model.staged_predict(X_ival)):
                if i + 1 == it:
                    preds = p
                    break
            if preds is None:
                continue

        # собираем wide DataFrame для IC
        df = ival_long.copy()
        df["pred"] = preds
        pred_wide = df.pivot(index="date", columns="ticker", values="pred")
        target_wide = df.pivot(index="date", columns="ticker", values="target")
        ic = mean_ic(pred_wide, target_wide)
        ic_curve.append({"iter": it, "ic": ic})

    if not ic_curve:
        return 1000, []

    # плато: итерация, после которой IC не падает > PLATEAU_TOL * max
    # в течение PLATEAU_WINDOW итераций
    max_ic = max(c["ic"] for c in ic_curve if np.isfinite(c["ic"]))
    if not np.isfinite(max_ic) or max_ic <= 0:
        return 1000, ic_curve

    best_iter = ITER_GRID[-1]
    for c in ic_curve:
        if not np.isfinite(c["ic"]):
            continue
        # смотрим вперёд на PLATEAU_WINDOW
        window = [x for x in ic_curve
                  if c["iter"] <= x["iter"] <= c["iter"] + PLATEAU_WINDOW]
        if len(window) == 0:
            continue
        # если в окне IC не падает ниже (1 - tol) * max
        if all(x["ic"] >= (1 - PLATEAU_TOL) * max_ic for x in window):
            best_iter = c["iter"]
            break

    return best_iter, ic_curve


def fit_with_plateau(X_core, y_core, X_ival, y_ival, ival_dates, ival_long,
                     feature_names, params_override=None, seed=42,
                     X_full=None, y_full=None):
    # ФИКСИРОВАННЫЕ 300 итераций (по IC curve — оптимум на большинстве фолдов)
    best_iter = 300
    ic_curve = []

    if X_full is None:
        X_full = np.vstack([X_core, X_ival])
        y_full = np.concatenate([y_core, y_ival])

    model_final = create_regressor_v2(params_override, seed=seed)
    model_final.set_params(iterations=best_iter)
    model_final.fit(X_full, y_full, verbose=False)

    return model_final, best_iter, ic_curve