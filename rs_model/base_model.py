# rs_model/base_model.py
# ============================================================
# CatBoostRegressor (pointwise, target = rank-gauss).
# ============================================================
from catboost import CatBoostRegressor
from .config import BASE_PARAMS


def create_base_model(params_override: dict = None, seed: int = None) -> CatBoostRegressor:
    p = dict(BASE_PARAMS)
    if params_override:
        p.update(params_override)
    if seed is not None:
        p["random_seed"] = seed
    return CatBoostRegressor(**p)


def fit_base_model(model: CatBoostRegressor, X_train, y_train, X_val, y_val):
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=100,
        verbose=False,
    )
    return model