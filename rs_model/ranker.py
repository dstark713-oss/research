
# rs_model/ranker.py
# ============================================================
# CatBoostRanker с YetiRank.
# Target = 5 relevance grades (0..4).
# Group = торговая дата.
# ============================================================

import numpy as np

from catboost import CatBoostRanker, Pool
from .config import RANKER_PARAMS


def create_ranker(params_override: dict = None,
                  seed: int = None) -> CatBoostRanker:
    p = dict(RANKER_PARAMS)

    if params_override:
        p.update(params_override)

    if seed is not None:
        p["random_seed"] = seed

    return CatBoostRanker(**p)


def _validate_groups(group_id, n_rows, name="dataset"):
    """Проверяет соответствие group_id строкам и непрерывность групп."""

    group_id = np.asarray(group_id)

    if len(group_id) != n_rows:
        raise ValueError(
            f"{name}: group_id length ({len(group_id)}) "
            f"!= number of rows ({n_rows})"
        )

    if group_id.ndim != 1:
        raise ValueError(f"{name}: group_id must be one-dimensional")

    if pd_is_missing(group_id):
        raise ValueError(f"{name}: group_id contains missing values")

    # Одна группа должна занимать непрерывный блок строк.
    _, first_indices = np.unique(group_id, return_index=True)
    ordered_groups = group_id[np.sort(first_indices)]

    if len(np.unique(group_id)) != len(ordered_groups):
        raise ValueError(f"{name}: invalid group structure")

    # Проверяем, что группа не появляется повторно после перехода
    # к следующей группе.
    seen = set()
    previous = None

    for gid in group_id:
        gid = gid.item() if hasattr(gid, "item") else gid

        if gid != previous:
            if gid in seen:
                raise ValueError(
                    f"{name}: group_id={gid} appears in "
                    "multiple non-contiguous blocks"
                )
            seen.add(gid)
            previous = gid

    return group_id.astype(np.int64)


def pd_is_missing(values):
    """Проверка пропусков без зависимости от pandas."""
    try:
        import pandas as pd
        return pd.isna(values).any()
    except (TypeError, ValueError):
        return False


def fit_ranker(model, X_train, y_train, group_train,
               X_val=None, y_val=None, group_val=None):

    group_train = _validate_groups(
        group_train, len(X_train), "train"
    )

    train_pool = Pool(
        data=X_train,
        label=y_train,
        group_id=group_train
    )

    if X_val is not None:
        if y_val is None or group_val is None:
            raise ValueError(
                "X_val, y_val and group_val must be provided together"
            )

        group_val = _validate_groups(
            group_val, len(X_val), "validation"
        )

        val_pool = Pool(
            data=X_val,
            label=y_val,
            group_id=group_val
        )

        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=False,
        )

    else:
        model.fit(
            train_pool,
            verbose=False
        )

    return model
