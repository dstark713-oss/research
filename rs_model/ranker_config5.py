# rs_model/ranker_config5.py
# ============================================================
# Ranker (YetiRank) с параметрами config 5 из grid search:
#   depth=4, l2_leaf_reg=100, random_strength=3, learning_rate=0.03
#
# Остальные параметры — как в базовом ranker.py.
# Это challenger-модель, не замена.
# ============================================================
from catboost import CatBoostRanker, Pool


CONFIG5_PARAMS = {
    "depth": 4,
    "learning_rate": 0.03,
    "iterations": 1500,
    "l2_leaf_reg": 100,
    "random_strength": 3,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "border_count": 64,
    "rsm": 0.8,
    "loss_function": "YetiRank",
    "eval_metric": "NDCG",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}


def create_ranker_config5(seed: int = 42) -> CatBoostRanker:
    p = dict(CONFIG5_PARAMS)
    p["random_seed"] = seed
    return CatBoostRanker(**p)


def fit_ranker_config5(model, X_train, y_train, group_train,
                       X_val=None, y_val=None, group_val=None):
    """
    Train Pool + val Pool с group_id. Работает во всех версиях CatBoost.
    """
    train_pool = Pool(X_train, y_train, group_id=group_train)

    if X_val is not None and group_val is not None:
        val_pool = Pool(X_val, y_val, group_id=group_val)
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=False,
        )
    else:
        model.fit(train_pool, verbose=False)
    return model