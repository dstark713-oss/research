# rs_model/ensemble.py
# ============================================================
# 5-seed ансамбль CatBoostRegressor.
# Усредняет predictions.
# ============================================================
import numpy as np
import pandas as pd

from .base_model import create_base_model, fit_base_model
from .config import ENSEMBLE_SEEDS


class SeedEnsemble:
    def __init__(self, params_override=None, seeds=None):
        self.params_override = params_override
        self.seeds = seeds or ENSEMBLE_SEEDS
        self.models = []

    def fit(self, X_train, y_train, X_val, y_val):
        self.models = []
        for s in self.seeds:
            m = create_base_model(self.params_override, seed=s)
            fit_base_model(m, X_train, y_train, X_val, y_val)
            self.models.append(m)
        return self

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return preds.mean(axis=0)