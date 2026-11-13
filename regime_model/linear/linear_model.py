# regime_model/linear/linear_model.py
# ============================================================
# Линейная модель (Ridge) с median imputation.
# ============================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def fit_ridge(X_train: pd.DataFrame, y_train: pd.Series,
              alpha: float = 1.0) -> Pipeline:
    """
    Fit Ridge с median imputation.
    Возвращает Pipeline (imputer + ridge).
    """
    df = pd.concat([X_train, y_train.rename("y")], axis=1).dropna(subset=["y"])
    if len(df) < 100:
        raise ValueError(f"Мало данных для fit: {len(df)} < 100")
    X = df.drop(columns="y").values
    y = df["y"].values

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("ridge", Ridge(alpha=alpha)),
    ])
    pipe.fit(X, y)
    return pipe


def predict_ridge(model: Pipeline, X: pd.DataFrame) -> pd.Series:
    """
    Predict. NaN в X → impute медианой (из train), prediction — число.
    """
    X_vals = X.values
    out = model.predict(X_vals)
    return pd.Series(out, index=X.index, name="vol_pred")


def model_coefs(model: Pipeline, feature_names: list) -> pd.Series:
    """Коэффициенты Ridge (после imputation)."""
    coef = model.named_steps["ridge"].coef_
    return pd.Series(coef, index=feature_names, name="coef").sort_values(
        key=abs, ascending=False
    )