# features/genetic_gp/fitness_cs.py
# ============================================================
# Кастомный fitness для gplearn: ежедневный cross-sectional Spearman IC.
#
# ВЕКТОРИЗОВАННАЯ версия: без цикла по датам, через pandas groupby.
# Математически эквивалентна scipy.stats.spearmanr по каждой дате,
# но в 10-50 раз быстрее.
# ============================================================
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning

from gplearn.fitness import make_fitness

warnings.filterwarnings("ignore", category=ConstantInputWarning)


def _cross_sectional_ic(y, y_pred, w):
    """
    Векторизованный ежедневный cross-sectional Spearman IC.

    y, y_pred: плоские массивы (n_samples,)
    w: date_ids (n_samples,)

    Возвращает: mean_ic - 0.3 * std_ic.
    """
    date_ids = w.astype(np.int64)

    # фильтр NaN/Inf
    mask = np.isfinite(y) & np.isfinite(y_pred)
    if mask.sum() < 100:
        return -1.0

    y = y[mask]
    y_pred = y_pred[mask]
    date_ids = date_ids[mask]

    # DataFrame для groupby (C-оптимизировано в pandas)
    df = pd.DataFrame({
        "d": date_ids,
        "y": y,
        "p": y_pred,
    })

    # ранги внутри каждой даты (два вызова, векторизовано)
    df["yr"] = df.groupby("d")["y"].rank(method="average")
    df["pr"] = df.groupby("d")["p"].rank(method="average")

    # центрируем ранги внутри дат
    df["dy"] = df["yr"] - df.groupby("d")["yr"].transform("mean")
    df["dp"] = df["pr"] - df.groupby("d")["pr"].transform("mean")

    # произведения
    df["dydp"] = df["dy"] * df["dp"]
    df["dy2"] = df["dy"] * df["dy"]
    df["dp2"] = df["dp"] * df["dp"]

    # агрегируем по датам
    g = df.groupby("d")
    cov = g["dydp"].mean()
    var_y = g["dy2"].mean()
    var_p = g["dp2"].mean()
    counts = g.size()

    # IC = cov / sqrt(var_y * var_p)
    denom = np.sqrt(var_y * var_p)
    ic = cov / denom.replace(0, np.nan)

    # фильтр: только даты с >= 20 наблюдений и конечным IC
    valid = (counts >= 20) & np.isfinite(ic)
    ic = ic[valid]

    if len(ic) < 30:
        return -1.0

    mean_ic = ic.mean()
    std_ic = ic.std()
    fitness = mean_ic - 0.3 * std_ic
    return float(fitness)


# обёртка для gplearn
# wrap=True — для мультипроцессинга (см. Патч 2)
cross_sectional_ic_fitness = make_fitness(
    function=_cross_sectional_ic,
    greater_is_better=True,
    wrap=True,
)