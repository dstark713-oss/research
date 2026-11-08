# features/genetic_gp/fitness_cs.py
# ============================================================
# Кастомный fitness для gplearn: ежедневный cross-sectional Spearman IC.
#
# gplearn передаёт y, y_pred, w как плоские массивы.
# sample_weight (w) = date_ids (номер даты для каждой строки).
#
# Внутри: группируем по date_id, считаем Spearman по каждой дате,
# возвращаем mean_ic - 0.3 * std_ic (robust fitness).
# ============================================================
import numpy as np
from scipy import stats
from gplearn.fitness import make_fitness


def _cross_sectional_ic(y, y_pred, w):
    """
    y, y_pred: плоские массивы.
    w: date_ids (целые числа, номер даты).
    Возвращает: mean_ic - 0.3 * std_ic.
    """
    date_ids = w.astype(int)
    unique_dates = np.unique(date_ids)

    ics = []
    for d in unique_dates:
        mask = date_ids == d
        if mask.sum() < 20:
            continue
        yv = y[mask]
        pv = y_pred[mask]
        # убираем NaN/Inf
        m2 = np.isfinite(yv) & np.isfinite(pv)
        if m2.sum() < 20:
            continue
        try:
            ic = stats.spearmanr(yv[m2], pv[m2]).correlation
            if np.isfinite(ic):
                ics.append(ic)
        except Exception:
            pass

    if len(ics) < 30:
        return -1.0  # штраф за мало данных

    ics = np.array(ics)
    mean_ic = ics.mean()
    std_ic = ics.std()
    fitness = mean_ic - 0.3 * std_ic
    return float(fitness)


# обёртка для gplearn
cross_sectional_ic_fitness = make_fitness(
    function=_cross_sectional_ic,
    greater_is_better=True,
    wrap=False,  # не нужно pickle, работаем в одном процессе
)