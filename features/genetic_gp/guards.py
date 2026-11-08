# features/genetic_gp/guards.py
# ============================================================
# Защита от вырождения после gplearn.
#
# gplearn сам по себе не отбраковывает константные/дублирующие
# выражения. Делаем пост-фильтр.
# ============================================================
import numpy as np
import pandas as pd


def is_degenerate(feature_values: np.ndarray, y: np.ndarray,
                  min_std_ratio: float = 1e-6, min_unique_ratio: float = 0.01) -> tuple:
    """
    Проверяет фичу на вырожденность.
    Возвращает (is_degenerate: bool, reason: str).
    """
    x = feature_values.copy()
    x = x[np.isfinite(x)]
    if len(x) < 100:
        return True, "too few finite values"

    # константа / почти константа
    std = x.std()
    if std < min_std_ratio * (np.abs(x).mean() + 1e-10):
        return True, f"near-constant (std={std:.2e})"

    # слишком мало уникальных значений
    n_unique = len(np.unique(x))
    if n_unique < min_unique_ratio * len(x):
        return True, f"low unique ratio ({n_unique}/{len(x)})"

    # Inf / NaN
    if not np.isfinite(x).all():
        return True, "contains Inf/NaN after filter"

    # экстремальные значения (численная нестабильность)
    if np.abs(x).max() > 1e12:
        return True, f"extreme max value ({np.abs(x).max():.2e})"

    return False, "ok"


def filter_degenerate(programs: list, X: np.ndarray, y: np.ndarray):
    """
    Принимает список gplearn-программ (или кортежей (program, values)),
    возвращает отфильтрованные.
    """
    good = []
    reasons = []
    for item in programs:
        if isinstance(item, tuple):
            prog, vals = item
        else:
            prog = item
            try:
                vals = prog.execute(X)
            except Exception:
                reasons.append((prog, "execute failed"))
                continue
        deg, reason = is_degenerate(vals, y)
        if deg:
            reasons.append((prog, reason))
        else:
            good.append((prog, vals))
    return good, reasons