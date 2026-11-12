# random_audit/statistical_tests.py
# ============================================================
# Статистические тесты:
#   - распределение Monte Carlo;
#   - доля random, превзошедших ML/IMOEX;
#   - позиция ML в распределении.
# ============================================================
import numpy as np
import pandas as pd


def distribution_summary(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return {}
    return {
        "n": int(len(v)),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v, ddof=1)),
        "p5": float(np.percentile(v, 5)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def fraction_better(random_values: np.ndarray, ml_value: float,
                    higher_is_better: bool = True) -> float:
    """
    Доля random, которые строго лучше ML.
    Если higher_is_better=False (MDD): random лучше = меньший |MDD|.
    """
    v = random_values[np.isfinite(random_values)]
    if len(v) == 0 or not np.isfinite(ml_value):
        return np.nan
    if higher_is_better:
        return float((v > ml_value).mean())
    else:
        # для MDD: ML MDD отрицательный, лучше = меньше по модулю
        return float((np.abs(v) < np.abs(ml_value)).mean())


def ml_percentile_in_distribution(random_values: np.ndarray,
                                  ml_value: float) -> float:
    """Позиция ML в распределении random (0..100)."""
    v = random_values[np.isfinite(random_values)]
    if len(v) == 0 or not np.isfinite(ml_value):
        return np.nan
    return float((v < ml_value).mean() * 100)