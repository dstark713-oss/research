# random_audit/bootstrap/mdd_percentile.py
# ============================================================
# Фикс MDD percentile: для MDD "лучше" = меньший |MDD|.
# ml_percentile_in_distribution для MDD должен считать долю random
# с |MDD| < |ML MDD| (т.е. random ЛУЧШЕ ML по MDD).
# ============================================================
import numpy as np
import pandas as pd


def mdd_percentile_correct(random_mdd: np.ndarray, ml_mdd: float) -> float:
    """
    Доля random с |MDD| < |ML MDD| — это позиция ML в распределении
    (100% = ML лучший, 0% = ML худший).
    """
    v = random_mdd[np.isfinite(random_mdd)]
    if len(v) == 0 or not np.isfinite(ml_mdd):
        return np.nan
    return float((np.abs(v) < abs(ml_mdd)).mean() * 100)


def frac_random_better_mdd(random_mdd: np.ndarray, ml_mdd: float) -> float:
    """Доля random, которые ЛУЧШЕ ML по MDD (меньший |MDD|)."""
    return mdd_percentile_correct(random_mdd, ml_mdd) / 100.0


def recompute_mdd_from_results(results_df: pd.DataFrame,
                               ml_metrics: dict) -> dict:
    """
    Пересчёт MDD percentile из сохранённого random_audit_results.csv.
    results_df должен содержать колонку 'MDD' и 'scenario'.
    """
    ml_mdd = ml_metrics.get("MDD", np.nan)
    out = {}
    for scenario in results_df["scenario"].unique():
        sub = results_df[results_df["scenario"] == scenario]
        if "MDD" not in sub.columns:
            continue
        mdd_vals = sub["MDD"].dropna().values
        out[scenario] = {
            "n": len(mdd_vals),
            "ml_mdd": ml_mdd,
            "ml_percentile_mdd": mdd_percentile_correct(mdd_vals, ml_mdd),
            "frac_random_better_mdd": frac_random_better_mdd(mdd_vals, ml_mdd),
            "random_median_abs_mdd": float(np.median(np.abs(mdd_vals))),
        }
    return out