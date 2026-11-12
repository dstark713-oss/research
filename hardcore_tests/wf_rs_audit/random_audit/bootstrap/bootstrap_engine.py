# random_audit/bootstrap/bootstrap_engine.py
# ============================================================
# Block bootstrap: парное сравнение ML vs random.
#
# Для каждой пары (ML, random):
#   diff_t = ml_ret_t - random_ret_t  (на общих датах)
#   block bootstrap по diff_t (сохраняем временную структуру)
#   CI для mean(diff), p-value (H0: mean=0)
#
# Векторизовано через numpy. Без numba — 100 портфелей × 4 блока × 2000 итераций
# = 800k × 100 = 80M операций, numpy справляется за ~1-2 минуты.
# ============================================================
import numpy as np
import pandas as pd

from .config import BLOCK_SIZES, N_BOOTSTRAP, BASE_SEED


def block_bootstrap_mean(series: np.ndarray, block_size: int,
                         n_boot: int, seed: int) -> dict:
    """
    Block bootstrap для mean(series).
    Возвращает mean, ci_low, ci_high, p_value (two-sided H0: mean=0).
    """
    n = len(series)
    if n < block_size * 2:
        return {"mean": float(series.mean()) if n > 0 else np.nan,
                "ci_low": np.nan, "ci_high": np.nan,
                "p_value": np.nan, "n_obs": n, "n_blocks": 0}

    rng = np.random.RandomState(seed)
    n_blocks = n // block_size
    # индексы блоков: (n_boot, n_blocks)
    starts = rng.randint(0, n - block_size + 1, size=(n_boot, n_blocks))
    # (n_boot, n_blocks, block_size)
    idx = starts[:, :, None] + np.arange(block_size)[None, None, :]
    # (n_boot, n_blocks * block_size)
    idx = idx.reshape(n_boot, -1)
    # (n_boot,)
    boot_means = series[idx].mean(axis=1)

    mean_obs = float(series.mean())
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    # two-sided p-value
    p = 2 * min((boot_means <= 0).mean(), (boot_means >= 0).mean())
    p = float(min(max(p, 0.0), 1.0))

    return {
        "mean": mean_obs,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p,
        "n_obs": n,
        "n_blocks": n_blocks,
    }


def paired_bootstrap(ml_series: pd.Series,
                     random_series: dict,
                     block_sizes: list = BLOCK_SIZES,
                     n_boot: int = N_BOOTSTRAP,
                     base_seed: int = BASE_SEED) -> pd.DataFrame:
    """
    Парное сравнение ML vs каждый random.
    Возвращает DataFrame: portfolio_id, block_size, mean_diff, ci_low, ci_high, p_value, n_obs.
    """
    rows = []
    for pid, rs in random_series.items():
        common = ml_series.index.intersection(rs.index)
        if len(common) < 20:
            continue
        diff = (ml_series.loc[common] - rs.loc[common]).values
        for bs in block_sizes:
            res = block_bootstrap_mean(diff, block_size=bs,
                                       n_boot=n_boot,
                                       seed=base_seed + pid + bs * 1000)
            rows.append({
                "portfolio_id": pid,
                "block_size": bs,
                **res,
            })
    return pd.DataFrame(rows)


def aggregate_bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегация по block_size: mean(mean_diff), mean(ci_low), mean(ci_high), доля p<0.05."""
    if len(df) == 0:
        return pd.DataFrame()
    agg = df.groupby("block_size").agg(
        mean_diff=("mean", "mean"),
        mean_ci_low=("ci_low", "mean"),
        mean_ci_high=("ci_high", "mean"),
        median_p_value=("p_value", "median"),
        frac_p_lt_05=("p_value", lambda s: (s < 0.05).mean()),
        n_portfolios=("portfolio_id", "count"),
    ).reset_index()
    return agg