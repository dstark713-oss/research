# random_audit/bootstrap.py
# ============================================================
# Block bootstrap по времени для парного сравнения ML vs random.
#
# Парное сравнение: на одинаковых датах разница port_rs(ML) - port_rs(random).
# Block bootstrap: сохраняем временную структуру (блоки по BLOCK_SIZE).
# ============================================================
import numpy as np
import pandas as pd


def block_bootstrap_ci(series: pd.Series, block_size: int,
                       n_boot: int = 2000, seed: int = 42,
                       alpha: float = 0.05) -> dict:
    """
    Block bootstrap для среднего значения ряда.
    Возвращает: mean, ci_low, ci_high, p_value (H0: mean=0).
    """
    s = series.dropna().values
    n = len(s)
    if n < block_size * 2:
        return {"mean": float(np.mean(s)) if n > 0 else np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
                "block_size": block_size, "n_blocks": 0}

    rng = np.random.RandomState(seed)
    n_blocks = n // block_size
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([s[st:st + block_size] for st in starts])
        boot_means[i] = sample.mean()

    mean_obs = float(np.mean(s))
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    # двусторонний p-value: доля bootstrap, где mean ≤ 0 (для positive H1)
    # или |mean| >= |mean_obs|
    p = 2 * min((boot_means <= 0).mean(), (boot_means >= 0).mean())
    p = float(min(max(p, 0.0), 1.0))

    return {
        "mean": mean_obs, "ci_low": ci_low, "ci_high": ci_high,
        "p_value": p, "block_size": block_size, "n_blocks": n_blocks,
        "n_obs": n,
    }


def paired_bootstrap_ml_vs_random(ml_series: pd.Series,
                                  random_series_dict: dict,
                                  block_sizes: list,
                                  n_boot: int = 2000,
                                  seed: int = 42) -> dict:
    """
    Парное сравнение ML с каждым random портфелем (по средним датам).
    Возвращает словарь: block_size -> {random_id -> ci_dict}.
    """
    out = {}
    for bs in block_sizes:
        out[bs] = {}
        for rid, rs in random_series_dict.items():
            common = ml_series.index.intersection(rs.index)
            if len(common) < bs * 2:
                continue
            diff = (ml_series.loc[common] - rs.loc[common]).dropna()
            ci = block_bootstrap_ci(diff, block_size=bs,
                                    n_boot=n_boot, seed=seed)
            out[bs][rid] = ci
    return out