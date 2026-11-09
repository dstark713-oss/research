# hardcore_tests/negative_control.py
# ============================================================
# ТЕСТ 7: Negative Control.
#
# Перемешиваем таргет между тикерами внутри каждой даты
# (сохраняя распределение RS на дату).
# Повторяем IC-фильтрацию.
#
# PASS: < 2 фич проходят порог |IC| > 0.015 на shuffled target
# FAIL: >= 2 фич проходят порог (ложные сигналы)
# ============================================================
import numpy as np
import pandas as pd

from features.rs_pool import RS_POOL_NAMES
from .data import load_all, daily_ic


def _shuffle_row(row, rng):
    x = row.dropna()
    vals = x.values.copy()
    rng.shuffle(vals)
    out = row.copy() * np.nan
    out.loc[x.index] = vals
    return out

def shuffle_target_within_dates(target, seed=42):
    """Векторный shuffle внутри каждой даты (без apply)."""
    rng = np.random.RandomState(seed)
    arr = target.values.copy()
    for i in range(arr.shape[0]):
        row = arr[i]
        mask = ~np.isnan(row)
        if mask.sum() < 2: continue
        idx = np.where(mask)[0]
        shuffled = row[idx].copy()
        rng.shuffle(shuffled)
        arr[i, idx] = shuffled
    return pd.DataFrame(arr, index=target.index, columns=target.columns)


def run(split_date=None, n_repeats=5) -> dict:
    raw, imoex, features, target = load_all(split_date)

    # baseline: сколько фич проходят порог на реальном таргете
    print("[neg_ctrl] baseline на реальном таргете...")
    base_pass = 0
    for name in RS_POOL_NAMES:
        ics = daily_ic(features[name], target)
        if len(ics) > 100 and abs(ics.mean()) > 0.015:
            base_pass += 1
    print(f"  {base_pass}/{len(RS_POOL_NAMES)} фич проходят |IC| > 0.015")

    # shuffled
    print(f"[neg_ctrl] shuffled target, {n_repeats} повторов...")
    shuffle_pass = []
    for r in range(n_repeats):
        target_shuf = shuffle_target_within_dates(target, seed=42+r)
        n_pass = 0
        for name in RS_POOL_NAMES:
            ics = daily_ic(features[name], target_shuf)
            if len(ics) > 100 and abs(ics.mean()) > 0.015:
                n_pass += 1
        shuffle_pass.append(n_pass)
        print(f"  repeat {r+1}: {n_pass}/{len(RS_POOL_NAMES)} фич проходят порог")

    mean_shuffle_pass = np.mean(shuffle_pass)
    print(f"\n[neg_ctrl] Среднее по shuffled: {mean_shuffle_pass:.1f} фич (baseline: {base_pass})")

    if mean_shuffle_pass >= 2:
        status = "FAIL"
    elif mean_shuffle_pass >= 1:
        status = "WARN"
    else:
        status = "PASS"

    print(f"[neg_ctrl] СТАТУС: {status}")
    return {"status": status, "base_pass": base_pass,
            "shuffle_pass": shuffle_pass, "mean_shuffle": mean_shuffle_pass}