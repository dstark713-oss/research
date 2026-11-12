# random_audit/random_generator.py
# ============================================================
# Генераторы случайных портфелей.
#
# Scenario A: uniform random из полного availability-пула.
# Scenario B: persistent random с заданным уровнем сохранения.
# Scenario C: uniform random из ML-пула (target non-NaN + close non-NaN).
#
# Все генераторы:
#   - независимы от будущих доходностей (только availability/target на t);
#   - без возвращения внутри портфеля (один тикер = одна позиция);
#   - воспроизводимы по seed.
# ============================================================
import numpy as np
import pandas as pd

from .config import TOP_N


def gen_uniform(rng: np.random.RandomState, pool: list, top_n: int = TOP_N) -> list:
    """Random выбор без возвращения из pool."""
    if len(pool) < top_n:
        return []
    return rng.choice(pool, size=top_n, replace=False).tolist()


def gen_persistent(rng: np.random.RandomState, pool: list, prev: list,
                   persistence: float, top_n: int = TOP_N) -> list:
    """
    Persistent random:
      - сохраняем round(persistence * top_n) из prev (если они в pool);
      - остальные добираем случайно из pool без возвращения.
    """
    if len(pool) < top_n:
        return []

    n_keep = int(round(persistence * top_n))
    keep = []
    if prev and n_keep > 0:
        # оставляем те prev, которые всё ещё в pool
        prev_in_pool = [t for t in prev if t in pool]
        if len(prev_in_pool) > n_keep:
            keep = rng.choice(prev_in_pool, size=n_keep, replace=False).tolist()
        else:
            keep = prev_in_pool
    # добираем из pool без возвращения
    remaining = [t for t in pool if t not in keep]
    need = top_n - len(keep)
    if need <= 0:
        return keep[:top_n]
    if len(remaining) < need:
        # fallback: если не хватает — возвращаем то, что есть + случайные из pool
        return (keep + remaining)[:top_n]
    add = rng.choice(remaining, size=need, replace=False).tolist()
    return keep + add