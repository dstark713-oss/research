# random_audit/validation.py
# ============================================================
# Проверки корректности самой рандомизации (Блок 10 из ТЗ).
# ============================================================
import numpy as np
import pandas as pd

from .config import TOP_N


def check_reproducibility(gen_fn, seed: int, pool: list, n_runs: int = 3) -> dict:
    """Один и тот же seed → одинаковые выборки."""
    rng1 = np.random.RandomState(seed)
    results = [tuple(gen_fn(rng1, pool, None)) for _ in range(n_runs)]
    consistent = all(r == results[0] for r in results)
    return {"consistent": consistent, "sample": results[0] if results else None}


def check_no_duplicates(holdings: list, top_n: int = TOP_N) -> dict:
    """Никаких дубликатов внутри одного портфеля."""
    ok = all(len(set(h)) == len(h) == top_n for h in holdings)
    return {"ok": ok, "n_holdings": len(holdings)}


def check_uniformity(holdings: list, pool_sizes: list) -> dict:
    """
    Проверка равномерности: каждый тикер должен попадать в портфель
    примерно с одинаковой частотой (в пределах статистического шума).
    """
    from collections import Counter
    counter = Counter()
    for h in holdings:
        for t in h:
            counter[t] += 1
    counts = np.array(list(counter.values()))
    if len(counts) < 5:
        return {"ok": False, "reason": "too few unique tickers"}
    # коэффициент вариации частот
    cv = counts.std() / counts.mean()
    # при равномерном выборе CV должна быть < 0.5 (грубая проверка)
    return {"ok": cv < 0.5, "cv": float(cv), "n_tickers": len(counter)}


def check_no_future_use(holdings: list, rebal_dates: list,
                        availability: pd.DataFrame) -> dict:
    """
    Проверка: все тикеры в портфеле были доступны на дату ребаланса.
    """
    ok = True
    violations = []
    for h, d in zip(holdings, rebal_dates):
        if d not in availability.index:
            violations.append((str(d), "date_missing"))
            continue
        avail_row = availability.loc[d]
        for t in h:
            if t not in avail_row.index or not avail_row[t]:
                violations.append((str(d), t))
    if violations:
        ok = False
    return {"ok": ok, "n_violations": len(violations),
            "first_violations": violations[:10]}


def check_rebalance_sequence(dates: list, rebal_step: int = 20) -> dict:
    """
    Проверка: даты ребаланса идут с шагом REBAL_STEP.
    """
    if len(dates) < 2:
        return {"ok": False, "reason": "too few dates"}
    diffs = np.diff([d.value for d in dates]) / (1e9 * 3600 * 24)
    # должно быть ~ rebal_step * 1.4 (календарные дни ~ торговые * 1.4)
    expected = rebal_step * 1.4
    ok = bool(np.all(np.abs(diffs - expected) / expected < 0.5))
    return {"ok": ok, "median_gap_days": float(np.median(diffs))}