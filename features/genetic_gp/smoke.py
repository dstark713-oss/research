# features/genetic_gp/smoke.py
# ============================================================
# Быстрый смок-тест (10–20 секунд): gplearn на маленькой популяции
# и 2 поколениях. Проверяет, что весь пайплайн работает end-to-end.
#
# Запускается ПЕРЕД полным прогоном. Если падает — полный не стартует.
# ============================================================
import time

import numpy as np
from gplearn.genetic import SymbolicTransformer

from .data_prep import (
    load_split_date, load_raw, load_imoex, build_features, build_target, flatten_panel,
)
from .fitness_cs import cross_sectional_ic_fitness


def smoke(split_date: str, verbose: bool = True):
    t0 = time.time()
    if verbose:
        print("[smoke] загрузка данных...")
    raw = load_raw(split_date)
    imoex = load_imoex(split_date)
    features = build_features(raw)
    target = build_target(raw, imoex)

    X, y, date_ids, names, valid_dates, tickers = flatten_panel(features, target)

    if verbose:
        print(f"[smoke] X={X.shape}  y={y.shape}  dates={len(valid_dates)}")

    # обрезаем до последних 300 дат, чтобы быстро
    max_date_id = min(300, len(valid_dates))
    mask = date_ids < max_date_id
    X_s, y_s, ids_s = X[mask], y[mask], date_ids[mask]

    if len(np.unique(ids_s)) < 50:
        raise RuntimeError(f"smoke: слишком мало дат ({len(np.unique(ids_s))})")

    if verbose:
        print(f"[smoke] запуск gplearn: pop=50, gen=2, dates={len(np.unique(ids_s))}")

    est = SymbolicTransformer(
        population_size=50,
        generations=2,
        tournament_size=10,
        function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv'),
        metric=cross_sectional_ic_fitness,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.05,
        parsimony_coefficient=0.005,
        max_samples=0.9,
        n_components=5,
        hall_of_fame=20,
        random_state=42,
        n_jobs=1,
        verbose=0,
    )
    est.fit(X_s, y_s, sample_weight=ids_s)

    # проверяем, что есть хотя бы одна программа с конечным fitness
    best = est._best_programs
    if not best:
        raise RuntimeError("smoke: gplearn вернул пустой список программ")

    finite = [p for p in best if np.isfinite(p.fitness_)]
    if not finite:
        raise RuntimeError("smoke: нет программ с конечным fitness")

    if verbose:
        top = max(finite, key=lambda p: p.fitness_)
        print(f"[smoke] лучший fitness: {top.fitness_:.4f}")
        print(f"[smoke] лучший expr: {str(top)[:80]}")
        print(f"[smoke] время: {time.time() - t0:.1f} сек")

    return est, best


if __name__ == "__main__":
    smoke(load_split_date())