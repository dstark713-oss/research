# features/genetic_gp/run_gp.py
# ============================================================
# Оркестратор: guard tests → smoke → полный прогон gplearn.
# При провале guards или smoke — полный прогон НЕ запускается.
# ============================================================
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer

from .data_prep import (
    load_split_date, load_raw, load_imoex, build_features, build_target, flatten_panel,
)
from .fitness_cs import cross_sectional_ic_fitness
from .guards import is_degenerate
from .diversity import select_diverse
from .guards_data import run_all as run_data_guards
from .smoke import smoke


def main():
    split_date = load_split_date()
    print(f"[run] split_date = {split_date}")

    # 1. Guard tests — падаем при lookahead/leak
    run_data_guards(split_date)

    # 2. Smoke — 10-20 сек
    print("[run] === SMOKE ===")
    smoke(split_date, verbose=True)
    print("[run] smoke OK\n")

    # 3. Полный прогон
    print("[run] === FULL RUN ===")
    t0 = time.time()

    raw = load_raw(split_date)
    imoex = load_imoex(split_date)
    features = build_features(raw)
    target = build_target(raw, imoex)
    X, y, date_ids, feature_names, dates, tickers = flatten_panel(features, target)
    print(f"[run] X={X.shape}  y={y.shape}  dates={len(dates)}")

    est = SymbolicTransformer(
        population_size=1000,
        generations=30,
        tournament_size=7,           # было 20
        function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg',
                      'max', 'min'),  # убрали inv, cos, sin, tan
        metric=cross_sectional_ic_fitness,
        p_crossover=0.7,
        p_subtree_mutation=0.05,
        p_hoist_mutation=0.05,
        p_point_mutation=0.05,
        p_point_replace=0.1,
        parsimony_coefficient=0.02,   # было 0.005
        max_samples=0.9,
        n_components=50,
        hall_of_fame=100,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    est.fit(X, y, sample_weight=date_ids)
    print(f"[run] обучение: {time.time()-t0:.1f} сек")

    # 4. Guards на программы
    programs = []
    for prog in est._best_programs:
        try:
            vals = prog.execute(X)
            programs.append({
                "expr": str(prog),
                "fitness": float(prog.fitness_),
                "length": int(prog.length_),
                "values": vals,
            })
        except Exception as e:
            print(f"[run] программа не выполнилась: {e}")

    good = []
    for p in programs:
        deg, reason = is_degenerate(p["values"], y)
        if not deg:
            good.append(p)
    print(f"[run] после guards: {len(good)}/{len(programs)}")

    # 5. Diversity
    diverse = select_diverse(good, max_n=20, corr_threshold=0.7)
    print(f"[run] после diversity: {len(diverse)}")

    # 6. Сохранение
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    df_div = pd.DataFrame([{
        "rank": i + 1, "expr": p["expr"],
        "fitness": p["fitness"], "length": p["length"],
    } for i, p in enumerate(diverse)])
    df_div.to_csv(out_dir / "diverse_features.csv", index=False)

    df_all = pd.DataFrame([{
        "expr": p["expr"], "fitness": p["fitness"], "length": p["length"],
    } for p in good])
    df_all.to_csv(out_dir / "all_good_features.csv", index=False)

    print(f"\n[run] сохранено: {out_dir}")
    print(df_div.to_string(index=False))


if __name__ == "__main__":
    main()