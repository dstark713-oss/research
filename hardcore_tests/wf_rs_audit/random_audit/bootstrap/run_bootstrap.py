# random_audit/bootstrap/run_bootstrap.py
# ============================================================
# Точка запуска: только bootstrap + MDD + turnover.
# Без Монте-Карло. 100 random портфелей × 4 блока × 2000 итераций.
# ============================================================
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    REPORTS_DIR, N_RANDOM_PORTFOLIOS, BLOCK_SIZES, N_BOOTSTRAP,
    ML_EQUITY_CSV, RANDOM_RESULTS_CSV,
)
from .data_loader import (
    load_split_date, load_ml_equity, load_random_results,
    load_ohlcv, load_imoex, compute_rs,
)
from .random_series import generate_random_portfolios
from .bootstrap_engine import paired_bootstrap, aggregate_bootstrap
from .mdd_percentile import recompute_mdd_from_results
from .turnover_fix import recompute_ml_turnover
from .report import render_report


def main():
    t0 = time.time()
    split_date = load_split_date()
    print("=" * 70)
    print("BOOTSTRAP AUDIT: ML vs Random (без Monte Carlo)")
    print("=" * 70)
    print(f"split_date = {split_date}")
    print(f"random portfolios: {N_RANDOM_PORTFOLIOS}")
    print(f"block sizes: {BLOCK_SIZES}")
    print(f"bootstrap iterations: {N_BOOTSTRAP}\n")

    # --- 1. ML equity ---
    print("[1] Загрузка ML equity...")
    ml_df = load_ml_equity()
    ml_series = ml_df.set_index("date")["port_rs"]
    print(f"    ML: {len(ml_series)} ребалансов, "
          f"{ml_series.index.min().date()} → {ml_series.index.max().date()}")

    ml_turnover = recompute_ml_turnover(ml_df)
    print(f"    ML turnover annualized: {ml_turnover.get('turnover_annual', 0)*100:.1f}%")

    # --- 2. ML metrics (независимый пересчёт) ---
    print("[2] Пересчёт ML метрик...")
    from .metrics_local import compute_metrics_from_port_rs
    ml_metrics = compute_metrics_from_port_rs(ml_series, ml_df.get("turnover"))
    print(f"    ML: CAGR={ml_metrics['CAGR']*100:+.2f}%, "
          f"Sharpe={ml_metrics['Sharpe']:.3f}, "
          f"MDD={ml_metrics['MDD']*100:+.2f}%")

    # --- 3. Random portfolios (100 seeded, НА ТЕХ ЖЕ ДАТАХ, что ML) ---
    print(f"\n[3] Генерация {N_RANDOM_PORTFOLIOS} random портфелей "
          f"(на датах ML: {len(ml_series)})...")
    raw = load_ohlcv(split_date)
    imoex = load_imoex(split_date)
    target_rs = compute_rs(raw["close"], imoex)
    
    # ИСПРАВЛЕНО: rebal_dates = ML-даты, а не get_rebalance_dates
    rebal_dates = pd.DatetimeIndex(ml_series.index)
    print(f"    rebal_dates: {len(rebal_dates)}, "
          f"{rebal_dates[0].date()} → {rebal_dates[-1].date()}")
    
    random_series = generate_random_portfolios(
        raw["close"], target_rs,
        rebal_dates=rebal_dates,       # <-- явно
        n_portfolios=N_RANDOM_PORTFOLIOS,
    )
    print(f"    получено {len(random_series)} портфелей")

    # общие даты — теперь должны быть = len(rebal_dates)
    common_counts = []
    for pid, rs in random_series.items():
        common_counts.append(len(ml_series.index.intersection(rs.index)))
    median_common = int(np.median(common_counts)) if common_counts else 0
    print(f"    common dates (median): {median_common}")

    # --- 4. Bootstrap ---
    print(f"\n[4] Bootstrap ({N_BOOTSTRAP} iterations × {len(BLOCK_SIZES)} blocks × "
          f"{len(random_series)} portfolios)...")
    raw_boot = paired_bootstrap(
        ml_series, random_series,
        block_sizes=BLOCK_SIZES,
        n_boot=N_BOOTSTRAP,
    )
    agg_boot = aggregate_bootstrap(raw_boot)
    print(f"    raw: {len(raw_boot)} rows")
    print(f"\n    Агрегация:")
    if len(agg_boot) > 0:
        for _, r in agg_boot.iterrows():
            print(f"      block={int(r['block_size'])}: "
                  f"mean_diff={r['mean_diff']*100:+.3f}%, "
                  f"CI=[{r['mean_ci_low']*100:+.3f}%, "
                  f"{r['mean_ci_high']*100:+.3f}%], "
                  f"median_p={r['median_p_value']:.3f}")

    # --- 5. MDD fix ---
    print("\n[5] Пересчёт MDD percentile из сохранённых результатов...")
    random_results = load_random_results()
    mdd_fix = recompute_mdd_from_results(random_results, ml_metrics)
    for scenario, m in mdd_fix.items():
        print(f"    {scenario}: ML percentile MDD = "
              f"{m['ml_percentile_mdd']:.0f}%, "
              f"frac better = {m['frac_random_better_mdd']*100:.1f}%")

    # --- 6. Отчёт ---
    print("\n[6] Генерация отчёта...")
    report = render_report(
        split_date, ml_metrics, ml_turnover,
        agg_boot, raw_boot, mdd_fix,
        n_random_portfolios=len(random_series),
        n_common_dates_median=median_common,
    )
    out_md = REPORTS_DIR / "bootstrap_audit_report.md"
    out_md.write_text(report, encoding="utf-8")

    # --- 7. CSV ---
    raw_boot.to_csv(REPORTS_DIR / "bootstrap_raw.csv", index=False)
    agg_boot.to_csv(REPORTS_DIR / "bootstrap_agg.csv", index=False)

    print(f"\n[bootstrap] отчёт: {out_md}")
    print(f"[bootstrap] raw: {REPORTS_DIR / 'bootstrap_raw.csv'}")
    print(f"[bootstrap] agg: {REPORTS_DIR / 'bootstrap_agg.csv'}")
    print(f"[bootstrap] время: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()