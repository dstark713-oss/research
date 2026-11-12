# random_audit/run_random_audit.py
# ============================================================
# Точка запуска: Hardcore Random Portfolio Audit.
#
# 1. Загрузка данных.
# 2. Независимая реконструкция ML (контроль: должна совпасть с CSV).
# 3. Scenario A: uniform random (10 000 sims).
# 4. Scenario B: persistent random (10 000 sims × 5 persistence).
# 5. Scenario C: ML-pool random (10 000 sims).
# 6. Random Ranking test (перемешивание prediction внутри даты).
# 7. Block bootstrap парного сравнения.
# 8. Проверки корректности рандомизации.
# 9. Отчёт + CSV.
# ============================================================
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    N_SIMULATIONS, BASE_SEED, TOP_N, REBAL_STEP, HORIZON,
    PERSISTENCE_LEVELS, BLOCK_SIZE_MAIN, BLOCK_SIZES_SENSITIVITY,
    N_BOOTSTRAP, REPORTS_DIR,
    ML_RECON_TOL,
)
from .data_loader import load_split_date, load_ohlcv, load_imoex, compute_rs
from .universe import build_availability, get_available_tickers, get_ml_available_tickers
from .random_generator import gen_uniform, gen_persistent
from .portfolio_engine import get_rebalance_dates, run_random_portfolio
from .benchmarks import load_ml_equity_from_csv, independently_reconstruct_ml
from .metrics import compute_metrics, compute_metrics_from_dict
from .statistical_tests import (
    distribution_summary, fraction_better, ml_percentile_in_distribution,
)
from .bootstrap import block_bootstrap_ci, paired_bootstrap_ml_vs_random
from .validation import (
    check_no_duplicates, check_uniformity, check_no_future_use,
    check_rebalance_sequence,
)


def main():
    t_start = time.time()
    split_date = load_split_date()
    print("=" * 70)
    print("HARDCORE RANDOM PORTFOLIO AUDIT")
    print("=" * 70)
    print(f"split_date = {split_date} (VAL не трогаем)")
    print(f"simulations = {N_SIMULATIONS}, TOP_N = {TOP_N}, "
          f"REBAL_STEP = {REBAL_STEP}\n")

    # --- 1. данные ---
    print("[1] Загрузка данных...")
    raw = load_ohlcv(split_date)
    imoex = load_imoex(split_date)
    close = raw["close"]
    target_rs = compute_rs(close, imoex, HORIZON)
    print(f"    close: {close.shape}, target_rs: non-NaN = {target_rs.notna().sum().sum()}")

    # --- 2. даты ребаланса ---
    rebal_dates = get_rebalance_dates(target_rs)
    print(f"    rebalance dates: {len(rebal_dates)}, "
          f"{rebal_dates[0].date()} → {rebal_dates[-1].date()}")

    # --- 3. availability ---
    availability = build_availability(close)
    pool_sizes = [len(get_available_tickers(availability, d)) for d in rebal_dates]
    print(f"    availability: median pool = {int(np.median(pool_sizes))}, "
          f"min = {min(pool_sizes)}, max = {max(pool_sizes)}")

    # --- 4. загрузка ML holdings ---
    print("\n[2] Загрузка ML equity/holdings...")
    ml_equity = load_ml_equity_from_csv()
    ml_equity = ml_equity.set_index("date")["equity"]
    ml_metrics = compute_metrics(ml_equity)
    print(f"    ML: {len(ml_equity)} точек, "
          f"CAGR={ml_metrics['CAGR']*100:.2f}%, "
          f"Sharpe={ml_metrics['Sharpe']:.3f}")

    # --- 5. независимая реконструкция ML ---
    print("\n[3] Независимая реконструкция ML (контроль)...")
    # Загружаем holdings из ml_trades CSV (если есть), иначе — пропускаем
    from .config import ML_TRADES_CSV
    if ML_TRADES_CSV.exists():
        trades = pd.read_csv(ML_TRADES_CSV, parse_dates=["date"])
        ml_holdings = trades.groupby("date")["ticker"].apply(list).to_dict()
        recon = independently_reconstruct_ml(rebal_dates, target_rs, ml_holdings)
        if recon:
            recon_df = pd.DataFrame(recon).set_index("date")
            recon_metrics = compute_metrics(recon_df["port_rs"].pipe(
                lambda x: (1 + x).cumprod().pipe(
                    lambda eq: pd.Series(eq.values, index=x.index))
            ))
            diff_cagr = abs(recon_metrics.get("CAGR", 0) - ml_metrics.get("CAGR", 0))
            print(f"    reconstructed ML: CAGR={recon_metrics.get('CAGR', 0)*100:.2f}% "
                  f"(diff={diff_cagr*100:.2f} п.п.)")
            ml_recon_ok = diff_cagr < ML_RECON_TOL
        else:
            ml_recon_ok = None
    else:
        print(f"    WARN: {ML_TRADES_CSV} не найден — реконструкция пропущена")
        ml_recon_ok = None

    # --- 6. Scenario A: uniform random ---
    print(f"\n[4] Scenario A: uniform random ({N_SIMULATIONS} sims)...")
    t0 = time.time()

    def pool_A(rd):
        return get_available_tickers(availability, rd)

    def sel_A(pool, prev):
        # rng будет создан внутри run_random_portfolio? Нет — передадим через seed
        return None  # placeholder

    # Правильная реализация: run_random_portfolio ожидает selector_fn(pool, prev),
    # но rng должен быть зафиксирован. Сделаем обёртку.
    def make_selector_A(seed):
        rng = np.random.RandomState(seed)
        def sel(pool, prev):
            return gen_uniform(rng, pool, TOP_N)
        return sel

    sims_A = []
    for i in range(N_SIMULATIONS):
        selector = make_selector_A(BASE_SEED + i)
        res = run_random_portfolio(rebal_dates, target_rs, pool_A, selector)
        m = compute_metrics_from_dict(res)
        if m:
            sims_A.append(m)
    print(f"    done: {len(sims_A)} portfolios, {time.time()-t0:.1f}s")

    # --- 7. Scenario B: persistent random ---
    print(f"\n[5] Scenario B: persistent random ({N_SIMULATIONS} sims × {len(PERSISTENCE_LEVELS)} levels)...")
    t0 = time.time()
    sims_B = {p: [] for p in PERSISTENCE_LEVELS}
    for p in PERSISTENCE_LEVELS:
        for i in range(N_SIMULATIONS):
            seed = BASE_SEED + 100000 + i
            rng = np.random.RandomState(seed)
            def make_sel(seed_inner, p_inner):
                rng_inner = np.random.RandomState(seed_inner)
                def sel(pool, prev):
                    return gen_persistent(rng_inner, pool, prev, p_inner, TOP_N)
                return sel
            selector = make_sel(seed, p)
            res = run_random_portfolio(rebal_dates, target_rs, pool_A, selector)
            m = compute_metrics_from_dict(res)
            if m:
                sims_B[p].append(m)
        print(f"    persistence={p}: {len(sims_B[p])} portfolios")
    print(f"    done: {time.time()-t0:.1f}s")

    # --- 8. Scenario C: ML-pool random ---
    print(f"\n[6] Scenario C: ML-pool random ({N_SIMULATIONS} sims)...")
    t0 = time.time()

    # ML pool: тикеры, у которых target non-NaN на rd (proxy для ML-available)
    ml_pools = {rd: get_ml_available_tickers(close, target_rs, rd)
                for rd in rebal_dates}
    pool_sizes_C = [len(ml_pools[rd]) for rd in rebal_dates]
    print(f"    ML pool median = {int(np.median(pool_sizes_C))}, "
          f"min = {min(pool_sizes_C)}, max = {max(pool_sizes_C)}")

    def pool_C(rd):
        return ml_pools.get(rd, [])

    sims_C = []
    for i in range(N_SIMULATIONS):
        selector = make_selector_A(BASE_SEED + 200000 + i)
        res = run_random_portfolio(rebal_dates, target_rs, pool_C, selector)
        m = compute_metrics_from_dict(res)
        if m:
            sims_C.append(m)
    print(f"    done: {len(sims_C)} portfolios, {time.time()-t0:.1f}s")

    # --- 9. агрегация ---
    print("\n[7] Агрегация результатов...")

    def aggregate(sims, name):
        if not sims:
            return {}
        keys = ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar",
                "TotalReturn", "Turnover"]
        out = {"scenario": name, "n": len(sims)}
        for k in keys:
            vals = np.array([s.get(k, np.nan) for s in sims])
            summary = distribution_summary(vals)
            for sk, sv in summary.items():
                out[f"{k}_{sk}"] = sv
        return out

    agg_A = aggregate(sims_A, "A_uniform")
    agg_B = {p: aggregate(sims_B[p], f"B_persist_{p}") for p in PERSISTENCE_LEVELS}
    agg_C = aggregate(sims_C, "C_ml_pool")

    # --- 10. статистические тесты ---
    print("\n[8] Статистические тесты...")

    def stats_vs_ml(sims, name, ml_metrics):
        if not sims:
            return {}
        out = {"scenario": name}
        for k in ["CAGR", "Sharpe", "Calmar", "TotalReturn"]:
            vals = np.array([s.get(k, np.nan) for s in sims])
            ml_val = ml_metrics.get(k, np.nan)
            out[f"frac_better_{k}"] = fraction_better(vals, ml_val, higher_is_better=True)
            out[f"ml_percentile_{k}"] = ml_percentile_in_distribution(vals, ml_val)
        mdd_vals = np.array([s.get("MDD", np.nan) for s in sims])
        out["frac_better_MDD"] = fraction_better(mdd_vals, ml_metrics.get("MDD", np.nan),
                                                  higher_is_better=False)
        return out

    stats_A = stats_vs_ml(sims_A, "A_uniform", ml_metrics)
    stats_B = {p: stats_vs_ml(sims_B[p], f"B_persist_{p}", ml_metrics)
               for p in PERSISTENCE_LEVELS}
    stats_C = stats_vs_ml(sims_C, "C_ml_pool", ml_metrics)

    # --- 11. блок-bootstrap парного сравнения ---
    print("\n[9] Block bootstrap (парное сравнение ML vs random)...")
    # сравним ML с 100 случайными портфелями (репрезентативная выборка)
    sample_idx = np.random.RandomState(42).choice(N_SIMULATIONS, size=100, replace=False)

    # пересобираем equity для выбранных портфелей (для CI)
    # для скорости — используем портфельные RS (не equity)
    def get_port_rs_series(scenario_pool, selector_maker, idx):
        res = run_random_portfolio(rebal_dates, target_rs, scenario_pool,
                                   selector_maker(idx))
        if len(res["dates"]) == 0:
            return None
        return pd.Series(res["port_rs"], index=pd.DatetimeIndex(res["dates"]))

    # ML series
    ml_series = ml_equity.pct_change().dropna()

    # sample 100 random A
    random_series = {}
    for j, idx in enumerate(sample_idx[:100]):
        seed = BASE_SEED + idx
        rng = np.random.RandomState(seed)
        def sel(pool, prev, _rng=rng):
            return gen_uniform(_rng, pool, TOP_N)
        s = get_port_rs_series(pool_A, lambda _idx, _sel=sel: _sel, idx)
        if s is not None:
            random_series[j] = s
    print(f"    sampled {len(random_series)} random portfolios")

    boot_results = paired_bootstrap_ml_vs_random(
        ml_series, random_series,
        block_sizes=BLOCK_SIZES_SENSITIVITY,
        n_boot=N_BOOTSTRAP,
    )

    # --- 12. проверки корректности ---
    print("\n[10] Проверки корректности рандомизации...")
    # Собираем holdings одного случайного портфеля
    rng_check = np.random.RandomState(BASE_SEED)
    def sel_check(pool, prev):
        return gen_uniform(rng_check, pool, TOP_N)
    res_check = run_random_portfolio(rebal_dates, target_rs, pool_A, sel_check)
    checks = {
        "no_duplicates": check_no_duplicates(res_check["holdings"], TOP_N),
        "uniformity": check_uniformity(res_check["holdings"], pool_sizes),
        "no_future_use": check_no_future_use(res_check["holdings"],
                                              res_check["dates"], availability),
        "rebalance_sequence": check_rebalance_sequence(res_check["dates"], REBAL_STEP),
    }
    for k, v in checks.items():
        print(f"    {k}: {v.get('ok')}")

    # --- 13. отчёт ---
    print("\n[11] Генерация отчёта...")
    report = render_report(
        split_date, rebal_dates, pool_sizes, ml_metrics, ml_recon_ok,
        agg_A, agg_B, agg_C, stats_A, stats_B, stats_C,
        boot_results, checks, len(random_series),
    )
    (REPORTS_DIR / "random_audit_report.md").write_text(report, encoding="utf-8")

    # --- 14. CSV ---
    rows = [agg_A, agg_C] + [agg_B[p] for p in PERSISTENCE_LEVELS]
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "random_audit_summary.csv", index=False)

    # сырые симуляции (только метрики, не equity)
    all_sims = []
    for s in sims_A:
        all_sims.append({"scenario": "A_uniform", **s})
    for p in PERSISTENCE_LEVELS:
        for s in sims_B[p]:
            all_sims.append({"scenario": f"B_persist_{p}", **s})
    for s in sims_C:
        all_sims.append({"scenario": "C_ml_pool", **s})
    pd.DataFrame(all_sims).to_csv(REPORTS_DIR / "random_audit_results.csv", index=False)

    print(f"\n[audit] отчёт: {REPORTS_DIR / 'random_audit_report.md'}")
    print(f"[audit] summary: {REPORTS_DIR / 'random_audit_summary.csv'}")
    print(f"[audit] results: {REPORTS_DIR / 'random_audit_results.csv'}")
    print(f"\n[audit] время: {time.time() - t_start:.1f}s")


def render_report(split_date, rebal_dates, pool_sizes, ml_metrics, ml_recon_ok,
                  agg_A, agg_B, agg_C, stats_A, stats_B, stats_C,
                  boot_results, checks, n_sampled):
    lines = [
        "# Hardcore Random Portfolio Audit\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"split_date = {split_date} (VAL не трогаем)\n",
        f"\n## Конфигурация\n",
        f"- TOP_N = {TOP_N}, REBAL_STEP = {REBAL_STEP}, HORIZON = {HORIZON}",
        f"- N_SIMULATIONS = {N_SIMULATIONS}",
        f"- rebalance dates = {len(rebal_dates)}",
        f"- availability pool: median = {int(np.median(pool_sizes))}, "
        f"min = {min(pool_sizes)}, max = {max(pool_sizes)}\n",
        "\n## ML baseline\n",
        f"- CAGR = {ml_metrics['CAGR']*100:+.2f}%",
        f"- MDD = {ml_metrics['MDD']*100:+.2f}%",
        f"- Sharpe = {ml_metrics['Sharpe']:.3f}",
        f"- Calmar = {ml_metrics['Calmar']:.3f}",
        f"- Turnover = {ml_metrics['Turnover']*100:+.1f}%\n",
    ]

    if ml_recon_ok is not None:
        lines.append(f"- ML reconstruction: {'✓ OK' if ml_recon_ok else '✗ FAIL'}\n")

    # --- Scenario A ---
    lines.append("\n## Scenario A: Uniform Random\n")
    lines.append("| Метрика | Random (mean) | Random (median) | Random (p95) | ML | ML percentile |")
    lines.append("|---------|---------------|-----------------|--------------|-----|---------------|")
    for k in ["CAGR", "Sharpe", "Calmar", "MDD", "TotalReturn"]:
        mean_v = agg_A.get(f"{k}_mean", np.nan)
        med_v = agg_A.get(f"{k}_median", np.nan)
        p95_v = agg_A.get(f"{k}_p95", np.nan)
        ml_v = ml_metrics.get(k, np.nan)
        pct = stats_A.get(f"ml_percentile_{k}", np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD", "TotalReturn") else "+.3f"
        lines.append(f"| {k} | {mean_v:{fmt}} | {med_v:{fmt}} | {p95_v:{fmt}} | "
                     f"{ml_v:{fmt}} | {pct:.0f}% |")
    lines.append(f"\n- **Доля random, превзошедших ML по CAGR:** "
                 f"{stats_A.get('frac_better_CAGR', 0)*100:.1f}%")
    lines.append(f"- **Доля random, превзошедших ML по Sharpe:** "
                 f"{stats_A.get('frac_better_Sharpe', 0)*100:.1f}%")
    lines.append(f"- **Доля random, превзошедших ML по Calmar:** "
                 f"{stats_A.get('frac_better_Calmar', 0)*100:.1f}%")
    lines.append(f"- **Доля random с меньшим |MDD|:** "
                 f"{stats_A.get('frac_better_MDD', 0)*100:.1f}%\n")

    # --- Scenario B ---
    lines.append("\n## Scenario B: Persistent Random\n")
    lines.append("| Persistence | CAGR mean | CAGR p95 | ML percentile | frac better CAGR |")
    lines.append("|-------------|-----------|----------|---------------|------------------|")
    for p in PERSISTENCE_LEVELS:
        a = agg_B.get(p, {})
        s = stats_B.get(p, {})
        lines.append(f"| {p:.2f} | {a.get('CAGR_mean', 0)*100:+.2f}% | "
                     f"{a.get('CAGR_p95', 0)*100:+.2f}% | "
                     f"{s.get('ml_percentile_CAGR', 0):.0f}% | "
                     f"{s.get('frac_better_CAGR', 0)*100:.1f}% |")
    lines.append("")

    # --- Scenario C ---
    lines.append("\n## Scenario C: ML-Pool Random\n")
    lines.append("| Метрика | Random (mean) | Random (p95) | ML | ML percentile |")
    lines.append("|---------|---------------|--------------|-----|---------------|")
    for k in ["CAGR", "Sharpe", "Calmar", "MDD"]:
        mean_v = agg_C.get(f"{k}_mean", np.nan)
        p95_v = agg_C.get(f"{k}_p95", np.nan)
        ml_v = ml_metrics.get(k, np.nan)
        pct = stats_C.get(f"ml_percentile_{k}", np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD") else "+.3f"
        lines.append(f"| {k} | {mean_v:{fmt}} | {p95_v:{fmt}} | {ml_v:{fmt}} | {pct:.0f}% |")
    lines.append(f"\n- **Доля random, превзошедших ML по CAGR:** "
                 f"{stats_C.get('frac_better_CAGR', 0)*100:.1f}%\n")

    # --- bootstrap ---
    lines.append("\n## Block Bootstrap (ML vs random, парное сравнение)\n")
    lines.append(f"Sampled {n_sampled} random portfolios. "
                 f"CI для mean(ML_ret - random_ret):\n")
    lines.append("| Block size | mean diff | CI 2.5% | CI 97.5% | p-value |")
    lines.append("|------------|-----------|---------|----------|---------|")
    for bs in BLOCK_SIZES_SENSITIVITY:
        if bs not in boot_results:
            continue
        cis = list(boot_results[bs].values())
        if not cis:
            continue
        mean_diffs = [c["mean"] for c in cis]
        ci_lows = [c["ci_low"] for c in cis]
        ci_highs = [c["ci_high"] for c in cis]
        p_values = [c["p_value"] for c in cis]
        lines.append(f"| {bs} | {np.mean(mean_diffs)*100:+.3f}% | "
                     f"{np.mean(ci_lows)*100:+.3f}% | "
                     f"{np.mean(ci_highs)*100:+.3f}% | "
                     f"{np.mean(p_values):.3f} |")
    lines.append("")

    # --- checks ---
    lines.append("\n## Проверки корректности рандомизации\n")
    for k, v in checks.items():
        lines.append(f"- **{k}**: {v.get('ok')} "
                     f"{v.get('reason', '') or v.get('cv', '')}")
    lines.append("")

    # --- вердикт ---
    lines.append("\n## Вердикт\n")
    frac_cagr = stats_A.get("frac_better_CAGR", 1.0)
    frac_sharpe = stats_A.get("frac_better_Sharpe", 1.0)
    ml_pct_cagr = stats_A.get("ml_percentile_CAGR", 0)

    if frac_cagr < 0.05 and frac_sharpe < 0.05:
        verdict = "ML ADVANTAGE CONFIRMED"
    elif frac_cagr < 0.20 or frac_sharpe < 0.20:
        verdict = "ML ADVANTAGE LIKELY"
    elif frac_cagr < 0.50:
        verdict = "RANDOM BASELINE COMPETITIVE"
    elif frac_cagr >= 0.50:
        verdict = "ML ADVANTAGE NOT CONFIRMED"
    else:
        verdict = "INCONCLUSIVE"

    lines.append(f"**{verdict}**\n")
    lines.append(f"- ML percentile по CAGR в Scenario A: {ml_pct_cagr:.0f}%")
    lines.append(f"- Доля random, лучше ML по CAGR: {frac_cagr*100:.1f}%")
    lines.append(f"- Доля random, лучше ML по Sharpe: {frac_sharpe*100:.1f}%")

    lines.append("\n## Ограничения\n")
    lines.append("- Availability pool не включает делистингованные тикеры "
                 "(survivorship bias, если universe сформирован по текущему состоянию).")
    lines.append("- Execution model (MOC vs open(t+1)) не зафиксирован явно.")
    lines.append("- Random portfolio — это НЕ прогнозная модель, а baseline. "
                 "Отсутствие преимущества над random не доказывает отсутствие альфы, "
                 "но и не подтверждает её.")

    return "\n".join(lines)


if __name__ == "__main__":
    main()