# backtests/is_wf_regressor_plateau.py
# ============================================================
# WF IS прогон: CatBoostRegressor с выбором итераций по плато.
# 6 фолдов, fold 6 — смотрим.
#
# Для каждого фолда:
#   1. train core (80%) + internal-val (20%, purged 21d)
#   2. обучение до 1000, снятие IC на internal-val на 100..1000
#   3. выбор плато
#   4. переобучение на полном train с best_iter
#   5. оценка на внешнем WF val
#
# Сохраняет:
#   - backtests/is_wf_regressor_plateau_report.md
#   - backtests/is_wf_regressor_plateau_equity.csv
#   - backtests/is_wf_regressor_plateau_ic_curves.csv
# ============================================================
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from rs_model.config import FEATURES, HORIZON, PURGE
from rs_model.data import build_dataset, to_long
from rs_model.metrics import evaluate, top_n_rs
from rs_model.wf import make_folds
from rs_model.base_model_v2 import (
    fit_with_plateau, create_regressor_v2, ITER_GRID
)

OUT_DIR = Path(__file__).parent
OUT_REPORT = OUT_DIR / "is_wf_regressor_plateau_report.md"
OUT_EQUITY = OUT_DIR / "is_wf_regressor_plateau_equity.csv"
OUT_CURVES = OUT_DIR / "is_wf_regressor_plateau_ic_curves.csv"

MIN_GROUP = 5


def _predict_wide(df_long, preds):
    df = df_long.copy()
    df["pred"] = preds
    return df.pivot(index="date", columns="ticker", values="pred")


def main():
    split_date = json.loads(
        (Path(__file__).parent / "split_info.json").read_text()
    )["split_date"]
    print(f"[reg-plateau] split_date = {split_date}")

    print("[reg-plateau] загрузка данных...")
    features_raw, features_rank, target_rs, target_gauss, target_grades = build_dataset(split_date)

    dates = target_rs.dropna(how="all").index
    folds = make_folds(dates, n_folds=6, purge=PURGE)
    print(f"[reg-plateau] folds: {len(folds)}")

    fold_metrics = []
    equity_records = []
    curve_records = []

    for i, (tr_dates, va_dates) in enumerate(folds):
        print(f"\n[reg-plateau] === fold {i+1}: "
              f"train {tr_dates[0].date()} → {tr_dates[-1].date()} | "
              f"val {va_dates[0].date()} → {va_dates[-1].date()} ===")

        # --- split train на core + internal-val ---
        n = len(tr_dates)
        val_start = int(n * 0.8)
        ival_dates = tr_dates[val_start:]
        core_end = max(0, val_start - PURGE)
        core_dates = tr_dates[:core_end]

        if len(core_dates) < 100 or len(ival_dates) < 20:
            print(f"  SKIP: мало данных (core={len(core_dates)}, ival={len(ival_dates)})")
            continue

        # --- core ---
        core_long = to_long(features_rank, target_gauss)
        core_long = core_long[core_long["date"].isin(core_dates)].reset_index(drop=True)
        X_core = core_long[FEATURES].values
        y_core = core_long["target"].values

        # --- internal-val ---
        ival_long = to_long(features_rank, target_gauss)
        ival_long = ival_long[ival_long["date"].isin(ival_dates)].reset_index(drop=True)
        X_ival = ival_long[FEATURES].values
        y_ival = ival_long["target"].values

        # --- внешний val ---
        va_long = to_long(features_rank, target_gauss)
        va_long = va_long[va_long["date"].isin(va_dates)].reset_index(drop=True)
        X_va = va_long[FEATURES].values
        y_va = va_long["target"].values

        # --- full train ---
        full_long = to_long(features_rank, target_gauss)
        full_long = full_long[full_long["date"].isin(tr_dates)].reset_index(drop=True)
        X_full = full_long[FEATURES].values
        y_full = full_long["target"].values

        print(f"  core={len(core_dates)}d, ival={len(ival_dates)}d, "
              f"full train={len(tr_dates)}d, val={len(va_dates)}d")

        # --- поиск плато + финальное обучение ---
        model, best_iter, ic_curve = fit_with_plateau(
            X_core, y_core, X_ival, y_ival, ival_dates, ival_long,
            FEATURES, seed=42, X_full=X_full, y_full=y_full
        )
        print(f"  best_iter = {best_iter}")
        if ic_curve:
            print(f"  IC curve: " + ", ".join(
                f"{c['iter']}:{c['ic']:+.4f}" for c in ic_curve
            ))
            for c in ic_curve:
                curve_records.append({
                    "fold": i + 1, "iter": c["iter"], "ic_ival": c["ic"],
                })

        # --- оценка на внешнем val ---
        pred = _predict_wide(va_long, model.predict(X_va))
        rs_va = target_rs.loc[va_long["date"].unique()]
        m = evaluate(pred, rs_va)
        m["fold"] = i + 1
        m["best_iter"] = best_iter
        fold_metrics.append(m)
        print(f"  IC={m['mean_ic']:+.4f}, NDCG@10={m['ndcg@10']:.3f}, "
              f"TOP10 mean RS={m['top10_mean_rs']:+.4f}, hit={m['hit_rate_ic']:.1%}")

        # --- equity ---
        top10_series = top_n_rs(pred, rs_va, n=10)
        for d, v in top10_series.items():
            equity_records.append({"date": d, "fold": i + 1, "top10_rs": v})

    df = pd.DataFrame(fold_metrics)
    print("\n[reg-plateau] === AGGREGATE ===")
    print(f"  mean IC:       {df['mean_ic'].mean():+.4f}")
    print(f"  NDCG@10:       {df['ndcg@10'].mean():.3f}")
    print(f"  TOP10 mean RS: {df['top10_mean_rs'].mean():+.4f}")
    print(f"  hit_rate:      {df['hit_rate_ic'].mean():.1%}")
    print(f"  median best_iter: {df['best_iter'].median():.0f}")

    # --- отчёт ---
    lines = [
        "# WF IS: CatBoostRegressor с выбором итераций по плато\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"split_date = {split_date}\n",
        f"\n## Конфигурация\n",
        f"- Модель: CatBoostRegressor, MSE на rank-gauss",
        f"- depth=3, l2_leaf_reg=50, random_strength=3, lr=0.03",
        f"- iterations: до 1000, выбор по плато на internal-val",
        f"- internal-val: 20% последних train-дат, purge={PURGE}",
        f"- PLATEAU_TOL=5%, PLATEAU_WINDOW=200 итераций",
        f"- Фичи: {len(FEATURES)}",
        f"- Purge внешний: {PURGE}\n",
        "\n## Walk-Forward folds\n",
        "| fold | train | val | best_iter |",
        "|------|-------|-----|-----------|",
    ]
    for i, (tr, va) in enumerate(folds):
        sub = df[df["fold"] == i + 1]
        bi = int(sub.iloc[0]["best_iter"]) if len(sub) else "-"
        lines.append(f"| {i+1} | {tr[0].date()} → {tr[-1].date()} | {va[0].date()} → {va[-1].date()} | {bi} |")

    lines.append("\n## Метрики по фолдам\n")
    lines.append(df[["fold", "best_iter", "mean_ic", "hit_rate_ic", "ndcg@10",
                     "top_decile_spread", "top10_mean_rs", "top10_median_rs"]].round(4).to_string(index=False))

    lines.append(f"\n**Aggregate:** mean IC = {df['mean_ic'].mean():+.4f}, "
                 f"NDCG@10 = {df['ndcg@10'].mean():.3f}, "
                 f"TOP10 mean RS = {df['top10_mean_rs'].mean():+.4f}, "
                 f"hit_rate = {df['hit_rate_ic'].mean():.1%}\n")

    lines.append("\n## Fold 6 (стрессовый период)\n")
    fold6 = df[df["fold"] == 6]
    if len(fold6) > 0:
        r = fold6.iloc[0]
        lines.append(f"- best_iter = {int(r['best_iter'])}")
        lines.append(f"- IC = {r['mean_ic']:+.4f}")
        lines.append(f"- NDCG@10 = {r['ndcg@10']:.3f}")
        lines.append(f"- TOP10 mean RS = {r['top10_mean_rs']:+.4f}")
        lines.append(f"- top_decile_spread = {r['top_decile_spread']:+.4f}")
        lines.append(f"\nFold 6 — известный стрессовый период (2023-03 → 2023-12). "
                     f"Не использовался для выбора параметров.\n")

    lines.append("\n## График\n")
    lines.append(f"- Equity: `is_wf_regressor_plateau_equity.csv`")
    lines.append(f"- IC curves: `is_wf_regressor_plateau_ic_curves.csv`")
    lines.append(f"- График: `plot_regressor_vs_benchmark.py` → `is_wf_regressor_vs_benchmark.png`\n")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[reg-plateau] отчёт: {OUT_REPORT}")

    pd.DataFrame(equity_records).sort_values("date").to_csv(OUT_EQUITY, index=False)
    print(f"[reg-plateau] equity: {OUT_EQUITY} ({len(equity_records)} строк)")

    if curve_records:
        pd.DataFrame(curve_records).to_csv(OUT_CURVES, index=False)
        print(f"[reg-plateau] IC curves: {OUT_CURVES} ({len(curve_records)} строк)")


if __name__ == "__main__":
    main()