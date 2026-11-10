# backtests/wf_is_rs.py
# ============================================================
# Точка запуска: базовый WF IS эксперимент.
#
# Прогоняет:
#   - base CatBoostRegressor
#   - seed ensemble (5 моделей)
#   - ranker (challenger)
#   - grid search (8 конфигураций, только внутренние фолды)
# Сохраняет Markdown-отчёт в backtests/.
# ============================================================
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from rs_model.config import FEATURES, HORIZON, PURGE, ENSEMBLE_SEEDS, USE_ENSEMBLE
from rs_model.data import build_dataset, to_long
from rs_model.metrics import evaluate
from rs_model.wf import make_folds
from rs_model.base_model import create_base_model, fit_base_model
from rs_model.ranker import create_ranker, fit_ranker
from rs_model.ensemble import SeedEnsemble
from rs_model.grid_search import run_grid

OUT_REPORT = Path(__file__).parent / "wf_is_rs_report.md"


def _predict_wide(df_long: pd.DataFrame, preds) -> pd.DataFrame:
    df = df_long.copy()
    df["pred"] = preds
    return df.pivot(index="date", columns="ticker", values="pred")


def _target_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    return df_long.pivot(index="date", columns="ticker", values="target")


def main():
    split_date = json.loads(
        (Path(__file__).parent / "split_info.json").read_text()
    )["split_date"]
    print(f"[wf] split_date = {split_date}")
    print("[wf] загрузка данных и построение фич...")
    features_raw, features_rank, target_rs, target_gauss, target_grades = build_dataset(split_date)

    dates = target_rs.dropna(how="all").index
    folds = make_folds(dates, n_folds=6, purge=PURGE)
    print(f"[wf] folds: {len(folds)}")
    for i, (tr, va) in enumerate(folds):
        print(f"  fold {i+1}: train {tr[0].date()} → {tr[-1].date()} | "
              f"val {va[0].date()} → {va[-1].date()}")

    # --- базовый регрессор ---
    print("\n[wf] === BASE CatBoostRegressor ===")
    base_results = []
    base_fold_metrics = []
    for i, (tr, va) in enumerate(folds):
        tr_long = to_long(features_rank, target_gauss)
        tr_long = tr_long[tr_long["date"].isin(tr)]
        va_long = to_long(features_rank, target_gauss)
        va_long = va_long[va_long["date"].isin(va)]

        X_tr = tr_long[FEATURES].values; y_tr = tr_long["target"].values
        X_va = va_long[FEATURES].values; y_va = va_long["target"].values

        model = create_base_model(seed=42)
        fit_base_model(model, X_tr, y_tr, X_va, y_va)

        pred = _predict_wide(va_long, model.predict(X_va))
        # для метрик — RS (не gauss), чтобы интерпретировать
        rs_va = target_rs.loc[va_long["date"].unique()]
        m = evaluate(pred, rs_va)
        m["fold"] = i + 1
        base_fold_metrics.append(m)
        print(f"  fold {i+1}: IC={m['mean_ic']:+.4f}, NDCG@10={m['ndcg@10']:.3f}, "
              f"TOP10 mean RS={m['top10_mean_rs']:+.4f}")

    base_df = pd.DataFrame(base_fold_metrics)
    print(f"  aggregate: IC={base_df['mean_ic'].mean():+.4f}, "
          f"NDCG@10={base_df['ndcg@10'].mean():.3f}")

    # --- seed ensemble ---
    print("\n[wf] === SEED ENSEMBLE (5 seeds) ===")
    ens_fold_metrics = []
    if USE_ENSEMBLE:
        for i, (tr, va) in enumerate(folds):
            tr_long = to_long(features_rank, target_gauss)
            tr_long = tr_long[tr_long["date"].isin(tr)]
            va_long = to_long(features_rank, target_gauss)
            va_long = va_long[va_long["date"].isin(va)]

            X_tr = tr_long[FEATURES].values; y_tr = tr_long["target"].values
            X_va = va_long[FEATURES].values; y_va = va_long["target"].values

            ens = SeedEnsemble()
            ens.fit(X_tr, y_tr, X_va, y_va)
            pred = _predict_wide(va_long, ens.predict(X_va))
            rs_va = target_rs.loc[va_long["date"].unique()]
            m = evaluate(pred, rs_va)
            m["fold"] = i + 1
            ens_fold_metrics.append(m)
            print(f"  fold {i+1}: IC={m['mean_ic']:+.4f}, NDCG@10={m['ndcg@10']:.3f}, "
                  f"TOP10 mean RS={m['top10_mean_rs']:+.4f}")
        ens_df = pd.DataFrame(ens_fold_metrics)
        print(f"  aggregate: IC={ens_df['mean_ic'].mean():+.4f}, "
              f"NDCG@10={ens_df['ndcg@10'].mean():.3f}")

    # --- ranker (challenger) ---
    print("\n[wf] === RANKER (YetiRank, challenger) ===")
    ranker_fold_metrics = []
    for i, (tr, va) in enumerate(folds):
        tr_long = to_long(features_rank, target_grades)
        tr_long = tr_long[tr_long["date"].isin(tr)]
        va_long = to_long(features_rank, target_grades)
        va_long = va_long[va_long["date"].isin(va)]

        # группы по датам: для CatBoostRanker group_id — метка группы ПО СТРОКАМ
        # ФИЛЬТР: даты с < 2 строк не годятся для ranking (nontrivial groups)
        MIN_GROUP = 5  # осмысленный ranking — хотя бы 5 акций на дату

        tr_long = tr_long.sort_values(["date", "ticker"]).reset_index(drop=True)
        va_long = va_long.sort_values(["date", "ticker"]).reset_index(drop=True)

        tr_counts = tr_long.groupby("date")["ticker"].transform("count")
        va_counts = va_long.groupby("date")["ticker"].transform("count")
        tr_long = tr_long[tr_counts >= MIN_GROUP].reset_index(drop=True)
        va_long = va_long[va_counts >= MIN_GROUP].reset_index(drop=True)

        if len(tr_long) == 0 or len(va_long) == 0:
            print(f"  fold {i+1}: SKIP (нет групп с >= {MIN_GROUP} акций)")
            continue

        # маппинг дата -> номер группы (для train и val — отдельные пространства)
        tr_date_codes = pd.Categorical(tr_long["date"]).codes
        va_date_codes = pd.Categorical(va_long["date"]).codes

        X_tr = tr_long[FEATURES].values; y_tr = tr_long["target"].values
        X_va = va_long[FEATURES].values; y_va = va_long["target"].values

        # для val group_id — тоже метки строк (не размеры), CatBoost примет
        model = create_ranker(seed=42)
        fit_ranker(model, X_tr, y_tr, tr_date_codes, X_va, y_va, va_date_codes)

        pred = _predict_wide(va_long, model.predict(X_va))
        rs_va = target_rs.loc[va_long["date"].unique()]
        m = evaluate(pred, rs_va)
        m["fold"] = i + 1
        ranker_fold_metrics.append(m)
        print(f"  fold {i+1}: IC={m['mean_ic']:+.4f}, NDCG@10={m['ndcg@10']:.3f}, "
              f"TOP10 mean RS={m['top10_mean_rs']:+.4f}")

    ranker_df = pd.DataFrame(ranker_fold_metrics)
    print(f"  aggregate: IC={ranker_df['mean_ic'].mean():+.4f}, "
          f"NDCG@10={ranker_df['ndcg@10'].mean():.3f}")

    # --- grid search ---
    print("\n[wf] === GRID SEARCH (8 configs, internal folds) ===")
    grid_df = run_grid(features_rank, target_gauss, folds, exclude_last=True)
    print(grid_df[["config_idx", "mean_ic", "params"]].to_string(index=False))
    best_cfg = grid_df.loc[grid_df["mean_ic"].idxmax()]
    print(f"\n  best config: {best_cfg['params']}, IC={best_cfg['mean_ic']:+.4f}")

    # --- отчёт ---
    lines = [
        "# WF IS: Stock Selection RS Model (v1)\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"split_date = {split_date} (VAL не трогаем)\n",
        f"\n## Конфигурация\n",
        f"- Фичи: {len(FEATURES)} (rs_pool v2)",
        f"- Горизонт: {HORIZON} дней",
        f"- Purge: {PURGE} дней",
        f"- Fold'ов: {len(folds)}",
        f"- Ансамбль: {USE_ENSEMBLE} ({ENSEMBLE_SEEDS})\n",
        "\n## Walk-Forward folds\n",
        "| fold | train | val |",
        "|------|-------|-----|",
    ]
    for i, (tr, va) in enumerate(folds):
        lines.append(f"| {i+1} | {tr[0].date()} → {tr[-1].date()} | {va[0].date()} → {va[-1].date()} |")

    lines.append("\n## Base CatBoostRegressor\n")
    lines.append(base_df[["fold", "mean_ic", "hit_rate_ic", "ndcg@10",
                           "top_decile_spread", "top10_mean_rs", "top10_median_rs"]].round(4).to_string(index=False))
    lines.append(f"\n**Aggregate:** mean IC = {base_df['mean_ic'].mean():+.4f}, "
                 f"NDCG@10 = {base_df['ndcg@10'].mean():.3f}, "
                 f"TOP10 mean RS = {base_df['top10_mean_rs'].mean():+.4f}\n")

    if USE_ENSEMBLE:
        lines.append("\n## Seed Ensemble (5 seeds)\n")
        lines.append(ens_df[["fold", "mean_ic", "hit_rate_ic", "ndcg@10",
                              "top_decile_spread", "top10_mean_rs", "top10_median_rs"]].round(4).to_string(index=False))
        lines.append(f"\n**Aggregate:** mean IC = {ens_df['mean_ic'].mean():+.4f}, "
                     f"NDCG@10 = {ens_df['ndcg@10'].mean():.3f}, "
                     f"TOP10 mean RS = {ens_df['top10_mean_rs'].mean():+.4f}\n")

    lines.append("\n## Ranker (YetiRank, challenger)\n")
    lines.append(ranker_df[["fold", "mean_ic", "hit_rate_ic", "ndcg@10",
                             "top_decile_spread", "top10_mean_rs", "top10_median_rs"]].round(4).to_string(index=False))
    lines.append(f"\n**Aggregate:** mean IC = {ranker_df['mean_ic'].mean():+.4f}, "
                 f"NDCG@10 = {ranker_df['ndcg@10'].mean():.3f}, "
                 f"TOP10 mean RS = {ranker_df['top10_mean_rs'].mean():+.4f}\n")

    lines.append("\n## Grid Search (8 configs, internal folds)\n")
    grid_view = grid_df[["config_idx", "mean_ic"]].copy()
    grid_view["params"] = grid_df["params"].astype(str)
    lines.append(grid_view.round(4).to_string(index=False))
    lines.append(f"\n**Best:** config {best_cfg['config_idx']} → {best_cfg['params']}, IC={best_cfg['mean_ic']:+.4f}\n")

    # сравнение
    lines.append("\n## Сравнение моделей\n")
    lines.append("| Модель | Mean IC | NDCG@10 | TOP10 mean RS |")
    lines.append("|--------|---------|---------|---------------|")
    lines.append(f"| Base | {base_df['mean_ic'].mean():+.4f} | {base_df['ndcg@10'].mean():.3f} | {base_df['top10_mean_rs'].mean():+.4f} |")
    if USE_ENSEMBLE:
        lines.append(f"| Ensemble | {ens_df['mean_ic'].mean():+.4f} | {ens_df['ndcg@10'].mean():.3f} | {ens_df['top10_mean_rs'].mean():+.4f} |")
    lines.append(f"| Ranker | {ranker_df['mean_ic'].mean():+.4f} | {ranker_df['ndcg@10'].mean():.3f} | {ranker_df['top10_mean_rs'].mean():+.4f} |")
    lines.append("\n## Замечания\n")
    lines.append("- **Fold 6** — известный стрессовый период (2022–2024), не независимый OOS. "
                 "Использовать для диагностики, не для выбора модели.")
    lines.append("- **Rank-Gauss** — target, MSE на transformed target. Метрики — на исходной RS.")
    lines.append("- **Grid search** — только на внутренних фолдах (fold 1–5), fold 6 не используется для выбора.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[wf] отчёт: {OUT_REPORT}")


if __name__ == "__main__":
    main()