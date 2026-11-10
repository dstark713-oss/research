# backtests/is_wf_ranker.py
# ============================================================
# WF IS прогон: Ranker config 5 (YetiRank, depth=4, l2=100).
# 6 фолдов, fold 6 — смотрим (известный стрессовый период).
# Без ансамбля. Без портфельных метрик.
#
# Сохраняет:
#   - backtests/is_wf_ranker_report.md
#   - backtests/is_wf_ranker_equity.csv (для графика)
# ============================================================
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from rs_model.config import FEATURES, HORIZON, PURGE
from rs_model.data import build_dataset, to_long
from rs_model.metrics import evaluate
from rs_model.wf import make_folds
from rs_model.ranker_config5 import create_ranker_config5, fit_ranker_config5

OUT_DIR = Path(__file__).parent
OUT_REPORT = OUT_DIR / "is_wf_ranker_report.md"
OUT_EQUITY = OUT_DIR / "is_wf_ranker_equity.csv"

MIN_GROUP = 5


def _predict_wide(df_long, preds):
    df = df_long.copy()
    df["pred"] = preds
    return df.pivot(index="date", columns="ticker", values="pred")


def main():
    split_date = json.loads(
        (Path(__file__).parent / "split_info.json").read_text()
    )["split_date"]
    print(f"[ranker5] split_date = {split_date}")

    print("[ranker5] загрузка данных...")
    features_raw, features_rank, target_rs, target_gauss, target_grades = build_dataset(split_date)

    dates = target_rs.dropna(how="all").index
    folds = make_folds(dates, n_folds=6, purge=PURGE)
    print(f"[ranker5] folds: {len(folds)}")

    fold_metrics = []
    equity_records = []  # (date, fold, top10_mean_rs_cum)

    for i, (tr_dates, va_dates) in enumerate(folds):
        print(f"\n[ranker5] === fold {i+1}: "
              f"train {tr_dates[0].date()} → {tr_dates[-1].date()} | "
              f"val {va_dates[0].date()} → {va_dates[-1].date()} ===")

        tr_long = to_long(features_rank, target_grades)
        tr_long = tr_long[tr_long["date"].isin(tr_dates)].reset_index(drop=True)
        va_long = to_long(features_rank, target_grades)
        va_long = va_long[va_long["date"].isin(va_dates)].reset_index(drop=True)

        tr_long = tr_long.sort_values(["date", "ticker"]).reset_index(drop=True)
        va_long = va_long.sort_values(["date", "ticker"]).reset_index(drop=True)

        tr_counts = tr_long.groupby("date")["ticker"].transform("count")
        va_counts = va_long.groupby("date")["ticker"].transform("count")
        tr_long = tr_long[tr_counts >= MIN_GROUP].reset_index(drop=True)
        va_long = va_long[va_counts >= MIN_GROUP].reset_index(drop=True)

        if len(tr_long) == 0 or len(va_long) == 0:
            print(f"  SKIP: нет групп >= {MIN_GROUP}")
            continue

        tr_codes = pd.Categorical(tr_long["date"]).codes
        va_codes = pd.Categorical(va_long["date"]).codes

        X_tr = tr_long[FEATURES].values; y_tr = tr_long["target"].values
        X_va = va_long[FEATURES].values; y_va = va_long["target"].values

        model = create_ranker_config5(seed=42)
        fit_ranker_config5(model, X_tr, y_tr, tr_codes, X_va, y_va, va_codes)

        pred = _predict_wide(va_long, model.predict(X_va))
        rs_va = target_rs.loc[va_long["date"].unique()]
        m = evaluate(pred, rs_va)
        m["fold"] = i + 1
        fold_metrics.append(m)
        print(f"  IC={m['mean_ic']:+.4f}, NDCG@10={m['ndcg@10']:.3f}, "
              f"TOP10 mean RS={m['top10_mean_rs']:+.4f}, "
              f"hit={m['hit_rate_ic']:.1%}")

        # --- equity для графика: топ-10 RS по датам ---
        from rs_model.metrics import top_n_rs
        top10_series = top_n_rs(pred, rs_va, n=10)
        # копируем в общий список с меткой фолда
        for d, v in top10_series.items():
            equity_records.append({
                "date": d,
                "fold": i + 1,
                "top10_rs": v,
            })

    df = pd.DataFrame(fold_metrics)
    print("\n[ranker5] === AGGREGATE ===")
    print(f"  mean IC:       {df['mean_ic'].mean():+.4f}")
    print(f"  NDCG@10:       {df['ndcg@10'].mean():.3f}")
    print(f"  TOP10 mean RS: {df['top10_mean_rs'].mean():+.4f}")
    print(f"  hit_rate:      {df['hit_rate_ic'].mean():.1%}")

    # --- отчёт ---
    lines = [
        "# WF IS: Ranker config 5 (YetiRank, depth=4, l2=100)\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"split_date = {split_date}\n",
        f"\n## Конфигурация\n",
        f"- Модель: CatBoostRanker, YetiRank",
        f"- depth=4, l2_leaf_reg=100, random_strength=3, lr=0.03",
        f"- iterations=1500, early_stopping=100, eval_metric=NDCG",
        f"- Фичи: {len(FEATURES)}",
        f"- Purge: {PURGE} дней",
        f"- Ансамбль: нет (single seed=42)\n",
        "\n## Walk-Forward folds\n",
        "| fold | train | val |",
        "|------|-------|-----|",
    ]
    for i, (tr, va) in enumerate(folds):
        lines.append(f"| {i+1} | {tr[0].date()} → {tr[-1].date()} | {va[0].date()} → {va[-1].date()} |")

    lines.append("\n## Метрики по фолдам\n")
    lines.append(df[["fold", "mean_ic", "hit_rate_ic", "ndcg@10",
                     "top_decile_spread", "top10_mean_rs", "top10_median_rs"]].round(4).to_string(index=False))

    lines.append(f"\n**Aggregate:** mean IC = {df['mean_ic'].mean():+.4f}, "
                 f"NDCG@10 = {df['ndcg@10'].mean():.3f}, "
                 f"TOP10 mean RS = {df['top10_mean_rs'].mean():+.4f}, "
                 f"hit_rate = {df['hit_rate_ic'].mean():.1%}\n")

    lines.append("\n## Fold 6 (стрессовый период)\n")
    fold6 = df[df["fold"] == 6]
    if len(fold6) > 0:
        r = fold6.iloc[0]
        lines.append(f"- IC = {r['mean_ic']:+.4f}")
        lines.append(f"- NDCG@10 = {r['ndcg@10']:.3f}")
        lines.append(f"- TOP10 mean RS = {r['top10_mean_rs']:+.4f}")
        lines.append(f"- top_decile_spread = {r['top_decile_spread']:+.4f}")
        lines.append(f"\nFold 6 — известный стрессовый период (пост-СВО, 2023-03 → 2023-12). "
                     f"Не использовался для выбора параметров. Смотрим как диагностику.\n")

    lines.append("\n## График\n")
    lines.append(f"- Equity: `is_wf_ranker_equity.csv`")
    lines.append(f"- График vs бенчмарки: `plot_ranker_vs_benchmark.py` → `is_wf_ranker_vs_benchmark.png`\n")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[ranker5] отчёт: {OUT_REPORT}")

    # --- equity CSV ---
    eq_df = pd.DataFrame(equity_records).sort_values("date")
    eq_df.to_csv(OUT_EQUITY, index=False)
    print(f"[ranker5] equity: {OUT_EQUITY} ({len(eq_df)} строк)")


if __name__ == "__main__":
    main()