# backtests/smoke_wf_is_rs.py
# ============================================================
# Быстрый smoke: 1 фолд, все модели (base, ensemble, ranker), grid на 2 конфигах.
# Цель — отловить ошибки формы/типов за 30-60 секунд.
# НЕ для оценки качества. Полный прогон — backtests/wf_is_rs.py.
# ============================================================
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from rs_model.config import FEATURES, HORIZON, PURGE, GRID
from rs_model.data import build_dataset, to_long
from rs_model.metrics import evaluate
from rs_model.wf import make_folds
from rs_model.base_model import create_base_model, fit_base_model
from rs_model.ranker import create_ranker, fit_ranker
from rs_model.ensemble import SeedEnsemble


MIN_GROUP = 5


def _predict_wide(df_long, preds):
    df = df_long.copy()
    df["pred"] = preds
    return df.pivot(index="date", columns="ticker", values="pred")


def smoke():
    print("=" * 60)
    print("SMOKE: WF IS RS")
    print("=" * 60)

    split_date = json.loads(
        (Path(__file__).parent / "split_info.json").read_text()
    )["split_date"]
    print(f"split_date = {split_date}")

    print("\n[1] загрузка данных...")
    features_raw, features_rank, target_rs, target_gauss, target_grades = build_dataset(split_date)
    print(f"  фич: {len(features_rank)}, дат: {len(target_rs)}")

    dates = target_rs.dropna(how="all").index
    folds = make_folds(dates, n_folds=6, purge=PURGE)
    print(f"  фолдов: {len(folds)}")

    # --- берём ОДИН фолд: fold 5 (предпоследний, не стрессовый) ---
    fold_idx = min(4, len(folds) - 1)  # 0-based, fold 5
    train_dates, val_dates = folds[fold_idx]
    print(f"\n[2] SMOKE на fold {fold_idx+1}: "
          f"train {train_dates[0].date()} → {train_dates[-1].date()} | "
          f"val {val_dates[0].date()} → {val_dates[-1].date()}")

    # ---------- BASE ----------
    print("\n[3] BASE CatBoostRegressor...")
    try:
        tr_long = to_long(features_rank, target_gauss)
        tr_long = tr_long[tr_long["date"].isin(train_dates)]
        va_long = to_long(features_rank, target_gauss)
        va_long = va_long[va_long["date"].isin(val_dates)]
        print(f"  train: {len(tr_long)}, val: {len(va_long)}")

        X_tr = tr_long[FEATURES].values; y_tr = tr_long["target"].values
        X_va = va_long[FEATURES].values; y_va = va_long["target"].values

        model = create_base_model(seed=42)
        fit_base_model(model, X_tr, y_tr, X_va, y_va)

        pred = _predict_wide(va_long, model.predict(X_va))
        rs_va = target_rs.loc[va_long["date"].unique()]
        m = evaluate(pred, rs_va)
        print(f"  OK: IC={m['mean_ic']:+.4f}, NDCG@10={m['ndcg@10']:.3f}, "
              f"TOP10 mean RS={m['top10_mean_rs']:+.4f}")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False

    # ---------- ENSEMBLE ----------
    print("\n[4] SEED ENSEMBLE (5 seeds, но в smoke — 2 seeds для скорости)...")
    try:
        ens = SeedEnsemble(seeds=[42, 43])  # smoke: 2 seeds
        ens.fit(X_tr, y_tr, X_va, y_va)
        pred_ens = _predict_wide(va_long, ens.predict(X_va))
        m_ens = evaluate(pred_ens, rs_va)
        print(f"  OK: IC={m_ens['mean_ic']:+.4f}, NDCG@10={m_ens['ndcg@10']:.3f}")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False

    # ---------- RANKER ----------
    print("\n[5] RANKER (YetiRank)...")
    try:
        tr_rank = to_long(features_rank, target_grades)
        tr_rank = tr_rank[tr_rank["date"].isin(train_dates)].reset_index(drop=True)
        va_rank = to_long(features_rank, target_grades)
        va_rank = va_rank[va_rank["date"].isin(val_dates)].reset_index(drop=True)

        tr_rank = tr_rank.sort_values(["date", "ticker"]).reset_index(drop=True)
        va_rank = va_rank.sort_values(["date", "ticker"]).reset_index(drop=True)

        # фильтр групп < MIN_GROUP
        tr_counts = tr_rank.groupby("date")["ticker"].transform("count")
        va_counts = va_rank.groupby("date")["ticker"].transform("count")
        tr_rank = tr_rank[tr_counts >= MIN_GROUP].reset_index(drop=True)
        va_rank = va_rank[va_counts >= MIN_GROUP].reset_index(drop=True)

        print(f"  после фильтра MIN_GROUP={MIN_GROUP}: train {len(tr_rank)}, val {len(va_rank)}")
        print(f"  train групп: {tr_rank['date'].nunique()}, val групп: {va_rank['date'].nunique()}")
        if len(tr_rank) == 0 or len(va_rank) == 0:
            print("  FAIL: нет данных после фильтра")
            return False

        tr_codes = pd.Categorical(tr_rank["date"]).codes
        va_codes = pd.Categorical(va_rank["date"]).codes
        print(f"  group_train: len={len(tr_codes)}, n_groups={tr_codes.max()+1}")
        print(f"  group_val:   len={len(va_codes)}, n_groups={va_codes.max()+1}")

        X_tr_r = tr_rank[FEATURES].values; y_tr_r = tr_rank["target"].values
        X_va_r = va_rank[FEATURES].values; y_va_r = va_rank["target"].values

        model_r = create_ranker(seed=42)
        fit_ranker(model_r, X_tr_r, y_tr_r, tr_codes, X_va_r, y_va_r, va_codes)

        pred_r = _predict_wide(va_rank, model_r.predict(X_va_r))
        rs_va_r = target_rs.loc[va_rank["date"].unique()]
        m_r = evaluate(pred_r, rs_va_r)
        print(f"  OK: IC={m_r['mean_ic']:+.4f}, NDCG@10={m_r['ndcg@10']:.3f}, "
              f"TOP10 mean RS={m_r['top10_mean_rs']:+.4f}")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False

    # ---------- GRID (2 конфига) ----------
    print("\n[6] GRID SEARCH (2 конфига, 1 фолд)...")
    try:
        from rs_model.grid_search import _prepare_xy  # проверим, что модуль работает
        # прогоняем только 2 конфига вручную (без run_grid, чтобы не гонять все фолды)
        for cfg_idx, cfg in enumerate(GRID[:2]):
            m_cfg = create_base_model(cfg, seed=42)
            fit_base_model(m_cfg, X_tr, y_tr, X_va, y_va)
            pred_cfg = _predict_wide(va_long, m_cfg.predict(X_va))
            m_eval = evaluate(pred_cfg, rs_va)
            print(f"  config {cfg_idx}: IC={m_eval['mean_ic']:+.4f}  params={cfg}")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("SMOKE: ВСЁ OK")
    print("=" * 60)
    return True


if __name__ == "__main__":
    ok = smoke()
    exit(0 if ok else 1)