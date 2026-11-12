# hardcore_tests/wf_rs_audit/independent_rebuild.py
# ============================================================
# Блок 7: независимая реконструкция портфеля.
# Заново обучаем CatBoostRegressor (iter=300), на каждом val —
# ребаланс раз в 20 дней, топ-10, equal-weight.
# Сравниваем equity с rs_is_wf_equity.csv.
#
# ИСПРАВЛЕНО: pd.Series.intersection → Index.intersection.
# ============================================================
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from rs_model.config import FEATURES, BASE_PARAMS
from rs_model.data import build_dataset, to_long
from .common import (
    load_split_date, independent_folds,
    HORIZON, PURGE, TOP_N, REBAL_STEP, verdict,
)


def run() -> dict:
    csv = (Path(__file__).parent.parent.parent
           / "backtests" / "portfolio_tests" / "output" / "rs_is_wf_equity.csv")
    if not csv.exists():
        return verdict(
            "7.1 Независимая реконструкция портфеля",
            "NOT_VERIFIED", "HIGH",
            evidence=f"Нет {csv}",
        )

    split_date = load_split_date()
    features_raw, features_rank, target_rs, target_gauss, _ = build_dataset(split_date)
    dates = target_rs.dropna(how="all").index
    folds = independent_folds(dates, n_folds=6, purge=PURGE)

    params = dict(BASE_PARAMS)
    params["iterations"] = 300
    params["verbose"] = False
    params["allow_writing_files"] = False

    records = []
    for i, (tr, va) in enumerate(folds):
        tr_long = to_long(features_rank, target_gauss)
        tr_long = tr_long[tr_long["date"].isin(tr)].reset_index(drop=True)
        X_tr = tr_long[FEATURES].values
        y_tr = tr_long["target"].values

        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, verbose=False)

        va_long = to_long(features_rank, target_gauss)
        va_long = va_long[va_long["date"].isin(va)].reset_index(drop=True)
        X_va = va_long[FEATURES].values
        va_long["pred"] = model.predict(X_va)
        pred_wide = va_long.pivot(index="date", columns="ticker", values="pred")

        rebal_dates = va[::REBAL_STEP]
        for rd in rebal_dates:
            if rd not in pred_wide.index:
                continue
            pred_row = pred_wide.loc[rd].dropna()
            if len(pred_row) < TOP_N:
                continue
            selected = pred_row.nlargest(TOP_N).index.tolist()
            if rd not in target_rs.index:
                continue
            rs_row = target_rs.loc[rd, selected].dropna()
            if len(rs_row) < TOP_N:
                continue
            records.append({"date": rd, "fold": i + 1, "port_rs": rs_row.mean()})

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    if df.empty:
        return verdict(
            "7.1 Независимая реконструкция портфеля",
            "FAIL", "CRITICAL",
            evidence="Реконструкция пуста — нет ребалансов.",
        )

    # --- сравнение с оригиналом ---
    orig = pd.read_csv(csv, parse_dates=["date"])
    orig = orig.sort_values("date").reset_index(drop=True)
    if "equity" not in orig.columns:
        return verdict(
            "7.1 Независимая реконструкция портфеля",
            "FAIL", "HIGH",
            evidence=f"В {csv} нет колонки equity",
        )

    # ИСПРАВЛЕНО: Index.intersection, а не Series.intersection
    dates_orig = pd.Index(orig["date"])
    dates_recon = pd.Index(df["date"])
    common_dates = dates_orig.intersection(dates_recon)
    if len(common_dates) == 0:
        return verdict(
            "7.1 Независимая реконструкция портфеля",
            "FAIL", "CRITICAL",
            evidence=f"Нет общих дат. "
                     f"orig: {orig['date'].min().date()} → {orig['date'].max().date()} "
                     f"({len(orig)} ребалансов); "
                     f"recon: {df['date'].min().date()} → {df['date'].max().date()} "
                     f"({len(df)} ребалансов)",
        )

    a = orig.set_index("date").loc[common_dates, "equity"]
    b = df.set_index("date").loc[common_dates, "port_rs"]
    b_equity = (1 + b).cumprod()

    # нормируем обе от 1.0 на первой общей дате
    a_norm = a / a.iloc[0]
    b_norm = b_equity / b_equity.iloc[0]

    diff = (a_norm - b_norm).abs()
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    # допуск: 1% (разные seed не используем — seed=42 тот же; если модель
    # детерминирована, diff должен быть ~0)
    if max_diff > 0.01:
        # показываем первые расхождения
        worst = diff.nlargest(3)
        worst_str = "; ".join(
            f"{d.date()}: orig={a_norm.loc[d]:.4f}, recon={b_norm.loc[d]:.4f}"
            for d in worst.index
        )
        return verdict(
            "7.1 Независимая реконструкция портфеля",
            "FAIL", "CRITICAL",
            evidence=f"max diff equity = {max_diff:.4f}, "
                     f"mean diff = {mean_diff:.4f} на {len(common_dates)} датах. "
                     f"Худшие: {worst_str}",
        )

    return verdict(
        "7.1 Независимая реконструкция портфеля",
        "PASS",
        evidence=f"max diff equity = {max_diff:.6f}, "
                 f"mean diff = {mean_diff:.6f} на {len(common_dates)} датах. "
                 f"Реконструкция совпадает с оригиналом.",
    )