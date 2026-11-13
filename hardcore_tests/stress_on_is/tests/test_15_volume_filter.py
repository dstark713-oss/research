# hardcore_tests/stress_on_is/tests/test_15_volume_filter.py
# ============================================================
# Тест 15: объёмный фильтр.
#
# Модель обучается на всём универсуме (185 акций),
# но покупает только из ТОП-N по объёму.
# N = 25/50/75/100.
#
# ТОП-N на прошлом фолде (для первого — за год до начала).
# ============================================================
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from rs_model.config import FEATURES
from rs_model.data import build_dataset, to_long
from rs_model.wf import make_folds
from regime_model.garch.data_loader import (
    load_split_date, load_stock_close, load_ml_holdings,
)
from regime_model.garch.portfolio_returns import simulate_buy_and_hold
from regime_model.garch.garch_model import fit_garch
from regime_model.garch.sizing import vol_to_exposure
from backtests.portfolio_tests.rs_is_wf_v2 import MODEL_PARAMS, PURGE, TOP_N, REBAL_STEP

from ..common.metrics import compute_metrics
from ..common.reporting import save_report, save_figure

VOLUME_TOP_N = [25, 50, 75, 100]
GARCH_MIN_TRAIN = 150
TARGET_VOL = 0.15


def _get_volume_rank(close: pd.DataFrame, rebal_date: pd.Timestamp,
                     prev_fold_start: pd.Timestamp = None) -> pd.Series:
    """
    Возвращает Series: тикер -> средний объём за окно.
    Окно: прошлый фолд (или за год до rebal_date, если prev_fold_start=None).
    """
    if prev_fold_start is None:
        # для первого фолда — за год до начала
        start = rebal_date - pd.Timedelta(days=365)
    else:
        start = prev_fold_start
    end = rebal_date
    mask = (close.index >= start) & (close.index < end)
    if mask.sum() < 20:
        # fallback: за 20 дней до rebal_date
        mask = (close.index >= rebal_date - pd.Timedelta(days=30)) & (close.index < rebal_date)
    return close.loc[mask].mean().dropna()


def _run_rs_wf_with_volume_filter(split_date: str, vol_top_n: int,
                                   verbose: bool = False) -> pd.DataFrame:
    """
    RS WF с объёмным фильтром:
    - Модель обучается на всём универсуме.
    - На каждой rebal_date — предсказание, но selected только из ТОП-N по объёму.
    Возвращает DataFrame с date, ret, fold.
    """
    features_raw, features_rank, target_rs, target_gauss, _ = build_dataset(split_date)
    dates = target_rs.dropna(how="all").index
    folds = make_folds(dates, n_folds=6, purge=PURGE)

    close = load_stock_close(split_date)
    daily_ret_all = close.pct_change()
    volume = close  # для объёма используем close (прокси; или volume из raw)

    # Настоящий объём — из raw (но load_stock_close возвращает close).
    # Загрузим volume отдельно.
    import sqlite3
    from ..config import ROOT
    db = ROOT / "dataset" / "sql" / "stocks.sqlite"
    conn = sqlite3.connect(db)
    vol_df = pd.read_sql(
        "SELECT ticker, date, volume FROM prices WHERE date < ?",
        conn, params=(split_date,),
    )
    conn.close()
    vol_df["date"] = pd.to_datetime(vol_df["date"])
    volume = vol_df.pivot(index="date", columns="ticker", values="volume").sort_index()

    records = []
    prev_fold_start = None

    for i, (tr_dates, va_dates) in enumerate(folds):
        if verbose:
            print(f"fold {i+1}: train {tr_dates[0].date()} → {tr_dates[-1].date()} | "
                  f"val {va_dates[0].date()} → {va_dates[-1].date()}")

        tr_long = to_long(features_rank, target_gauss)
        tr_long = tr_long[tr_long["date"].isin(tr_dates)].reset_index(drop=True)
        if len(tr_long) < 100:
            continue
        X_tr = tr_long[FEATURES].values
        y_tr = tr_long["target"].values

        model = CatBoostRegressor(**MODEL_PARAMS)
        model.fit(X_tr, y_tr, verbose=False)

        va_long = to_long(features_rank, target_gauss)
        va_long = va_long[va_long["date"].isin(va_dates)].reset_index(drop=True)
        if len(va_long) == 0:
            continue
        X_va = va_long[FEATURES].values
        va_long["pred"] = model.predict(X_va)
        pred_wide = va_long.pivot(index="date", columns="ticker", values="pred")

        rebal_dates = va_dates[::REBAL_STEP]
        for j, rd in enumerate(rebal_dates):
            if rd not in pred_wide.index:
                continue
            pred_row = pred_wide.loc[rd].dropna()
            if len(pred_row) < TOP_N:
                continue

            # --- объёмный фильтр ---
            vol_rank = _get_volume_rank(volume, rd, prev_fold_start)
            top_n_tickers = set(vol_rank.nlargest(vol_top_n).index)

            # оставляем только те, что в ТОП-N по объёму
            pred_filtered = pred_row[pred_row.index.isin(top_n_tickers)]
            if len(pred_filtered) < TOP_N:
                # если меньше TOP_N — берём всё, что есть
                selected = pred_filtered.nlargest(len(pred_filtered)).index.tolist()
            else:
                selected = pred_filtered.nlargest(TOP_N).index.tolist()

            if len(selected) == 0:
                continue

            if j + 1 < len(rebal_dates):
                next_rd = rebal_dates[j + 1]
            else:
                next_rd = va_dates[-1]

            bh = simulate_buy_and_hold(daily_ret_all, selected, rd, next_rd)
            for d, r in bh.items():
                records.append({"date": d, "ret": r, "fold": i + 1})

        prev_fold_start = va_dates[0]

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df
    df = df.drop_duplicates("date", keep="first").sort_values("date").reset_index(drop=True)
    return df


def _apply_garch_daily(daily_df: pd.DataFrame) -> pd.Series:
    """GARCH daily WF поверх daily_df (как в freeze_model)."""
    daily_df = daily_df.copy()
    daily_df["sized_ret"] = daily_df["ret"].copy()
    daily_df["exposure"] = 1.0
    unique_folds = sorted(daily_df["fold"].unique())

    for fold_idx in unique_folds:
        if fold_idx == 1:
            continue
        train_mask = daily_df["fold"] < fold_idx
        val_mask = daily_df["fold"] == fold_idx
        train_ret = pd.Series(
            daily_df.loc[train_mask, "ret"].values,
            index=pd.DatetimeIndex(daily_df.loc[train_mask, "date"])
        )
        val_ret = pd.Series(
            daily_df.loc[val_mask, "ret"].values,
            index=pd.DatetimeIndex(daily_df.loc[val_mask, "date"])
        )
        if len(train_ret) < GARCH_MIN_TRAIN:
            continue
        try:
            res = fit_garch(train_ret)
            p = res.params
            omega = p.get("omega", 0.0) / (100 ** 2)
            alpha = p.get("alpha[1]", 0.0)
            beta = p.get("beta[1]", 0.0)
            mu = p.get("mu", 0.0) / 100
            last_var = float(res.conditional_volatility.iloc[-1] ** 2) / (100 ** 2)
            exposure = {}
            for d in val_ret.index:
                vol_ann = np.sqrt(last_var) * np.sqrt(252)
                e = vol_to_exposure(vol_ann, target_vol=TARGET_VOL)
                exposure[d] = e
                ret = val_ret.loc[d]
                if not np.isfinite(ret): ret = 0.0
                eps = ret - mu
                last_var = max(omega + alpha * eps**2 + beta * last_var, 1e-10)
            exp_series = pd.Series(exposure).sort_index()
            val_indices = daily_df.index[val_mask]
            daily_df.loc[val_indices, "exposure"] = exp_series.values
            daily_df.loc[val_indices, "sized_ret"] = (
                daily_df.loc[val_indices, "ret"].values * exp_series.values
            )
        except Exception:
            continue

    return pd.Series(
        daily_df["sized_ret"].values,
        index=pd.DatetimeIndex(daily_df["date"])
    ).sort_index()


def run(verbose: bool = True) -> dict:
    t0 = time.time()
    split_date = load_split_date()
    if verbose:
        print(f"split_date = {split_date}")

    results = {}
    for n in VOLUME_TOP_N:
        if verbose:
            print(f"\n[volume_top_{n}] прогон RS WF...")
        df = _run_rs_wf_with_volume_filter(split_date, n, verbose=verbose)
        if len(df) == 0:
            if verbose:
                print(f"  нет данных, пропуск")
            continue
        sized = _apply_garch_daily(df)
        m = compute_metrics(sized)
        results[f"top_{n}"] = m
        if verbose:
            print(f"  CAGR={m['CAGR']*100:+.2f}%, MDD={m['MDD']*100:+.2f}%, "
                  f"Sharpe={m['Sharpe']:.3f}")

    # baseline (frozen)
    from ..common.data_loader import load_frozen_equity
    frozen = load_frozen_equity()
    baseline = pd.Series(
        frozen["sized_ret"].values,
        index=pd.DatetimeIndex(frozen["date"])
    ).sort_index()
    m_baseline = compute_metrics(baseline)
    results["baseline"] = m_baseline

    # отчёт
    lines = [
        "# Тест 15: объёмный фильтр",
        f"Дата: {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        f"\n## Гипотеза",
        "Ограничение покупок ТОП-N по объёму (25/50/75/100) снизит ликвидностный риск, "
        "но может ухудшить Sharpe, если модель теряет сильные сигналы в неликвидных бумагах.",
        f"\n## Результаты",
        "| Фильтр | CAGR | MDD | VOL | Sharpe | Sortino | Calmar |",
        "|--------|------|-----|-----|--------|---------|--------|",
    ]
    for name, m in results.items():
        if not m: continue
        lines.append(
            f"| {name} | {m['CAGR']*100:+.2f}% | {m['MDD']*100:+.2f}% | "
            f"{m['VOL']*100:+.2f}% | {m['Sharpe']:.3f} | {m['Sortino']:.3f} | {m['Calmar']:.3f} |"
        )
    lines.append(f"\n## Время: {time.time()-t0:.1f}s")
    lines.append(f"\n## Вердикт: см. таблицу (сравнение с baseline)")

    save_report("test_15_volume_filter", lines)
    return results


if __name__ == "__main__":
    run()