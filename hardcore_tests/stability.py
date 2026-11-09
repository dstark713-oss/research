# hardcore_tests/stability.py
# ============================================================
# ТЕСТ 3: IS Subperiod Stability (6 фолдов + purge 20d).
#
# PASS: все 6 фолдов имеют IC того же знака, пул не collapse
# WARN: 1-2 фолда с другим знаком
# FAIL: pool collapse (>=50% фич теряют знак одновременно)
#
# Метрики на фолд:
#   mean IC, ICIR, hit_rate, n_dates, n_tickers
# Плюс: блочный bootstrap CI (20-дневные блоки)
# Плюс: сводные pool-метрики:
#   positive_ic_ratio, median|IC|, mean|IC|, effective_features
# ============================================================
import numpy as np
import pandas as pd

from features.rs_pool import RS_POOL_NAMES
from .data import load_all, daily_ic, HORIZON


def split_into_folds(dates, n_folds=6, purge=HORIZON):
    """Разбивает даты на n_folds непересекающихся блоков с purge."""
    n = len(dates)
    fold_size = n // n_folds
    folds = []
    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_folds - 1 else n
        # purge: обрезаем начало на purge (чтобы не пересекаться с предыдущим)
        if k > 0:
            start = min(start + purge, end)
        folds.append(dates[start:end])
    return folds


def block_bootstrap_ci(ics, block_size=HORIZON, n_boot=200):
    """95% CI для mean IC с блочным bootstrap."""
    ics = ics.dropna().values
    if len(ics) < block_size * 2:
        return np.nan, np.nan
    n = len(ics)
    means = []
    rng = np.random.RandomState(42)
    n_blocks = n // block_size
    for _ in range(n_boot):
        starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([ics[s:s+block_size] for s in starts])
        means.append(sample.mean())
    return np.percentile(means, [2.5, 97.5])


def run(split_date=None, n_folds=6) -> dict:
    raw, imoex, features, target = load_all(split_date)
    dates = target.index

    folds = split_into_folds(dates, n_folds=n_folds)

    # per fold per feature
    rows = []
    fold_summary = []

    for fold_idx, fold_dates in enumerate(folds):
        fold_target = target.loc[fold_dates]
        fold_ics = {}
        for name in RS_POOL_NAMES:
            ics = daily_ic(features[name].loc[fold_dates], fold_target)
            fold_ics[name] = ics
            if len(ics) < 30:
                continue
            mean_ic = ics.mean()
            std_ic = ics.std()
            ir = mean_ic / std_ic if std_ic > 0 else np.nan
            hit = (ics > 0).mean()
            ci_low, ci_high = block_bootstrap_ci(ics)
            rows.append({
                "fold": fold_idx + 1,
                "feature": name,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "ir": ir,
                "hit_rate": hit,
                "n_dates": len(ics),
                "ci_low": ci_low,
                "ci_high": ci_high,
            })

        # pool-метрики для фолда
        fold_df = pd.DataFrame([r for r in rows if r["fold"] == fold_idx + 1])
        if len(fold_df) == 0:
            continue
        # "положительный IC" — после фиксации направления (знак из общего IC на full IS)
        full_ic_signs = {}
        for name in RS_POOL_NAMES:
            full_ics = daily_ic(features[name], target)
            full_ic_signs[name] = np.sign(full_ics.mean()) if len(full_ics) else 0
        fold_df["signed_ic"] = fold_df.apply(
            lambda r: r["mean_ic"] * full_ic_signs.get(r["feature"], 1), axis=1
        )
        pos_ratio = (fold_df["signed_ic"] > 0).mean()
        med_abs = fold_df["mean_ic"].abs().median()
        mean_abs = fold_df["mean_ic"].abs().mean()
        # effective: сколько фич сохраняют положительный signed_ic и |IC| > 0.01
        effective = ((fold_df["signed_ic"] > 0) & (fold_df["mean_ic"].abs() > 0.01)).sum()
        fold_summary.append({
            "fold": fold_idx + 1,
            "n_dates": fold_df["n_dates"].mean(),
            "positive_ic_ratio": pos_ratio,
            "median_abs_ic": med_abs,
            "mean_abs_ic": mean_abs,
            "effective_features": effective,
        })

    # pool collapse: >=50% фич с отрицательным signed_ic одновременно
    pool_collapse = any(s["positive_ic_ratio"] < 0.5 for s in fold_summary)

    # сводка
    print("\n[stability] Pool-метрики по фолдам:")
    df_sum = pd.DataFrame(fold_summary)
    print(df_sum.to_string(index=False))

    # per-feature stability
    df_rows = pd.DataFrame(rows)
    print("\n[stability] Стабильность по фичам (знак IC по фолдам):")
    unstable = []
    for name in RS_POOL_NAMES:
        sub = df_rows[df_rows["feature"] == name]
        if len(sub) < 3: continue
        signs = np.sign(sub["mean_ic"])
        if len(set(signs)) > 1:
            unstable.append(name)
            print(f"  ⚠ {name}: знаки IC по фолдам {signs.tolist()}")

    if pool_collapse:
        status = "FAIL"
    elif len(unstable) > 0:
        status = "WARN"
    else:
        status = "PASS"

    print(f"\n[stability] СТАТУС: {status}")
    return {
        "status": status,
        "fold_summary": fold_summary,
        "per_feature": rows,
        "unstable_features": unstable,
        "pool_collapse": pool_collapse,
    }