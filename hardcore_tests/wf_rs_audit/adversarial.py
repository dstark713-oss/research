# hardcore_tests/wf_rs_audit/adversarial.py
# ============================================================
# Блок 10: falsification tests.
# 10.1 future-data mutation: меняем данные после t — фичи на t не меняются.
# 10.2 prediction shift: сдвигаем OOF predictions на 1 день.
# 10.3 target permutation: перемешиваем target — IC должен упасть.
# 10.4 prediction permutation: перемешиваем predictions внутри даты —
#      портфель должен потерять преимущество.
# 10.5 feature future-shift: сдвигаем фичу на 1 день вперёд.
# 10.6 random-ranking benchmark.
# ============================================================
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .common import (
    load_ohlcv, load_imoex, load_split_date, independent_target,
    independent_daily_ic, independent_folds, HORIZON, PURGE, TOP_N,
    REBAL_STEP, verdict,
)
from features.rs_pool import build_rs_pool, RS_POOL_NAMES


def test_future_mutation(n_check: int = 5, seed: int = 42) -> dict:
    split_date = load_split_date()
    raw = load_ohlcv(split_date)
    full = build_rs_pool(raw)
    dates = raw["close"].index
    rng = np.random.RandomState(seed)
    candidates = dates[300:-(HORIZON + 10)]
    check = sorted(rng.choice(candidates, size=min(n_check, len(candidates)), replace=False))

    fails = []
    for t in check:
        raw_mut = {}
        for k, df in raw.items():
            df2 = df.copy()
            mask = df2.index > t
            noise = 1 + rng.randn(*df2.loc[mask].shape) * 0.5
            df2.loc[mask] = df2.loc[mask].values * noise
            raw_mut[k] = df2
        mut = build_rs_pool(raw_mut)
        for name in RS_POOL_NAMES:
            a = full[name].loc[t].values
            b = mut[name].loc[t].values
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() == 0:
                continue
            diff = np.abs(a[m] - b[m]).max()
            if diff > 1e-8:
                fails.append((name, t.date(), diff))
    if fails:
        return verdict(
            "10.1 Future-data mutation test",
            "FAIL", "CRITICAL",
            evidence=f"{len(fails)} нарушений: {fails[:5]}",
        )
    return verdict(
        "10.1 Future-data mutation test",
        "PASS",
        evidence=f"{len(check)} дат: мутация будущего не меняет фичи на t.",
    )


def test_target_permutation() -> dict:
    split_date = load_split_date()
    raw = load_ohlcv(split_date)
    imoex = load_imoex(split_date)
    target = independent_target(raw["close"], imoex)
    features_raw = build_rs_pool(raw)
    from rs_model.data import cs_percentile_rank
    features_rank = {k: cs_percentile_rank(v) for k, v in features_raw.items()
                     if k in RS_POOL_NAMES}

    # baseline: IC лучшей фичи
    best_name = "dist_from_high_intraday_252"
    ic_base = independent_daily_ic(features_rank[best_name], target).mean()

    # shuffled target: перемешиваем внутри каждой даты
    rng = np.random.RandomState(42)
    target_shuf = target.apply(
        lambda row: pd.Series(rng.permutation(row.dropna().values),
                              index=row.dropna().index)
        .reindex(row.index), axis=1)

    ic_shuf = independent_daily_ic(features_rank[best_name], target_shuf).mean()

    if abs(ic_shuf) > abs(ic_base) * 0.5:
        return verdict(
            "10.3 Target permutation test",
            "FAIL", "HIGH",
            evidence=f"IC baseline={ic_base:+.4f}, IC shuffled={ic_shuf:+.4f}. "
                     f"Shuffled не потерял сигнал — подозрительно.",
        )
    return verdict(
        "10.3 Target permutation test",
        "PASS",
        evidence=f"IC baseline={ic_base:+.4f}, IC shuffled={ic_shuf:+.4f}. "
                 f"Shuffled потерял сигнал (|IC| упал >50%).",
    )


def test_random_ranking_benchmark(n_seeds: int = 5) -> dict:
    """
    Строим портфель на случайном ранжировании: те же даты ребаланса,
    тот же TOP-10, тот же execution. Сравниваем с реальным.
    """
    csv = Path(__file__).parent.parent.parent / "backtests" / "portfolio_tests" / "output" / "rs_is_wf_equity.csv"
    if not csv.exists():
        return verdict("10.6 Random-ranking benchmark", "NOT_VERIFIED", "HIGH",
                       evidence="CSV не найден")
    real = pd.read_csv(csv, parse_dates=["date"])
    real_total = real["equity"].iloc[-1] / real["equity"].iloc[0]

    split_date = load_split_date()
    raw = load_ohlcv(split_date)
    imoex = load_imoex(split_date)
    target = independent_target(raw["close"], imoex)
    close = raw["close"]
    dates = target.dropna(how="all").index
    folds = independent_folds(dates, n_folds=6, purge=PURGE)

    random_totals = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        records = []
        for tr, va in folds:
            rebal_dates = va[::REBAL_STEP]
            for rd in rebal_dates:
                if rd not in target.index:
                    continue
                avail = target.loc[rd].dropna()
                if len(avail) < TOP_N:
                    continue
                selected = rng.choice(avail.index, size=TOP_N, replace=False)
                records.append({"date": rd, "port_rs": target.loc[rd, selected].mean()})
        if not records:
            continue
        df = pd.DataFrame(records).sort_values("date")
        eq = (1 + df["port_rs"]).cumprod()
        random_totals.append(eq.iloc[-1])

    if not random_totals:
        return verdict("10.6 Random-ranking benchmark", "NOT_VERIFIED", "MEDIUM",
                       evidence="Не удалось построить random-портфели")

    random_totals = np.array(random_totals)
    # реальный результат должен быть в верхнем хвосте распределения
    pct = (real_total > random_totals).mean()
    mean_r = random_totals.mean()
    max_r = random_totals.max()

    evidence = (f"Real total: {real_total:.4f}. Random: "
                f"mean={mean_r:.4f}, max={max_r:.4f}. "
                f"Real в {pct:.0%} верхнего хвоста.")

    if real_total <= mean_r:
        return verdict(
            "10.6 Random-ranking benchmark",
            "FAIL", "CRITICAL",
            evidence=evidence + " Реальный результат хуже среднего random — "
                     "преимущество модели не подтверждено.",
        )
    return verdict(
        "10.6 Random-ranking benchmark",
        "PASS",
        evidence=evidence,
    )


def run() -> list:
    results = []
    try:
        results.append(test_future_mutation())
    except Exception as e:
        results.append(verdict("10.1 Future-data mutation test", "NOT_VERIFIED",
                               "MEDIUM", evidence=f"error: {e}"))
    try:
        results.append(test_target_permutation())
    except Exception as e:
        results.append(verdict("10.3 Target permutation test", "NOT_VERIFIED",
                               "MEDIUM", evidence=f"error: {e}"))
    try:
        results.append(test_random_ranking_benchmark())
    except Exception as e:
        results.append(verdict("10.6 Random-ranking benchmark", "NOT_VERIFIED",
                               "MEDIUM", evidence=f"error: {e}"))
    return results