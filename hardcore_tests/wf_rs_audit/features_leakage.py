# hardcore_tests/wf_rs_audit/features_leakage.py
# ============================================================
# Блок 2: look-ahead в признаках.
# 1. Проверка кода rs_pool.py на shift(-n), center=True.
# 2. Causality test: обрезаем данные после t, пересчитываем фичи,
#    сравниваем с полными на дату t.
# 3. Проверка: cs_percentile_rank — per-date, не глобальный.
# ============================================================
import numpy as np
import pandas as pd
from pathlib import Path

from features.rs_pool import build_rs_pool, RS_POOL_NAMES
from .common import load_ohlcv, load_split_date, verdict


def test_static_code() -> dict:
    pool = Path(__file__).parent.parent.parent / "features" / "rs_pool.py"
    src = pool.read_text(encoding="utf-8")
    bad = []
    for token in ["shift(-", ".shift(-", "center=True"]:
        for line_no, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            if token in code:
                bad.append((token, line_no, line.strip()))
    if bad:
        return verdict(
            "2.1 Static: нет shift(-n)/center=True в rs_pool.py",
            "FAIL", "CRITICAL",
            evidence=f"Найдено: {bad}",
            files=["features/rs_pool.py"],
        )
    return verdict(
        "2.1 Static: нет shift(-n)/center=True в rs_pool.py",
        "PASS",
        evidence="Все rolling/shift — только с положительными окнами.",
        files=["features/rs_pool.py"],
    )


def test_causality(n_check: int = 10, seed: int = 0) -> dict:
    split_date = load_split_date()
    raw = load_ohlcv(split_date)
    full = build_rs_pool(raw)
    dates = raw["close"].index
    if len(dates) < 400:
        return verdict("2.2 Causality test", "NOT_VERIFIED", "MEDIUM",
                       evidence="мало дат")

    rng = np.random.RandomState(seed)
    candidates = dates[300:-(20 + 10)]
    check = sorted(rng.choice(candidates, size=min(n_check, len(candidates)), replace=False))

    fails = []
    for t in check:
        cutoff = t + pd.Timedelta(days=1)
        raw_tr = {k: df.loc[df.index < cutoff] for k, df in raw.items()}
        tr = build_rs_pool(raw_tr)
        for name in RS_POOL_NAMES:
            a = full[name].loc[t]
            b = tr[name].loc[t] if t in tr[name].index else None
            if b is None:
                continue
            av = a.values
            bv = b.values
            m = np.isfinite(av) & np.isfinite(bv)
            if m.sum() == 0:
                continue
            diff = np.abs(av[m] - bv[m]).max()
            if diff > 1e-8:
                fails.append((name, t.date(), diff))
    if fails:
        return verdict(
            "2.2 Causality test (truncate after t)",
            "FAIL", "CRITICAL",
            evidence=f"Найдено {len(fails)} нарушений: {fails[:5]}",
            files=["features/rs_pool.py"],
        )
    return verdict(
        "2.2 Causality test (truncate after t)",
        "PASS",
        evidence=f"{len(check)} дат: фичи на t не меняются при обрезке после t.",
        files=["features/rs_pool.py"],
    )


def test_cs_rank_per_date() -> dict:
    """Проверка: cs_percentile_rank — per-date, не глобальный."""
    # Создаём синтетику: 5 дат × 5 тикеров, значения = дата * 100 + тикер
    dates = pd.date_range("2020-01-01", periods=5)
    cols = [f"T{i}" for i in range(5)]
    data = np.array([[d * 100 + i for i in range(5)] for d in range(5)])
    df = pd.DataFrame(data, index=dates, columns=cols)

    from rs_model.data import cs_percentile_rank
    ranked = cs_percentile_rank(df)

    # Если per-date: на каждой дате ранги = [0.2, 0.4, 0.6, 0.8, 1.0]
    expected = np.array([[0.2, 0.4, 0.6, 0.8, 1.0]] * 5)
    diff = np.abs(ranked.values - expected).max()
    if diff > 1e-9:
        return verdict(
            "2.3 cs_percentile_rank — per-date",
            "FAIL", "CRITICAL",
            evidence=f"Ранги не совпадают с per-date. max diff = {diff:.4f}",
            files=["rs_model/data.py::cs_percentile_rank"],
        )
    return verdict(
        "2.3 cs_percentile_rank — per-date",
        "PASS",
        evidence="На синтетике: ранги per-date = [0.2,0.4,0.6,0.8,1.0] на каждой дате.",
        files=["rs_model/data.py::cs_percentile_rank"],
    )


def run() -> list:
    return [test_static_code(), test_causality(), test_cs_rank_per_date()]