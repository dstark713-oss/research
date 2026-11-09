# hardcore_tests/leakage.py
# ============================================================
# ТЕСТ 1: Leakage Audit (жёсткий, FAIL при нарушении).
#
# 1A. Point-in-time invariance (truncation)
# 1B. Future perturbation (изменяем будущее — фича на t не меняется)
# 1C. Target isolation (фича и таргет не пересекаются)
# 1D. Static: нет shift(-n), center=True
# 1E. NaN/Inf sanity
# ============================================================
import numpy as np
import pandas as pd

from features.rs_pool import build_rs_pool, RS_POOL_NAMES
from .data import HORIZON, load_all, build_target


class LeakageFail(Exception): pass


def _truncate_raw(raw, cutoff):
    return {k: df.loc[df.index < cutoff] for k, df in raw.items()}


def test_truncation(raw, features_full, n_check=15, seed=0):
    """1A: фича на T не меняется при обрезке после T."""
    dates = raw["close"].index
    rng = np.random.RandomState(seed)
    candidates = dates[300:-(HORIZON + 10)]
    check = sorted(rng.choice(candidates, size=min(n_check, len(candidates)), replace=False))

    fails = []
    for T in check:
        cutoff = T + pd.Timedelta(days=1)
        raw_tr = _truncate_raw(raw, cutoff)
        feats_tr = build_rs_pool(raw_tr)
        for name in RS_POOL_NAMES:
            a = features_full[name]; b = feats_tr[name]
            common = a.index.intersection(b.index)
            common = common[common <= T]
            if len(common) == 0: continue
            av = a.loc[common].values; bv = b.loc[common].values
            m = np.isfinite(av) & np.isfinite(bv)
            if m.sum() == 0: continue
            diff = np.abs(av[m] - bv[m]).max()
            if diff > 1e-8:
                fails.append((name, T, diff))
    return fails


def test_future_perturbation(raw, features_full, n_check=10, seed=42):
    """1B: меняем будущее (после T) — фича на T не должна меняться."""
    dates = raw["close"].index
    rng = np.random.RandomState(seed)
    candidates = dates[300:-(HORIZON + 10)]
    check = sorted(rng.choice(candidates, size=min(n_check, len(candidates)), replace=False))

    fails = []
    for T in check:
        raw_pert = {}
        for k, df in raw.items():
            df2 = df.copy()
            future_mask = df2.index > T
            # мультипликативный шум на будущее
            noise = 1 + rng.randn(*df2.loc[future_mask].shape) * 0.1
            df2.loc[future_mask] = df2.loc[future_mask].values * noise
            raw_pert[k] = df2
        feats_pert = build_rs_pool(raw_pert)
        for name in RS_POOL_NAMES:
            a = features_full[name].loc[T].values
            b = feats_pert[name].loc[T].values
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() == 0: continue
            diff = np.abs(a[m] - b[m]).max()
            if diff > 1e-8:
                fails.append((name, T, diff))
    return fails


def test_target_isolation(features_full, target):
    """1C: фича и таргет не должны быть идентичны (нет утечки таргета в фичу)."""
    fails = []
    for name in RS_POOL_NAMES:
        f = features_full[name]
        common_dates = f.index.intersection(target.index)
        common_tickers = f.columns.intersection(target.columns)
        fv = f.loc[common_dates, common_tickers].values.flatten()
        tv = target.loc[common_dates, common_tickers].values.flatten()
        m = np.isfinite(fv) & np.isfinite(tv)
        if m.sum() < 500: continue
        c = np.corrcoef(fv[m], tv[m])[0, 1]
        if abs(c) > 0.99:
            fails.append((name, c))
    return fails


def test_static_code():
    """1D: статический анализ rs_pool.py."""
    from pathlib import Path
    pool = Path(__file__).parent.parent / "features" / "rs_pool.py"
    src = pool.read_text(encoding="utf-8")
    fails = []
    for token in ["shift(-", ".shift(-", "center=True"]:
        for line in src.splitlines():
            code = line.split("#", 1)[0]
            if token in code:
                fails.append((token, line.strip()))
    return fails


def test_nan_inf(features_full):
    """1E: NaN/Inf sanity."""
    fails = []
    for name, f in features_full.items():
        arr = f.values
        if np.isinf(arr).any():
            fails.append((name, "has Inf"))
        elif np.isnan(arr).mean() > 0.9:
            fails.append((name, f"{np.isnan(arr).mean():.1%} NaN"))
    return fails


def run(split_date=None) -> dict:
    raw, imoex, features, target = load_all(split_date)

    results = {}

    print("[leakage] 1A truncation...")
    f = test_truncation(raw, features)
    results["1A_truncation"] = {"status": "FAIL" if f else "PASS", "fails": f}
    print(f"  {'FAIL' if f else 'PASS'} ({len(f)} нарушений)")

    print("[leakage] 1B future perturbation...")
    f = test_future_perturbation(raw, features)
    results["1B_future_perturbation"] = {"status": "FAIL" if f else "PASS", "fails": f}
    print(f"  {'FAIL' if f else 'PASS'} ({len(f)} нарушений)")

    print("[leakage] 1C target isolation...")
    f = test_target_isolation(features, target)
    results["1C_target_isolation"] = {"status": "FAIL" if f else "PASS", "fails": f}
    print(f"  {'FAIL' if f else 'PASS'} ({len(f)} нарушений)")

    print("[leakage] 1D static code...")
    f = test_static_code()
    results["1D_static_code"] = {"status": "FAIL" if f else "PASS", "fails": f}
    print(f"  {'FAIL' if f else 'PASS'} ({len(f)} нарушений)")

    print("[leakage] 1E nan/inf...")
    f = test_nan_inf(features)
    results["1E_nan_inf"] = {"status": "FAIL" if f else "PASS", "fails": f}
    print(f"  {'FAIL' if f else 'PASS'} ({len(f)} нарушений)")

    overall = "FAIL" if any(r["status"] == "FAIL" for r in results.values()) else "PASS"
    results["_overall"] = overall
    print(f"\n[leakage] ОБЩИЙ СТАТУС: {overall}")
    return results