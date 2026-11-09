# hardcore_tests/correlation.py
# ============================================================
# ТЕСТ 2: Correlation & Redundancy.
# PASS: все |corr| < 0.7
# WARN: есть пары 0.7 <= |corr| < 0.95
# FAIL: есть пары |corr| >= 0.95 (функциональные дубликаты)
#
# Плюс: проверка на монотонные преобразования (Spearman vs Pearson).
# Плюс: стабильность корреляций по 3 подпериодам.
# ============================================================
import numpy as np
import pandas as pd

from features.rs_pool import RS_POOL_NAMES
from .data import load_all


def compute_corr_matrices(features):
    names = list(features.keys())
    mats = np.array([features[n].values.flatten() for n in names])
    m = np.isfinite(mats).all(axis=0)
    if m.sum() < 100:
        return None, None, names
    pearson = np.corrcoef(mats[:, m])
    # Spearman = Pearson на рангах (для скорости — rank по flatten)
    ranked = np.argsort(np.argsort(mats[:, m], axis=1), axis=1).astype(float)
    spearman = np.corrcoef(ranked)
    return pearson, spearman, names


def find_pairs(C, names, lo, hi=None):
    pairs = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            c = C[i, j]
            if abs(c) >= lo and (hi is None or abs(c) < hi):
                pairs.append((names[i], names[j], c))
    return pairs


def correlation_stability(features, n_splits=3):
    """std(pearson corr) по 3 подпериодам для каждой пары."""
    names = list(features.keys())
    n_dates = min(f.shape[0] for f in features.values())
    step = n_dates // n_splits
    cors = {n: [] for n in names}
    per_period = []
    for k in range(n_splits):
        a, b = k*step, (k+1)*step
        mats = np.array([features[n].iloc[a:b].values.flatten() for n in names])
        m = np.isfinite(mats).all(axis=0)
        if m.sum() < 100:
            per_period.append(None); continue
        per_period.append(np.corrcoef(mats[:, m]))

    unstable = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            vals = [P[i, j] for P in per_period if P is not None]
            if len(vals) >= 2:
                std = np.std(vals)
                if std > 0.3:
                    unstable.append((names[i], names[j], std, vals))
    return unstable


def run(split_date=None) -> dict:
    raw, imoex, features, target = load_all(split_date)
    print("[correlation] расчёт матриц...")
    pearson, spearman, names = compute_corr_matrices(features)
    if pearson is None:
        return {"status": "FAIL", "reason": "мало общих данных"}

    warn_pairs = find_pairs(pearson, names, 0.7, 0.95)
    fail_pairs = find_pairs(pearson, names, 0.95)
    # проверка монотонных дубликатов: если |pearson| высокая, но |spearman| ещё выше
    mono_dupes = find_pairs(spearman, names, 0.95)

    print(f"  |corr| 0.7-0.95: {len(warn_pairs)} пар")
    for a, b, c in warn_pairs:
        print(f"    {a} ↔ {b}: {c:+.3f}")
    print(f"  |corr| >= 0.95: {len(fail_pairs)} пар")
    for a, b, c in fail_pairs:
        print(f"    {a} ↔ {b}: {c:+.3f}")
    print(f"  монотонные дубликаты (|spearman| >= 0.95): {len(mono_dupes)}")
    for a, b, c in mono_dupes:
        print(f"    {a} ↔ {b}: {c:+.3f}")

    print("[correlation] стабильность по 3 подпериодам...")
    unstable = correlation_stability(features)
    print(f"  нестабильных пар (std > 0.3): {len(unstable)}")
    for a, b, s, vals in unstable[:10]:
        print(f"    {a} ↔ {b}: std={s:.3f}, values={[f'{v:+.3f}' for v in vals]}")

    if fail_pairs or mono_dupes:
        status = "FAIL"
    elif warn_pairs or unstable:
        status = "WARN"
    else:
        status = "PASS"

    print(f"\n[correlation] СТАТУС: {status}")
    return {
        "status": status,
        "warn_pairs": warn_pairs,
        "fail_pairs": fail_pairs,
        "mono_dupes": mono_dupes,
        "unstable_pairs": unstable,
    }