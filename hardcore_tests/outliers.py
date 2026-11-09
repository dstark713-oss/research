# hardcore_tests/outliers.py
# ============================================================
# ТЕСТ 5: Outlier & Data Quality Stress.
#
# Сценарии:
#   A. Обычный IC.
#   B. Winsorize фичи на 1%/99% внутри каждой даты.
#   C. Исключить 1% дат с экстремальным cross-sectional std RS.
#
# PASS: IC меняется < 20% во всех сценариях
# WARN: IC меняется 20-50%
# FAIL: IC меняется > 50% (сигнал держится на выбросах)
# ============================================================
import numpy as np
import pandas as pd

from features.rs_pool import RS_POOL_NAMES
from .data import load_all, daily_ic


# outliers.py — вместо lambda в winsorize_cs
def _winsorize_row(row, low=0.01, high=0.99):
    x = row.dropna()
    if len(x) < 20: return row
    lo = x.quantile(low); hi = x.quantile(high)
    return row.clip(lo, hi)

def winsorize_cs(df, low=0.01, high=0.99):
    return df.apply(_winsorize_row, axis=1, args=(low, high))


def run(split_date=None) -> dict:
    raw, imoex, features, target = load_all(split_date)

    # базовый IC
    base_ic = {}
    for name in RS_POOL_NAMES:
        ics = daily_ic(features[name], target)
        base_ic[name] = ics.mean() if len(ics) else np.nan

    # сценарий B: winsorize фичи
    print("[outliers] winsorize фичи 1%/99%...")
    feats_w = {name: winsorize_cs(f) for name, f in features.items()}
    ic_w = {}
    for name in RS_POOL_NAMES:
        ics = daily_ic(feats_w[name], target)
        ic_w[name] = ics.mean() if len(ics) else np.nan

    # сценарий C: исключить 1% дат с экстремальным cs std RS
    print("[outliers] исключение 1% дат с экстремальным cs std RS...")
    cs_std = target.std(axis=1)
    threshold = cs_std.quantile(0.99)
    keep_dates = cs_std[cs_std <= threshold].index
    target_c = target.loc[keep_dates]
    ic_c = {}
    for name in RS_POOL_NAMES:
        ics = daily_ic(features[name].loc[keep_dates], target_c)
        ic_c[name] = ics.mean() if len(ics) else np.nan

    # сравнение
    print("\n[outliers] Сравнение IC:")
    rows = []
    for name in RS_POOL_NAMES:
        b = base_ic[name]; w = ic_w[name]; c = ic_c[name]
        if not np.isfinite(b) or b == 0: continue
        dw = abs(w - b) / abs(b) if np.isfinite(w) else np.nan
        dc = abs(c - b) / abs(b) if np.isfinite(c) else np.nan
        rows.append({"feature": name, "ic_base": b, "ic_wins": w, "ic_noext": c,
                     "delta_wins_pct": dw, "delta_noext_pct": dc})
        print(f"  {name}: base={b:+.4f}, wins={w:+.4f} (Δ{dw:.0%}), noext={c:+.4f} (Δ{dc:.0%})")

    df = pd.DataFrame(rows)
    max_delta = df[["delta_wins_pct", "delta_noext_pct"]].max().max()
    high_delta = df[(df["delta_wins_pct"] > 0.5) | (df["delta_noext_pct"] > 0.5)]

    if len(high_delta) > 0:
        status = "FAIL"
    elif max_delta > 0.2:
        status = "WARN"
    else:
        status = "PASS"

    print(f"\n[outliers] СТАТУС: {status} (max Δ={max_delta:.0%})")
    return {"status": status, "per_feature": rows, "max_delta": max_delta}