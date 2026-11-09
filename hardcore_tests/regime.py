# hardcore_tests/regime.py
# ============================================================
# ТЕСТ 4: Market Regime Stress.
# Разбиваем IS по состоянию IMOEX:
#   - Bull / Bear (IMOEX vs MA200)
#   - Low / High Vol (rolling std 60, медиана)
#
# PASS: IC всех фич сохраняет знак во всех режимах
# WARN: 1-2 фичи теряют знак в одном режиме
# FAIL: >=50% фич теряют знак в одном режиме (системная зависимость)
# ============================================================
import numpy as np
import pandas as pd

from features.rs_pool import RS_POOL_NAMES
from .data import load_all, daily_ic


def build_regimes(imoex: pd.Series) -> dict:
    """Метки режимов по датам."""
    ma200 = imoex.rolling(200, min_periods=100).mean()
    bull = imoex > ma200  # True = bull, False = bear
    vol60 = imoex.pct_change().rolling(60, min_periods=30).std()
    vol_med = vol60.median()
    high_vol = vol60 > vol_med

    return {
        "Bull": bull,
        "Bear": ~bull,
        "LowVol": ~high_vol,
        "HighVol": high_vol,
    }


def run(split_date=None) -> dict:
    raw, imoex, features, target = load_all(split_date)
    regimes = build_regimes(imoex)

    full_ic = {}
    for name in RS_POOL_NAMES:
        ics = daily_ic(features[name], target)
        full_ic[name] = np.sign(ics.mean()) if len(ics) else 0

    rows = []
    for reg_name, reg_mask in regimes.items():
        reg_dates = reg_mask[reg_mask].index
        reg_dates = reg_dates.intersection(target.index)
        if len(reg_dates) < 100:
            continue
        reg_target = target.loc[reg_dates]
        for name in RS_POOL_NAMES:
            ics = daily_ic(features[name].loc[reg_dates], reg_target)
            if len(ics) < 50: continue
            mean_ic = ics.mean()
            signed = mean_ic * full_ic[name]
            rows.append({
                "regime": reg_name,
                "feature": name,
                "mean_ic": mean_ic,
                "signed_ic": signed,
                "n_dates": len(ics),
            })

    df = pd.DataFrame(rows)

    # сколько фич теряют знак в каждом режиме
    print("[regime] Потеря знака по режимам:")
    fail_regimes = []
    for reg in regimes:
        sub = df[df["regime"] == reg]
        if len(sub) == 0: continue
        lost = (sub["signed_ic"] < 0).sum()
        total = len(sub)
        ratio = lost / total if total > 0 else 0
        print(f"  {reg}: {lost}/{total} фич теряют знак ({ratio:.0%})")
        if ratio >= 0.5:
            fail_regimes.append(reg)

    # сводка по фичам: в скольких режимах теряют знак
    print("\n[regime] Фичи, теряющие знак:")
    feat_lost = df.groupby("feature")["signed_ic"].apply(lambda s: (s < 0).sum()).sort_values(ascending=False)
    for name, n in feat_lost.items():
        if n > 0:
            print(f"  {name}: теряет знак в {n} режимах")

    if fail_regimes:
        status = "FAIL"
    elif (feat_lost > 0).any():
        status = "WARN"
    else:
        status = "PASS"

    print(f"\n[regime] СТАТУС: {status}")
    return {"status": status, "per_regime": rows, "fail_regimes": fail_regimes}