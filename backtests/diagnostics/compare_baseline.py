# backtests/diagnostics/compare_baseline.py
# ============================================================
# Сравнение baseline из rs_is_wf_v2.run_rs_wf и freeze_model.
# Ищем, на каких датах расходятся дневные доходности.
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path

from regime_model.garch.data_loader import load_split_date

# 1. Baseline из рабочего отчета
from backtests.portfolio_tests.rs_is_wf_v2 import run_rs_wf

# 2. Baseline из freeze_model (наша реализация)
from freeze_model.pipeline import _run_rs_wf_collect
from backtests.rm_iswf.metrics import compute_metrics_from_daily

def main():
    split_date = load_split_date()
    print(f"split_date = {split_date}\n")

    # --- 1. Из рабочего отчета ---
    print("[1] run_rs_wf из rs_is_wf_v2...")
    res_ref = run_rs_wf(split_date, rebalance_step=20, verbose=False)
    daily_ref = res_ref["daily_ret"]
    print(f"    n={len(daily_ref)}, "
          f"{daily_ref.index.min().date()} → {daily_ref.index.max().date()}")

    # --- 2. Из freeze_model ---
    print("\n[2] _run_rs_wf_collect из freeze_model...")
    df_frz = _run_rs_wf_collect(split_date, verbose=False)
    daily_frz = pd.Series(
        df_frz["ret"].values, 
        index=pd.DatetimeIndex(df_frz["date"])
    )
    print(f"    n={len(daily_frz)}, "
          f"{daily_frz.index.min().date()} → {daily_frz.index.max().date()}")

    # --- 3. Сравнение ---
    print("\n[3] Сравнение на общих датах...")
    common = daily_ref.index.intersection(daily_frz.index)
    print(f"    common dates: {len(common)}")
    
    a = daily_ref.loc[common]
    b = daily_frz.loc[common]
    diff = (a - b).abs()
    
    print(f"    max diff: {diff.max():.6f}")
    print(f"    mean diff: {diff.mean():.6f}")
    print(f"    identical: {np.allclose(a, b, atol=1e-6)}")

    # --- 4. Где расходится ---
    if diff.max() > 1e-6:
        print("\n[4] Первые 10 расхождений (дата, ref, freeze, diff):")
        bad = diff[diff > 1e-6].sort_index().head(10)
        for d, _ in bad.items():
            print(f"    {d.date()}: ref={a.loc[d]:+.6f}, frz={b.loc[d]:+.6f}, diff={diff.loc[d]:.6f}")
    
    # --- 5. Даты, которые есть только в одном ---
    only_ref = daily_ref.index.difference(daily_frz.index)
    only_frz = daily_frz.index.difference(daily_ref.index)
    
    if len(only_ref) > 0:
        print(f"\n[5] Даты только в ref ({len(only_ref)}):")
        for d in only_ref[:10]:
            print(f"    {d.date()}: ret={daily_ref.loc[d]:+.6f}")
            
    if len(only_frz) > 0:
        print(f"\n[6] Даты только в freeze ({len(only_frz)}):")
        for d in only_frz[:10]:
            print(f"    {d.date()}: ret={daily_frz.loc[d]:+.6f}")

    # --- 7. Метрики ---
    print("\n[7] Метрики (для проверки):")
    m_ref = compute_metrics_from_daily(daily_ref)
    m_frz = compute_metrics_from_daily(daily_frz)
    
    print(f"    {'Метрика':<12} {'Ref (rs_is_wf)':>15} {'Freeze':>15}")
    print("    " + "-" * 45)
    for k in ["CAGR", "MDD", "VOL", "Sharpe"]:
        v_ref = m_ref.get(k, np.nan)
        v_frz = m_frz.get(k, np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD", "VOL") else "+.3f"
        print(f"    {k:<12} {v_ref:{fmt}} {v_frz:{fmt}}")

if __name__ == "__main__":
    main()