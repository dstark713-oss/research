# backtests/diagnostics/compare_garch_runs.py
# ============================================================
# Диагностика: почему GARCH daily даёт разные Sharpe
# в garch_tuning vs freeze_model.
# ============================================================
from pathlib import Path

import numpy as np
import pandas as pd

from regime_model.garch.data_loader import (
    load_split_date, load_stock_close, load_ml_holdings, load_ml_equity,
)
from regime_model.garch.portfolio_returns import build_daily_returns

from backtests.garch_tuning.daily_runner import run_garch_daily_fast
from freeze_model.pipeline import run_frozen_model

OUT_DIR = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    split_date = load_split_date()
    print("=" * 70)
    print("DIAGNOSTICS: GARCH daily в двух прогонах")
    print("=" * 70)
    print(f"split_date = {split_date}\n")

    # --- 1. Загрузка ---
    close = load_stock_close(split_date)
    holdings = load_ml_holdings()
    ml_eq = load_ml_equity()
    rebal_dates = sorted(ml_eq["date"].tolist())[1:]
    port_ret = build_daily_returns(close, holdings)
    for rd in rebal_dates:
        if rd not in port_ret.index:
            port_ret.loc[rd] = 0.0
    port_ret = port_ret.sort_index()

    print(f"[1] port_ret: {len(port_ret)}, "
          f"{port_ret.index.min().date()} → {port_ret.index.max().date()}")
    print(f"    rebal_dates: {len(rebal_dates)}")

    # --- 2. GARCH из garch_tuning ---
    print("\n[2] GARCH из garch_tuning (run_garch_daily_fast)...")
    wf_t = run_garch_daily_fast(port_ret, rebal_dates, verbose=True)
    exp_t = wf_t["exposure_daily"]
    print(f"    exposure: n={len(exp_t)}, "
          f"first={exp_t.index[0].date() if len(exp_t) else 'none'}, "
          f"last={exp_t.index[-1].date() if len(exp_t) else 'none'}")
    if len(exp_t) > 0:
        print(f"    mean={exp_t.mean():.4f}, min={exp_t.min():.4f}, max={exp_t.max():.4f}")

    # --- 3. GARCH из freeze_model ---
    print("\n[3] GARCH из freeze_model (run_frozen_model)...")
    result_f = run_frozen_model(split_date, verbose=True)
    exp_f = result_f["exposure"]
    print(f"    exposure: n={len(exp_f)}, "
          f"first={exp_f.index[0].date() if len(exp_f) else 'none'}, "
          f"last={exp_f.index[-1].date() if len(exp_f) else 'none'}")
    if len(exp_f) > 0:
        print(f"    mean={exp_f.mean():.4f}, min={exp_f.min():.4f}, max={exp_f.max():.4f}")

    # --- 4. Сравнение exposure ---
    print("\n[4] Сравнение exposure...")
    if len(exp_t) > 0 and len(exp_f) > 0:
        common = exp_t.index.intersection(exp_f.index)
        print(f"    common dates: {len(common)}")
        if len(common) > 0:
            a = exp_t.loc[common]
            b = exp_f.loc[common]
            diff = (a - b).abs()
            print(f"    max diff: {diff.max():.6f}")
            print(f"    mean diff: {diff.mean():.6f}")
            print(f"    identical: {np.allclose(a, b, atol=1e-6)}")

    # --- 5. Сравнение baseline daily_ret ---
    print("\n[5] Сравнение baseline daily_ret...")
    # baseline из garch_tuning: применяем scaled с exp_t (но baseline — исходный port_ret)
    # baseline из freeze_model: result_f["daily_ret_baseline"]
    base_f = result_f["daily_ret_baseline"]
    # из garch_tuning — берём common dates с exp_t
    if len(exp_t) > 0:
        common_t = exp_t.index.intersection(port_ret.index)
        base_t = port_ret.loc[common_t]
        common = base_t.index.intersection(base_f.index)
        print(f"    common dates: {len(common)}")
        if len(common) > 0:
            a = base_t.loc[common]
            b = base_f.loc[common]
            diff = (a - b).abs()
            print(f"    max diff: {diff.max():.6f}")
            print(f"    mean diff: {diff.mean():.6f}")
            print(f"    identical: {np.allclose(a, b, atol=1e-6)}")

    # --- 6. Метрики на ОДНИХ датах ---
    print("\n[6] Метрики на одинаковых датах...")
    if len(exp_t) > 0 and len(exp_f) > 0:
        common = exp_t.index.intersection(exp_f.index)
        common = common.sort_values()
        if len(common) > 10:
            from backtests.rm_iswf.metrics import compute_metrics_from_daily
            b_ret = port_ret.loc[common]
            t_ret = (port_ret.loc[common] * exp_t.loc[common])
            f_ret = (port_ret.loc[common] * exp_f.loc[common])
            m_b = compute_metrics_from_daily(b_ret)
            m_t = compute_metrics_from_daily(t_ret)
            m_f = compute_metrics_from_daily(f_ret)
            print(f"\n    {'Метрика':<12} {'Baseline':>12} {'garch_tuning':>15} {'freeze_model':>15}")
            print("    " + "-" * 58)
            for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar"]:
                b = m_b.get(k, np.nan)
                t = m_t.get(k, np.nan)
                f = m_f.get(k, np.nan)
                fmt = "+.2%" if k in ("CAGR", "MDD", "VOL") else "+.3f"
                print(f"    {k:<12} {b:{fmt}} {t:{fmt}} {f:{fmt}}")

    # --- 7. Сохранить для отладки ---
    print("\n[7] Сохранение CSV для отладки...")
    if len(exp_t) > 0:
        exp_t.to_csv(OUT_DIR / "exposure_garch_tuning.csv")
        print(f"    {OUT_DIR / 'exposure_garch_tuning.csv'}")
    if len(exp_f) > 0:
        exp_f.to_csv(OUT_DIR / "exposure_freeze_model.csv")
        print(f"    {OUT_DIR / 'exposure_freeze_model.csv'}")

    print("\n[diagnostics] готово. Отчёты в", OUT_DIR)


if __name__ == "__main__":
    main()