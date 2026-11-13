# backtests/garch_tuning/run_comparison.py
# ============================================================
# Сравнение GARCH: daily (rolling update) vs horizon (20d).
# ============================================================
from datetime import datetime

import numpy as np
import pandas as pd

from regime_model.garch.config import TARGET_VOL_ANNUAL
from regime_model.garch.data_loader import (
    load_split_date, load_stock_close, load_ml_holdings, load_ml_equity,
)
from regime_model.garch.portfolio_returns import build_daily_returns
from regime_model.garch.walk_forward import run_garch_wf, apply_exposure_to_daily
from backtests.rm_iswf.metrics import compute_metrics_from_daily, compute_exposure_stats

from .config import OUT_DIR
from .daily_runner import run_garch_daily_fast as run_garch_daily, apply_daily_exposure


def main():
    print("=" * 70)
    print("GARCH COMPARISON: DAILY vs HORIZON (20d)")
    print("=" * 70)

    split_date = load_split_date()
    print(f"split_date = {split_date}\n")

    # --- данные ---
    print("[1] Загрузка...")
    close = load_stock_close(split_date)
    holdings = load_ml_holdings()
    ml_eq = load_ml_equity()
    rebal_dates = sorted(ml_eq["date"].tolist())[1:]
    port_ret = build_daily_returns(close, holdings)
    for rd in rebal_dates:
        if rd not in port_ret.index:
            port_ret.loc[rd] = 0.0
    port_ret = port_ret.sort_index()
    print(f"    port_ret: {len(port_ret)}, rebal_dates: {len(rebal_dates)}")

    # --- Horizon GARCH ---
    print("\n[2] Horizon GARCH (20d, reuse run_garch_wf)...")
    wf_h = run_garch_wf(port_ret, rebal_dates, target_vol=TARGET_VOL_ANNUAL, verbose=False)
    exp_h = wf_h["exposure"]
    scaled_h = apply_exposure_to_daily(port_ret, exp_h, rebal_dates)
    print(f"    exposure: n={len(exp_h)}, mean={exp_h.mean():.3f}")

    # --- Daily GARCH ---
    print("\n[3] Daily GARCH (fast rolling)...")
    try:
        wf_d = run_garch_daily(port_ret, rebal_dates,
                               target_vol=TARGET_VOL_ANNUAL,
                               verbose=True)
        exp_d = wf_d["exposure_daily"]
        print(f"    [DEBUG] exp_d: n={len(exp_d)}, "
              f"first={exp_d.index[0].date() if len(exp_d) else 'none'}")
        scaled_d = apply_daily_exposure(port_ret, exp_d)
        print(f"    [DEBUG] scaled_d: n={scaled_d.notna().sum()}, "
              f"mean={scaled_d.mean():.6f}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        exp_d = pd.Series()
        scaled_d = pd.Series()

    # --- общие даты ---
    print("\n[4] Метрики на общих датах...")
    common_idx = None
    for s in [scaled_h, scaled_d]:
        idx = s.dropna().index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()

    baseline_ret = port_ret.loc[common_idx].dropna()
    common_idx = baseline_ret.index

    to_series = ml_eq.set_index("date")["turnover"] if "turnover" in ml_eq.columns else None

    m_baseline = compute_metrics_from_daily(baseline_ret, to_series)
    m_h = compute_metrics_from_daily(scaled_h.loc[common_idx].dropna(), to_series)
    m_d = compute_metrics_from_daily(scaled_d.loc[common_idx].dropna(), to_series)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 70)
    print(f"\n{'Метрика':<12} {'Baseline':>12} {'GARCH 20d':>12} {'GARCH daily':>12}")
    print("-" * 50)
    for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar"]:
        b = m_baseline.get(k, np.nan)
        h = m_h.get(k, np.nan)
        d = m_d.get(k, np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD", "VOL") else "+.3f"
        print(f"{k:<12} {b:{fmt}} {h:{fmt}} {d:{fmt}}")

    # --- отчёт ---
    lines = [
        "# GARCH: Daily vs Horizon (20d)\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"split_date = {split_date}\n",
        f"\n## Конфигурация\n",
        f"- GARCH(1,1, t), target_vol={TARGET_VOL_ANNUAL*100:.0f}%",
        f"- 20d: reuse run_garch_wf (exposure фиксирована на 20 дней)",
        f"- Daily: один fit на train + rolling update дисперсии каждый день\n",
        "\n## Метрики\n",
        "| Метрика | Baseline | GARCH 20d | GARCH Daily |",
        "|---------|----------|-----------|-------------|",
    ]
    for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar"]:
        b = m_baseline.get(k, np.nan)
        h = m_h.get(k, np.nan)
        d = m_d.get(k, np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD", "VOL") else "+.3f"
        lines.append(f"| {k} | {b:{fmt}} | {h:{fmt}} | {d:{fmt}} |")

    lines.append("\n## Exposure stats\n")
    for name, exp in [("GARCH 20d", exp_h), ("GARCH daily", exp_d)]:
        stats = compute_exposure_stats(exp)
        lines.append(f"\n### {name}")
        for k, v in stats.items():
            lines.append(f"- **{k}**: {v:.3f}")

    lines.append("\n## Выводы\n")
    sh_h = m_h.get("Sharpe", np.nan)
    sh_d = m_d.get("Sharpe", np.nan)
    lines.append(f"- 20d: Sharpe={sh_h:.3f}, MDD={m_h.get('MDD', np.nan)*100:+.2f}%")
    lines.append(f"- Daily: Sharpe={sh_d:.3f}, MDD={m_d.get('MDD', np.nan)*100:+.2f}%")
    if sh_d > sh_h + 0.02:
        lines.append("- **Daily лучше** (пересчёт чаще ловит режимы).")
    elif sh_d < sh_h - 0.02:
        lines.append("- **20d лучше** (daily шумит).")
    else:
        lines.append("- **Разница незначима**.")

    report_path = OUT_DIR / "garch_comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[garch-comparison] отчёт: {report_path}")

    pd.DataFrame({
        "date": common_idx,
        "baseline": baseline_ret.loc[common_idx].values,
        "garch_20d": scaled_h.loc[common_idx].values,
        "garch_daily": scaled_d.loc[common_idx].values,
    }).to_csv(OUT_DIR / "garch_comparison_equity.csv", index=False)
    print(f"[garch-comparison] csv: {OUT_DIR / 'garch_comparison_equity.csv'}")


if __name__ == "__main__":
    main()