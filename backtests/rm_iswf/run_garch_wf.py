# backtests/rm_iswf/run_garch_wf.py
# ============================================================
# WF GARCH risk sizing на IS.
#
# 1. Загрузка данных (close, holdings, ML equity, port_ret).
# 2. WF: на каждой rebal_date — fit GARCH, forecast vol(20d), exposure.
# 3. Применение exposure к дневным доходностям.
# 4. Метрики baseline vs GARCH, exposure stats.
# 5. Отчёт + графики + CSV.
# ============================================================
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from regime_model.garch.config import (
    TARGET_VOL_ANNUAL, HORIZON, REPORTS_DIR,
)
from regime_model.garch.data_loader import (
    load_split_date, load_stock_close, load_ml_holdings, load_ml_equity,
)
from regime_model.garch.portfolio_returns import build_daily_returns
from regime_model.garch.walk_forward import (
    run_garch_wf, apply_exposure_to_daily,
)
from .metrics import compute_metrics_from_daily, compute_exposure_stats
from .plot_wf import plot_equity, plot_vol_and_exposure

OUT_DIR = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("GARCH RISK SIZING — WALK-FORWARD (IS)")
    print("=" * 70)

    split_date = load_split_date()
    print(f"split_date = {split_date} (VAL не трогаем)\n")

    # --- 1. данные ---
    print("[1] Загрузка данных...")
    close = load_stock_close(split_date)
    holdings = load_ml_holdings()
    ml_equity_df = load_ml_equity()
    rebal_dates = sorted(holdings.keys())
    print(f"    close: {close.shape}, holdings: {len(rebal_dates)} rebalances")

    # --- 2. дневные доходности базового портфеля ---
    print("\n[2] Восстановление дневных доходностей портфеля...")
    port_ret = build_daily_returns(close, holdings)
    print(f"    port_ret: {len(port_ret)} дней, "
          f"{port_ret.index.min().date()} → {port_ret.index.max().date()}")

    # --- 3. WF GARCH ---
    print(f"\n[3] WF GARCH sizing на {len(rebal_dates)} датах ребаланса...")
    wf_result = run_garch_wf(
        port_ret, rebal_dates,
        target_vol=TARGET_VOL_ANNUAL,
        verbose=True,
    )
    exposure = wf_result["exposure"]
    vol_forecast = wf_result["vol_forecast"]
    errors = wf_result["errors"]

    print(f"\n    exposure: {len(exposure)} дат, "
          f"mean={exposure.mean():.3f}, "
          f"min={exposure.min():.3f}, max={exposure.max():.3f}")
    print(f"    forecast vol: mean={vol_forecast.mean()*100:.2f}%, "
          f"min={vol_forecast.min()*100:.2f}%, "
          f"max={vol_forecast.max()*100:.2f}%")
    if errors:
        print(f"    errors: {len(errors)}")
        for e in errors[:5]:
            print(f"      {e['date'].date()}: {e['reason']}")

    # --- 4. применение exposure ---
    print("\n[4] Применение exposure к дневным доходностям...")
    scaled_ret = apply_exposure_to_daily(port_ret, exposure, rebal_dates)
    valid_days = scaled_ret.notna().sum()
    print(f"    scaled_ret: {valid_days} дней с масштабированием")

    # --- 5. метрики ---
    print("\n[5] Метрики...")
    # baseline: только дни, где есть scaled (для честного сравнения)
    common_idx = scaled_ret.dropna().index
    baseline_ret = port_ret.loc[common_idx].dropna()
    garch_ret = scaled_ret.loc[common_idx].dropna()
    common = baseline_ret.index.intersection(garch_ret.index)
    baseline_ret = baseline_ret.loc[common]
    garch_ret = garch_ret.loc[common]

    # turnover baseline: из ML equity
    if "turnover" in ml_equity_df.columns:
        to_series = ml_equity_df.set_index("date")["turnover"]
    else:
        to_series = None

    baseline_metrics = compute_metrics_from_daily(baseline_ret, to_series)
    garch_metrics = compute_metrics_from_daily(garch_ret, to_series)
    exp_stats = compute_exposure_stats(exposure)

    # --- 6. вывод ---
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 70)
    print(f"\n{'Метрика':<15} {'Baseline':>15} {'RS + GARCH':>15}")
    print("-" * 47)
    for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar"]:
        b = baseline_metrics.get(k, np.nan)
        g = garch_metrics.get(k, np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD", "VOL") else "+.3f"
        print(f"{k:<15} {b:{fmt}} {g:{fmt}}")
    print(f"\nExposure stats:")
    for k, v in exp_stats.items():
        print(f"  {k:<20} = {v:.3f}")

    # --- 7. отчёт ---
    print("\n[6] Генерация отчёта...")
    lines = [
        "# GARCH Risk Sizing — WF (IS)\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"split_date = {split_date} (VAL не трогаем)\n",
        "\n## Конфигурация\n",
        f"- GARCH(1,1), t-distribution",
        f"- Target vol (annualized): {TARGET_VOL_ANNUAL*100:.1f}%",
        f"- Horizon: {HORIZON} дней",
        f"- Fit: на каждой rebal_date, train = все дни до rd",
        f"- Exposure: min(1, target_vol / forecast_vol)",
        f"- Apply: exposure фиксирована на 20 дней до следующего ребаланса\n",
        f"\n## Данные\n",
        f"- rebalances: {len(rebal_dates)}",
        f"- port_ret: {len(port_ret)} дней, "
        f"{port_ret.index.min().date()} → {port_ret.index.max().date()}",
        f"- exposure dates: {len(exposure)}",
        f"- errors: {len(errors)}\n",
        "\n## Метрики\n",
        "| Метрика | Baseline | RS + GARCH | Δ |",
        "|---------|----------|------------|---|",
    ]
    for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar"]:
        b = baseline_metrics.get(k, np.nan)
        g = garch_metrics.get(k, np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD", "VOL") else "+.3f"
        lines.append(f"| {k} | {b:{fmt}} | {g:{fmt}} | {g-b:+.3f} |")

    lines.append("\n## Exposure stats\n")
    for k, v in exp_stats.items():
        lines.append(f"- **{k}**: {v:.3f}")

    if errors:
        lines.append("\n## Errors\n")
        for e in errors:
            lines.append(f"- {e['date'].date()}: {e['reason']}")

    lines.append("\n## Выводы\n")
    delta_sharpe = garch_metrics.get("Sharpe", 0) - baseline_metrics.get("Sharpe", 0)
    delta_mdd = garch_metrics.get("MDD", 0) - baseline_metrics.get("MDD", 0)
    lines.append(f"- ΔSharpe: {delta_sharpe:+.3f}")
    lines.append(f"- ΔMDD: {delta_mdd*100:+.2f} п.п.")
    if delta_sharpe > 0:
        lines.append("- GARCH улучшает risk-adjusted return (Sharpe ↑).")
    else:
        lines.append("- GARCH не улучшает Sharpe на этом прогоне.")

    report_path = OUT_DIR / "garch_wf_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"    отчёт: {report_path}")

    # --- 8. CSV ---
    eq_df = pd.DataFrame({
        "date": common,
        "baseline_ret": baseline_ret.loc[common].values,
        "garch_ret": garch_ret.loc[common].values,
    })
    eq_df.to_csv(OUT_DIR / "garch_wf_equity.csv", index=False)

    pd.DataFrame({
        "date": exposure.index,
        "exposure": exposure.values,
        "vol_forecast": vol_forecast.reindex(exposure.index).values,
    }).to_csv(OUT_DIR / "garch_wf_exposure.csv", index=False)

    # --- 9. графики ---
    print("[7] Графики...")
    plot_equity(baseline_ret, garch_ret, OUT_DIR / "garch_wf_equity.png")
    plot_vol_and_exposure(vol_forecast, exposure,
                          OUT_DIR / "garch_wf_vol_exposure.png",
                          target_vol=TARGET_VOL_ANNUAL)

    print(f"\n[garch-wf] готово. Отчёты в {OUT_DIR}")


if __name__ == "__main__":
    main()