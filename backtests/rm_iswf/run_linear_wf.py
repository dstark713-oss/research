# backtests/rm_iswf/run_linear_wf.py
# ============================================================
# WF линейной модели сайзинга (полный прогон + метрики).
# По аналогии с run_garch_wf.py, но с Ridge.
# ============================================================
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from regime_model.garch.config import (
    TARGET_VOL_ANNUAL, HORIZON, REPORTS_DIR as GARCH_REPORTS,
)
from regime_model.garch.data_loader import (
    load_split_date, load_stock_close, load_ml_holdings, load_ml_equity,
)
from regime_model.garch.portfolio_returns import build_daily_returns
from regime_model.garch.data_loader import load_imoex
from regime_model.linear.walk_forward import run_linear_wf
from regime_model.linear.sizing import apply_exposure_to_daily
from .metrics import compute_metrics_from_daily, compute_exposure_stats
from .plot_wf import plot_equity, plot_vol_and_exposure

OUT_DIR = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("LINEAR RISK SIZING — WALK-FORWARD BACKTEST (IS)")
    print("=" * 70)

    split_date = load_split_date()
    print(f"split_date = {split_date}\n")

    # --- 1. данные ---
    print("[1] Загрузка данных...")
    close = load_stock_close(split_date)
    holdings = load_ml_holdings()
    ml_equity_df = load_ml_equity()
    rebal_dates_all = sorted(ml_equity_df["date"].tolist())
    rebal_dates = rebal_dates_all[1:]  # первая не имеет train
    print(f"    holdings: {len(holdings)} rebalances, "
          f"rebal_dates: {len(rebal_dates)}")

    # --- 2. доходности базового портфеля ---
    print("\n[2] Восстановление дневных доходностей...")
    port_ret = build_daily_returns(close, holdings)
    # добавить rd с 0
    for rd in rebal_dates:
        if rd not in port_ret.index:
            port_ret.loc[rd] = 0.0
    port_ret = port_ret.sort_index()
    print(f"    port_ret: {len(port_ret)} дней")

    # --- 3. IMOEX fallback ---
    imoex_close = load_imoex(split_date)
    imoex_ret = imoex_close.pct_change().dropna()
    print(f"    imoex_ret: {len(imoex_ret)} дней")

    # --- 4. WF Linear ---
    print(f"\n[3] WF Linear sizing на {len(rebal_dates)} датах...")
    wf_result = run_linear_wf(
        port_ret, rebal_dates,
        imoex_ret=imoex_ret,
        target_vol=TARGET_VOL_ANNUAL,
        verbose=True,
    )
    exposure = wf_result["exposure"]
    vol_forecast = wf_result["vol_forecast"]
    errors = wf_result["errors"]
    source_history = wf_result["source_history"]

    print(f"\n    exposure: n={len(exposure)}, "
          f"mean={exposure.mean():.3f}, min={exposure.min():.3f}, "
          f"max={exposure.max():.3f}")
    if errors:
        print(f"    errors: {len(errors)}")

    # --- 5. применение exposure ---
    print("\n[4] Применение exposure к дневным доходностям...")
    scaled_ret = apply_exposure_to_daily(port_ret, exposure, rebal_dates)
    valid_days = scaled_ret.notna().sum()
    print(f"    scaled_ret: {valid_days} дней")

    # --- 6. метрики ---
    print("\n[5] Метрики...")
    common_idx = scaled_ret.dropna().index
    baseline_ret = port_ret.loc[common_idx].dropna()
    garch_ret = scaled_ret.loc[common_idx].dropna()
    common = baseline_ret.index.intersection(garch_ret.index)
    baseline_ret = baseline_ret.loc[common]
    garch_ret = garch_ret.loc[common]

    if "turnover" in ml_equity_df.columns:
        to_series = ml_equity_df.set_index("date")["turnover"]
    else:
        to_series = None

    baseline_metrics = compute_metrics_from_daily(baseline_ret, to_series)
    garch_metrics = compute_metrics_from_daily(garch_ret, to_series)
    exp_stats = compute_exposure_stats(exposure)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 70)
    print(f"\n{'Метрика':<15} {'Baseline':>15} {'RS + LINEAR':>15}")
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
    print("\n[6] Отчёт...")
    lines = [
        "# Linear Risk Sizing — WF (IS)\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"split_date = {split_date} (VAL не трогаем)\n",
        "\n## Конфигурация\n",
        f"- Ridge regression (alpha=1.0), median imputation",
        f"- Target vol (annualized): {TARGET_VOL_ANNUAL*100:.1f}%",
        f"- Horizon: {HORIZON} дней",
        f"- Fit: на каждой rebal_date, train = все дни до rd−20",
        f"- Fallback: IMOEX для ранних дат (train < 100 дней на портфеле)",
        f"- Exposure: min(1, target_vol / forecast_vol)\n",
        f"\n## Данные\n",
        f"- rebal_dates: {len(rebal_dates)}",
        f"- exposure dates: {len(exposure)}",
        f"- errors: {len(errors)}",
        f"- source: portfolio={sum(1 for s in source_history if s['source']=='portfolio')}, "
        f"imoex={sum(1 for s in source_history if s['source']=='imoex')}\n",
        "\n## Метрики\n",
        "| Метрика | Baseline | RS + Linear | Δ |",
        "|---------|----------|-------------|---|",
    ]
    for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar"]:
        b = baseline_metrics.get(k, np.nan)
        g = garch_metrics.get(k, np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD", "VOL") else "+.3f"
        lines.append(f"| {k} | {b:{fmt}} | {g:{fmt}} | {g-b:+.3f} |")

    lines.append("\n## Exposure stats\n")
    for k, v in exp_stats.items():
        lines.append(f"- **{k}**: {v:.3f}")

    lines.append("\n## Выводы\n")
    delta_sharpe = garch_metrics.get("Sharpe", 0) - baseline_metrics.get("Sharpe", 0)
    delta_mdd = garch_metrics.get("MDD", 0) - baseline_metrics.get("MDD", 0)
    lines.append(f"- ΔSharpe: {delta_sharpe:+.3f}")
    lines.append(f"- ΔMDD: {delta_mdd*100:+.2f} п.п.")
    if delta_sharpe > 0:
        lines.append("- Linear улучшает Sharpe (risk-adjusted return ↑).")
    else:
        lines.append("- Linear не улучшает Sharpe.")

    report_path = OUT_DIR / "linear_wf_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"    отчёт: {report_path}")

    # --- 8. CSV ---
    pd.DataFrame({
        "date": common,
        "baseline_ret": baseline_ret.loc[common].values,
        "linear_ret": garch_ret.loc[common].values,
    }).to_csv(OUT_DIR / "linear_wf_equity.csv", index=False)

    pd.DataFrame({
        "date": exposure.index,
        "exposure": exposure.values,
        "vol_forecast": vol_forecast.reindex(exposure.index).values,
        "source": [s["source"] for s in source_history],
    }).to_csv(OUT_DIR / "linear_wf_exposure.csv", index=False)

    # --- 9. графики ---
    print("[7] Графики...")
    plot_equity(baseline_ret, garch_ret, OUT_DIR / "linear_wf_equity.png")
    plot_vol_and_exposure(vol_forecast, exposure,
                          OUT_DIR / "linear_wf_vol_exposure.png",
                          target_vol=TARGET_VOL_ANNUAL)

    print(f"\n[linear-wf] готово. Отчёты в {OUT_DIR}")


if __name__ == "__main__":
    main()