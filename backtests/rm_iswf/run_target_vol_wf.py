# backtests/rm_iswf/run_target_vol_wf.py
# ============================================================
# WF Target Volatility (тупая модель): полный прогон + метрики.
# По аналогии с run_garch_wf.py и run_linear_wf.py.
# ============================================================
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from regime_model.garch.config import HORIZON
from regime_model.garch.data_loader import (
    load_split_date, load_stock_close, load_ml_holdings, load_ml_equity,
)
from regime_model.garch.portfolio_returns import build_daily_returns
from regime_model.target_vol.config import (
    TARGET_VOL_ANNUAL, REALIZED_VOL_WINDOW,
)
from regime_model.target_vol.sizing import (
    realized_vol_annualized, compute_exposure_series, apply_exposure_to_daily,
)
from .metrics import compute_metrics_from_daily, compute_exposure_stats
from .plot_wf import plot_equity, plot_vol_and_exposure

OUT_DIR = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("TARGET VOLATILITY — WALK-FORWARD BACKTEST (IS)")
    print("=" * 70)

    split_date = load_split_date()
    print(f"split_date = {split_date}\n")

    # --- 1. данные ---
    print("[1] Загрузка данных...")
    close = load_stock_close(split_date)
    holdings = load_ml_holdings()
    ml_equity_df = load_ml_equity()
    rebal_dates_all = sorted(ml_equity_df["date"].tolist())
    rebal_dates = rebal_dates_all[1:]
    print(f"    rebal_dates: {len(rebal_dates)}")

    # --- 2. доходности портфеля ---
    print("\n[2] Восстановление дневных доходностей...")
    port_ret = build_daily_returns(close, holdings)
    for rd in rebal_dates:
        if rd not in port_ret.index:
            port_ret.loc[rd] = 0.0
    port_ret = port_ret.sort_index()
    print(f"    port_ret: {len(port_ret)} дней")

    # --- 3. realized vol + exposure ---
    print("\n[3] Расчёт realized vol и exposure...")
    rv = realized_vol_annualized(port_ret, window=REALIZED_VOL_WINDOW)
    exposure = compute_exposure_series(
        port_ret, rebal_dates,
        target_vol=TARGET_VOL_ANNUAL,
        window=REALIZED_VOL_WINDOW,
    )
    print(f"    realized vol: mean={rv.dropna().mean()*100:.2f}%, "
          f"min={rv.dropna().min()*100:.2f}%, "
          f"max={rv.dropna().max()*100:.2f}%")
    print(f"    exposure: n={len(exposure)}, mean={exposure.mean():.3f}, "
          f"min={exposure.min():.3f}, max={exposure.max():.3f}")

    # --- 4. применение ---
    print("\n[4] Применение exposure к дневным доходностям...")
    scaled_ret = apply_exposure_to_daily(port_ret, exposure, rebal_dates)
    valid_days = scaled_ret.notna().sum()
    print(f"    scaled_ret: {valid_days} дней")

    # --- 5. метрики ---
    print("\n[5] Метрики...")
    common_idx = scaled_ret.dropna().index
    baseline_ret = port_ret.loc[common_idx].dropna()
    tv_ret = scaled_ret.loc[common_idx].dropna()
    common = baseline_ret.index.intersection(tv_ret.index)
    baseline_ret = baseline_ret.loc[common]
    tv_ret = tv_ret.loc[common]

    if "turnover" in ml_equity_df.columns:
        to_series = ml_equity_df.set_index("date")["turnover"]
    else:
        to_series = None

    baseline_metrics = compute_metrics_from_daily(baseline_ret, to_series)
    tv_metrics = compute_metrics_from_daily(tv_ret, to_series)
    exp_stats = compute_exposure_stats(exposure)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 70)
    print(f"\n{'Метрика':<15} {'Baseline':>15} {'RS + Target Vol':>15}")
    print("-" * 47)
    for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar"]:
        b = baseline_metrics.get(k, np.nan)
        g = tv_metrics.get(k, np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD", "VOL") else "+.3f"
        print(f"{k:<15} {b:{fmt}} {g:{fmt}}")
    print(f"\nExposure stats:")
    for k, v in exp_stats.items():
        print(f"  {k:<20} = {v:.3f}")

    # --- 6. отчёт ---
    print("\n[6] Отчёт...")
    lines = [
        "# Target Volatility — WF (IS)\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"split_date = {split_date} (VAL не трогаем)\n",
        "\n## Конфигурация\n",
        f"- Модель: без ML, реализованная vol за {REALIZED_VOL_WINDOW} дней",
        f"- Target vol (annualized): {TARGET_VOL_ANNUAL*100:.1f}%",
        f"- Exposure: min(1, target_vol / realized_vol_20d)\n",
        f"\n## Данные\n",
        f"- rebal_dates: {len(rebal_dates)}",
        f"- exposure dates: {len(exposure)}\n",
        "\n## Метрики\n",
        "| Метрика | Baseline | RS + Target Vol | Δ |",
        "|---------|----------|-----------------|---|",
    ]
    for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar"]:
        b = baseline_metrics.get(k, np.nan)
        g = tv_metrics.get(k, np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD", "VOL") else "+.3f"
        lines.append(f"| {k} | {b:{fmt}} | {g:{fmt}} | {g-b:+.3f} |")

    lines.append("\n## Exposure stats\n")
    for k, v in exp_stats.items():
        lines.append(f"- **{k}**: {v:.3f}")

    lines.append("\n## Выводы\n")
    delta_sharpe = tv_metrics.get("Sharpe", 0) - baseline_metrics.get("Sharpe", 0)
    delta_mdd = tv_metrics.get("MDD", 0) - baseline_metrics.get("MDD", 0)
    lines.append(f"- ΔSharpe: {delta_sharpe:+.3f}")
    lines.append(f"- ΔMDD: {delta_mdd*100:+.2f} п.п.")
    if delta_sharpe > 0:
        lines.append("- Target Vol улучшает Sharpe.")
    else:
        lines.append("- Target Vol не улучшает Sharpe.")

    report_path = OUT_DIR / "target_vol_wf_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"    отчёт: {report_path}")

    # --- 7. CSV ---
    pd.DataFrame({
        "date": common,
        "baseline_ret": baseline_ret.loc[common].values,
        "tv_ret": tv_ret.loc[common].values,
    }).to_csv(OUT_DIR / "target_vol_wf_equity.csv", index=False)

    pd.DataFrame({
        "date": exposure.index,
        "exposure": exposure.values,
        "realized_vol": rv.reindex(exposure.index).values,
    }).to_csv(OUT_DIR / "target_vol_wf_exposure.csv", index=False)

    # --- 8. графики ---
    print("[7] Графики...")
    plot_equity(baseline_ret, tv_ret, OUT_DIR / "target_vol_wf_equity.png")
    plot_vol_and_exposure(rv, exposure,
                          OUT_DIR / "target_vol_wf_vol_exposure.png",
                          target_vol=TARGET_VOL_ANNUAL)

    print(f"\n[target-vol-wf] готово. Отчёты в {OUT_DIR}")


if __name__ == "__main__":
    main()