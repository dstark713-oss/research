# backtests/rm_iswf/run_regime_wf.py
# ============================================================
# WF Regime-based risk sizing (PC1/PC2/Hurst/cluster).
# По аналогии с run_garch_wf.py, run_linear_wf.py, run_target_vol_wf.py.
# ============================================================
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from regime_model.garch.data_loader import (
    load_split_date, load_stock_close, load_ml_holdings, load_ml_equity,
)
from regime_model.garch.portfolio_returns import build_daily_returns
from regime_model.regime_based.state_builder import (
    build_state_matrix_from_prices, build_wf_pc_and_cluster,
)
from regime_model.regime_based.sizing import (
    compute_exposure_series, apply_exposure_to_daily,
    TARGET_VOL_ANNUAL, MIN_EXPOSURE, MAX_EXPOSURE,
)
from .metrics import compute_metrics_from_daily, compute_exposure_stats
from .plot_wf import plot_equity, plot_vol_and_exposure

OUT_DIR = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("REGIME-BASED RISK SIZING — WALK-FORWARD BACKTEST (IS)")
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

    # --- 2. S_t на 185 акциях ---
    print("\n[2] Построение S_t (PC1/PC2/Hurst/cluster)...")
    states = build_state_matrix_from_prices(close)
    state = build_wf_pc_and_cluster(states, rebal_dates, n_components=3, n_clusters=4)
    print(f"    states: {len(state)} окон, "
          f"{state.index.min().date()} → {state.index.max().date()}")
    print(f"    cluster counts: {state['cluster'].value_counts().to_dict()}")

    # --- 3. доходности портфеля ---
    print("\n[3] Восстановление дневных доходностей...")
    port_ret = build_daily_returns(close, holdings)
    for rd in rebal_dates:
        if rd not in port_ret.index:
            port_ret.loc[rd] = 0.0
    port_ret = port_ret.sort_index()
    print(f"    port_ret: {len(port_ret)} дней")

    # --- 4. exposure ---
    print("\n[4] Расчёт exposure (по правилу режима)...")
    exposure = compute_exposure_series(state, port_ret, rebal_dates)
    print(f"    exposure: n={len(exposure)}, mean={exposure.mean():.3f}, "
          f"min={exposure.min():.3f}, max={exposure.max():.3f}")

    # --- 5. применение ---
    print("\n[5] Применение exposure к дневным доходностям...")
    scaled_ret = apply_exposure_to_daily(port_ret, exposure, rebal_dates)
    valid_days = scaled_ret.notna().sum()
    print(f"    scaled_ret: {valid_days} дней")

    # --- 6. метрики ---
    print("\n[6] Метрики...")
    common_idx = scaled_ret.dropna().index
    baseline_ret = port_ret.loc[common_idx].dropna()
    regime_ret = scaled_ret.loc[common_idx].dropna()
    common = baseline_ret.index.intersection(regime_ret.index)
    baseline_ret = baseline_ret.loc[common]
    regime_ret = regime_ret.loc[common]

    if "turnover" in ml_equity_df.columns:
        to_series = ml_equity_df.set_index("date")["turnover"]
    else:
        to_series = None

    baseline_metrics = compute_metrics_from_daily(baseline_ret, to_series)
    regime_metrics = compute_metrics_from_daily(regime_ret, to_series)
    exp_stats = compute_exposure_stats(exposure)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 70)
    print(f"\n{'Метрика':<15} {'Baseline':>15} {'RS + Regime':>15}")
    print("-" * 47)
    for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar"]:
        b = baseline_metrics.get(k, np.nan)
        g = regime_metrics.get(k, np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD", "VOL") else "+.3f"
        print(f"{k:<15} {b:{fmt}} {g:{fmt}}")
    print(f"\nExposure stats:")
    for k, v in exp_stats.items():
        print(f"  {k:<20} = {v:.3f}")

    # --- 7. отчёт ---
    print("\n[7] Отчёт...")
    lines = [
        "# Regime-Based Risk Sizing — WF (IS)\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"split_date = {split_date} (VAL не трогаем)\n",
        "\n## Конфигурация\n",
        f"- Состояние: PC1/PC2/Hurst/cluster на S_t (185 акций, rolling 252, step 20)",
        f"- Мультипликатор: regime_multiplier(PC1, PC2, Hurst, cluster)",
        f"- Базовая экспозиция: target_vol / realized_vol_20d",
        f"- Итоговая: clip(base * multiplier, {MIN_EXPOSURE}, {MAX_EXPOSURE})",
        f"- Target vol (annualized): {TARGET_VOL_ANNUAL*100:.1f}%\n",
        f"\n## Данные\n",
        f"- state: {len(state)} окон",
        f"- exposure dates: {len(exposure)}",
        f"- cluster counts: {state['cluster'].value_counts().to_dict()}\n",
        "\n## Метрики\n",
        "| Метрика | Baseline | RS + Regime | Δ |",
        "|---------|----------|-------------|---|",
    ]
    for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar"]:
        b = baseline_metrics.get(k, np.nan)
        g = regime_metrics.get(k, np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD", "VOL") else "+.3f"
        lines.append(f"| {k} | {b:{fmt}} | {g:{fmt}} | {g-b:+.3f} |")

    lines.append("\n## Exposure stats\n")
    for k, v in exp_stats.items():
        lines.append(f"- **{k}**: {v:.3f}")

    lines.append("\n## Выводы\n")
    delta_sharpe = regime_metrics.get("Sharpe", 0) - baseline_metrics.get("Sharpe", 0)
    delta_mdd = regime_metrics.get("MDD", 0) - baseline_metrics.get("MDD", 0)
    lines.append(f"- ΔSharpe: {delta_sharpe:+.3f}")
    lines.append(f"- ΔMDD: {delta_mdd*100:+.2f} п.п.")
    if delta_sharpe > 0:
        lines.append("- Regime-based улучшает Sharpe.")
    else:
        lines.append("- Regime-based не улучшает Sharpe.")

    report_path = OUT_DIR / "regime_wf_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"    отчёт: {report_path}")

    # --- 8. CSV ---
    pd.DataFrame({
        "date": common,
        "baseline_ret": baseline_ret.loc[common].values,
        "regime_ret": regime_ret.loc[common].values,
    }).to_csv(OUT_DIR / "regime_wf_equity.csv", index=False)

    # Для графиков — exposure и vol (realized)
    rv = port_ret.rolling(20, min_periods=20).std() * np.sqrt(252)
    pd.DataFrame({
        "date": exposure.index,
        "exposure": exposure.values,
        "realized_vol": rv.reindex(exposure.index).values,
    }).to_csv(OUT_DIR / "regime_wf_exposure.csv", index=False)

    # --- 9. графики ---
    print("[8] Графики...")
    plot_equity(baseline_ret, regime_ret, OUT_DIR / "regime_wf_equity.png")
    plot_vol_and_exposure(rv, exposure,
                          OUT_DIR / "regime_wf_vol_exposure.png",
                          target_vol=TARGET_VOL_ANNUAL)

    print(f"\n[regime-wf] готово. Отчёты в {OUT_DIR}")


if __name__ == "__main__":
    main()