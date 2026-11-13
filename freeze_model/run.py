# freeze_model/run.py
# ============================================================
# Точка запуска замороженной модели + сохранение holdings в JSON.
# ============================================================
import json
from datetime import datetime

import numpy as np
import pandas as pd

from regime_model.garch.data_loader import load_split_date, load_ml_equity
from backtests.rm_iswf.metrics import compute_metrics_from_daily, compute_exposure_stats

from .config import OUT_DIR
from .pipeline import run_frozen_model


def main():
    print("=" * 70)
    print("FROZEN MODEL: RS + GARCH daily sizing")
    print("=" * 70)

    split_date = load_split_date()
    print(f"split_date = {split_date}\n")

    result = run_frozen_model(split_date, verbose=True)

    daily_baseline = result["daily_ret_baseline"]
    daily_sized = result["daily_ret_sized"]
    exposure = result["exposure"]
    holdings = result["holdings"]

    # --- turnover из ml_equity ---
    ml_eq = load_ml_equity()
    to_series = ml_eq.set_index("date")["turnover"] if "turnover" in ml_eq.columns else None

    # --- метрики ---
    m_baseline = compute_metrics_from_daily(daily_baseline, to_series)
    m_sized = compute_metrics_from_daily(daily_sized, to_series)
    exp_stats = compute_exposure_stats(exposure)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ (FROZEN)")
    print("=" * 70)
    print(f"\n{'Метрика':<12} {'Baseline':>15} {'Frozen (RS+GARCH)':>20}")
    print("-" * 50)
    for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar"]:
        b = m_baseline.get(k, np.nan)
        s = m_sized.get(k, np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD", "VOL") else "+.3f"
        print(f"{k:<12} {b:{fmt}} {s:{fmt}}")

    print(f"\nExposure stats:")
    for k, v in exp_stats.items():
        print(f"  {k:<20} = {v:.3f}")

    # --- отчёт ---
    lines = [
        "# Frozen Model: RS + GARCH Daily Sizing\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"split_date = {split_date}\n",
        "\n## Конфигурация (заморожена)\n",
        f"- RS: CatBoostRegressor, depth=3, lr=0.03, iter=300, target=20d, rebalance=20d, TOP-10",
        f"- GARCH: GARCH(1,1, t), daily rolling update, target_vol=15%",
        f"- Exposure: min(1, target_vol / garch_vol)",
        f"\n## Метрики (IS)\n",
        "| Метрика | Baseline | Frozen (RS+GARCH) |",
        "|---------|----------|-------------------|",
    ]
    for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar"]:
        b = m_baseline.get(k, np.nan)
        s = m_sized.get(k, np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD", "VOL") else "+.3f"
        lines.append(f"| {k} | {b:{fmt}} | {s:{fmt}} |")

    lines.append("\n## Exposure stats\n")
    for k, v in exp_stats.items():
        lines.append(f"- **{k}**: {v:.3f}")

    lines.append("\n## Ключевые улучшения\n")
    d_sharpe = m_sized.get("Sharpe", 0) - m_baseline.get("Sharpe", 0)
    d_mdd = (m_sized.get("MDD", 0) - m_baseline.get("MDD", 0)) * 100
    d_vol = (m_sized.get("VOL", 0) - m_baseline.get("VOL", 0)) * 100
    lines.append(f"- ΔSharpe: **{d_sharpe:+.3f}**")
    lines.append(f"- ΔMDD: **{d_mdd:+.2f} п.п.**")
    lines.append(f"- ΔVOL: **{d_vol:+.2f} п.п.**")

    report_path = OUT_DIR / "frozen_model_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[freeze] отчёт: {report_path}")

    # --- CSV equity ---
    pd.DataFrame({
        "date": daily_baseline.index,
        "baseline_ret": daily_baseline.values,
        "sized_ret": daily_sized.reindex(daily_baseline.index).values,
        "exposure": exposure.reindex(daily_baseline.index).values,
    }).to_csv(OUT_DIR / "frozen_equity.csv", index=False)
    print(f"[freeze] equity: {OUT_DIR / 'frozen_equity.csv'}")

    # --- JSON holdings ---
    holdings_json = {str(k): v for k, v in holdings.items()}
    holdings_path = OUT_DIR / "frozen_holdings.json"
    with open(holdings_path, "w", encoding="utf-8") as f:
        json.dump(holdings_json, f)
    print(f"[freeze] holdings: {holdings_path}")


if __name__ == "__main__":
    main()