# backtests/rm_iswf/run_regime_decomposition.py
# ============================================================
# Decomposition: разложение Regime на компоненты.
#
# Варианты (на одних датах):
#   A. Target Vol only: exposure = clip(target/vol, min, max)
#   B. Regime multiplier only: exposure = clip(multiplier, min, max)
#   C. Combined: exposure = clip(vol_exp * multiplier, min, max)
#   Baseline: 100%
#
# Цель: понять, что даёт multiplier поверх Target Vol.
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
    build_state_matrix_from_prices, build_pc_and_cluster,
)
from regime_model.regime_based.sizing import (
    vol_to_exposure, regime_multiplier,
    TARGET_VOL_ANNUAL, MIN_EXPOSURE, MAX_EXPOSURE,
)
from .metrics import compute_metrics_from_daily, compute_exposure_stats
from .plot_wf import plot_equity

OUT_DIR = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_exposures(state, port_ret, rebal_dates, window_vol=20):
    """
    Возвращает dict:
      A_target_vol: exposure по realized vol
      B_multiplier: exposure по regime multiplier
      C_combined:   exposure по vol * multiplier
    """
    rv = port_ret.rolling(window_vol, min_periods=window_vol).std() * np.sqrt(252)
    exp_A = {}
    exp_B = {}
    exp_C = {}
    for rd in sorted(rebal_dates):
        rd = pd.Timestamp(rd)
        # realized vol до rd
        rv_before = rv.loc[rv.index < rd].dropna()
        if len(rv_before) == 0:
            continue
        vol = float(rv_before.iloc[-1])
        base_exp = vol_to_exposure(vol)
        # state на ближайшую дату <= rd
        state_before = state.loc[state.index <= rd]
        if len(state_before) == 0:
            mult = 1.0
        else:
            st = state_before.iloc[-1]
            mult = regime_multiplier(st["PC1"], st["PC2"], st["hurst"],
                                     int(st["cluster"]))
        # A: только vol
        exp_A[rd] = float(np.clip(base_exp, MIN_EXPOSURE, MAX_EXPOSURE))
        # B: только multiplier (без vol)
        exp_B[rd] = float(np.clip(mult, MIN_EXPOSURE, MAX_EXPOSURE))
        # C: vol * multiplier
        exp_C[rd] = float(np.clip(base_exp * mult, MIN_EXPOSURE, MAX_EXPOSURE))
    return {
        "A_target_vol": pd.Series(exp_A).sort_index(),
        "B_multiplier": pd.Series(exp_B).sort_index(),
        "C_combined": pd.Series(exp_C).sort_index(),
    }


def apply_exposure(port_ret, exposure, rebal_dates):
    rebal_sorted = sorted(rebal_dates)
    scaled = pd.Series(np.nan, index=port_ret.index)
    for i, rd in enumerate(rebal_sorted):
        rd = pd.Timestamp(rd)
        if rd not in exposure.index:
            continue
        exp = exposure.loc[rd]
        if i + 1 < len(rebal_sorted):
            next_rd = pd.Timestamp(rebal_sorted[i + 1])
        else:
            next_rd = port_ret.index[-1]
        mask = (port_ret.index > rd) & (port_ret.index <= next_rd)
        scaled.loc[mask] = port_ret.loc[mask] * exp
    return scaled


def main():
    print("=" * 70)
    print("REGIME DECOMPOSITION: Target Vol vs Multiplier vs Combined")
    print("=" * 70)

    split_date = load_split_date()
    print(f"split_date = {split_date}\n")

    # --- 1. данные ---
    print("[1] Загрузка данных...")
    close = load_stock_close(split_date)
    holdings = load_ml_holdings()
    ml_equity_df = load_ml_equity()
    rebal_dates = sorted(ml_equity_df["date"].tolist())[1:]
    print(f"    rebal_dates: {len(rebal_dates)}")

    # --- 2. state ---
    print("\n[2] Построение state (PC1/PC2/Hurst/cluster)...")
    states = build_state_matrix_from_prices(close)
    state = build_pc_and_cluster(states, n_components=3, n_clusters=4)
    print(f"    state: {len(state)} окон")

    # --- 3. port_ret ---
    print("\n[3] Восстановление доходностей портфеля...")
    port_ret = build_daily_returns(close, holdings)
    for rd in rebal_dates:
        if rd not in port_ret.index:
            port_ret.loc[rd] = 0.0
    port_ret = port_ret.sort_index()
    print(f"    port_ret: {len(port_ret)} дней")

    # --- 4. exposures ---
    print("\n[4] Расчёт 3 вариантов экспозиции...")
    exp_dict = build_exposures(state, port_ret, rebal_dates)
    for name, exp in exp_dict.items():
        print(f"    {name}: n={len(exp)}, mean={exp.mean():.3f}, "
              f"min={exp.min():.3f}, max={exp.max():.3f}")

    # --- 5. применяем к дневным доходностям ---
    print("\n[5] Применение экспозиций...")
    scaled_dict = {}
    for name, exp in exp_dict.items():
        scaled_dict[name] = apply_exposure(port_ret, exp, rebal_dates)

    # --- 6. метрики на общих датах ---
    print("\n[6] Метрики...")
    # общие даты: все варианты + baseline
    common_idx = None
    for s in scaled_dict.values():
        idx = s.dropna().index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()

    baseline_ret = port_ret.loc[common_idx].dropna()
    common_idx = baseline_ret.index  # после dropna

    if "turnover" in ml_equity_df.columns:
        to_series = ml_equity_df.set_index("date")["turnover"]
    else:
        to_series = None

    results = {}
    results["Baseline"] = compute_metrics_from_daily(baseline_ret, to_series)
    results["A_target_vol"] = compute_metrics_from_daily(
        scaled_dict["A_target_vol"].loc[common_idx], to_series
    )
    results["B_multiplier"] = compute_metrics_from_daily(
        scaled_dict["B_multiplier"].loc[common_idx], to_series
    )
    results["C_combined"] = compute_metrics_from_daily(
        scaled_dict["C_combined"].loc[common_idx], to_series
    )

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 70)
    header = f"{'Метрика':<12} {'Baseline':>12} {'A: TV':>12} {'B: Mult':>12} {'C: Comb':>12}"
    print(header)
    print("-" * len(header))
    for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar"]:
        b = results["Baseline"].get(k, np.nan)
        a = results["A_target_vol"].get(k, np.nan)
        m = results["B_multiplier"].get(k, np.nan)
        c = results["C_combined"].get(k, np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD", "VOL") else "+.3f"
        print(f"{k:<12} {b:{fmt}} {a:{fmt}} {m:{fmt}} {c:{fmt}}")

    # exposure stats
    print("\nExposure stats:")
    for name, exp in exp_dict.items():
        stats = compute_exposure_stats(exp)
        print(f"  {name}:")
        print(f"    mean={stats.get('exposure_mean', np.nan):.3f}, "
              f"min={stats.get('exposure_min', np.nan):.3f}, "
              f"max={stats.get('exposure_max', np.nan):.3f}, "
              f"frac_below_1={stats.get('frac_below_1', np.nan):.3f}")

    # --- 7. отчёт ---
    print("\n[7] Отчёт...")
    lines = [
        "# Regime Decomposition: Target Vol vs Multiplier vs Combined\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"split_date = {split_date}\n",
        "\n## Цель\n",
        "Разложить Regime-модель на компоненты, чтобы понять:",
        "- A: Target Vol only — что даёт простой vol-targeting",
        "- B: Regime multiplier only — что даёт только классификация режима",
        "- C: Combined — что даёт комбинация\n",
        "\n## Варианты\n",
        "| Вариант | Exposure |",
        "|---------|----------|",
        "| A: Target Vol | clip(target/vol_20d, 0.2, 1.0) |",
        "| B: Multiplier | clip(multiplier(PC1,PC2,Hurst,cluster), 0.2, 1.0) |",
        "| C: Combined | clip(vol_exp × multiplier, 0.2, 1.0) |",
        "| Baseline | 1.0 (100%) |\n",
        "\n## Метрики\n",
        "| Метрика | Baseline | A: TV | B: Multiplier | C: Combined |",
        "|---------|----------|-------|---------------|-------------|",
    ]
    for k in ["CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar"]:
        b = results["Baseline"].get(k, np.nan)
        a = results["A_target_vol"].get(k, np.nan)
        m = results["B_multiplier"].get(k, np.nan)
        c = results["C_combined"].get(k, np.nan)
        fmt = "+.2%" if k in ("CAGR", "MDD", "VOL") else "+.3f"
        lines.append(f"| {k} | {b:{fmt}} | {a:{fmt}} | {m:{fmt}} | {c:{fmt}} |")

    lines.append("\n## Exposure stats\n")
    for name, exp in exp_dict.items():
        stats = compute_exposure_stats(exp)
        lines.append(f"\n### {name}")
        for key, v in stats.items():
            lines.append(f"- **{key}**: {v:.3f}")

    lines.append("\n## Выводы\n")
    # сравнение A vs C
    a_sharpe = results["A_target_vol"].get("Sharpe", np.nan)
    c_sharpe = results["C_combined"].get("Sharpe", np.nan)
    a_mdd = results["A_target_vol"].get("MDD", np.nan)
    c_mdd = results["C_combined"].get("MDD", np.nan)
    lines.append(f"\n**A vs C (что даёт multiplier поверх Target Vol):**")
    lines.append(f"- ΔSharpe: {c_sharpe - a_sharpe:+.3f}")
    lines.append(f"- ΔMDD: {(c_mdd - a_mdd)*100:+.2f} п.п.")
    if c_sharpe > a_sharpe + 0.02:
        lines.append("- **Multiplier улучшает Sharpe** (полезен).")
    elif c_sharpe < a_sharpe - 0.02:
        lines.append("- **Multiplier ухудшает Sharpe** (не полезен).")
    else:
        lines.append("- **Multiplier не даёт значимого улучшения Sharpe.** "
                     "Target Vol достаточно.")

    # сравнение B vs Baseline
    b_sharpe = results["B_multiplier"].get("Sharpe", np.nan)
    b_mdd = results["B_multiplier"].get("MDD", np.nan)
    lines.append(f"\n**B vs Baseline (что даёт только multiplier):**")
    lines.append(f"- ΔSharpe: {b_sharpe - results['Baseline'].get('Sharpe', np.nan):+.3f}")
    lines.append(f"- ΔMDD: {(b_mdd - results['Baseline'].get('MDD', np.nan))*100:+.2f} п.п.")

    report_path = OUT_DIR / "regime_decomposition_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"    отчёт: {report_path}")

    # --- 8. CSV ---
    pd.DataFrame({
        "date": common_idx,
        "baseline": baseline_ret.loc[common_idx].values,
        "A_target_vol": scaled_dict["A_target_vol"].loc[common_idx].values,
        "B_multiplier": scaled_dict["B_multiplier"].loc[common_idx].values,
        "C_combined": scaled_dict["C_combined"].loc[common_idx].values,
    }).to_csv(OUT_DIR / "regime_decomposition_equity.csv", index=False)

    # --- 9. график ---
    print("[8] График...")
    curves = {
        "Baseline": baseline_ret.loc[common_idx],
        "A: Target Vol": scaled_dict["A_target_vol"].loc[common_idx],
        "B: Multiplier": scaled_dict["B_multiplier"].loc[common_idx],
        "C: Combined": scaled_dict["C_combined"].loc[common_idx],
    }
    # используем plot_equity, но передадим 4 серии
    from .plot_wf import setup_dark_style
    import matplotlib.pyplot as plt
    setup_dark_style()
    MONO = ["#ffffff", "#cccccc", "#999999", "#666666"]
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (name, ret) in enumerate(curves.items()):
        eq = (1 + ret.dropna()).cumprod()
        ax.plot(eq.index, eq.values, color=MONO[i], linewidth=1.3, label=name)
    ax.set_xlabel("Дата")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.set_title("Regime Decomposition: Target Vol vs Multiplier",
                 color="#eeeeee", fontsize=13)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="#555555", linewidth=0.5, linestyle=":")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "regime_decomposition_equity.png",
                dpi=140, facecolor="#1a1a1a")
    plt.close()
    print(f"    график: {OUT_DIR / 'regime_decomposition_equity.png'}")

    print(f"\n[decomposition] готово. Отчёты в {OUT_DIR}")


if __name__ == "__main__":
    main()