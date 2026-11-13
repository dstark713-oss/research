# backtests/rm_tuning/run_horizon.py
# ============================================================
# Оркестратор тюнинга горизонта РЕБАЛАНСА RS-модели.
# Использует run_rs_wf (без дублирования модели/target/фич).
# ============================================================
from datetime import datetime

import pandas as pd

from regime_model.garch.data_loader import load_split_date
from .config import HORIZONS, OUT_DIR
from .horizon_runner import run_horizon


def main():
    print("=" * 70)
    print("REBALANCE HORIZON TUNING (RS model, target=20d fixed)")
    print("=" * 70)

    split_date = load_split_date()
    print(f"split_date = {split_date}\n")

    results = []
    for h in HORIZONS:
        print(f"\n{'='*60}")
        print(f"REBALANCE HORIZON = {h}")
        print(f"{'='*60}")
        try:
            res = run_horizon(split_date, h, verbose=True)
            results.append(res)
            print(f"  CAGR={res.get('CAGR', 0)*100:+.2f}%, "
                  f"MDD={res.get('MDD', 0)*100:+.2f}%, "
                  f"Sharpe={res.get('Sharpe', 0):.3f}, "
                  f"Calmar={res.get('Calmar', 0):.3f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"horizon": h, "error": str(e)})

    # --- сводка ---
    print("\n" + "=" * 70)
    print("СВОДКА")
    print("=" * 70)
    df = pd.DataFrame(results)
    cols = ["horizon", "CAGR", "MDD", "VOL", "Sharpe", "Sortino", "Calmar"]
    print(df[cols].to_string(index=False))

    # --- отчёт ---
    lines = [
        "# Rebalance Horizon Tuning (RS model, target=20d fixed)\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"split_date = {split_date}\n",
        "\n## Цель\n",
        "Найти оптимальный горизонт РЕБАЛАНСА при фиксированном target 20d.\n",
        f"\nГоризонты: {HORIZONS}\n",
        "\n## Результаты\n",
        "| Horizon | CAGR | MDD | VOL | Sharpe | Sortino | Calmar |",
        "|---------|------|-----|-----|--------|---------|--------|",
    ]
    for r in results:
        if "error" in r and r["error"]:
            lines.append(f"| {r['horizon']} | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {r['horizon']} | {r['CAGR']*100:+.2f}% | {r['MDD']*100:+.2f}% | "
            f"{r['VOL']*100:+.2f}% | {r['Sharpe']:.3f} | {r['Sortino']:.3f} | {r['Calmar']:.3f} |"
        )

    valid = df[df["error"].isna()] if "error" in df.columns else df
    lines.append("\n## Выводы\n")
    if len(valid) > 0:
        best_sharpe = valid.loc[valid["Sharpe"].idxmax()]
        best_calmar = valid.loc[valid["Calmar"].idxmax()]
        lines.append(f"- **Лучший по Sharpe:** rebalance={int(best_sharpe['horizon'])} "
                     f"(Sharpe={best_sharpe['Sharpe']:.3f})")
        lines.append(f"- **Лучший по Calmar:** rebalance={int(best_calmar['horizon'])} "
                     f"(Calmar={best_calmar['Calmar']:.3f})")
        lines.append(f"\n**Рекомендация:** rebalance={int(best_sharpe['horizon'])}")

    report_path = OUT_DIR / "horizon_tuning_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[horizon] отчёт: {report_path}")

    df.to_csv(OUT_DIR / "horizon_tuning.csv", index=False)
    print(f"[horizon] csv: {OUT_DIR / 'horizon_tuning.csv'}")


if __name__ == "__main__":
    main()