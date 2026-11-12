# random_audit/bootstrap/report.py
# ============================================================
# Генерация Markdown-отчёта по bootstrap + MDD + turnover.
# ============================================================
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BLOCK_SIZES, N_BOOTSTRAP


def render_report(split_date: str,
                  ml_metrics: dict,
                  ml_turnover: dict,
                  bootstrap_agg: pd.DataFrame,
                  bootstrap_raw: pd.DataFrame,
                  mdd_fix: dict,
                  n_random_portfolios: int,
                  n_common_dates_median: int) -> str:
    lines = [
        "# Bootstrap Audit: ML vs Random\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"split_date = {split_date} (VAL не трогаем)\n",
        "\n## Конфигурация\n",
        f"- Bootstrap iterations: {N_BOOTSTRAP}",
        f"- Block sizes: {BLOCK_SIZES}",
        f"- Random portfolios sampled: {n_random_portfolios}",
        f"- Common dates (median): {n_common_dates_median}",
        "\n## ML baseline\n",
        f"- CAGR = {ml_metrics.get('CAGR', np.nan)*100:+.2f}%",
        f"- MDD = {ml_metrics.get('MDD', np.nan)*100:+.2f}%",
        f"- Sharpe = {ml_metrics.get('Sharpe', np.nan):.3f}",
        f"- Calmar = {ml_metrics.get('Calmar', np.nan):.3f}",
        f"- Turnover (annualized) = {ml_turnover.get('turnover_annual', np.nan)*100:+.1f}%",
        f"- Turnover (mean per rebalance) = {ml_turnover.get('mean_turnover', np.nan)*100:+.1f}%",
        f"- Rebalances per year = {ml_turnover.get('rebalances_per_year', np.nan):.1f}\n",
        "\n## MDD Percentile (фикс)\n",
        "| Scenario | ML MDD | Random median |MDD|| ML percentile (MDD) | frac random better MDD |",
        "|----------|--------|---------------------|---------------------|------------------------|",
    ]
    for scenario, m in mdd_fix.items():
        lines.append(
            f"| {scenario} | {m['ml_mdd']*100:+.2f}% | "
            f"{m['random_median_abs_mdd']*100:.2f}% | "
            f"{m['ml_percentile_mdd']:.0f}% | "
            f"{m['frac_random_better_mdd']*100:.1f}% |"
        )

    lines.extend([
        "\n## Block Bootstrap (парное сравнение ML vs random)\n",
        f"Для каждой пары (ML, random): diff_t = ml_ret_t - random_ret_t "
        f"на общих датах. Block bootstrap с блоками {BLOCK_SIZES}.\n",
        "| Block size | mean diff | CI 2.5% | CI 97.5% | median p | frac p<0.05 | n portfolios |",
        "|------------|-----------|---------|----------|----------|-------------|--------------|",
    ])
    if len(bootstrap_agg) > 0:
        for _, r in bootstrap_agg.iterrows():
            lines.append(
                f"| {int(r['block_size'])} | {r['mean_diff']*100:+.3f}% | "
                f"{r['mean_ci_low']*100:+.3f}% | {r['mean_ci_high']*100:+.3f}% | "
                f"{r['median_p_value']:.3f} | {r['frac_p_lt_05']*100:.1f}% | "
                f"{int(r['n_portfolios'])} |"
            )
    else:
        lines.append("| — | — | — | — | — | — | — |")

    # --- интерпретация ---
    lines.append("\n## Интерпретация\n")
    if len(bootstrap_agg) > 0:
        # основной block_size = 20 (горизонт target)
        main = bootstrap_agg[bootstrap_agg["block_size"] == 20]
        if len(main) > 0:
            m = main.iloc[0]
            ci_low = m["mean_ci_low"]
            ci_high = m["mean_ci_high"]
            p = m["median_p_value"]
            if ci_low > 0 and p < 0.05:
                verdict = ("**ML ADVANTAGE CONFIRMED.** "
                           f"CI [{ci_low*100:+.3f}%, {ci_high*100:+.3f}%] "
                           f"не включает 0, p={p:.3f} < 0.05.")
            elif ci_high < 0:
                verdict = ("**RANDOM BASELINE COMPETITIVE.** "
                           f"CI [{ci_low*100:+.3f}%, {ci_high*100:+.3f}%] "
                           f"целиком отрицательный — random лучше.")
            else:
                verdict = ("**INCONCLUSIVE.** "
                           f"CI [{ci_low*100:+.3f}%, {ci_high*100:+.3f}%] "
                           f"включает 0, p={p:.3f}. "
                           f"Разница не значима на уровне 5%.")
            lines.append(f"**Block size = 20 (горизонт target):**\n\n{verdict}\n")

        # чувствительность
        lines.append("\n**Чувствительность к длине блока:**\n")
        for _, r in bootstrap_agg.iterrows():
            bs = int(r["block_size"])
            sign = "положительный" if r["mean_ci_low"] > 0 else (
                "отрицательный" if r["mean_ci_high"] < 0 else "включает 0")
            lines.append(f"- block_size={bs}: CI {sign}, p={r['median_p_value']:.3f}")
    else:
        lines.append("Bootstrap не дал результатов — недостаточно общих дат.")

    lines.append("\n## Ограничения\n")
    lines.append("- Random portfolios — seeded выборка 100 из 10 000. "
                 "Для точных квантилей нужны все 10 000, но для CI 100 достаточно.")
    lines.append("- Block bootstrap сохраняет временную структуру, но не "
                 "учитывает cross-sectional корреляцию (все пары (ML, random) "
                 "считаются независимыми — это оптимистично).")
    lines.append("- Turnover ML: 976% в год (из rs_is_wf_equity.csv). "
                 "Random turnover — разный, но не влияет на сравнение (обе "
                 "стратегии rebal 20d).")
    lines.append("- MDD percentile исправлен: для MDD 'лучше' = меньший |MDD|.")

    return "\n".join(lines)