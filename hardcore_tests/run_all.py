# hardcore_tests/run_all.py
# ============================================================
# Оркестратор: запускает все 7 тестов, собирает сводный отчёт
# и CSV.
# ============================================================
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from .data import load_split_date, OUT_DIR
from . import leakage, correlation, stability, regime, outliers, coverage, negative_control

REPORT = OUT_DIR / "stress_report.md"
CSV = OUT_DIR / "stress_results.csv"


# hardcore_tests/run_all.py — заменить main() на эту версию

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from .data import load_split_date, OUT_DIR
from . import leakage, correlation, stability, regime, outliers, coverage, negative_control

REPORT = OUT_DIR / "stress_report.md"
CSV = OUT_DIR / "stress_results.csv"


def _run_test(args):
    """Обёртка для parallel: (name, func, split_date)."""
    name, func, split_date = args
    try:
        result = func(split_date)
        return name, result, None
    except Exception as e:
        import traceback
        return name, None, traceback.format_exc()


def main():
    split_date = load_split_date()
    print(f"[hardcore] split_date = {split_date}\n")

    # порядок вывода (важно для отчёта)
    test_names = ["leakage", "correlation", "stability", "regime",
                  "outliers", "coverage", "negative_control"]
    test_funcs = {
        "leakage": leakage.run,
        "correlation": correlation.run,
        "stability": stability.run,
        "regime": regime.run,
        "outliers": outliers.run,
        "coverage": coverage.run,
        "negative_control": negative_control.run,
    }

    # ЗАПУСК: все тесты параллельно (7 процессов)
    results = {}
    print(f"[hardcore] запуск {len(test_names)} тестов параллельно...\n")

    with ProcessPoolExecutor(max_workers=len(test_names)) as executor:
        futures = {
            executor.submit(_run_test, (name, test_funcs[name], split_date)): name
            for name in test_names
        }
        for future in as_completed(futures):
            name, result, err = future.result()
            if err:
                print(f"[{name}] ERROR:\n{err}")
                results[name] = {"status": "ERROR", "error": err}
            else:
                results[name] = result
                status = result.get("status") or result.get("_overall")
                print(f"[{name}] СТАТУС: {status}")

    # --- сводный отчёт ---
    lines = [f"# Feature Pool Stress Test v1\n\nДата: {datetime.now():%Y-%m-%d %H:%M}\n",
             f"split_date = {split_date}\n"]

    lines.append("\n## Сводная таблица\n")
    lines.append("| Тест | Статус | Что выявлено |")
    lines.append("|------|--------|--------------|")

    status_leak = results["leakage"]["_overall"]
    leak_fails = [f"{k}: {len(v['fails'])}" for k, v in results["leakage"].items()
                  if k != "_overall" and v["status"] == "FAIL"]
    lines.append(f"| Leakage | **{status_leak}** | {', '.join(leak_fails) if leak_fails else 'нет нарушений'} |")

    c = results["correlation"]
    c_msg = f"fail pairs: {len(c.get('fail_pairs', []))}, warn pairs: {len(c.get('warn_pairs', []))}, unstable: {len(c.get('unstable_pairs', []))}"
    lines.append(f"| Correlation | **{c['status']}** | {c_msg} |")

    s = results["stability"]
    s_msg = f"pool_collapse={s['pool_collapse']}, unstable_features={len(s['unstable_features'])}"
    lines.append(f"| IS Stability | **{s['status']}** | {s_msg} |")

    r = results["regime"]
    lines.append(f"| Regime Stress | **{r['status']}** | fail_regimes={r.get('fail_regimes', [])} |")

    o = results["outliers"]
    lines.append(f"| Outlier Stress | **{o['status']}** | max_delta={o.get('max_delta', 0):.0%} |")

    cov = results["coverage"]
    cov_fails = [x["feature"] for x in cov.get("per_feature", []) if x["top5_share"] > 0.5]
    lines.append(f"| Coverage | **{cov['status']}** | concentrated: {cov_fails if cov_fails else 'нет'} |")

    nc = results["negative_control"]
    lines.append(f"| Negative Control | **{nc['status']}** | baseline_pass={nc['base_pass']}, shuffled_mean={nc['mean_shuffle']:.1f} |")

    # детали
    lines.append("\n## Детали\n")

    lines.append("\n### 1. Leakage\n")
    for k, v in results["leakage"].items():
        if k == "_overall": continue
        lines.append(f"- {k}: **{v['status']}** ({len(v['fails'])} нарушений)")

    lines.append("\n### 2. Correlation\n")
    lines.append(f"- WARN pairs (0.7-0.95): {len(c.get('warn_pairs', []))}")
    for a, b, cv in c.get("warn_pairs", [])[:10]:
        lines.append(f"  - {a} ↔ {b}: {cv:+.3f}")
    lines.append(f"- FAIL pairs (>= 0.95): {len(c.get('fail_pairs', []))}")
    for a, b, cv in c.get("fail_pairs", []):
        lines.append(f"  - {a} ↔ {b}: {cv:+.3f}")

    lines.append("\n### 3. IS Stability\n")
    lines.append(pd.DataFrame(s["fold_summary"]).to_string(index=False))
    if s["unstable_features"]:
        lines.append(f"\n- Нестабильные фичи: {', '.join(s['unstable_features'])}")

    lines.append("\n### 4. Regime Stress\n")
    df_r = pd.DataFrame(r["per_regime"])
    if len(df_r) > 0:
        pivot = df_r.pivot(index="feature", columns="regime", values="signed_ic")
        lines.append(pivot.round(4).to_string())

    lines.append("\n### 5. Outlier Stress\n")
    df_o = pd.DataFrame(o["per_feature"])
    if len(df_o) > 0:
        lines.append(df_o[["feature", "ic_base", "ic_wins", "ic_noext",
                            "delta_wins_pct", "delta_noext_pct"]].round(4).to_string(index=False))

    lines.append("\n### 6. Coverage\n")
    df_c = pd.DataFrame(cov["per_feature"])
    if len(df_c) > 0:
        lines.append(df_c[["feature", "n_tickers", "pos_share", "top5_share",
                            "delta_excl_top_pct"]].round(4).to_string(index=False))

    lines.append("\n### 7. Negative Control\n")
    lines.append(f"- Baseline (real target): {nc['base_pass']} фич проходят |IC|>0.015")
    lines.append(f"- Shuffled target: {nc['shuffle_pass']}")
    lines.append(f"- Mean shuffled: {nc['mean_shuffle']:.1f}")

    # --- CSV ---
    csv_rows = []
    for name in results["stability"]["unstable_features"] or []:
        pass
    # Собираем per-feature метрики из всех тестов
    feat_metrics = {}
    for name in [x["feature"] for x in s["per_feature"]]:
        feat_metrics.setdefault(name, {})
    for name in feat_metrics:
        # stability: сколько фолдов с тем же знаком
        sub = [r for r in s["per_feature"] if r["feature"] == name]
        signs = [np.sign(r["mean_ic"]) for r in sub]
        feat_metrics[name]["stable_folds"] = len(set(signs)) == 1
        # outlier
        o_row = next((x for x in o["per_feature"] if x["feature"] == name), None)
        if o_row:
            feat_metrics[name]["delta_wins_pct"] = o_row["delta_wins_pct"]
            feat_metrics[name]["delta_noext_pct"] = o_row["delta_noext_pct"]
        # coverage
        c_row = next((x for x in cov["per_feature"] if x["feature"] == name), None)
        if c_row:
            feat_metrics[name]["top5_share"] = c_row["top5_share"]
            feat_metrics[name]["delta_excl_top_pct"] = c_row["delta_excl_top_pct"]

    df_csv = pd.DataFrame.from_dict(feat_metrics, orient="index").reset_index()
    df_csv = df_csv.rename(columns={"index": "feature"})
    df_csv.to_csv(CSV, index=False)
    print(f"\n[hardcore] CSV: {CSV}")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[hardcore] отчёт: {REPORT}")

    # итог
    statuses = [results[k].get("status") or results[k].get("_overall")
                for k in ["leakage", "correlation", "stability",
                          "regime", "outliers", "coverage", "negative_control"]]
    print(f"\n[hardcore] ИТОГ: {statuses}")


if __name__ == "__main__":
    main()