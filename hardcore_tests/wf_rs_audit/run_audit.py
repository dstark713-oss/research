# hardcore_tests/wf_rs_audit/run_audit.py
# ============================================================
# Оркестратор forensic audit. Все блоки последовательно.
# ============================================================
from pathlib import Path

from .common import load_split_date
from . import (
    trace_temporal, features_leakage, target_purge, cs_ranking,
    wf_split, oof_portfolio, independent_rebuild, survivorship,
    metrics_audit, adversarial, report,
)


def main():
    split_date = load_split_date()
    print("=" * 70)
    print("FORENSIC AUDIT: WF IS RS BACKTEST")
    print("=" * 70)
    print(f"split_date = {split_date} (VAL не трогаем)\n")

    all_results = []

    # --- Блок 1: трассировка ---
    print("[1/10] Трассировка временной информации...")
    try:
        results, temporal_lines = trace_temporal.audit_temporal()
        all_results.extend(results)
    except Exception as e:
        print(f"  ERROR: {e}")
        temporal_lines = [f"ERROR: {e}"]

    # --- Блок 2: features leakage ---
    print("[2/10] Look-ahead в признаках...")
    try:
        all_results.extend(features_leakage.run())
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Блок 3: target и purge ---
    print("[3/10] Target и purge...")
    try:
        all_results.extend(target_purge.run())
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Блок 4: cs ranking ---
    print("[4/10] Cross-sectional ranking...")
    try:
        all_results.extend(cs_ranking.run())
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Блок 5: WF split ---
    print("[5/10] WF split и обучение...")
    try:
        all_results.extend(wf_split.run())
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Блок 6: OOF и портфель ---
    print("[6/10] OOF predictions и портфель...")
    try:
        all_results.extend(oof_portfolio.run())
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Блок 7: независимая реконструкция ---
    print("[7/10] Независимая реконструкция портфеля...")
    try:
        all_results.append(independent_rebuild.run())
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Блок 8: survivorship ---
    print("[8/10] Survivorship bias...")
    try:
        all_results.extend(survivorship.run())
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Блок 9: метрики ---
    print("[9/10] Пересчёт метрик...")
    try:
        all_results.extend(metrics_audit.run())
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Блок 10: adversarial ---
    print("[10/10] Falsification tests...")
    try:
        all_results.extend(adversarial.run())
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- отчёт ---
    print("\nГенерация отчёта...")
    text = report.render_report(all_results, temporal_lines, split_date)
    report.save_report(text)

    # --- сводка в консоль ---
    print("\n" + "=" * 70)
    print("СВОДКА")
    print("=" * 70)
    for r in all_results:
        marker = {"PASS": "✓", "FAIL": "✗", "NOT_VERIFIED": "?"}[r["status"]]
        sev = f"[{r['severity']}]" if r["status"] != "PASS" else ""
        print(f"  {marker} {r['name']} {sev}")

    overall = report.overall_verdict(all_results)
    print(f"\n** ВЕРДИКТ: {overall} **")


if __name__ == "__main__":
    main()