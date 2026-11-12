# hardcore_tests/wf_rs_audit/run_rebuild.py
# ============================================================
# Точка запуска ТОЛЬКО для блока 7 (independent rebuild).
# Не перезапускает весь аудит — только этот модуль.
# ============================================================
from .common import load_split_date
from . import independent_rebuild


def main():
    split_date = load_split_date()
    print("=" * 70)
    print("AUDIT: Independent Portfolio Reconstruction (Block 7)")
    print("=" * 70)
    print(f"split_date = {split_date}\n")

    try:
        result = independent_rebuild.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return

    marker = {"PASS": "✓", "FAIL": "✗", "NOT_VERIFIED": "?"}[result["status"]]
    sev = f"[{result['severity']}]" if result["status"] != "PASS" else ""
    print(f"\n{marker} {result['name']} {sev}")
    print(f"\nEvidence:\n  {result['evidence']}")
    if result["recommendation"]:
        print(f"\nRecommendation:\n  {result['recommendation']}")


if __name__ == "__main__":
    main()