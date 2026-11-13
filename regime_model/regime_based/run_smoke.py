# regime_model/regime_based/run_smoke.py
# ============================================================
# Точка запуска smoke Regime-based модели.
# ============================================================
from .smoke import smoke


def main():
    print("=" * 70)
    print("REGIME-BASED RISK SIZING — SMOKE TEST")
    print("=" * 70)
    result = smoke(verbose=True)
    print("\n" + "=" * 70)
    print("SMOKE PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()