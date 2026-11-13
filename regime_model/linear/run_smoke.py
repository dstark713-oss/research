# regime_model/linear/run_smoke.py
# ============================================================
# Точка запуска smoke линейной модели.
# ============================================================
from .smoke import smoke


def main():
    print("=" * 70)
    print("LINEAR RISK SIZING — SMOKE TEST")
    print("=" * 70)
    result = smoke(verbose=True)
    print("\n" + "=" * 70)
    print("SMOKE PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()