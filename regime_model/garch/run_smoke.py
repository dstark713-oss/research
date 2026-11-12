# риск_модели/гарч/run_smoke.py
# ============================================================
# Точка запуска smoke.
# ============================================================
from .smoke import smoke


def main():
    print("=" * 70)
    print("GARCH RISK SIZING — SMOKE TEST")
    print("=" * 70)
    result = smoke(verbose=True)
    print("\n" + "=" * 70)
    print("SMOKE PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()