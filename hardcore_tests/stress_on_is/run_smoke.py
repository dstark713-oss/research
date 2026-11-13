# hardcore_tests/stress_on_is/run_smoke.py
# ============================================================
# Точка запуска smoke.
# ============================================================
from .smoke import smoke


def main():
    ok = smoke()
    if not ok:
        exit(1)


if __name__ == "__main__":
    main()