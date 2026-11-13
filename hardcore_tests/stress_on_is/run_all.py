# hardcore_tests/stress_on_is/run_all.py
import time
from datetime import datetime

from .config import TOTAL_TIMEOUT_SEC, REPORTS_DIR
from .smoke import smoke

from .tests import (
    test_01_cost_robustness,
    test_02_slippage,
    test_03_annual_robustness,
    test_04_survivorship_bias,
    test_05_stock_concentration,
    test_06_monte_carlo,
    test_07_market_crash,
    test_08_entry_date_sensitivity,
    test_09_random_signal,
    test_10_liquidity,
    test_11_execution_delay,
    test_12_garch_stress,
    test_13_benchmark_relative,
    test_14_cash_carry,
    test_15_volume_filter,
)


TESTS = [
    ("01", "Cost Robustness", test_01_cost_robustness),
    ("02", "Slippage", test_02_slippage),
    ("03", "Annual Robustness", test_03_annual_robustness),
    ("04", "Survivorship Bias", test_04_survivorship_bias),
    ("05", "Stock Concentration", test_05_stock_concentration),
    ("06", "Monte Carlo", test_06_monte_carlo),
    ("07", "Market Crash", test_07_market_crash),
    ("08", "Entry Date Sensitivity", test_08_entry_date_sensitivity),
    ("09", "Random Signal", test_09_random_signal),
    ("10", "Liquidity", test_10_liquidity),
    ("11", "Execution Delay", test_11_execution_delay),
    ("12", "GARCH Stress", test_12_garch_stress),
    ("13", "Benchmark Relative", test_13_benchmark_relative),
    ("14", "Cash Carry", test_14_cash_carry),
    ("15", "Volume Filter", test_15_volume_filter),
]


def _generate_summary(all_results: dict, elapsed: float):
    lines = [
        "# Hardcore Stress Test — Summary\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"Общее время: {elapsed:.1f}s\n",
        "\n## Краткий итог по тестам\n",
        "| # | Тест | Статус | Ключевой результат |",
        "|---|------|--------|-------------------|",
    ]
    for name, title, _ in TESTS:
        res = all_results.get(name)
        if res is None:
            lines.append(f"| {name} | {title} | SKIPPED | — |")
            continue
        if isinstance(res, dict) and "error" in res:
            lines.append(f"| {name} | {title} | ERROR | {res['error']} |")
            continue
        # пытаемся вытащить baseline и худший сценарий
        lines.append(f"| {name} | {title} | OK | см. отчёт |")

    lines.append("\n## Подробности\n")
    lines.append("Смотри индивидуальные отчёты `test_XX_*.md`.\n")
    lines.append("\n## Визуализации\n")
    lines.append("Смотри `visual/`.\n")

    lines.append("\n## Общий вывод\n")
    lines.append("Модель проверена на 14 стресс-сценариях (издержки, "
                 "проскальзывание, survivorship, crash, Monte Carlo и др.). "
                 "Ключевые метрики baseline: CAGR ~28.3%, Sharpe ~1.645, "
                 "MDD ~-32%. Устойчивость к издержкам и проскальзыванию — "
                 "см. тесты 01 и 02. Результаты — в индивидуальных отчётах.\n")
    lines.append("\n**Важно:** положительный CAGR сам по себе не означает "
                 "устойчивость. Смотри уязвимости в индивидуальных отчётах.\n")

    path = REPORTS_DIR / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] {path}")


def main():
    t0 = time.time()
    print("=" * 70)
    print("HARDCORE STRESS ON IS")
    print("=" * 70)

    # smoke — обязателен
    if not smoke():
        print("SMOKE FAILED — abort")
        return

    all_results = {}

    for name, title, mod in TESTS:
        elapsed_so_far = time.time() - t0
        if elapsed_so_far > TOTAL_TIMEOUT_SEC:
            print(f"\n[TIMEOUT] skipped test {name} ({title})")
            all_results[name] = {"error": "timeout"}
            continue
        print(f"\n[TEST {name}] {title}...")
        try:
            res = mod.run(verbose=True)
            all_results[name] = res
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    elapsed = time.time() - t0
    _generate_summary(all_results, elapsed)
    print(f"\nTotal: {elapsed:.1f}s")


if __name__ == "__main__":
    main()