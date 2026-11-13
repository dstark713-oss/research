# Hardcore Stress Test — Summary

Дата: 2026-11-13 23:24

Общее время: 366.4s


## Краткий итог по тестам

| # | Тест | Статус | Ключевой результат |
|---|------|--------|-------------------|
| 01 | Cost Robustness | OK | см. отчёт |
| 02 | Slippage | OK | см. отчёт |
| 03 | Annual Robustness | OK | см. отчёт |
| 04 | Survivorship Bias | OK | см. отчёт |
| 05 | Stock Concentration | OK | см. отчёт |
| 06 | Monte Carlo | OK | см. отчёт |
| 07 | Market Crash | OK | см. отчёт |
| 08 | Entry Date Sensitivity | OK | см. отчёт |
| 09 | Random Signal | OK | см. отчёт |
| 10 | Liquidity | OK | см. отчёт |
| 11 | Execution Delay | OK | см. отчёт |
| 12 | GARCH Stress | OK | см. отчёт |
| 13 | Benchmark Relative | OK | см. отчёт |
| 14 | Cash Carry | OK | см. отчёт |
| 15 | Volume Filter | OK | см. отчёт |

## Подробности

Смотри индивидуальные отчёты `test_XX_*.md`.


## Визуализации

Смотри `visual/`.


## Общий вывод

Модель проверена на 14 стресс-сценариях (издержки, проскальзывание, survivorship, crash, Monte Carlo и др.). Ключевые метрики baseline: CAGR ~28.3%, Sharpe ~1.645, MDD ~-32%. Устойчивость к издержкам и проскальзыванию — см. тесты 01 и 02. Результаты — в индивидуальных отчётах.


**Важно:** положительный CAGR сам по себе не означает устойчивость. Смотри уязвимости в индивидуальных отчётах.
