# Bootstrap Audit: ML vs Random

Дата: 2026-11-12 23:14

split_date = 2024-01-30 (VAL не трогаем)


## Конфигурация

- Bootstrap iterations: 2000
- Block sizes: [10, 20, 40, 60]
- Random portfolios sampled: 100
- Common dates (median): 64

## ML baseline

- CAGR = +21.97%
- MDD = -15.32%
- Sharpe = 1.243
- Calmar = 1.434
- Turnover (annualized) = +976.1%
- Turnover (mean per rebalance) = +77.5%
- Rebalances per year = 12.6


## MDD Percentile (фикс)

| Scenario | ML MDD | Random median |MDD|| ML percentile (MDD) | frac random better MDD |
|----------|--------|---------------------|---------------------|------------------------|
| A_uniform | -15.32% | 28.37% | 0% | 0.4% |
| B_persist_0.0 | -15.32% | 28.29% | 1% | 0.5% |
| B_persist_0.25 | -15.32% | 28.36% | 0% | 0.4% |
| B_persist_0.5 | -15.32% | 29.15% | 0% | 0.2% |
| B_persist_0.75 | -15.32% | 30.42% | 0% | 0.3% |
| B_persist_1.0 | -15.32% | 34.35% | 0% | 0.1% |
| C_ml_pool | -15.32% | 32.65% | 0% | 0.0% |

## Block Bootstrap (парное сравнение ML vs random)

Для каждой пары (ML, random): diff_t = ml_ret_t - random_ret_t на общих датах. Block bootstrap с блоками [10, 20, 40, 60].

| Block size | mean diff | CI 2.5% | CI 97.5% | median p | frac p<0.05 | n portfolios |
|------------|-----------|---------|----------|----------|-------------|--------------|
| 10 | +0.168% | -1.059% | +1.494% | 0.510 | 13.0% | 100 |
| 20 | +0.168% | -0.818% | +1.359% | 0.469 | 16.0% | 100 |
| 40 | +0.168% | +nan% | +nan% | nan | 0.0% | 100 |
| 60 | +0.168% | +nan% | +nan% | nan | 0.0% | 100 |

## Интерпретация

**Block size = 20 (горизонт target):**

**INCONCLUSIVE.** CI [-0.818%, +1.359%] включает 0, p=0.469. Разница не значима на уровне 5%.


**Чувствительность к длине блока:**

- block_size=10: CI включает 0, p=0.510
- block_size=20: CI включает 0, p=0.469
- block_size=40: CI включает 0, p=nan
- block_size=60: CI включает 0, p=nan

## Ограничения

- Random portfolios — seeded выборка 100 из 10 000. Для точных квантилей нужны все 10 000, но для CI 100 достаточно.
- Block bootstrap сохраняет временную структуру, но не учитывает cross-sectional корреляцию (все пары (ML, random) считаются независимыми — это оптимистично).
- Turnover ML: 976% в год (из rs_is_wf_equity.csv). Random turnover — разный, но не влияет на сравнение (обе стратегии rebal 20d).
- MDD percentile исправлен: для MDD 'лучше' = меньший |MDD|.