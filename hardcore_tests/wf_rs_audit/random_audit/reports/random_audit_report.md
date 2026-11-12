# Hardcore Random Portfolio Audit

Дата: 2026-11-12 03:10

split_date = 2024-01-30 (VAL не трогаем)


## Конфигурация

- TOP_N = 10, REBAL_STEP = 20, HORIZON = 20
- N_SIMULATIONS = 10000
- rebalance dates = 130
- availability pool: median = 126, min = 5, max = 172


## ML baseline

- CAGR = +21.97%
- MDD = -15.32%
- Sharpe = 1.243
- Calmar = 1.434


## Scenario A: Uniform Random

| Метрика | Random (mean) | Random (median) | Random (p95) | ML | ML percentile |
|---------|---------------|-----------------|--------------|-----|---------------|
| CAGR | +11.28% | +11.08% | +19.01% | +21.97% | 98% |
| Sharpe | +0.608 | +0.616 | +0.880 | +1.243 | 100% |
| Calmar | +0.421 | +0.391 | +0.831 | +1.434 | 100% |
| MDD | -29.36% | -28.37% | -19.97% | -15.32% | nan% |
| TotalReturn | +231.39% | +198.77% | +513.09% | +197.94% | 50% |

- **Доля random, превзошедших ML по CAGR:** 1.6%
- **Доля random, превзошедших ML по Sharpe:** 0.0%
- **Доля random, превзошедших ML по Calmar:** 0.1%
- **Доля random с меньшим |MDD|:** 0.4%


## Scenario B: Persistent Random

| Persistence | CAGR mean | CAGR p95 | ML percentile | frac better CAGR |
|-------------|-----------|----------|---------------|------------------|
| 0.00 | +11.40% | +19.21% | 99% | 1.4% |
| 0.25 | +11.41% | +19.15% | 99% | 1.4% |
| 0.50 | +11.28% | +18.95% | 99% | 1.4% |
| 0.75 | +10.93% | +18.54% | 99% | 1.4% |
| 1.00 | +3.01% | +6.48% | 100% | 0.0% |


## Scenario C: ML-Pool Random

| Метрика | Random (mean) | Random (p95) | ML | ML percentile |
|---------|---------------|--------------|-----|---------------|
| CAGR | +11.80% | +19.78% | +21.97% | 98% |
| Sharpe | +0.596 | +0.859 | +1.243 | 100% |
| Calmar | +0.384 | +0.768 | +1.434 | 100% |
| MDD | -33.68% | -22.64% | -15.32% | nan% |

- **Доля random, превзошедших ML по CAGR:** 2.2%


## Block Bootstrap (ML vs random, парное сравнение)

Sampled 100 random portfolios. CI для mean(ML_ret - random_ret):

| Block size | mean diff | CI 2.5% | CI 97.5% | p-value |
|------------|-----------|---------|----------|---------|


## Проверки корректности рандомизации

- **no_duplicates**: True 
- **uniformity**: False 0.5308329421713966
- **no_future_use**: True 
- **rebalance_sequence**: False 


## Вердикт

**ML ADVANTAGE CONFIRMED**

- ML percentile по CAGR в Scenario A: 98%
- Доля random, лучше ML по CAGR: 1.6%
- Доля random, лучше ML по Sharpe: 0.0%

## Ограничения

- Availability pool не включает делистингованные тикеры (survivorship bias, если universe сформирован по текущему состоянию).
- Execution model (MOC vs open(t+1)) не зафиксирован явно.
- Random portfolio — это НЕ прогнозная модель, а baseline. Отсутствие преимущества над random не доказывает отсутствие альфы, но и не подтверждает её.