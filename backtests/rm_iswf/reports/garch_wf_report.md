# GARCH Risk Sizing — WF (IS)

Дата: 2026-11-13 00:02

split_date = 2024-01-30 (VAL не трогаем)


## Конфигурация

- GARCH(1,1), t-distribution
- Target vol (annualized): 15.0%
- Horizon: 20 дней
- Fit: на каждой rebal_date, train = все дни до rd
- Exposure: min(1, target_vol / forecast_vol)
- Apply: exposure фиксирована на 20 дней до следующего ребаланса


## Данные

- rebalances: 71
- port_ret: 1277 дней, 2018-09-04 → 2024-01-26
- exposure dates: 65
- errors: 6


## Метрики

| Метрика | Baseline | RS + GARCH | Δ |
|---------|----------|------------|---|
| CAGR | +38.10% | +28.16% | -0.099 |
| MDD | -32.34% | -26.05% | +0.063 |
| VOL | +23.34% | +19.30% | -0.040 |
| Sharpe | +1.504 | +1.385 | -0.120 |
| Sortino | +2.062 | +1.890 | -0.172 |
| Calmar | +1.178 | +1.081 | -0.097 |

## Exposure stats

- **exposure_mean**: 0.845
- **exposure_median**: 0.877
- **exposure_min**: 0.232
- **exposure_max**: 1.000
- **exposure_std**: 0.151
- **frac_below_1**: 0.754
- **frac_full_exposure**: 0.246

## Errors

- 2018-09-03: train too short (0 < 100)
- 2018-10-01: train too short (19 < 100)
- 2018-10-29: train too short (38 < 100)
- 2018-11-27: train too short (57 < 100)
- 2018-12-25: train too short (76 < 100)
- 2019-01-25: train too short (95 < 100)

## Выводы

- ΔSharpe: -0.120
- ΔMDD: +6.29 п.п.
- GARCH не улучшает Sharpe на этом прогоне.