# Target Volatility — WF (IS)

Дата: 2026-11-13 01:45

split_date = 2024-01-30 (VAL не трогаем)


## Конфигурация

- Модель: без ML, реализованная vol за 20 дней
- Target vol (annualized): 15.0%
- Exposure: min(1, target_vol / realized_vol_20d)


## Данные

- rebal_dates: 70
- exposure dates: 69


## Метрики

| Метрика | Baseline | RS + Target Vol | Δ |
|---------|----------|-----------------|---|
| CAGR | +35.05% | +25.37% | -0.097 |
| MDD | -32.34% | -24.82% | +0.075 |
| VOL | +22.29% | +17.43% | -0.049 |
| Sharpe | +1.464 | +1.386 | -0.078 |
| Sortino | +2.016 | +1.919 | -0.096 |
| Calmar | +1.084 | +1.022 | -0.062 |

## Exposure stats

- **exposure_mean**: 0.864
- **exposure_median**: 1.000
- **exposure_min**: 0.134
- **exposure_max**: 1.000
- **exposure_std**: 0.215
- **frac_below_1**: 0.449
- **frac_full_exposure**: 0.551

## Выводы

- ΔSharpe: -0.078
- ΔMDD: +7.52 п.п.
- Target Vol не улучшает Sharpe.