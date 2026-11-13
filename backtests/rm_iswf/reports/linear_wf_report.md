# Linear Risk Sizing — WF (IS)

Дата: 2026-11-13 01:41

split_date = 2024-01-30 (VAL не трогаем)


## Конфигурация

- Ridge regression (alpha=1.0), median imputation
- Target vol (annualized): 15.0%
- Horizon: 20 дней
- Fit: на каждой rebal_date, train = все дни до rd−20
- Fallback: IMOEX для ранних дат (train < 100 дней на портфеле)
- Exposure: min(1, target_vol / forecast_vol)


## Данные

- rebal_dates: 70
- exposure dates: 70
- errors: 0
- source: portfolio=64, imoex=6


## Метрики

| Метрика | Baseline | RS + Linear | Δ |
|---------|----------|-------------|---|
| CAGR | +33.67% | +26.14% | -0.075 |
| MDD | -32.34% | -32.49% | -0.002 |
| VOL | +22.18% | +21.05% | -0.011 |
| Sharpe | +1.423 | +1.212 | -0.211 |
| Sortino | +1.960 | +1.642 | -0.318 |
| Calmar | +1.041 | +0.804 | -0.237 |

## Exposure stats

- **exposure_mean**: 0.900
- **exposure_median**: 1.000
- **exposure_min**: 0.240
- **exposure_max**: 1.000
- **exposure_std**: 0.168
- **frac_below_1**: 0.471
- **frac_full_exposure**: 0.529

## Выводы

- ΔSharpe: -0.211
- ΔMDD: -0.15 п.п.
- Linear не улучшает Sharpe.