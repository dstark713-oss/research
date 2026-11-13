# Frozen Model: RS + GARCH Daily Sizing

Дата: 2026-11-13 22:55

split_date = 2024-01-30


## Конфигурация (заморожена)

- RS: CatBoostRegressor, depth=3, lr=0.03, iter=300, target=20d, rebalance=20d, TOP-10
- GARCH: GARCH(1,1, t), daily rolling update, target_vol=15%
- Exposure: min(1, target_vol / garch_vol)

## Метрики (IS)

| Метрика | Baseline | Frozen (RS+GARCH) |
|---------|----------|-------------------|
| CAGR | +31.83% | +28.30% |
| MDD | -34.97% | -31.99% |
| VOL | +22.73% | +15.93% |
| Sharpe | +1.334 | +1.645 |
| Sortino | +1.803 | +2.405 |
| Calmar | +0.910 | +0.885 |

## Exposure stats

- **exposure_mean**: 0.882
- **exposure_median**: 1.000
- **exposure_min**: 0.086
- **exposure_max**: 1.000
- **exposure_std**: 0.185
- **frac_below_1**: 0.454
- **frac_full_exposure**: 0.546

## Ключевые улучшения

- ΔSharpe: **+0.311**
- ΔMDD: **+2.98 п.п.**
- ΔVOL: **-6.80 п.п.**