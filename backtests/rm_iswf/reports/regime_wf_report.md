# Regime-Based Risk Sizing — WF (IS)

Дата: 2026-11-13 02:14

split_date = 2024-01-30 (VAL не трогаем)


## Конфигурация

- Состояние: PC1/PC2/Hurst/cluster на S_t (185 акций, rolling 252, step 20)
- Мультипликатор: regime_multiplier(PC1, PC2, Hurst, cluster)
- Базовая экспозиция: target_vol / realized_vol_20d
- Итоговая: clip(base * multiplier, 0.2, 1.0)
- Target vol (annualized): 15.0%


## Данные

- state: 70 окон
- exposure dates: 69
- cluster counts: {3: 19, 1: 18, 0: 17, 2: 16}


## Метрики

| Метрика | Baseline | RS + Regime | Δ |
|---------|----------|-------------|---|
| CAGR | +35.05% | +16.75% | -0.183 |
| MDD | -32.34% | -24.27% | +0.081 |
| VOL | +22.29% | +15.53% | -0.068 |
| Sharpe | +1.464 | +1.076 | -0.388 |
| Sortino | +2.016 | +1.428 | -0.588 |
| Calmar | +1.084 | +0.690 | -0.394 |

## Exposure stats

- **exposure_mean**: 0.690
- **exposure_median**: 0.663
- **exposure_min**: 0.214
- **exposure_max**: 1.000
- **exposure_std**: 0.228
- **frac_below_1**: 0.797
- **frac_full_exposure**: 0.203

## Выводы

- ΔSharpe: -0.388
- ΔMDD: +8.07 п.п.
- Regime-based не улучшает Sharpe.