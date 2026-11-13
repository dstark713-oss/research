# Regime Decomposition: Target Vol vs Multiplier vs Combined

Дата: 2026-11-13 02:05

split_date = 2024-01-30


## Цель

Разложить Regime-модель на компоненты, чтобы понять:
- A: Target Vol only — что даёт простой vol-targeting
- B: Regime multiplier only — что даёт только классификация режима
- C: Combined — что даёт комбинация


## Варианты

| Вариант | Exposure |
|---------|----------|
| A: Target Vol | clip(target/vol_20d, 0.2, 1.0) |
| B: Multiplier | clip(multiplier(PC1,PC2,Hurst,cluster), 0.2, 1.0) |
| C: Combined | clip(vol_exp × multiplier, 0.2, 1.0) |
| Baseline | 1.0 (100%) |


## Метрики

| Метрика | Baseline | A: TV | B: Multiplier | C: Combined |
|---------|----------|-------|---------------|-------------|
| CAGR | +35.05% | +25.42% | +29.60% | +23.57% |
| MDD | -32.34% | -24.65% | -32.34% | -23.65% |
| VOL | +22.29% | +17.43% | +20.51% | +15.67% |
| Sharpe | +1.464 | +1.388 | +1.371 | +1.431 |
| Sortino | +2.016 | +1.923 | +1.872 | +1.984 |
| Calmar | +1.084 | +1.031 | +0.915 | +0.997 |

## Exposure stats


### A_target_vol
- **exposure_mean**: 0.865
- **exposure_median**: 1.000
- **exposure_min**: 0.200
- **exposure_max**: 1.000
- **exposure_std**: 0.212
- **frac_below_1**: 0.449
- **frac_full_exposure**: 0.551

### B_multiplier
- **exposure_mean**: 0.857
- **exposure_median**: 1.000
- **exposure_min**: 0.300
- **exposure_max**: 1.000
- **exposure_std**: 0.281
- **frac_below_1**: 0.217
- **frac_full_exposure**: 0.783

### C_combined
- **exposure_mean**: 0.774
- **exposure_median**: 1.000
- **exposure_min**: 0.200
- **exposure_max**: 1.000
- **exposure_std**: 0.307
- **frac_below_1**: 0.493
- **frac_full_exposure**: 0.507

## Выводы


**A vs C (что даёт multiplier поверх Target Vol):**
- ΔSharpe: +0.042
- ΔMDD: +1.00 п.п.
- **Multiplier улучшает Sharpe** (полезен).

**B vs Baseline (что даёт только multiplier):**
- ΔSharpe: -0.093
- ΔMDD: -0.00 п.п.