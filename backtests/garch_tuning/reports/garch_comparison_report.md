# GARCH: Daily vs Horizon (20d)

Дата: 2026-11-13 03:08

split_date = 2024-01-30


## Конфигурация

- GARCH(1,1, t), target_vol=15%
- 20d: reuse run_garch_wf (exposure фиксирована на 20 дней)
- Daily: один fit на train + rolling update дисперсии каждый день


## Метрики

| Метрика | Baseline | GARCH 20d | GARCH Daily |
|---------|----------|-----------|-------------|
| CAGR | +35.79% | +25.46% | +29.77% |
| MDD | -32.34% | -25.75% | -13.06% |
| VOL | +22.73% | +18.16% | +11.03% |
| Sharpe | +1.464 | +1.342 | +2.420 |
| Sortino | +2.007 | +1.819 | +3.802 |
| Calmar | +1.107 | +0.989 | +2.280 |

## Exposure stats


### GARCH 20d
- **exposure_mean**: 0.822
- **exposure_median**: 0.851
- **exposure_min**: 0.218
- **exposure_max**: 1.000
- **exposure_std**: 0.155
- **frac_below_1**: 0.831
- **frac_full_exposure**: 0.169

### GARCH daily
- **exposure_mean**: 0.874
- **exposure_median**: 1.000
- **exposure_min**: 0.079
- **exposure_max**: 1.000
- **exposure_std**: 0.192
- **frac_below_1**: 0.481
- **frac_full_exposure**: 0.519

## Выводы

- 20d: Sharpe=1.342, MDD=-25.75%
- Daily: Sharpe=2.420, MDD=-13.06%
- **Daily лучше** (пересчёт чаще ловит режимы).