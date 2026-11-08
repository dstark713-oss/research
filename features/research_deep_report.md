# Углублённое исследование (IS)

Дата: 2026-11-08 16:49

split_date = 2024-01-30 (VAL не трогаем)

stocks: 2718 дат × 185 тикеров
global: 2884 дат × 5 инструментов
IS: 2013-03-26 → 2024-01-29

======================================================================
1. ГЕОМЕТРИЯ РАСПРЕДЕЛЕНИЙ
======================================================================

[1a. Левый/правый хвосты, quantile asymmetry]
  STOCKS: median tail_ratio_1pct=1.368  median bowley=-0.009
  GLOBAL: median tail_ratio_1pct=0.957  median bowley=+0.015

[1b. Tail index (Hill α) по порогам]
  STOCKS q=0.900: median α=2.010
  STOCKS q=0.950: median α=2.215
  STOCKS q=0.975: median α=2.400
  STOCKS q=0.990: median α=2.681
  GLOBAL q=0.900: median α=2.174
  GLOBAL q=0.950: median α=2.301
  GLOBAL q=0.975: median α=2.547
  GLOBAL q=0.990: median α=2.356

[1c. Стабильность skew/kurt (rolling 252)]
  STOCKS: skew std=1.403  kurt std=12.1
  GLOBAL: skew std=1.145  kurt std=9.7

[1d. Fit: Normal vs Student-t vs Skewed-t]
  STOCKS: доля t лучше Normal: 100.0%  median df=1.9
  GLOBAL: доля t лучше Normal: 100.0%  median df=2.5

======================================================================
2. ПАМЯТЬ И ЗАВИСИМОСТЬ
======================================================================

[2a. ACF(1..10) для ret, |ret|, ret²]
  STOCKS ret: median |ACF(1)|=0.0714  |ACF(10)|=0.0169
  STOCKS abs: median |ACF(1)|=0.3305  |ACF(10)|=0.1013
  STOCKS sq: median |ACF(1)|=0.2242  |ACF(10)|=0.0239
  GLOBAL ret: median |ACF(1)|=0.0113  |ACF(10)|=0.0152
  GLOBAL abs: median |ACF(1)|=0.3333  |ACF(10)|=0.1980
  GLOBAL sq: median |ACF(1)|=0.2553  |ACF(10)|=0.0871

[2b. Short vs long memory]
  STOCKS ret: median decay_lag=2  median ACF(20)=0.0087
  STOCKS abs: median decay_lag=19  median ACF(20)=0.0658
  GLOBAL ret: median decay_lag=1  median ACF(20)=0.0147
  GLOBAL abs: median decay_lag=50  median ACF(20)=0.1038

[2c. Variance Ratio на горизонтах]
  STOCKS VR(2): median=0.954
  STOCKS VR(5): median=0.922
  STOCKS VR(10): median=0.885
  STOCKS VR(20): median=0.865
  STOCKS VR(60): median=0.797
  GLOBAL VR(2): median=1.010
  GLOBAL VR(5): median=1.160
  GLOBAL VR(10): median=1.242
  GLOBAL VR(20): median=1.235
  GLOBAL VR(60): median=1.171

[2d. DFA и устойчивость Hurst]
  STOCKS: median Hurst=0.621  median std по окнам=0.025
  GLOBAL: median Hurst=0.620  median std по окнам=0.062

[2e. Спектральный анализ (top-5 пиков)]
  → таблица spectral_peaks (топ-5 частот на тикер)

======================================================================
3. УСЛОВНАЯ ДИНАМИКА РИСКА
======================================================================

[STOCKS]
  GARCH persist: median=0.9877
  GJR gamma (leverage): median=+0.0444
  EGARCH gamma (asym): median=-0.0192

[GLOBAL]
  GARCH persist: median=0.9951
  GJR gamma (leverage): median=+0.0544
  EGARCH gamma (asym): median=-0.0556

[Устойчивость GARCH по 3 подпериодам]
  IMOEX: persist по окнам = [0.975, 0.937, 0.979]
  SBER: persist по окнам = [0.972, 0.914, 0.993]

======================================================================
4. НЕСТАЦИОНАРНОСТЬ И СТРУКТУРНЫЕ ИЗМЕНЕНИЯ
======================================================================

[4a. Rolling moments (252d) для IMOEX и SBER]
  IMOEX: std ratio=4.76  skew [-8.99..2.38]  kurt max=111.7
  SBER: std ratio=3.33  skew [-5.88..1.75]  kurt max=58.8

[4b. Change-point detection (CUSUM)]
  STOCKS: доля p<0.05 (есть сдвиг): 10.0%
  GLOBAL: доля p<0.05 (есть сдвиг): 0.0%

[4c. Сравнение статистик по 3 равным периодам]

  IMOEX:
    P1: mean=+0.000379  std=0.012306  skew=-0.57  kurt=7.0
    P2: mean=+0.000345  std=0.011053  skew=-0.94  kurt=15.0
    P3: mean=+0.000410  std=0.018943  skew=-5.06  kurt=117.9

  SBER:
    P1: mean=+0.000659  std=0.021381  skew=-0.11  kurt=5.9
    P2: mean=+0.000636  std=0.019260  skew=-0.62  kurt=10.5
    P3: mean=+0.000590  std=0.025083  skew=-3.96  kurt=55.1

[4d. Связь rolling std и rolling skew (IMOEX)]
  corr(rolling std, rolling skew) = -0.799

======================================================================
ИТОГ
======================================================================
SQL: c:\Users\Danil\Desktop\research\features\sql\research_deep.sqlite
Report: c:\Users\Danil\Desktop\research\features\research_deep_report.md
  acf_ret_abs_sq               522 строк
  cusum_breaks                  45 строк
  dist_fit                     174 строк
  dist_tails                   174 строк
  garch_models                  45 строк
  garch_stability                6 строк
  hurst_stability               45 строк
  memory_short_long            348 строк
  period_comparison              6 строк
  rolling_moments                2 строк
  skew_kurt_stability          174 строк
  spectral_peaks               190 строк
  tail_index_by_threshold      696 строк
  variance_ratio               174 строк