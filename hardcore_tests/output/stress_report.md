# Feature Pool Stress Test v1

Дата: 2026-11-09 23:33

split_date = 2024-01-30


## Сводная таблица

| Тест | Статус | Что выявлено |
|------|--------|--------------|
| Leakage | **PASS** | нет нарушений |
| Correlation | **FAIL** | fail pairs: 1, warn pairs: 6, unstable: 0 |
| IS Stability | **WARN** | pool_collapse=False, unstable_features=5 |
| Regime Stress | **WARN** | fail_regimes=[] |
| Outlier Stress | **PASS** | max_delta=8% |
| Coverage | **PASS** | concentrated: нет |
| Negative Control | **PASS** | baseline_pass=16, shuffled_mean=0.0 |

## Детали


### 1. Leakage

- 1A_truncation: **PASS** (0 нарушений)
- 1B_future_perturbation: **PASS** (0 нарушений)
- 1C_target_isolation: **PASS** (0 нарушений)
- 1D_static_code: **PASS** (0 нарушений)
- 1E_nan_inf: **PASS** (0 нарушений)

### 2. Correlation

- WARN pairs (0.7-0.95): 6
  - upside_semivariance_20 ↔ path_curvature_20: +0.745
  - upside_semivariance_20 ↔ max_recovery_20: +0.783
  - downside_semivariance_20 ↔ path_curvature_20: +0.709
  - downside_semivariance_20 ↔ max_recovery_20: +0.727
  - distance_to_high_60 ↔ drawdown_depth_60: +0.923
  - path_curvature_20 ↔ max_recovery_20: +0.836
- FAIL pairs (>= 0.95): 1
  - distance_to_low_60 ↔ recovery_speed_60: +0.997

### 3. IS Stability

 fold  n_dates  positive_ic_ratio  median_abs_ic  mean_abs_ic  effective_features
    1 298.6875             0.8750       0.036320     0.035586                  12
    2 432.3750             0.9375       0.021477     0.020544                  11
    3 432.1250             1.0000       0.038712     0.039343                  16
    4 431.9375             1.0000       0.036751     0.044306                  15
    5 432.1875             1.0000       0.044625     0.044526                  16
    6 413.1875             0.8750       0.011798     0.016764                   8

- Нестабильные фичи: upside_semivariance_20, distance_to_high_60, path_curvature_20, max_recovery_20, volume_abs_return_correlation_20

### 4. Regime Stress

regime                              Bear    Bull  HighVol  LowVol
feature                                                          
dist_from_high_intraday_252      -0.0241  0.0681   0.0093  0.0792
distance_to_high_60              -0.0438  0.0552   0.0046  0.0557
distance_to_low_60                0.0630  0.0337   0.0505  0.0317
downside_semivariance_20          0.0288  0.0441   0.0501  0.0311
drawdown_depth_60                -0.0496  0.0521  -0.0045  0.0573
intraday_return_skew_20           0.0427  0.0221   0.0374  0.0171
max_recovery_20                   0.0205  0.0347   0.0414  0.0210
negative_return_fraction_20      -0.0059  0.0330   0.0135  0.0330
path_curvature_20                 0.0234  0.0345   0.0359  0.0276
range_position_change_5           0.0573  0.0140   0.0321  0.0177
range_volatility_ratio_20         0.0318  0.0313   0.0333  0.0296
recovery_speed_60                 0.0637  0.0313   0.0508  0.0282
return_autocorr_2_20              0.0260  0.0140   0.0128  0.0213
up_down_day_ratio_20              0.0037  0.0268   0.0135  0.0285
upside_semivariance_20            0.0146  0.0503   0.0368  0.0466
volume_abs_return_correlation_20  0.0103  0.0258   0.0168  0.0270

### 5. Outlier Stress

                         feature  ic_base  ic_wins  ic_noext  delta_wins_pct  delta_noext_pct
          upside_semivariance_20  -0.0419  -0.0419   -0.0449          0.0001           0.0727
              distance_to_low_60  -0.0411  -0.0411   -0.0435          0.0001           0.0591
        downside_semivariance_20  -0.0403  -0.0403   -0.0423          0.0001           0.0500
       range_volatility_ratio_20  -0.0314  -0.0314   -0.0324          0.0002           0.0297
             distance_to_high_60   0.0303   0.0303    0.0318          0.0001           0.0476
         range_position_change_5  -0.0249  -0.0248   -0.0252          0.0004           0.0144
     negative_return_fraction_20  -0.0232  -0.0232   -0.0231          0.0005           0.0038
            up_down_day_ratio_20   0.0210   0.0209    0.0209          0.0042           0.0064
     dist_from_high_intraday_252   0.0442   0.0442    0.0449          0.0000           0.0166
               recovery_speed_60  -0.0394  -0.0394   -0.0416          0.0001           0.0550
               path_curvature_20  -0.0317  -0.0317   -0.0341          0.0001           0.0750
                 max_recovery_20  -0.0312  -0.0312   -0.0332          0.0003           0.0649
         intraday_return_skew_20  -0.0272  -0.0272   -0.0279          0.0003           0.0257
               drawdown_depth_60   0.0266   0.0266    0.0267          0.0003           0.0060
volume_abs_return_correlation_20  -0.0219  -0.0219   -0.0234          0.0007           0.0661
            return_autocorr_2_20   0.0170   0.0170    0.0171          0.0007           0.0052

### 6. Coverage

                         feature  n_tickers  pos_share  top5_share  delta_excl_top_pct
          upside_semivariance_20        166     0.6024      0.1044              0.0552
              distance_to_low_60        174     0.5632      0.1289              0.2331
        downside_semivariance_20        173     0.5318      0.1185              0.2807
       range_volatility_ratio_20        174     0.5977      0.1050              0.1261
             distance_to_high_60        174     0.6609      0.1249              0.1087
         range_position_change_5        174     0.6092      0.1192              0.0885
     negative_return_fraction_20        174     0.6092      0.1279              0.1267
            up_down_day_ratio_20        174     0.6724      0.1237              0.0906
     dist_from_high_intraday_252        171     0.5322      0.1484              0.2032
               recovery_speed_60        174     0.5345      0.1300              0.4174
               path_curvature_20        174     0.5920      0.1343              0.1078
                 max_recovery_20        174     0.5402      0.1331              0.1892
         intraday_return_skew_20        141     0.6028      0.1272              0.1155
               drawdown_depth_60        174     0.6724      0.1238              0.0814
volume_abs_return_correlation_20        174     0.5287      0.1078              5.8173
            return_autocorr_2_20        174     0.5977      0.1204              0.1399

### 7. Negative Control

- Baseline (real target): 16 фич проходят |IC|>0.015
- Shuffled target: [0, 0, 0, 0, 0]
- Mean shuffled: 0.0