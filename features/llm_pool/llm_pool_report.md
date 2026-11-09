# LLM Feature Pool Evaluation

Дата: 2026-11-09 19:19


## Батч 001_050 — 56 фич

После IC/IR фильтра: **17/56**

Топ-15 по |IC|:
                    feature   mean_ic        ir  hit_rate stable_3sub
     upside_semivariance_20 -0.041881 -0.220704  0.403372        True
         distance_to_low_60 -0.041066 -0.264179  0.369159        True
      downside_deviation_20 -0.040296 -0.218009  0.397809        True
   downside_semivariance_20 -0.040296 -0.218009  0.397809        True
volatility_of_volatility_20 -0.036598 -0.237637  0.373592        True
         distance_to_low_20 -0.034285 -0.227811  0.376649        True
      negative_tail_mean_20  0.033839  0.200922  0.602639        True
  range_volatility_ratio_20 -0.031435 -0.216575  0.396043        True
        distance_to_high_60  0.030341  0.161417  0.604751        True
      positive_tail_mean_20 -0.029236 -0.187963  0.384556        True
    range_position_change_5 -0.024856 -0.197000  0.404214        True
negative_return_fraction_20 -0.023211 -0.162043  0.410128        True
          sign_imbalance_20  0.022598  0.165595  0.584012        True
positive_return_fraction_20  0.021447  0.164184  0.569385        True
       up_down_day_ratio_20  0.021018  0.159517  0.571705        True

После ортогонализации (corr<0.7): **8** фич
  - upside_semivariance_20: IC=-0.0419, IR=-0.221, hit=40.34%
  - distance_to_low_60: IC=-0.0411, IR=-0.264, hit=36.92%
  - downside_semivariance_20: IC=-0.0403, IR=-0.218, hit=39.78%
  - range_volatility_ratio_20: IC=-0.0314, IR=-0.217, hit=39.60%
  - distance_to_high_60: IC=+0.0303, IR=+0.161, hit=60.48%
  - range_position_change_5: IC=-0.0249, IR=-0.197, hit=40.42%
  - negative_return_fraction_20: IC=-0.0232, IR=-0.162, hit=41.01%
  - up_down_day_ratio_20: IC=+0.0210, IR=+0.160, hit=57.17%

## Батч 051_100 — 53 фич

После IC/IR фильтра: **13/53**

Топ-15 по |IC|:
                         feature   mean_ic        ir  hit_rate stable_3sub
     dist_from_high_intraday_252  0.044177  0.238628  0.641479        True
              dist_from_high_252  0.041602  0.227672  0.647910        True
               recovery_speed_60 -0.039425 -0.250804  0.386293        True
              dist_from_high_120  0.036692  0.199414  0.640427        True
               path_curvature_20 -0.031737 -0.194018  0.392081        True
                  path_length_20 -0.031531 -0.185954  0.402794        True
                 max_recovery_20 -0.031162 -0.182747  0.396819        True
         intraday_return_skew_20 -0.027236 -0.216939  0.406905        True
               drawdown_depth_60  0.026580  0.147326  0.608645        True
              price_position_252  0.025275  0.158553  0.581190        True
volume_abs_return_correlation_20 -0.021915 -0.174137  0.428405        True
            return_autocorr_2_20  0.017035  0.151183  0.570819        True
              path_efficiency_60 -0.016651 -0.120662  0.448358        True

После ортогонализации (corr<0.7): **8** фич
  - dist_from_high_intraday_252: IC=+0.0442, IR=+0.239, hit=64.15%
  - recovery_speed_60: IC=-0.0394, IR=-0.251, hit=38.63%
  - path_curvature_20: IC=-0.0317, IR=-0.194, hit=39.21%
  - max_recovery_20: IC=-0.0312, IR=-0.183, hit=39.68%
  - intraday_return_skew_20: IC=-0.0272, IR=-0.217, hit=40.69%
  - drawdown_depth_60: IC=+0.0266, IR=+0.147, hit=60.86%
  - volume_abs_return_correlation_20: IC=-0.0219, IR=-0.174, hit=42.84%
  - return_autocorr_2_20: IC=+0.0170, IR=+0.151, hit=57.08%

## ИТОГ

Ортогональных фич: **16**

Список:
  - upside_semivariance_20
  - distance_to_low_60
  - downside_semivariance_20
  - range_volatility_ratio_20
  - distance_to_high_60
  - range_position_change_5
  - negative_return_fraction_20
  - up_down_day_ratio_20
  - dist_from_high_intraday_252
  - recovery_speed_60
  - path_curvature_20
  - max_recovery_20
  - intraday_return_skew_20
  - drawdown_depth_60
  - volume_abs_return_correlation_20
  - return_autocorr_2_20