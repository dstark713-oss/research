# WF IS: Stock Selection RS Model (v1)

Дата: 2026-11-11 00:29

split_date = 2024-01-30 (VAL не трогаем)


## Конфигурация

- Фичи: 11 (rs_pool v2)
- Горизонт: 20 дней
- Purge: 21 дней
- Fold'ов: 6
- Ансамбль: True ([42, 43, 44, 45, 46])


## Walk-Forward folds

| fold | train | val |
|------|-------|-----|
| 1 | 2013-03-25 → 2018-08-02 | 2018-09-03 → 2019-07-24 |
| 2 | 2013-03-25 → 2019-06-25 | 2019-07-25 → 2020-06-17 |
| 3 | 2013-03-25 → 2020-05-18 | 2020-06-18 → 2021-05-10 |
| 4 | 2013-03-25 → 2021-04-08 | 2021-05-11 → 2022-04-20 |
| 5 | 2013-03-25 → 2022-02-24 | 2022-04-21 → 2023-03-14 |
| 6 | 2013-03-25 → 2023-02-09 | 2023-03-15 → 2023-12-28 |

## Base CatBoostRegressor

 fold  mean_ic  hit_rate_ic  ndcg@10  top_decile_spread  top10_mean_rs  top10_median_rs
    1   0.0707       0.7333   0.4870             0.0286         0.0011          -0.0021
    2   0.0675       0.7689   0.4794             0.0018         0.0118           0.0116
    3   0.1299       0.8756   0.5167             0.0272         0.0167           0.0204
    4   0.0752       0.8178   0.4848             0.0133         0.0279           0.0122
    5   0.0150       0.5778   0.4377            -0.0050         0.0120           0.0220
    6   0.0736       0.5784   0.4903            -0.0637         0.0230           0.0123

**Aggregate:** mean IC = +0.0720, NDCG@10 = 0.483, TOP10 mean RS = +0.0154


## Seed Ensemble (5 seeds)

 fold  mean_ic  hit_rate_ic  ndcg@10  top_decile_spread  top10_mean_rs  top10_median_rs
    1   0.0725       0.7333   0.4888             0.0283         0.0016          -0.0015
    2   0.0685       0.7822   0.4821             0.0027         0.0125           0.0118
    3   0.1323       0.8667   0.5160             0.0282         0.0147           0.0180
    4   0.0779       0.8178   0.4864             0.0123         0.0283           0.0172
    5   0.0178       0.6089   0.4431            -0.0066         0.0118           0.0200
    6   0.0746       0.5735   0.4863            -0.0633         0.0208           0.0104

**Aggregate:** mean IC = +0.0739, NDCG@10 = 0.484, TOP10 mean RS = +0.0150


## Ranker (YetiRank, challenger)

 fold  mean_ic  hit_rate_ic  ndcg@10  top_decile_spread  top10_mean_rs  top10_median_rs
    1   0.0707       0.7022   0.4821             0.0143        -0.0066          -0.0055
    2   0.0624       0.7556   0.4868             0.0042         0.0129           0.0136
    3   0.1420       0.8978   0.5603             0.0182         0.0142           0.0129
    4   0.0534       0.6444   0.5043             0.0102         0.0282           0.0154
    5   0.0262       0.6222   0.4670            -0.0018         0.0165           0.0212
    6   0.0764       0.5637   0.5017            -0.0571         0.0281           0.0160

**Aggregate:** mean IC = +0.0719, NDCG@10 = 0.500, TOP10 mean RS = +0.0155


## Grid Search (8 configs, internal folds)

 config_idx  mean_ic                                                                        params
          0   0.0720  {'depth': 3, 'l2_leaf_reg': 30, 'random_strength': 3, 'learning_rate': 0.03}
          1   0.0717  {'depth': 3, 'l2_leaf_reg': 50, 'random_strength': 3, 'learning_rate': 0.03}
          2   0.0727 {'depth': 3, 'l2_leaf_reg': 100, 'random_strength': 3, 'learning_rate': 0.03}
          3   0.0736  {'depth': 3, 'l2_leaf_reg': 50, 'random_strength': 1, 'learning_rate': 0.03}
          4   0.0750  {'depth': 4, 'l2_leaf_reg': 50, 'random_strength': 3, 'learning_rate': 0.03}
          5   0.0763 {'depth': 4, 'l2_leaf_reg': 100, 'random_strength': 3, 'learning_rate': 0.03}
          6   0.0753  {'depth': 3, 'l2_leaf_reg': 50, 'random_strength': 3, 'learning_rate': 0.05}
          7   0.0757 {'depth': 5, 'l2_leaf_reg': 100, 'random_strength': 3, 'learning_rate': 0.03}

**Best:** config 5 → {'depth': 4, 'l2_leaf_reg': 100, 'random_strength': 3, 'learning_rate': 0.03}, IC=+0.0763


## Сравнение моделей

| Модель | Mean IC | NDCG@10 | TOP10 mean RS |
|--------|---------|---------|---------------|
| Base | +0.0720 | 0.483 | +0.0154 |
| Ensemble | +0.0739 | 0.484 | +0.0150 |
| Ranker | +0.0719 | 0.500 | +0.0155 |

## Замечания

- **Fold 6** — известный стрессовый период (2022–2024), не независимый OOS. Использовать для диагностики, не для выбора модели.
- **Rank-Gauss** — target, MSE на transformed target. Метрики — на исходной RS.
- **Grid search** — только на внутренних фолдах (fold 1–5), fold 6 не используется для выбора.