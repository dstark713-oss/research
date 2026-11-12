# Forensic Audit of Walk-Forward Backtest

Дата: 2026-11-11 23:33

split_date = 2024-01-30 (VAL не трогаем)


## Executive Summary

- Всего проверок: 26
- PASS: 21
- FAIL: 2
- NOT_VERIFIED: 3

**Вердикт: WF INVALID**


## Таблица всех проверок

| # | Проверка | Статус | Severity | Доказательство |
|---|----------|--------|----------|----------------|
| 1 | 1.1 Target использует только будущие цены | PASS | LOW | target.loc[2019-02-27] = ret(close(t)→close(t+20)) - ret_IMOEX. Последние 20 дат — NaN (shift(-20)). |
| 2 | 1.2 Prediction date предшествует target horizon | PASS | LOW | Prediction на t=2019-02-27 использует features(t), target(t) = ret(t→t+20). t < t+20. OK. |
| 3 | 1.3 Entry price vs signal bar | NOT_VERIFIED | HIGH | В rs_is_wf.py портфель: selected на rd, port_rs = target_rs.loc[rd, selected] = ret(rd→rd+20). Это имплицитно предполагает вход по close(rd) (same-bar close). Если реальное исполнение по open(rd+1), т |
| 4 | 1.4 Off-by-one сдвиг в portfolio return | NOT_VERIFIED | HIGH | target_rs = close.shift(-20)/close - 1. Это ret(close(t)→close(t+20)). В портфеле port_rs = target_rs.loc[rd] = ret(rd→rd+20). Если вход по close(rd), всё ок. Если по open(rd) — ошибка: используется c |
| 5 | 2.1 Static: нет shift(-n)/center=True в rs_pool.py | PASS | LOW | Все rolling/shift — только с положительными окнами. |
| 6 | 2.2 Causality test (truncate after t) | PASS | LOW | 10 дат: фичи на t не меняются при обрезке после t. |
| 7 | 2.3 cs_percentile_rank — per-date | PASS | LOW | На синтетике: ранги per-date = [0.2,0.4,0.6,0.8,1.0] на каждой дате. |
| 8 | 3.1 Последние 20 дат target = NaN | PASS | LOW | Последние 20 дат target = NaN (shift(-20)). |
| 9 | 3.2 Target-интервалы train/val не пересекаются | PASS | LOW | Purge=21 >= HORIZON=20: last_train_target_end < first_val на всех 6 фолдах. |
| 10 | 4.1 Rank-Gauss — per-date | PASS | LOW | На синтетике: rank-gauss одинаков для одинаковых рангов на разных датах. |
| 11 | 4.2 Rank-Gauss не зависит от других дат (нет fit на val) | PASS | LOW | Изменение одной даты не меняет rank-gauss на других датах (детерминированное преобразование, не fit). |
| 12 | 4.3 Features percentile rank — per-date | PASS | LOW | На 5 датах: ранги совпадают с rank(pct) внутри даты. |
| 13 | 5.1 make_folds совпадает с независимой реконструкцией | PASS | LOW | 6 фолдов: train/val даты совпадают. |
| 14 | 5.2 Нет shuffle в WF | PASS | LOW | В wf.py нет shuffle/permutation/sample. |
| 15 | 5.3 Val не пересекается с train | PASS | LOW | 6 фолдов: нет общих дат. |
| 16 | 5.4 Early stopping не на внешнем val | PASS | LOW | В проверенных файлах нет eval_set с внешним val. |
| 17 | 5.5 Итерации фиксированы | PASS | LOW | BASE_PARAMS['iterations'] = 1500. В base_model_v2.py используется фикс. 300. |
| 18 | 6.1 Portfolio equity CSV существует | PASS | LOW | 71 строк, 2018-09-03 → 2023-12-25 |
| 19 | 6.2 Даты ребаланса в val-фолдах | PASS | LOW | Все 71 дат ребаланса попадают в val-периоды. |
| 20 | 6.3 Нет дубликатов дат ребаланса | PASS | LOW | 71 дат, все уникальны. |
| 21 | 6.4 Turnover formula — one-way (0..1) | PASS | LOW | mean=0.775, annualized=976.1%. Диапазон [0,1] — one-way turnover. |
| 22 | 8.1 Делистингованные тикеры в universe | FAIL | HIGH | Тикеров на дату: min=3, median=135, max=180. Тикеров, торговавшихся в начале, но не в конце: 0. Нет ни одного делистингованного — survivorship bias вероятен. Список тикеров, вероятно, сформирован по т |
| 23 | 9.1 Независимый пересчёт метрик | PASS | LOW | max diff = 0.000046. Метрики совпадают с заявленными. |
| 24 | 10.1 Future-data mutation test | PASS | LOW | 5 дат: мутация будущего не меняет фичи на t. |
| 25 | 10.3 Target permutation test | NOT_VERIFIED | MEDIUM | error: array is read-only |
| 26 | 10.6 Random-ranking benchmark | FAIL | CRITICAL | Real total: 2.9794. Random: mean=3.0981, max=3.5013. Real в 40% верхнего хвоста. Реальный результат хуже среднего random — преимущество модели не подтверждено. |

## CRITICAL и HIGH проблемы


### 1.3 Entry price vs signal bar — NOT_VERIFIED (HIGH)
- **Evidence:** В rs_is_wf.py портфель: selected на rd, port_rs = target_rs.loc[rd, selected] = ret(rd→rd+20). Это имплицитно предполагает вход по close(rd) (same-bar close). Если реальное исполнение по open(rd+1), то доходность должна быть ret(rd+1→rd+21), что ДРУГОЙ ряд. Нужно подтвердить, какой execution model заложен.
- **Files:** backtests/portfolio_tests/rs_is_wf.py
- **Recommendation:** Явно зафиксировать execution model: MOC (close rd) или open(rd+1). Если MOC — обосновать реалистичность. Если open(rd+1) — пересчитать port_rs.

### 1.4 Off-by-one сдвиг в portfolio return — NOT_VERIFIED (HIGH)
- **Evidence:** target_rs = close.shift(-20)/close - 1. Это ret(close(t)→close(t+20)). В портфеле port_rs = target_rs.loc[rd] = ret(rd→rd+20). Если вход по close(rd), всё ок. Если по open(rd) — ошибка: используется close(rd) как база вместо open(rd).
- **Files:** backtests/portfolio_tests/rs_is_wf.py

### 8.1 Делистингованные тикеры в universe — FAIL (HIGH)
- **Evidence:** Тикеров на дату: min=3, median=135, max=180. Тикеров, торговавшихся в начале, но не в конце: 0. Нет ни одного делистингованного — survivorship bias вероятен. Список тикеров, вероятно, сформирован по текущему состоянию.
- **Files:** dataset/loader.py
- **Recommendation:** Проверить источник списка тикеров: если это TQBR на текущую дату, то исторические делистингованные отсутствуют.

### 10.6 Random-ranking benchmark — FAIL (CRITICAL)
- **Evidence:** Real total: 2.9794. Random: mean=3.0981, max=3.5013. Real в 40% верхнего хвоста. Реальный результат хуже среднего random — преимущество модели не подтверждено.

## Трассировка временной информации

**Пример: t = 2019-02-27**

| Этап | Что используется | Когда доступно |
|------|------------------|----------------|
| Feature date | close(t) = 137.5000 (пример) | конец сессии t |
| Target horizon | close(t) → close(2019-03-28) | конец сессии t+20 |
| Prediction date | t (вечер) | вечер t |
| Entry | следующий торговый день | open(t+1) |
| Exit | через 20 торговых дней | close(t+20) |


## Итоговый вердикт

**WF INVALID**

Есть CRITICAL проблемы — результатам нельзя доверять. Требуется исправление до любых выводов.


## Ограничения аудита

- Не проверялась робастность стратегии (это отдельный этап).
- Не проверялись рыночные режимы (это отдельный этап).
- Survivorship bias: если universe сформирован по текущему состоянию, исторические делистингованные отсутствуют.
- Execution model (MOC vs open(t+1)) — требует явной фиксации.
- Отсутствие обнаруженной ошибки не доказывает её отсутствия.
