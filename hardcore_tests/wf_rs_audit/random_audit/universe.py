# random_audit/universe.py
# ============================================================
# Исторический universe на каждую дату.
#
# Определение доступности:
#   - тикер доступен на дату t, если у него есть close(t) и close(t-1).
#   - минимальная история: не менее MIN_HISTORY дней close до t.
#
# ВАЖНО: фильтры НЕ используют будущую доходность.
# ============================================================
import numpy as np
import pandas as pd

MIN_HISTORY = 60  # минимум торговых дней истории


def build_availability(close: pd.DataFrame) -> pd.DataFrame:
    """
    Boolean DataFrame: True, если тикер доступен на дату t.
    Условия: есть close(t), close(t-1), и минимум MIN_HISTORY дней истории.
    """
    has_today = close.notna()
    has_prev = close.shift(1).notna()
    # rolling count non-NaN до t включительно
    history_count = close.notna().cumsum()
    has_history = history_count >= MIN_HISTORY
    return has_today & has_prev & has_history


def get_available_tickers(availability: pd.DataFrame, date: pd.Timestamp) -> list:
    if date not in availability.index:
        return []
    row = availability.loc[date]
    return row[row].index.tolist()


def get_ml_available_tickers(close: pd.DataFrame, target: pd.DataFrame,
                             date: pd.Timestamp) -> list:
    """
    Пул, который был доступен ML-модели на дату date:
    - есть prediction (значит, фичи не NaN) — но prediction мы не импортируем,
      поэтому используем прокси: target non-NaN + close non-NaN.
    - Фактически ML использует features, которые требуют историю, поэтому
      пул ML ⊆ пул availability.
    Возвращаем: тикеры, у которых target не NaN и close не NaN на date.
    Это консервативная оценка пула ML (может быть шире фактического).
    """
    if date not in close.index or date not in target.index:
        return []
    close_ok = close.loc[date].notna()
    target_ok = target.loc[date].notna()
    avail = close_ok & target_ok
    return avail[avail].index.tolist()