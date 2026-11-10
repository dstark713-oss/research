# rs_model/wf.py
# ============================================================
# Anchored expanding walk-forward с purge.
# ============================================================
import numpy as np
import pandas as pd

from .config import PURGE, HORIZON


def make_folds(dates: pd.DatetimeIndex, n_folds: int = 6, purge: int = PURGE):
    """
    Anchored expanding WF:
      - fold 1: train = dates[:k], val = dates[k:k+w]
      - fold 2: train = dates[:k+w], val = dates[k+w:k+2w]
      - ...
    Purge = gap между train и val (в торговых днях).
    Возвращает список (train_dates, val_dates).
    """
    n = len(dates)
    # делим на n_folds+1 равных блоков; первый — train-start, последний — последний val
    # Начинаем с ~50% данных на первый train
    first_train_end = n // 2
    remaining = n - first_train_end
    fold_size = remaining // n_folds

    folds = []
    for k in range(n_folds):
        train_end = first_train_end + k * fold_size
        val_start = train_end + purge
        val_end = min(val_start + fold_size, n)
        if val_start >= n or val_end - val_start < 10:
            break
        train_dates = dates[:train_end]
        val_dates = dates[val_start:val_end]
        folds.append((train_dates, val_dates))
    return folds