# hardcore_tests/wf_rs_audit/target_purge.py
# ============================================================
# Блок 3: target leakage и purge.
# Проверяем: границы target, последние 20 дат = NaN, пересечение
# train/val target-интервалов.
# ============================================================
import numpy as np
import pandas as pd

from .common import (
    load_ohlcv, load_imoex, load_split_date,
    independent_target, independent_folds, HORIZON, PURGE, verdict,
)


def test_target_boundary() -> dict:
    split_date = load_split_date()
    raw = load_ohlcv(split_date)
    imoex = load_imoex(split_date)
    target = independent_target(raw["close"], imoex)

    last = target.index[-HORIZON:]
    n_notnan = int(target.loc[last].notna().sum().sum())
    if n_notnan > 0:
        return verdict(
            "3.1 Последние 20 дат target = NaN",
            "FAIL", "CRITICAL",
            evidence=f"Найдено {n_notnan} non-NaN в последних 20 датах IS. "
                     f"Это утечка в VAL.",
            files=["hardcore_tests/wf_rs_audit/common.py::independent_target"],
        )
    return verdict(
        "3.1 Последние 20 дат target = NaN",
        "PASS",
        evidence=f"Последние {HORIZON} дат target = NaN (shift(-20)).",
    )


def test_fold_target_overlap() -> dict:
    split_date = load_split_date()
    raw = load_ohlcv(split_date)
    imoex = load_imoex(split_date)
    target = independent_target(raw["close"], imoex)
    dates = target.dropna(how="all").index
    folds = independent_folds(dates, n_folds=6, purge=PURGE)

    fails = []
    for i, (tr, va) in enumerate(folds):
        if len(tr) == 0 or len(va) == 0:
            continue
        last_train_feature = tr[-1]
        # target для last_train использует close(last_train + 20)
        idx_last = dates.get_loc(last_train_feature)
        last_train_target_end = dates[min(idx_last + HORIZON, len(dates) - 1)]
        first_val = va[0]
        # Если last_train_target_end >= first_val — пересечение target-интервалов
        if last_train_target_end >= first_val:
            fails.append({
                "fold": i + 1,
                "last_train": str(last_train_feature.date()),
                "last_train_target_end": str(last_train_target_end.date()),
                "first_val": str(first_val.date()),
            })
    if fails:
        return verdict(
            "3.2 Target-интервалы train/val не пересекаются",
            "FAIL", "CRITICAL",
            evidence=f"Пересечения: {fails}",
            files=["hardcore_tests/wf_rs_audit/common.py::independent_folds"],
        )
    return verdict(
        "3.2 Target-интервалы train/val не пересекаются",
        "PASS",
        evidence=f"Purge={PURGE} >= HORIZON={HORIZON}: "
                 f"last_train_target_end < first_val на всех 6 фолдах.",
    )


def run() -> list:
    return [test_target_boundary(), test_fold_target_overlap()]