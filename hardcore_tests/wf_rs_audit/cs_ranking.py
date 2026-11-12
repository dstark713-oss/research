# hardcore_tests/wf_rs_audit/cs_ranking.py
# ============================================================
# Блок 4: leakage в cross-sectional ranking.
# Проверяем: rank-gauss target не использует val labels для fit;
# cs_percentile_rank не использует глобальные статистики.
# ============================================================
import numpy as np
import pandas as pd

from rs_model.data import rank_gauss_target, grades_target
from .common import verdict


def test_rank_gauss_per_date() -> dict:
    """
    Rank-Gauss должен быть per-date. Проверяем на синтетике:
    два разных набора target на одной дате → разные rank-gauss,
    но на другой дате — тоже per-date.
    """
    dates = pd.date_range("2020-01-01", periods=3)
    cols = [f"T{i}" for i in range(5)]
    # на каждой дате target = [1, 2, 3, 4, 5] * (d+1)
    data = np.array([[1, 2, 3, 4, 5]]) * (np.arange(3)[:, None] + 1)
    target = pd.DataFrame(data.astype(float), index=dates, columns=cols)

    rg = rank_gauss_target(target)
    # на каждой дате rank-gauss должен быть одинаковым (т.к. ранги одинаковы)
    d0 = rg.loc[dates[0]].values
    d1 = rg.loc[dates[1]].values
    d2 = rg.loc[dates[2]].values
    max_diff = max(np.abs(d0 - d1).max(), np.abs(d0 - d2).max())
    if max_diff > 1e-9:
        return verdict(
            "4.1 Rank-Gauss — per-date",
            "FAIL", "CRITICAL",
            evidence=f"Rank-gauss зависит от других дат: max diff = {max_diff:.4f}",
            files=["rs_model/data.py::rank_gauss_target"],
        )
    return verdict(
        "4.1 Rank-Gauss — per-date",
        "PASS",
        evidence="На синтетике: rank-gauss одинаков для одинаковых рангов на разных датах.",
        files=["rs_model/data.py::rank_gauss_target"],
    )


def test_rank_gauss_no_val_fit() -> dict:
    """
    Rank-Gauss — детерминированное преобразование (inverse normal CDF).
    Оно НЕ fit-ится на данных — только применяется. Значит, val labels
    не могут «повлиять» на преобразование.
    """
    # Проверяем: rank_gauss(x) не зависит от других дат
    dates = pd.date_range("2020-01-01", periods=3)
    cols = [f"T{i}" for i in range(5)]
    data1 = np.array([[1, 2, 3, 4, 5]] * 3, dtype=float)
    t1 = pd.DataFrame(data1, index=dates, columns=cols)

    data2 = data1.copy()
    data2[2] = [10, 20, 30, 40, 50]  # меняем ТОЛЬКО дату 3
    t2 = pd.DataFrame(data2, index=dates, columns=cols)

    rg1 = rank_gauss_target(t1)
    rg2 = rank_gauss_target(t2)

    # rank-gauss на датах 1 и 2 должен быть одинаковым
    diff_12 = max(np.abs(rg1.loc[dates[0]] - rg2.loc[dates[0]]).max(),
                  np.abs(rg1.loc[dates[1]] - rg2.loc[dates[1]]).max())
    if diff_12 > 1e-9:
        return verdict(
            "4.2 Rank-Gauss не зависит от других дат (нет fit на val)",
            "FAIL", "CRITICAL",
            evidence=f"Изменение даты 3 повлияло на даты 1-2: diff = {diff_12:.4f}",
            files=["rs_model/data.py::rank_gauss_target"],
        )
    return verdict(
        "4.2 Rank-Gauss не зависит от других дат (нет fit на val)",
        "PASS",
        evidence="Изменение одной даты не меняет rank-gauss на других датах "
                 "(детерминированное преобразование, не fit).",
        files=["rs_model/data.py::rank_gauss_target"],
    )


def test_features_rank_per_date() -> dict:
    """
    Features percentile ranks — per-date. Проверяем на реальных данных:
    изменение фичи на дате t не меняет ранги на дате t-1.
    """
    import sqlite3
    from .common import STOCKS_DB, load_split_date
    split = load_split_date()
    # простой тест: берём feature, ранжируем, проверяем что ранги на каждой
    # дате — это rank внутри этой даты (а не глобальный)
    conn = sqlite3.connect(STOCKS_DB)
    df = pd.read_sql(
        "SELECT ticker, date, close FROM prices WHERE date < ?",
        conn, params=(split,),
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot(index="date", columns="ticker", values="close").sort_index()
    ret = close.pct_change()

    from rs_model.data import cs_percentile_rank
    ranked = cs_percentile_rank(ret)

    # проверяем: на каждой дате ранги = rank(pct) внутри даты
    sample_dates = ranked.dropna(how="all").index[100:105]
    for d in sample_dates:
        row = ret.loc[d].dropna()
        if len(row) < 10:
            continue
        expected = row.rank(pct=True)
        actual = ranked.loc[d, expected.index]
        diff = np.abs(actual - expected).max()
        if diff > 1e-9:
            return verdict(
                "4.3 Features percentile rank — per-date",
                "FAIL", "CRITICAL",
                evidence=f"На дате {d.date()}: diff = {diff:.4f}",
                files=["rs_model/data.py::cs_percentile_rank"],
            )
    return verdict(
        "4.3 Features percentile rank — per-date",
        "PASS",
        evidence="На 5 датах: ранги совпадают с rank(pct) внутри даты.",
        files=["rs_model/data.py::cs_percentile_rank"],
    )


def run() -> list:
    return [
        test_rank_gauss_per_date(),
        test_rank_gauss_no_val_fit(),
        test_features_rank_per_date(),
    ]