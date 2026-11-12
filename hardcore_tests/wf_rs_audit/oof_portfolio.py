# hardcore_tests/wf_rs_audit/oof_portfolio.py
# ============================================================
# Блок 6: OOF predictions и портфельный тест.
# Проверяем: каждая prediction — out-of-sample, нет повторных дат,
# нет in-sample, правильная синхронизация.
# ============================================================
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    load_ohlcv, load_imoex, load_split_date, independent_target,
    independent_folds, HORIZON, PURGE, TOP_N, REBAL_STEP, verdict,
)


def test_portfolio_equity_exists() -> dict:
    """Проверяем: equity CSV существует и не пуст."""
    csv = Path(__file__).parent.parent.parent / "backtests" / "portfolio_tests" / "output" / "rs_is_wf_equity.csv"
    if not csv.exists():
        return verdict(
            "6.1 Portfolio equity CSV существует",
            "FAIL", "HIGH",
            evidence=f"Файл не найден: {csv}",
        )
    df = pd.read_csv(csv, parse_dates=["date"])
    if len(df) == 0:
        return verdict(
            "6.1 Portfolio equity CSV существует",
            "FAIL", "HIGH",
            evidence="Файл пуст.",
        )
    return verdict(
        "6.1 Portfolio equity CSV существует",
        "PASS",
        evidence=f"{len(df)} строк, {df['date'].min().date()} → {df['date'].max().date()}",
        files=[str(csv)],
    )


def test_rebalance_dates_in_folds() -> dict:
    """Проверяем: все даты ребаланса попадают в val-периоды фолдов."""
    csv = Path(__file__).parent.parent.parent / "backtests" / "portfolio_tests" / "output" / "rs_is_wf_equity.csv"
    if not csv.exists():
        return verdict("6.2 Даты ребаланса в val-фолдах", "NOT_VERIFIED", "MEDIUM",
                       evidence="CSV не найден")
    port = pd.read_csv(csv, parse_dates=["date"])

    split_date = load_split_date()
    raw = load_ohlcv(split_date)
    imoex = load_imoex(split_date)
    target = independent_target(raw["close"], imoex)
    dates = target.dropna(how="all").index
    folds = independent_folds(dates, n_folds=6, purge=PURGE)

    val_dates_all = set()
    for _, va in folds:
        val_dates_all.update(va)

    outside = [d for d in port["date"] if d not in val_dates_all]
    if outside:
        return verdict(
            "6.2 Даты ребаланса в val-фолдах",
            "FAIL", "CRITICAL",
            evidence=f"{len(outside)} дат ребаланса вне val: {outside[:5]}",
        )
    return verdict(
        "6.2 Даты ребаланса в val-фолдах",
        "PASS",
        evidence=f"Все {len(port)} дат ребаланса попадают в val-периоды.",
    )


def test_no_duplicate_dates() -> dict:
    csv = Path(__file__).parent.parent.parent / "backtests" / "portfolio_tests" / "output" / "rs_is_wf_equity.csv"
    if not csv.exists():
        return verdict("6.3 Нет дубликатов дат ребаланса", "NOT_VERIFIED", "LOW",
                       evidence="CSV не найден")
    port = pd.read_csv(csv, parse_dates=["date"])
    dup = port[port.duplicated("date", keep=False)]
    if len(dup) > 0:
        return verdict(
            "6.3 Нет дубликатов дат ребаланса",
            "FAIL", "HIGH",
            evidence=f"{len(dup)} дубликатов: {dup['date'].dt.date.unique()[:5]}",
        )
    return verdict(
        "6.3 Нет дубликатов дат ребаланса",
        "PASS",
        evidence=f"{len(port)} дат, все уникальны.",
    )


def test_turnover_formula() -> dict:
    """Проверяем: turnover = one-way (sold/TOP_N), не two-way."""
    csv = Path(__file__).parent.parent.parent / "backtests" / "portfolio_tests" / "output" / "rs_is_wf_equity.csv"
    if not csv.exists() or "turnover" not in pd.read_csv(csv, nrows=1).columns:
        return verdict("6.4 Turnover formula — one-way", "NOT_VERIFIED", "MEDIUM",
                       evidence="CSV не найден или нет колонки turnover")
    port = pd.read_csv(csv, parse_dates=["date"])
    # one-way: 0..1, first entry = 1.0
    to = port["turnover"].dropna()
    if (to < 0).any() or (to > 1.0 + 1e-9).any():
        return verdict(
            "6.4 Turnover formula — one-way (0..1)",
            "FAIL", "HIGH",
            evidence=f"turnover вне [0,1]: min={to.min():.3f}, max={to.max():.3f}",
        )
    mean_to = to.mean()
    annual = mean_to * (252 / REBAL_STEP)
    return verdict(
        "6.4 Turnover formula — one-way (0..1)",
        "PASS",
        evidence=f"mean={mean_to:.3f}, annualized={annual*100:.1f}%. "
                 f"Диапазон [0,1] — one-way turnover.",
    )


def run() -> list:
    return [
        test_portfolio_equity_exists(),
        test_rebalance_dates_in_folds(),
        test_no_duplicate_dates(),
        test_turnover_formula(),
    ]