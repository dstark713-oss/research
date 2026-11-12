# hardcore_tests/wf_rs_audit/common.py
# ============================================================
# Общие хелперы аудита. НЕЗАВИСИМЫ от rs_model — загружаем данные
# и фичи «с нуля», чтобы не доверять коду модели.
# ============================================================
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
STOCKS_DB = ROOT / "dataset" / "sql" / "stocks.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"
SPLIT_JSON = ROOT / "backtests" / "split_info.json"

HORIZON = 20
PURGE = 21
TOP_N = 10
REBAL_STEP = 20
TRADING_DAYS = 252


def load_split_date() -> str:
    return json.loads(SPLIT_JSON.read_text())["split_date"]


def load_ohlcv(split_date: str) -> dict:
    conn = sqlite3.connect(STOCKS_DB)
    df = pd.read_sql(
        "SELECT ticker, date, open, high, low, close, volume "
        "FROM prices WHERE date < ?",
        conn, params=(split_date,),
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = df.pivot(index="date", columns="ticker", values=col).sort_index()
    return out


def load_imoex(split_date: str) -> pd.Series:
    conn = sqlite3.connect(INDEXES_DB)
    df = pd.read_sql(
        "SELECT date, close FROM indexes WHERE ticker='IMOEX' AND date < ?",
        conn, params=(split_date,),
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()


def independent_target(close: pd.DataFrame, imoex: pd.Series,
                       horizon: int = HORIZON) -> pd.DataFrame:
    """Независимый target: RS = ret_i(t→t+h) - ret_IMOEX(t→t+h)."""
    fwd = close.shift(-horizon) / close - 1
    imoex_fwd = imoex.shift(-horizon) / imoex - 1
    imoex_fwd = imoex_fwd.reindex(close.index)
    return fwd.sub(imoex_fwd, axis=0)


def cs_percentile_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Независимый cross-sectional percentile rank."""
    return df.rank(axis=1, pct=True)


def independent_daily_ic(feature: pd.DataFrame, target: pd.DataFrame,
                         min_stocks: int = 20) -> pd.Series:
    """Векторизованный daily Spearman IC (независимая реализация)."""
    common_dates = feature.index.intersection(target.index)
    common_tickers = feature.columns.intersection(target.columns)
    if len(common_tickers) < min_stocks:
        return pd.Series(dtype=float)
    f = feature.loc[common_dates, common_tickers]
    t = target.loc[common_dates, common_tickers]
    fl = f.stack().rename("f")
    tl = t.stack().rename("t")
    df = pd.concat([fl, tl], axis=1).dropna()
    if len(df) == 0:
        return pd.Series(dtype=float)
    df["fr"] = df.groupby(level=0)["f"].rank(method="average")
    df["tr"] = df.groupby(level=0)["t"].rank(method="average")
    g = df.groupby(level=0)
    df["df"] = df["fr"] - g["fr"].transform("mean")
    df["dt"] = df["tr"] - g["tr"].transform("mean")
    df["dft"] = df["df"] * df["dt"]
    df["df2"] = df["df"] ** 2
    df["dt2"] = df["dt"] ** 2
    g2 = df.groupby(level=0)
    cov = g2["dft"].mean()
    vf = g2["df2"].mean()
    vt = g2["dt2"].mean()
    counts = g2.size()
    ic = cov / np.sqrt(vf * vt).replace(0, np.nan)
    ic = ic[(counts >= min_stocks) & np.isfinite(ic)]
    ic.index.name = "date"
    return ic.sort_index()


def independent_folds(dates: pd.DatetimeIndex, n_folds: int = 6,
                      purge: int = PURGE) -> list:
    """
    Независимая реконструкция WF folds.
    Anchored expanding: первый train заканчивается на 50%, далее val по fold_size.
    """
    n = len(dates)
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
        folds.append((dates[:train_end], dates[val_start:val_end]))
    return folds


# ============================================================
# Результат проверки — единый формат для отчёта
# ============================================================
def verdict(name: str, status: str, severity: str = "LOW",
            evidence: str = "", files: list = None,
            recommendation: str = "") -> dict:
    """
    status: PASS / FAIL / NOT_VERIFIED
    severity: CRITICAL / HIGH / MEDIUM / LOW (только для FAIL/NOT_VERIFIED)
    """
    assert status in ("PASS", "FAIL", "NOT_VERIFIED")
    assert severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    return {
        "name": name,
        "status": status,
        "severity": severity if status != "PASS" else "LOW",
        "evidence": evidence,
        "files": files or [],
        "recommendation": recommendation,
    }