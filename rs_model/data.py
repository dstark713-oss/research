# rs_model/data.py
# ============================================================
# Загрузка IS, построение фич (percentile rank), таргетов:
#   - rank-gauss (для регрессии)
#   - 5 grades (для ranker)
# Только IS, VAL не трогаем.
# ============================================================
import json
import sqlite3

import numpy as np
import pandas as pd

from features.rs_pool import build_rs_pool
from .config import SPLIT_JSON, STOCKS_DB, INDEXES_DB, HORIZON, N_GRADES, FEATURES


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


def build_target_rs(raw: dict, imoex: pd.Series) -> pd.DataFrame:
    """RS_{i,t} = ret_i(t→t+h) - ret_IMOEX(t→t+h). Последние HORIZON дней = NaN."""
    C = raw["close"]
    fwd = C.shift(-HORIZON) / C - 1
    imoex_fwd = imoex.shift(-HORIZON) / imoex - 1
    imoex_fwd = imoex_fwd.reindex(C.index)
    return fwd.sub(imoex_fwd, axis=0)


def cs_percentile_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank по каждой дате (0..1)."""
    return df.rank(axis=1, pct=True)


def rank_gauss_target(rs: pd.DataFrame) -> pd.DataFrame:
    """
    Rank-Gauss: внутри каждой даты:
      - ранжируем акции по RS,
      - percentile,
      - inverse normal CDF.
    """
    from scipy.stats import norm
    # ранги внутри даты
    ranks = rs.rank(axis=1, pct=True)
    # центрируем, чтобы не было 0 и 1 (inf)
    n = rs.notna().sum(axis=1)
    # безопасно: clip в (eps, 1-eps)
    eps = 1.0 / (2 * n.max()) if len(n) > 0 else 1e-6
    ranks = ranks.clip(eps, 1 - eps)
    return pd.DataFrame(norm.ppf(ranks.values), index=rs.index, columns=rs.columns)


def grades_target(rs: pd.DataFrame, n_grades: int = N_GRADES) -> pd.DataFrame:
    """
    5 grades из RS по квинтилям внутри каждой даты (0..4).
    """
    ranks = rs.rank(axis=1, pct=True)
    grades = np.floor(ranks * n_grades).clip(0, n_grades - 1)
    return pd.DataFrame(grades.values, index=rs.index, columns=rs.columns).astype(float)


def build_dataset(split_date: str = None):
    """
    Возвращает:
      features_raw: dict[name, DataFrame]   — сырые фичи
      features_rank: dict[name, DataFrame]  — percentile ranks фич
      target_rs: DataFrame                  — RS
      target_gauss: DataFrame               — rank-gauss (для регрессии)
      target_grades: DataFrame              — 5 grades (для ranker)
    """
    if split_date is None:
        split_date = load_split_date()
    raw = load_ohlcv(split_date)
    imoex = load_imoex(split_date)

    features_raw = build_rs_pool(raw)
    # только 11 утверждённых
    features_raw = {k: features_raw[k] for k in FEATURES}

    features_rank = {k: cs_percentile_rank(v) for k, v in features_raw.items()}

    target_rs = build_target_rs(raw, imoex)
    target_gauss = rank_gauss_target(target_rs)
    target_grades = grades_target(target_rs)

    return features_raw, features_rank, target_rs, target_gauss, target_grades


def to_long(features_rank: dict, target: pd.DataFrame) -> pd.DataFrame:
    """
    Длинный формат: date, ticker, features..., target.
    Только строки, где target не NaN.
    """
    f_names = list(features_rank.keys())
    dfs = []
    for name in f_names:
        dfs.append(features_rank[name].stack().rename(name))
    df = pd.concat(dfs, axis=1)
    df["target"] = target.stack()
    df = df.dropna(subset=["target"])
    df.index.names = ["date", "ticker"]
    return df.reset_index()