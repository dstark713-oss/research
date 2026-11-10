# rs_model/metrics.py
# ============================================================
# Метрики ранжирования.
# ============================================================
import numpy as np
import pandas as pd
from scipy import stats


def daily_ic(pred_df: pd.DataFrame, target_df: pd.DataFrame, min_stocks=20) -> pd.Series:
    """Spearman IC по датам (pred_df и target_df — wide)."""
    common_dates = pred_df.index.intersection(target_df.index)
    common_tickers = pred_df.columns.intersection(target_df.columns)
    if len(common_tickers) < min_stocks:
        return pd.Series(dtype=float)
    p = pred_df.loc[common_dates, common_tickers]
    t = target_df.loc[common_dates, common_tickers]
    ics = {}
    for d in common_dates:
        pv = p.loc[d]; tv = t.loc[d]
        m = pv.notna() & tv.notna()
        if m.sum() < min_stocks: continue
        try:
            ic = stats.spearmanr(pv[m], tv[m]).correlation
            if np.isfinite(ic): ics[d] = ic
        except Exception:
            pass
    return pd.Series(ics).sort_index()


def mean_ic(pred_df, target_df) -> float:
    ics = daily_ic(pred_df, target_df)
    return ics.mean() if len(ics) else np.nan


def ndcg_at_k(pred_df: pd.DataFrame, target_df: pd.DataFrame, k=10) -> pd.Series:
    """NDCG@k по датам. Релевантность = percentile rank target."""
    common_dates = pred_df.index.intersection(target_df.index)
    common_tickers = pred_df.columns.intersection(target_df.columns)
    p = pred_df.loc[common_dates, common_tickers]
    t = target_df.loc[common_dates, common_tickers]
    out = {}
    for d in common_dates:
        pv = p.loc[d]; tv = t.loc[d]
        m = pv.notna() & tv.notna()
        if m.sum() < k + 5: continue
        pv = pv[m]; tv = tv[m]
        # релевантность: percentile rank target
        rel = tv.rank(pct=True).values
        # топ-k по pred
        top_k_idx = pv.nlargest(k).index
        # DCG
        dcg = 0.0
        for i, idx in enumerate(top_k_idx):
            r = rel[pv.index.get_loc(idx)]
            dcg += (2**r - 1) / np.log2(i + 2)
        # IDCG — идеальный порядок
        ideal = np.sort(rel)[::-1][:k]
        idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
        if idcg > 0:
            out[d] = dcg / idcg
    return pd.Series(out).sort_index()


def top_decile_spread(pred_df, target_df) -> pd.Series:
    """Средний RS топ-10% минус средний RS боттом-10% по датам."""
    common_dates = pred_df.index.intersection(target_df.index)
    common_tickers = pred_df.columns.intersection(target_df.columns)
    p = pred_df.loc[common_dates, common_tickers]
    t = target_df.loc[common_dates, common_tickers]
    out = {}
    for d in common_dates:
        pv = p.loc[d]; tv = t.loc[d]
        m = pv.notna() & tv.notna()
        if m.sum() < 30: continue
        pv = pv[m]; tv = tv[m]
        n = len(pv)
        k = max(1, int(n * 0.1))
        top = pv.nlargest(k).index
        bot = pv.nsmallest(k).index
        out[d] = tv[top].mean() - tv[bot].mean()
    return pd.Series(out).sort_index()


def top_n_rs(pred_df, target_df, n=10) -> pd.Series:
    """Средний RS топ-n по датам."""
    common_dates = pred_df.index.intersection(target_df.index)
    common_tickers = pred_df.columns.intersection(target_df.columns)
    p = pred_df.loc[common_dates, common_tickers]
    t = target_df.loc[common_dates, common_tickers]
    out = {}
    for d in common_dates:
        pv = p.loc[d]; tv = t.loc[d]
        m = pv.notna() & tv.notna()
        if m.sum() < n: continue
        pv = pv[m]; tv = tv[m]
        top = pv.nlargest(n).index
        out[d] = tv[top].mean()
    return pd.Series(out).sort_index()


def evaluate(pred_df: pd.DataFrame, target_df: pd.DataFrame) -> dict:
    ics = daily_ic(pred_df, target_df)
    ndcg = ndcg_at_k(pred_df, target_df, k=10)
    tds = top_decile_spread(pred_df, target_df)
    top10 = top_n_rs(pred_df, target_df, n=10)
    return {
        "mean_ic": float(ics.mean()) if len(ics) else np.nan,
        "std_ic": float(ics.std()) if len(ics) else np.nan,
        "hit_rate_ic": float((ics > 0).mean()) if len(ics) else np.nan,
        "ndcg@10": float(ndcg.mean()) if len(ndcg) else np.nan,
        "top_decile_spread": float(tds.mean()) if len(tds) else np.nan,
        "top10_mean_rs": float(top10.mean()) if len(top10) else np.nan,
        "top10_median_rs": float(top10.median()) if len(top10) else np.nan,
        "n_dates": int(len(ics)),
    }