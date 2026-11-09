# hardcore_tests/coverage.py
# ============================================================
# ТЕСТ 6: Cross-sectional Coverage & Concentration.
#
# Для каждой фичи:
#   - IC по каждому тикеру (через время);
#   - распределение индивидуальных IC;
#   - доля суммарного сигнала на топ-5 тикеров;
#   - стабильность при исключении топ-тикера.
#
# PASS: нет концентрации (топ-5 < 20% сигнала)
# WARN: концентрация 20-50%
# FAIL: концентрация > 50% (сигнал на 1-2 тикерах)
# ============================================================
import numpy as np
import pandas as pd
from scipy import stats

from features.rs_pool import RS_POOL_NAMES
from .data import load_all


def run(split_date=None) -> dict:
    raw, imoex, features, target = load_all(split_date)

    rows = []
    for name in RS_POOL_NAMES:
        f = features[name]
        common_dates = f.index.intersection(target.index)
        common_tickers = f.columns.intersection(target.columns)
        f = f.loc[common_dates, common_tickers]
        t = target.loc[common_dates, common_tickers]

        # IC по каждому тикеру (через время)
        ticker_ics = {}
        for ticker in common_tickers:
            fv = f[ticker].values
            tv = t[ticker].values
            m = np.isfinite(fv) & np.isfinite(tv)
            if m.sum() < 100: continue
            try:
                ic = stats.spearmanr(fv[m], tv[m]).correlation
                if np.isfinite(ic): ticker_ics[ticker] = ic
            except Exception:
                pass

        if len(ticker_ics) < 10:
            continue

        ics_arr = np.array(list(ticker_ics.values()))
        # доля "положительных" вкладов (по знаку общего IC)
        overall_sign = np.sign(ics_arr.mean())
        pos_share = (ics_arr * overall_sign > 0).mean()

        # топ-5 концентрация: сумма |IC| топ-5 / сумма |IC| всех
        abs_ics = np.abs(ics_arr)
        top5_share = np.sort(abs_ics)[-5:].sum() / (abs_ics.sum() + 1e-9)

        # стабильность: исключаем топ-1 тикер по |IC|, пересчитываем mean IC
        top_ticker = list(ticker_ics.keys())[int(np.argmax(abs_ics))]
        rest = {k: v for k, v in ticker_ics.items() if k != top_ticker}
        mean_all = ics_arr.mean()
        mean_rest = np.mean(list(rest.values())) if rest else np.nan
        delta_excl = abs(mean_rest - mean_all) / abs(mean_all) if mean_all != 0 else np.nan

        rows.append({
            "feature": name,
            "n_tickers": len(ticker_ics),
            "pos_share": pos_share,
            "top5_share": top5_share,
            "mean_ic_by_ticker": mean_all,
            "mean_ic_excl_top": mean_rest,
            "delta_excl_top_pct": delta_excl,
            "top_ticker": top_ticker,
        })

    df = pd.DataFrame(rows)
    print("[coverage] Концентрация по тикерам:")
    for _, r in df.iterrows():
        flag = ""
        if r["top5_share"] > 0.5: flag = " ⚠ FAIL"
        elif r["top5_share"] > 0.2: flag = " ⚠ WARN"
        print(f"  {r['feature']:<35} top5={r['top5_share']:.1%} "
              f"pos={r['pos_share']:.1%} excl_top={r['top_ticker']} (Δ{r['delta_excl_top_pct']:.0%}){flag}")

    fails = df[df["top5_share"] > 0.5]
    warns = df[(df["top5_share"] > 0.2) & (df["top5_share"] <= 0.5)]
    if len(fails) > 0:
        status = "FAIL"
    elif len(warns) > 0:
        status = "WARN"
    else:
        status = "PASS"
    print(f"\n[coverage] СТАТУС: {status}")
    return {"status": status, "per_feature": rows}