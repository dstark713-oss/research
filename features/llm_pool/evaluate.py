# features/llm_pool/evaluate.py
# ============================================================
# Оценка батча:
#   1. IC (ежедневный cross-sectional Spearman с RS) + IR (mean/std IC)
#   2. Фильтр: |IC| > 0.015, IR > 0.1, stability по 3 подпериодам
#   3. Ортогональность: жадный отбор с |corr| < 0.7
# ============================================================
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .common import HORIZON, load_ohlcv, load_imoex, load_split_date, EPS

OUT_SQL = Path(__file__).parent / "sql" / "llm_pool.sqlite"
OUT_SQL.parent.mkdir(exist_ok=True)
OUT_MD = Path(__file__).parent / "llm_pool_report.md"


def build_target(raw: dict, imoex: pd.Series) -> pd.DataFrame:
    C = raw["close"]
    fwd = C.shift(-HORIZON) / C - 1
    imoex_fwd = imoex.shift(-HORIZON) / imoex - 1
    imoex_fwd = imoex_fwd.reindex(C.index)
    return fwd.sub(imoex_fwd, axis=0)


def daily_ic(feature: pd.DataFrame, target: pd.DataFrame, min_stocks=20):
    """Возвращает Series IC по датам."""
    common_dates = feature.index.intersection(target.index)
    common_tickers = feature.columns.intersection(target.columns)
    if len(common_tickers) < min_stocks:
        return pd.Series(dtype=float)
    f = feature.loc[common_dates, common_tickers]
    t = target.loc[common_dates, common_tickers]

    ics = {}
    for d in common_dates:
        fv = f.loc[d]; tv = t.loc[d]
        m = fv.notna() & tv.notna()
        if m.sum() < min_stocks: continue
        try:
            ic = stats.spearmanr(fv[m], tv[m]).correlation
            if np.isfinite(ic):
                ics[d] = ic
        except Exception:
            pass
    return pd.Series(ics).sort_index()


def evaluate_batch(features: dict, target: pd.DataFrame,
                   min_dates=500, min_ic_abs=0.015, min_ir=0.10) -> pd.DataFrame:
    rows = []
    for name, f in features.items():
        try:
            ics = daily_ic(f, target)
        except Exception as e:
            rows.append({"feature": name, "error": str(e)[:80]})
            continue
        if len(ics) < min_dates:
            rows.append({"feature": name, "error": f"too few IC dates ({len(ics)})"})
            continue

        mean_ic = ics.mean()
        std_ic = ics.std()
        ir = mean_ic / std_ic if std_ic > 0 else np.nan
        hit = (ics > 0).mean()

        # stability: IC по 3 подпериодам, знак должен совпадать
        n = len(ics)
        thirds = [ics.iloc[:n//3], ics.iloc[n//3:2*n//3], ics.iloc[2*n//3:]]
        signs = [np.sign(s.mean()) for s in thirds if len(s) > 0]
        stable = len(set(signs)) == 1

        rows.append({
            "feature": name,
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "ir": ir,
            "hit_rate": hit,
            "n_dates": n,
            "stable_3sub": stable,
            "pass": (abs(mean_ic) > min_ic_abs) and (abs(ir) > min_ir) and stable,
        })
    return pd.DataFrame(rows)


def orthogonalize(features_pass: dict, target: pd.DataFrame,
                  corr_threshold=0.7, max_n=20) -> list:
    """
    Жадный отбор по |IC| с фильтром корреляции.
    Возвращает список имён выбранных фич.
    """
    if not features_pass:
        return []
    # IC уже известен — пересчитаем или передадим. Проще передать через замыкание.
    # Здесь пересчёт не нужен: предполагаем, что вызывающий передал уже отсортированный список.
    # Но для простоты — сортируем по |IC|.
    ic_map = {}
    for name, f in features_pass.items():
        ics = daily_ic(f, target)
        ic_map[name] = abs(ics.mean()) if len(ics) else 0

    sorted_names = sorted(ic_map, key=lambda k: -ic_map[k])
    selected = []
    for name in sorted_names:
        if len(selected) >= max_n: break
        ok = True
        for sel in selected:
            a = features_pass[name]; b = features_pass[sel]
            common_dates = a.index.intersection(b.index)
            common_t = a.columns.intersection(b.columns)
            av = a.loc[common_dates, common_t].values.flatten()
            bv = b.loc[common_dates, common_t].values.flatten()
            m = np.isfinite(av) & np.isfinite(bv)
            if m.sum() < 500: continue
            c = np.corrcoef(av[m], bv[m])[0, 1]
            if np.isfinite(c) and abs(c) > corr_threshold:
                ok = False; break
        if ok:
            selected.append(name)
    return selected


def run_evaluate(batch_name: str, features: dict, rep_lines: list):
    split_date = load_split_date()
    raw = load_ohlcv(split_date)
    imoex = load_imoex(split_date)
    target = build_target(raw, imoex)

    rep_lines.append(f"\n## Батч {batch_name} — {len(features)} фич")
    df = evaluate_batch(features, target)
    passed = df[df["pass"] == True].copy() if "pass" in df else pd.DataFrame()
    rep_lines.append(f"\nПосле IC/IR фильтра: **{len(passed)}/{len(df)}**")

    if len(passed) > 0:
        rep_lines.append("\nТоп-15 по |IC|:")
        rep_lines.append(passed.reindex(passed["mean_ic"].abs().sort_values(ascending=False).index)
                         .head(15)[["feature","mean_ic","ir","hit_rate","stable_3sub"]].to_string(index=False))

    # ортогонализация
    passed_dict = {name: features[name] for name in passed["feature"].tolist()}
    selected = orthogonalize(passed_dict, target, corr_threshold=0.7, max_n=20)
    rep_lines.append(f"\nПосле ортогонализации (corr<0.7): **{len(selected)}** фич")
    for s in selected:
        row = passed[passed["feature"] == s].iloc[0]
        rep_lines.append(f"  - {s}: IC={row['mean_ic']:+.4f}, IR={row['ir']:+.3f}, hit={row['hit_rate']:.2%}")

    # сохраняем
    df["batch"] = batch_name
    with sqlite3.connect(OUT_SQL) as conn:
        df.to_sql(f"eval_{batch_name}", conn, if_exists="replace", index=False)

    return passed, selected