# research/features/state_analysis.py
# ============================================================
# Анализ статистических состояний рынка (только акции, только IS).
#
# Идея: каждое rolling-окно — «наблюдение», описанное вектором S_t.
# Дальше: корреляции, PCA, кластеры, переходы, leading/lagging.
#
# Глобальные инструменты (IMOEX, RGBI, GOLD, BRENT, CNYRUB_TOM)
# НЕ ИСПОЛЬЗУЮТСЯ — мы их не торгуем.
#
# Запуск: python state_analysis.py
# ============================================================
import json
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.stattools import acf
    SM_OK = True
except ImportError:
    SM_OK = False

# ============================================================
# КОНСТАНТЫ
# ============================================================
ROOT = Path(__file__).parent.parent
STOCKS_DB = ROOT / "dataset" / "sql" / "stocks.sqlite"
SPLIT_JSON = ROOT / "backtests" / "split_info.json"

OUT_SQL = Path(__file__).parent / "sql" / "state_analysis.sqlite"
OUT_SQL.parent.mkdir(exist_ok=True)
OUT_MD = Path(__file__).parent / "state_analysis_report.md"

WINDOW = 252      # окно статистики (1 год)
STEP = 5          # шаг между окнами
MIN_STOCKS = 30   # минимум акций в окне, чтобы считать «рыночное» состояние

# ============================================================
# HELPERS
# ============================================================
def section(t): return f"\n{'='*70}\n{t}\n{'='*70}"

class Report:
    def __init__(self): self.lines=[]
    def emit(self, s=""): print(s); self.lines.append(str(s))
    def save(self, p): p.write_text("\n".join(self.lines), encoding="utf-8"); print(f"\nОтчёт: {p}")

def load_split():
    return json.loads(SPLIT_JSON.read_text())["split_date"]

def load_stock_returns(split_date):
    conn = sqlite3.connect(STOCKS_DB)
    tabs = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "returns" in tabs:
        df = pd.read_sql("SELECT ticker, date, ret FROM returns WHERE date < ?",
                         conn, params=(split_date,))
    else:
        p = pd.read_sql("SELECT ticker, date, close FROM prices WHERE date < ?",
                        conn, params=(split_date,)).sort_values(["ticker","date"])
        p["ret"] = p.groupby("ticker")["close"].pct_change()
        df = p[["ticker","date","ret"]].dropna()
    conn.close()
    w = df.pivot(index="date", columns="ticker", values="ret")
    w.index = pd.to_datetime(w.index)
    return w.sort_index()

def dfa_hurst(x, order=1, max_lag=50):
    x = np.asarray(x); n = len(x)
    if n < 50: return np.nan
    y = np.cumsum(x - x.mean())
    lags = np.unique(np.floor(np.logspace(np.log10(4), np.log10(min(max_lag, n//4)), 10)).astype(int))
    F = []
    for lag in lags:
        if lag < 2: continue
        nc = n//lag
        if nc < 2: continue
        rms = []
        for i in range(nc):
            seg = y[i*lag:(i+1)*lag]; t = np.arange(lag)
            coef = np.polyfit(t, seg, order)
            rms.append(np.sqrt(np.mean((seg - np.polyval(coef, t))**2)))
        if rms: F.append((lag, np.mean(rms)))
    if len(F) < 3: return np.nan
    la = np.array([f[0] for f in F]); Fa = np.array([f[1] for f in F])
    try: return np.polyfit(np.log(la), np.log(Fa), 1)[0]
    except: return np.nan

def compute_window_stats(window_data: pd.DataFrame) -> dict:
    """
    window_data: DataFrame (WINDOW × n_stocks), доходности.
    Возвращает вектор S_t — «среднее состояние рынка в окне».
    """
    # усредняем по акциям: для каждого признака считаем медиану по акциям
    feats = {k: [] for k in [
        "mean","std","skew","kurt","acf_ret_1","acf_abs_1","acf_abs_5",
        "vr_5","vr_20","hurst","downside_vol","upside_vol","vol_asym","tail_ratio"
    ]}
    for col in window_data.columns:
        x = window_data[col].dropna().values
        if len(x) < WINDOW * 0.7:  # слишком мало данных
            continue
        mean = x.mean(); std = x.std(ddof=1)
        if std == 0: continue
        skew = stats.skew(x); kurt = stats.kurtosis(x)
        # ACF
        if SM_OK:
            a = acf(x, nlags=10, fft=False)
            acf_ret_1 = a[1] if len(a)>1 else np.nan
        else:
            acf_ret_1 = np.nan
        absx = np.abs(x)
        if SM_OK:
            a_abs = acf(absx, nlags=10, fft=False)
            acf_abs_1 = a_abs[1] if len(a_abs)>1 else np.nan
            acf_abs_5 = a_abs[5] if len(a_abs)>5 else np.nan
        else:
            acf_abs_1 = acf_abs_5 = np.nan
        # VR
        def vr(x, q):
            if len(x) < q+1: return np.nan
            v1 = np.var(x, ddof=1)
            xq = pd.Series(x).rolling(q).sum().dropna().values
            vq = np.var(xq, ddof=1)/q
            return vq/v1 if v1>0 else np.nan
        vr5 = vr(x, 5); vr20 = vr(x, 20)
        h = dfa_hurst(x)
        pos = x[x>0]; neg = x[x<0]
        down_vol = neg.std(ddof=1) if len(neg)>1 else np.nan
        up_vol = pos.std(ddof=1) if len(pos)>1 else np.nan
        vol_asym = down_vol/up_vol if up_vol and up_vol>0 else np.nan
        q = np.quantile(x, [0.01, 0.50, 0.99])
        left = q[1]-q[0]; right = q[2]-q[1]
        tail_ratio = right/left if left>0 else np.nan

        for k, v in [("mean",mean),("std",std),("skew",skew),("kurt",kurt),
                     ("acf_ret_1",acf_ret_1),("acf_abs_1",acf_abs_1),("acf_abs_5",acf_abs_5),
                     ("vr_5",vr5),("vr_20",vr20),("hurst",h),
                     ("downside_vol",down_vol),("upside_vol",up_vol),
                     ("vol_asym",vol_asym),("tail_ratio",tail_ratio)]:
            feats[k].append(v)

    # медиана по акциям для каждого признака
    out = {}
    for k, vals in feats.items():
        vals = [v for v in vals if not np.isnan(v)]
        out[k] = np.median(vals) if vals else np.nan
    return out


# ============================================================
# ПОСТРОЕНИЕ S_t
# ============================================================
def build_state_matrix(stock_ret: pd.DataFrame, rep: Report) -> pd.DataFrame:
    """Возвращает DataFrame: index=window_end_date, columns=features."""
    dates = stock_ret.index
    rows = []
    n = len(dates)
    starts = list(range(0, n - WINDOW + 1, STEP))
    rep.emit(f"Окно: {WINDOW} дней, шаг: {STEP}, всего окон: {len(starts)}")

    for i, s in enumerate(starts):
        if i % 50 == 0:
            rep.emit(f"  окно {i+1}/{len(starts)} ...", )
        e = s + WINDOW
        win = stock_ret.iloc[s:e]
        # оставляем акции с достаточной историей в окне
        valid = win.columns[win.notna().sum() >= WINDOW * 0.7]
        if len(valid) < MIN_STOCKS:
            continue
        st = compute_window_stats(win[valid])
        st["window_end"] = dates[e-1]
        st["window_start"] = dates[s]
        st["n_stocks"] = len(valid)
        rows.append(st)

    df = pd.DataFrame(rows).set_index("window_end")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ============================================================
# 1. КОРРЕЛЯЦИИ МЕЖДУ ПРИЗНАКАМИ
# ============================================================
def block_correlations(states: pd.DataFrame, rep: Report):
    rep.emit(section("1. КОРРЕЛЯЦИИ МЕЖДУ СТАТИСТИЧЕСКИМИ ХАРАКТЕРИСТИКАМИ"))

    feat_cols = [c for c in states.columns if c not in ("window_start","n_stocks")]
    corr = states[feat_cols].corr(method="spearman")
    rep.emit("\nSpearman корреляции (top-10 |corr|):")
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            pairs.append((corr.columns[i], corr.columns[j],
                          corr.iloc[i,j]))
    pairs_df = pd.DataFrame(pairs, columns=["f1","f2","corr"])
    pairs_df["abs_corr"] = pairs_df["corr"].abs()
    pairs_df = pairs_df.sort_values("abs_corr", ascending=False)
    for _, r in pairs_df.head(10).iterrows():
        rep.emit(f"  {r['f1']:<15} ↔ {r['f2']:<15} : {r['corr']:+.3f}")

    rep.emit("\nСильные связи (|corr| > 0.6):")
    strong = pairs_df[pairs_df["abs_corr"] > 0.6]
    if len(strong) == 0:
        rep.emit("  нет пар с |corr| > 0.6 — признаки почти независимы")
    else:
        for _, r in strong.iterrows():
            rep.emit(f"  {r['f1']:<15} ↔ {r['f2']:<15} : {r['corr']:+.3f}")

    # сохраняем матрицу корреляций
    corr_long = pairs_df.copy()
    with sqlite3.connect(OUT_SQL) as c:
        corr_long.to_sql("feature_correlations", c, if_exists="replace", index=False)


# ============================================================
# 2. PCA
# ============================================================
def block_pca(states: pd.DataFrame, rep: Report):
    rep.emit(section("2. PCA СТАТИСТИЧЕСКОГО СОСТОЯНИЯ"))

    feat_cols = [c for c in states.columns
                 if c not in ("window_start","n_stocks")]
    X = states[feat_cols].copy()
    # заполняем NaN медианой
    X = X.fillna(X.median())
    # стандартизация
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=min(10, len(feat_cols)))
    Z = pca.fit_transform(Xs)

    rep.emit(f"\nОбъяснённая дисперсия (первые 10 компонент):")
    cum = 0
    for i, ev in enumerate(pca.explained_variance_ratio_):
        cum += ev
        rep.emit(f"  PC{i+1}: {ev:.4f}  (cumulative {cum:.4f})")

    # сколько нужно для 80%, 90%, 95%
    for thr in [0.80, 0.90, 0.95]:
        n_comp = int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= thr)) + 1
        rep.emit(f"  для {thr:.0%} дисперсии нужно {n_comp} компонент")

    rep.emit("\nНагрузки первых 3 компонент (|loading| > 0.3):")
    loadings = pd.DataFrame(
        pca.components_[:3].T,
        index=feat_cols,
        columns=["PC1","PC2","PC3"]
    )
    for pc in ["PC1","PC2","PC3"]:
        rep.emit(f"\n  [{pc}]")
        s = loadings[pc].sort_values(key=abs, ascending=False)
        for name, val in s.items():
            if abs(val) > 0.3:
                rep.emit(f"    {name:<15} {val:+.3f}")

    # сохраняем
    pca_df = pd.DataFrame({
        "window_end": states.index,
        "PC1": Z[:,0],
        "PC2": Z[:,1] if Z.shape[1]>1 else np.nan,
        "PC3": Z[:,2] if Z.shape[1]>2 else np.nan,
    })
    with sqlite3.connect(OUT_SQL) as c:
        pca_df.to_sql("pca_windows", c, if_exists="replace", index=False)

    loadings_long = loadings.reset_index().rename(columns={"index":"feature"})
    with sqlite3.connect(OUT_SQL) as c:
        loadings_long.to_sql("pca_loadings", c, if_exists="replace", index=False)


# ============================================================
# 3. КЛАСТЕРЫ
# ============================================================
def block_clusters(states: pd.DataFrame, rep: Report):
    rep.emit(section("3. КЛАСТЕРЫ СТАТИСТИЧЕСКИХ СОСТОЯНИЙ"))

    feat_cols = [c for c in states.columns
                 if c not in ("window_start","n_stocks")]
    X = states[feat_cols].fillna(states[feat_cols].median())
    Xs = StandardScaler().fit_transform(X)

    # KMeans с разным k, выбираем по silhouette
    from sklearn.metrics import silhouette_score
    best_k, best_score, best_labels = None, -1, None
    for k in range(2, 8):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)
        score = silhouette_score(Xs, labels)
        rep.emit(f"  k={k}: silhouette = {score:.3f}")
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels
    rep.emit(f"\nВыбран k={best_k} (silhouette={best_score:.3f})")

    # средние признаки по кластерам
    states_cl = states.copy()
    states_cl["cluster"] = best_labels
    rep.emit(f"\nСредние признаки по кластерам:")
    cluster_means = states_cl.groupby("cluster")[feat_cols].median()
    for cl in cluster_means.index:
        row = cluster_means.loc[cl]
        rep.emit(f"\n  [Cluster {cl}] n={sum(best_labels==cl)}")
        for f in ["std","skew","kurt","acf_abs_1","vr_5","hurst","vol_asym"]:
            if f in row:
                rep.emit(f"    {f:<15} {row[f]:+.4f}")

    # сохраняем
    out = states_cl[["cluster"]].copy()
    out["window_end"] = states_cl.index
    with sqlite3.connect(OUT_SQL) as c:
        out.to_sql("clusters", c, if_exists="replace", index=False)
        cluster_means.reset_index().to_sql("cluster_means", c, if_exists="replace", index=False)


# ============================================================
# 4. ПЕРЕХОДЫ
# ============================================================
def block_transitions(states_cl: pd.DataFrame, rep: Report):
    rep.emit(section("4. ПЕРЕХОДЫ МЕЖДУ СОСТОЯНИЯМИ"))

    labels = states_cl["cluster"].values
    n = len(labels)
    if n < 5:
        rep.emit("  слишком мало окон")
        return

    # матрица переходов
    clusters = sorted(set(labels))
    tm = pd.DataFrame(0, index=clusters, columns=clusters)
    for i in range(1, n):
        tm.loc[labels[i-1], labels[i]] += 1
    # нормируем по строкам
    tm_prob = tm.div(tm.sum(axis=1), axis=0).fillna(0)

    rep.emit("\nМатрица переходов (вероятности):")
    rep.emit(tm_prob.round(3).to_string())

    # диагностика: диагональ высокая → устойчивые состояния
    rep.emit("\nДиагональ (устойчивость):")
    for cl in clusters:
        rep.emit(f"  Cluster {cl}: P(остаться) = {tm_prob.loc[cl,cl]:.3f}")

    # длины серий
    runs = []
    cur = labels[0]; length = 1
    for l in labels[1:]:
        if l == cur: length += 1
        else:
            runs.append((cur, length)); cur = l; length = 1
    runs.append((cur, length))
    runs_df = pd.DataFrame(runs, columns=["cluster","length"])
    rep.emit(f"\nСредняя длина серии по кластерам (в окнах):")
    for cl in clusters:
        sub = runs_df[runs_df["cluster"]==cl]
        if len(sub):
            rep.emit(f"  Cluster {cl}: mean={sub['length'].mean():.1f}  "
                     f"max={sub['length'].max()}")

    # сохраняем
    with sqlite3.connect(OUT_SQL) as c:
        tm_prob.reset_index().to_sql("transition_matrix", c, if_exists="replace", index=False)
        runs_df.to_sql("cluster_runs", c, if_exists="replace", index=False)


# ============================================================
# 5. LEADING / LAGGING
# ============================================================
def block_leading_lagging(states: pd.DataFrame, rep: Report):
    rep.emit(section("5. ВЕДУЩИЕ И ЗАПАЗДЫВАЮЩИЕ ИЗМЕНЕНИЯ"))

    # Для каждой пары признаков: cross-correlation на лагах -10..+10
    # Ищем, какой признак «идёт первым».
    feat_cols = [c for c in states.columns
                 if c not in ("window_start","n_stocks")]
    # фокусные признаки
    focus = ["std","skew","kurt","acf_abs_1","hurst"]
    focus = [f for f in focus if f in feat_cols]

    rep.emit(f"\nФокусные признаки: {focus}")
    rep.emit("Cross-correlation на лагах −10..+10 (относительно 'std'):")

    # для каждого фокусного признака считаем corr(std[t], feat[t+lag])
    # лаг > 0 → feat опережает std; лаг < 0 → feat отстаёт
    base = states["std"].values
    for f in focus:
        if f == "std": continue
        x = states[f].values
        rows = []
        for lag in range(-10, 11):
            if lag >= 0:
                a = base[:len(base)-lag] if lag > 0 else base
                b = x[lag:] if lag > 0 else x
            else:
                a = base[-lag:]
                b = x[:len(x)+lag]
            if len(a) > 10:
                r = np.corrcoef(a, b)[0,1]
                rows.append({"lag": lag, "corr": r})
        rdf = pd.DataFrame(rows)
        # лучший лаг
        best = rdf.iloc[rdf["corr"].abs().argmax()]
        rep.emit(f"  std ↔ {f}: best lag={best['lag']:+.0f}, corr={best['corr']:+.3f}  "
                 f"({'опережает' if best['lag']>0 else 'отстаёт' if best['lag']<0 else 'синхронно'})")

    # Сохраняем все cross-correlations
    all_cc = []
    for f in focus:
        if f == "std": continue
        x = states[f].values
        for lag in range(-10, 11):
            if lag >= 0:
                a = base[:len(base)-lag] if lag > 0 else base
                b = x[lag:] if lag > 0 else x
            else:
                a = base[-lag:]
                b = x[:len(x)+lag]
            if len(a) > 10:
                r = np.corrcoef(a, b)[0,1]
                all_cc.append({"pair": f"std_{f}", "lag": lag, "corr": r})
    with sqlite3.connect(OUT_SQL) as c:
        pd.DataFrame(all_cc).to_sql("cross_correlations", c, if_exists="replace", index=False)


# ============================================================
# MAIN
# ============================================================
def main():
    rep = Report()
    rep.emit("# Анализ статистических состояний рынка (IS, только акции)\n")
    rep.emit(f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n")

    split = load_split()
    rep.emit(f"split_date = {split} (VAL не трогаем)")
    rep.emit(f"Глобальные инструменты: ИСКЛЮЧЕНЫ (не торгуем)\n")

    stock = load_stock_returns(split)
    rep.emit(f"stocks: {stock.shape[0]} дат × {stock.shape[1]} тикеров")
    rep.emit(f"IS: {stock.index.min().date()} → {stock.index.max().date()}\n")

    # --- S_t ---
    rep.emit(section("ПОСТРОЕНИЕ ВЕКТОРА S_t"))
    states = build_state_matrix(stock, rep)
    rep.emit(f"\nПолучено окон: {len(states)}")
    rep.emit(f"Признаков: {len([c for c in states.columns if c not in ('window_start','n_stocks')])}")

    # сохраняем S_t
    with sqlite3.connect(OUT_SQL) as c:
        st_out = states.reset_index()
        st_out.columns = [str(x) for x in st_out.columns]
        st_out.to_sql("state_matrix", c, if_exists="replace", index=False)

    # --- 5 блоков ---
    block_correlations(states, rep)
    block_pca(states, rep)
    block_clusters(states, rep)

    # для transitions нужны кластеры — читаем из SQL
    with sqlite3.connect(OUT_SQL) as c:
        cl = pd.read_sql("SELECT window_end, cluster FROM clusters", c)
        cl["window_end"] = pd.to_datetime(cl["window_end"])
        cl = cl.set_index("window_end")
    block_transitions(cl, rep)

    block_leading_lagging(states, rep)

    rep.emit(section("ИТОГ"))
    rep.emit(f"SQL: {OUT_SQL}")
    rep.emit(f"Report: {OUT_MD}")
    with sqlite3.connect(OUT_SQL) as c:
        for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
            n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            rep.emit(f"  {t:<25} {n:>5} строк")
    rep.save(OUT_MD)


if __name__ == "__main__":
    main()