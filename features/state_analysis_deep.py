# research/features/state_analysis_deep.py
# ============================================================
# Углублённый анализ статистических состояний (только акции, только IS).
#
# 1. k=3,4,5 кластеры — подрежимы
# 2. Непересекающиеся окна (STEP = WINDOW)
# 3. Hurst как leading indicator (лаги -60..+60)
# 4. PC1/PC2/PC3 во времени
# 5. Спектральный анализ S_t
# 6. Сравнение overlap vs non-overlap
#
# Графики: тёмный фон, серые/чёрные/белые линии.
#
# Запуск: python state_analysis_deep.py
# ============================================================
import json
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

OUT_SQL = Path(__file__).parent / "sql" / "state_analysis_deep.sqlite"
OUT_SQL.parent.mkdir(exist_ok=True)
OUT_MD = Path(__file__).parent / "state_analysis_deep_report.md"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

WINDOW = 252
STEP_OVERLAP = 5       # перекрывающиеся окна (как в базовом)
MIN_STOCKS = 30

# ============================================================
# ГРАФИКИ — ТЁМНЫЙ ФОН, МОНОХРОМ
# ============================================================
def setup_dark_style():
    plt.rcParams.update({
        "figure.facecolor": "#1a1a1a",
        "axes.facecolor": "#1a1a1a",
        "savefig.facecolor": "#1a1a1a",
        "axes.edgecolor": "#888888",
        "axes.labelcolor": "#dddddd",
        "axes.titlecolor": "#eeeeee",
        "xtick.color": "#aaaaaa",
        "ytick.color": "#aaaaaa",
        "text.color": "#dddddd",
        "grid.color": "#333333",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "legend.facecolor": "#2a2a2a",
        "legend.edgecolor": "#888888",
        "legend.labelcolor": "#dddddd",
        "font.size": 10,
    })

# монохромная палитра
MONO = ["#ffffff", "#cccccc", "#999999", "#666666", "#444444",
        "#bbbbbb", "#777777", "#aaaaaa"]

def plot_pc_timeline(pca_df: pd.DataFrame, path: Path):
    setup_dark_style()
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, pc, color in zip(axes, ["PC1","PC2","PC3"], MONO[:3]):
        ax.plot(pca_df["window_end"], pca_df[pc], color=color, linewidth=0.8)
        ax.axhline(0, color="#555555", linewidth=0.5, linestyle=":")
        ax.set_ylabel(pc, color="#dddddd")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Дата окончания окна")
    axes[0].set_title("История статистических состояний (PC1/PC2/PC3)", color="#eeeeee")
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor="#1a1a1a")
    plt.close()
    print(f"  график: {path}")

def plot_clusters_timeline(df: pd.DataFrame, path: Path, k: int):
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(14, 4))
    for cl in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cl]
        ax.scatter(sub["window_end"], [cl]*len(sub),
                   color=MONO[int(cl) % len(MONO)], s=8, alpha=0.7,
                   label=f"Cluster {cl}")
    ax.set_yticks(sorted(df["cluster"].unique()))
    ax.set_xlabel("Дата окончания окна")
    ax.set_ylabel("Кластер")
    ax.set_title(f"Кластеры во времени (k={k})", color="#eeeeee")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor="#1a1a1a")
    plt.close()
    print(f"  график: {path}")

def plot_spectrum(freqs, power, path: Path, title: str):
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(freqs, power, color="#cccccc", linewidth=0.8)
    ax.set_xlabel("Частота (1/день)")
    ax.set_ylabel("Мощность")
    ax.set_title(title, color="#eeeeee")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor="#1a1a1a")
    plt.close()
    print(f"  график: {path}")

def plot_cross_corr(lags, corrs, path: Path, pair: str):
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(lags, corrs, color="#999999", width=0.8)
    ax.axhline(0, color="#555555", linewidth=0.5)
    ax.axvline(0, color="#555555", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Лаг (окон)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title(f"std ↔ {pair}", color="#eeeeee")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor="#1a1a1a")
    plt.close()
    print(f"  график: {path}")

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
    feats = {k: [] for k in [
        "mean","std","skew","kurt","acf_ret_1","acf_abs_1","acf_abs_5",
        "vr_5","vr_20","hurst","downside_vol","upside_vol","vol_asym","tail_ratio"
    ]}
    for col in window_data.columns:
        x = window_data[col].dropna().values
        if len(x) < WINDOW * 0.7: continue
        mean = x.mean(); std = x.std(ddof=1)
        if std == 0: continue
        skew = stats.skew(x); kurt = stats.kurtosis(x)
        if SM_OK:
            a = acf(x, nlags=10, fft=False)
            acf_ret_1 = a[1] if len(a)>1 else np.nan
        else: acf_ret_1 = np.nan
        absx = np.abs(x)
        if SM_OK:
            a_abs = acf(absx, nlags=10, fft=False)
            acf_abs_1 = a_abs[1] if len(a_abs)>1 else np.nan
            acf_abs_5 = a_abs[5] if len(a_abs)>5 else np.nan
        else: acf_abs_1 = acf_abs_5 = np.nan
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
    out = {}
    for k, vals in feats.items():
        vals = [v for v in vals if not np.isnan(v)]
        out[k] = np.median(vals) if vals else np.nan
    return out

def build_state_matrix(stock_ret: pd.DataFrame, step: int, rep: Report) -> pd.DataFrame:
    dates = stock_ret.index
    rows = []
    n = len(dates)
    starts = list(range(0, n - WINDOW + 1, step))
    rep.emit(f"  окно: {WINDOW}, шаг: {step}, окон: {len(starts)}")
    for i, s in enumerate(starts):
        if i % 100 == 0 and i > 0:
            rep.emit(f"    окно {i+1}/{len(starts)} ...")
        e = s + WINDOW
        win = stock_ret.iloc[s:e]
        valid = win.columns[win.notna().sum() >= WINDOW * 0.7]
        if len(valid) < MIN_STOCKS: continue
        st = compute_window_stats(win[valid])
        st["window_end"] = dates[e-1]
        st["window_start"] = dates[s]
        st["n_stocks"] = len(valid)
        rows.append(st)
    df = pd.DataFrame(rows).set_index("window_end")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

# ============================================================
# 1. КЛАСТЕРЫ k=3,4,5
# ============================================================
def block_clusters_deep(states: pd.DataFrame, rep: Report):
    rep.emit(section("1. КЛАСТЕРЫ k=3,4,5 — ПОДРЕЖИМЫ"))

    feat_cols = [c for c in states.columns if c not in ("window_start","n_stocks")]
    X = states[feat_cols].fillna(states[feat_cols].median())
    Xs = StandardScaler().fit_transform(X)

    results = {}
    for k in [2,3,4,5]:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        results[k] = {"labels": labels, "sil": sil, "km": km}
        rep.emit(f"\n[k={k}] silhouette = {sil:.3f}")

        # средние признаки
        df_cl = states[feat_cols].copy()
        df_cl["cluster"] = labels
        cm = df_cl.groupby("cluster")[feat_cols].median()
        for cl in cm.index:
            n_cl = int((labels==cl).sum())
            rep.emit(f"  Cluster {cl}: n={n_cl}  "
                     f"std={cm.loc[cl,'std']:.4f}  skew={cm.loc[cl,'skew']:+.2f}  "
                     f"kurt={cm.loc[cl,'kurt']:.1f}  acf_abs={cm.loc[cl,'acf_abs_1']:.3f}  "
                     f"vr5={cm.loc[cl,'vr_5']:.3f}  hurst={cm.loc[cl,'hurst']:.3f}")

    # выбираем k=3 для дальнейшего (компромисс детализации/устойчивости)
    # и k=4 для графика
    for k in [3, 4]:
        if k in results:
            labels = results[k]["labels"]
            df_out = pd.DataFrame({
                "window_end": states.index,
                f"cluster_k{k}": labels,
            })
            with sqlite3.connect(OUT_SQL) as c:
                df_out.to_sql(f"clusters_k{k}", c, if_exists="replace", index=False)

    # график k=4
    if 4 in results:
        labels = results[4]["labels"]
        df_plot = pd.DataFrame({"window_end": states.index, "cluster": labels})
        plot_clusters_timeline(df_plot, PLOTS_DIR / "clusters_k4.png", k=4)

    # сохраняем силуэт
    sil_df = pd.DataFrame([{"k":k, "silhouette":v["sil"]} for k,v in results.items()])
    with sqlite3.connect(OUT_SQL) as c:
        sil_df.to_sql("silhouette_by_k", c, if_exists="replace", index=False)


# ============================================================
# 2. НЕПЕРЕСЕКАЮЩИЕСЯ ОКНА
# ============================================================
def block_nonoverlap(stock_ret: pd.DataFrame, rep: Report):
    rep.emit(section("2. НЕПЕРЕСЕКАЮЩИЕСЯ ОКНА (STEP = WINDOW)"))

    states_no = build_state_matrix(stock_ret, step=WINDOW, rep=rep)
    rep.emit(f"  получено окон: {len(states_no)}")
    if len(states_no) < 5:
        rep.emit("  слишком мало окон, пропуск")
        return states_no

    feat_cols = [c for c in states_no.columns if c not in ("window_start","n_stocks")]
    X = states_no[feat_cols].fillna(states_no[feat_cols].median())
    Xs = StandardScaler().fit_transform(X)

    # кластеры k=2
    km = KMeans(n_clusters=2, n_init=20, random_state=42)
    labels = km.fit_predict(Xs)

    # матрица переходов
    tm = pd.DataFrame(0, index=[0,1], columns=[0,1])
    for i in range(1, len(labels)):
        tm.loc[labels[i-1], labels[i]] += 1
    tm_prob = tm.div(tm.sum(axis=1), axis=0).fillna(0)

    rep.emit("\n  Матрица переходов (непересекающиеся окна, k=2):")
    rep.emit(tm_prob.round(3).to_string())

    # длины серий
    runs = []
    cur = labels[0]; length = 1
    for l in labels[1:]:
        if l == cur: length += 1
        else: runs.append((cur, length)); cur = l; length = 1
    runs.append((cur, length))
    runs_df = pd.DataFrame(runs, columns=["cluster","length"])
    rep.emit(f"\n  Средняя длина серии:")
    for cl in [0,1]:
        sub = runs_df[runs_df["cluster"]==cl]
        if len(sub):
            rep.emit(f"    Cluster {cl}: mean={sub['length'].mean():.1f}  "
                     f"max={sub['length'].max()}")

    with sqlite3.connect(OUT_SQL) as c:
        tm_prob.reset_index().to_sql("transitions_nonoverlap", c, if_exists="replace", index=False)
        runs_df.to_sql("runs_nonoverlap", c, if_exists="replace", index=False)
        out = states_no.reset_index()
        out.columns = [str(x) for x in out.columns]
        out.to_sql("state_matrix_nonoverlap", c, if_exists="replace", index=False)

    return states_no


# ============================================================
# 3. HURST КАК LEADING INDICATOR (лаги -60..+60)
# ============================================================
def block_hurst_leading(states: pd.DataFrame, rep: Report):
    rep.emit(section("3. HURST КАК LEADING INDICATOR (лаги -60..+60)"))

    if "hurst" not in states or "std" not in states:
        rep.emit("  нет hurst или std")
        return

    base = states["std"].values
    x = states["hurst"].values
    rows = []
    for lag in range(-60, 61):
        if lag >= 0:
            a = base[:len(base)-lag] if lag > 0 else base
            b = x[lag:] if lag > 0 else x
        else:
            a = base[-lag:]
            b = x[:len(x)+lag]
        if len(a) > 20:
            r = np.corrcoef(a, b)[0,1]
            rows.append({"lag": lag, "corr": r})
    df = pd.DataFrame(rows)
    best = df.iloc[df["corr"].abs().argmax()]
    rep.emit(f"\n  std ↔ hurst: best lag={best['lag']:+.0f}  corr={best['corr']:+.3f}")
    rep.emit(f"  {'hurst ОПЕРЕЖАЕТ std' if best['lag']>0 else 'hurst ОТСТАЁТ от std' if best['lag']<0 else 'синхронно'}")

    # топ-5 лагов
    top = df.reindex(df["corr"].abs().sort_values(ascending=False).index).head(5)
    rep.emit("\n  Топ-5 лагов по |corr|:")
    for _, r in top.iterrows():
        rep.emit(f"    lag={r['lag']:+3.0f}  corr={r['corr']:+.3f}")

    plot_cross_corr(df["lag"], df["corr"],
                    PLOTS_DIR / "cross_corr_std_hurst.png", "hurst")

    with sqlite3.connect(OUT_SQL) as c:
        df.to_sql("cross_corr_hurst_std", c, if_exists="replace", index=False)


# ============================================================
# 4. PC1/PC2/PC3 ВО ВРЕМЕНИ
# ============================================================
def block_pca_timeline(states: pd.DataFrame, rep: Report):
    rep.emit(section("4. PC1/PC2/PC3 ВО ВРЕМЕНИ"))

    feat_cols = [c for c in states.columns if c not in ("window_start","n_stocks")]
    X = states[feat_cols].fillna(states[feat_cols].median())
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3)
    Z = pca.fit_transform(Xs)

    pca_df = pd.DataFrame({
        "window_end": states.index,
        "PC1": Z[:,0], "PC2": Z[:,1], "PC3": Z[:,2],
    })

    plot_pc_timeline(pca_df, PLOTS_DIR / "pc_timeline.png")

    # сохраняем
    with sqlite3.connect(OUT_SQL) as c:
        pca_df.to_sql("pca_timeline", c, if_exists="replace", index=False)

    # статистика по PC
    rep.emit("\n  Статистика PC по окнам:")
    for pc in ["PC1","PC2","PC3"]:
        s = pca_df[pc]
        rep.emit(f"  {pc}: mean={s.mean():+.3f}  std={s.std():.3f}  "
                 f"min={s.min():+.3f}  max={s.max():+.3f}")

    # корреляция PC с исходными признаками (интерпретация)
    rep.emit("\n  Корреляции PC с признаками (|corr| > 0.5):")
    for pc in ["PC1","PC2","PC3"]:
        corrs = {}
        for f in feat_cols:
            c = np.corrcoef(pca_df[pc], states[f].fillna(states[f].median()))[0,1]
            if abs(c) > 0.5:
                corrs[f] = c
        rep.emit(f"  [{pc}]")
        for f, c in sorted(corrs.items(), key=lambda x: -abs(x[1])):
            rep.emit(f"    {f:<15} {c:+.3f}")


# ============================================================
# 5. СПЕКТРАЛЬНЫЙ АНАЛИЗ S_t
# ============================================================
def block_spectrum_states(states: pd.DataFrame, rep: Report):
    rep.emit(section("5. СПЕКТРАЛЬНЫЙ АНАЛИЗ S_t"))

    # анализируем PC1 (главная ось) и отдельно std
    feat_cols = [c for c in states.columns if c not in ("window_start","n_stocks")]
    X = states[feat_cols].fillna(states[feat_cols].median())
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(Xs)[:,0]

    # спектр PC1
    f_pc1, P_pc1 = signal.periodogram(pc1, fs=1.0)
    # топ-5 пиков (исключая f=0)
    idx = np.argsort(P_pc1[1:])[::-1][:5] + 1
    rep.emit("\n  Топ-5 частот PC1 (период = 1/freq окон; окно=252д):")
    for i in idx:
        period_windows = 1.0/f_pc1[i]
        period_days = period_windows * WINDOW
        rep.emit(f"    freq={f_pc1[i]:.4f}  период={period_windows:.1f} окон "
                 f"(~{period_days:.0f} дней ~ {period_days/252:.1f} лет)  power={P_pc1[i]:.4f}")

    plot_spectrum(f_pc1, P_pc1, PLOTS_DIR / "spectrum_pc1.png", "Спектр PC1 (главная ось)")

    # спектр std
    f_std, P_std = signal.periodogram(states["std"].fillna(states["std"].median()).values, fs=1.0)
    idx = np.argsort(P_std[1:])[::-1][:5] + 1
    rep.emit("\n  Топ-5 частот std:")
    for i in idx:
        period_windows = 1.0/f_std[i]
        period_days = period_windows * WINDOW
        rep.emit(f"    freq={f_std[i]:.4f}  период={period_windows:.1f} окон "
                 f"(~{period_days:.0f} дней ~ {period_days/252:.1f} лет)  power={P_std[i]:.4f}")

    plot_spectrum(f_std, P_std, PLOTS_DIR / "spectrum_std.png", "Спектр std")

    with sqlite3.connect(OUT_SQL) as c:
        pd.DataFrame({"freq": f_pc1, "power": P_pc1}).to_sql("spectrum_pc1", c, if_exists="replace", index=False)
        pd.DataFrame({"freq": f_std, "power": P_std}).to_sql("spectrum_std", c, if_exists="replace", index=False)


# ============================================================
# 6. СРАВНЕНИЕ OVERLAP vs NON-OVERLAP
# ============================================================
def block_compare_overlap(states_ov: pd.DataFrame, states_no: pd.DataFrame, rep: Report):
    rep.emit(section("6. СРАВНЕНИЕ OVERLAP vs NON-OVERLAP"))

    if states_no is None or len(states_no) < 5:
        rep.emit("  нет непересекающихся окон, пропуск")
        return

    # базовая статистика S_t в двух режимах
    feat_cols = [c for c in states_ov.columns if c not in ("window_start","n_stocks")]
    rep.emit(f"\n  {'признак':<15} {'overlap median':>18} {'non-overlap median':>20}")
    rep.emit("  " + "-"*55)
    for f in feat_cols:
        a = states_ov[f].median()
        b = states_no[f].median() if f in states_no else np.nan
        rep.emit(f"  {f:<15} {a:>18.4f} {b:>20.4f}")

    # кластерная структура
    rep.emit("\n  Сравнение кластеров (k=2):")
    for name, st in [("overlap", states_ov), ("non-overlap", states_no)]:
        if st is None or len(st) < 5: continue
        X = st[feat_cols].fillna(st[feat_cols].median())
        Xs = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=2, n_init=20, random_state=42)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        # длины серий
        runs = []
        cur = labels[0]; length = 1
        for l in labels[1:]:
            if l == cur: length += 1
            else: runs.append((cur, length)); cur = l; length = 1
        runs.append((cur, length))
        runs_df = pd.DataFrame(runs, columns=["cluster","length"])
        rep.emit(f"    {name}: silhouette={sil:.3f}  "
                 f"mean run length={runs_df['length'].mean():.1f}  "
                 f"n_windows={len(st)}")


# ============================================================
# MAIN
# ============================================================
def main():
    rep = Report()
    rep.emit("# Углублённый анализ состояний (IS, только акции)\n")
    rep.emit(f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n")

    split = load_split()
    rep.emit(f"split_date = {split} (VAL не трогаем)\n")

    stock = load_stock_returns(split)
    rep.emit(f"stocks: {stock.shape[0]} дат × {stock.shape[1]} тикеров")
    rep.emit(f"IS: {stock.index.min().date()} → {stock.index.max().date()}\n")

    rep.emit(section("ПОСТРОЕНИЕ S_t (overlap)"))
    states_ov = build_state_matrix(stock, step=STEP_OVERLAP, rep=rep)
    rep.emit(f"  получено окон: {len(states_ov)}")

    # сохраняем
    with sqlite3.connect(OUT_SQL) as c:
        out = states_ov.reset_index()
        out.columns = [str(x) for x in out.columns]
        out.to_sql("state_matrix_overlap", c, if_exists="replace", index=False)

    block_clusters_deep(states_ov, rep)
    states_no = block_nonoverlap(stock, rep)
    block_hurst_leading(states_ov, rep)
    block_pca_timeline(states_ov, rep)
    block_spectrum_states(states_ov, rep)
    block_compare_overlap(states_ov, states_no, rep)

    rep.emit(section("ИТОГ"))
    rep.emit(f"SQL: {OUT_SQL}")
    rep.emit(f"Report: {OUT_MD}")
    rep.emit(f"Plots: {PLOTS_DIR}")
    with sqlite3.connect(OUT_SQL) as c:
        for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
            n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            rep.emit(f"  {t:<30} {n:>5} строк")
    rep.emit(f"\nГрафики (тёмный фон, монохром):")
    for p in sorted(PLOTS_DIR.glob("*.png")):
        rep.emit(f"  {p.name}")

    rep.save(OUT_MD)


if __name__ == "__main__":
    main()