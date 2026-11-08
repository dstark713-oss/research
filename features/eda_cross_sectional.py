# research/features/eda_cross_sectional.py
# ============================================================
# Кросс-секционный EDA: Relative Strength vs IMOEX (только IS).
#
# Таргет: RS_{i,t} = ret_i(t→t+20) - ret_IMOEX(t→t+20)
# Горизонт: 20 торговых дней.
#
# 6 направлений:
#   1. Геометрия RS
#   2. Структура кросс-секционного распределения
#   3. Устойчивость относительной силы
#   4. Cross-sectional dependence
#   5. Исследование исторической информации
#   6. Predictive EDA (без ML)
#
# Ограничения:
#   - только IS
#   - перекрытие будущих окон учитывается (блочный bootstrap)
#   - без ML, без оптимизации
#
# Графики: тёмный фон, монохром.
# Запуск: python eda_cross_sectional.py
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
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# КОНСТАНТЫ
# ============================================================
ROOT = Path(__file__).parent.parent
STOCKS_DB = ROOT / "dataset" / "sql" / "stocks.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"
SPLIT_JSON = ROOT / "backtests" / "split_info.json"

OUT_SQL = Path(__file__).parent / "sql" / "eda_cross_sectional.sqlite"
OUT_SQL.parent.mkdir(exist_ok=True)
OUT_MD = Path(__file__).parent / "eda_cross_sectional_report.md"
PLOTS_DIR = Path(__file__).parent / "plots_cs"
PLOTS_DIR.mkdir(exist_ok=True)

HORIZON = 20              # горизонт RS (дней)
MIN_BARS = 500            # минимум баров для акции
BOOTSTRAP_BLOCKS = 200    # число блочных bootstrap-итераций
BLOCK_SIZE = HORIZON      # размер блока = горизонт (учитывает перекрытие)

# ============================================================
# ГРАФИКИ — ТЁМНЫЙ ФОН, МОНОХРОМ
# ============================================================
MONO = ["#ffffff", "#cccccc", "#999999", "#666666", "#444444",
        "#bbbbbb", "#777777", "#aaaaaa"]

def setup_dark_style():
    plt.rcParams.update({
        "figure.facecolor": "#1a1a1a", "axes.facecolor": "#1a1a1a",
        "savefig.facecolor": "#1a1a1a", "axes.edgecolor": "#888888",
        "axes.labelcolor": "#dddddd", "axes.titlecolor": "#eeeeee",
        "xtick.color": "#aaaaaa", "ytick.color": "#aaaaaa",
        "text.color": "#dddddd", "grid.color": "#333333",
        "grid.linestyle": "--", "grid.linewidth": 0.5,
        "legend.facecolor": "#2a2a2a", "legend.edgecolor": "#888888",
        "legend.labelcolor": "#dddddd", "font.size": 10,
    })

def plot_hist(series, path, title, xlabel, bins=50):
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(series.dropna(), bins=bins, color="#999999", edgecolor="#444444", alpha=0.9)
    ax.set_xlabel(xlabel); ax.set_ylabel("Частота")
    ax.set_title(title, color="#eeeeee")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120, facecolor="#1a1a1a"); plt.close()
    print(f"  график: {path}")

def plot_timeline(x, y, path, title, ylabel):
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, y, color="#cccccc", linewidth=0.8)
    ax.set_xlabel("Дата"); ax.set_ylabel(ylabel)
    ax.set_title(title, color="#eeeeee")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120, facecolor="#1a1a1a"); plt.close()
    print(f"  график: {path}")

def plot_bar(labels, values, path, title, ylabel):
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [MONO[i % len(MONO)] for i in range(len(labels))]
    ax.bar(range(len(labels)), values, color=colors, edgecolor="#444444")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title, color="#eeeeee")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120, facecolor="#1a1a1a"); plt.close()
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

def load_prices(split_date):
    """Возвращает wide DataFrame: index=date, columns=tickers, values=close."""
    conn = sqlite3.connect(STOCKS_DB)
    df = pd.read_sql("SELECT ticker, date, close FROM prices WHERE date < ?",
                     conn, params=(split_date,))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    w = df.pivot(index="date", columns="ticker", values="close").sort_index()
    return w

def load_imoex(split_date):
    conn = sqlite3.connect(INDEXES_DB)
    df = pd.read_sql("SELECT date, close FROM indexes WHERE ticker='IMOEX' AND date < ?",
                     conn, params=(split_date,))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()

def compute_rs(prices: pd.DataFrame, imoex: pd.Series, horizon: int) -> pd.DataFrame:
    """
    RS_{i,t} = ret_i(t→t+h) - ret_IMOEX(t→t+h)
    Возвращает wide DataFrame: index=date, columns=tickers.
    NaN где нет будущего или нет данных.
    """
    fwd_ret = prices.shift(-horizon) / prices - 1
    imoex_fwd = imoex.shift(-horizon) / imoex - 1
    # выравниваем imoex по датам prices
    imoex_fwd = imoex_fwd.reindex(prices.index)
    rs = fwd_ret.sub(imoex_fwd, axis=0)
    return rs

def block_bootstrap_corr(x, y, block_size=BLOCK_SIZE, n_iter=BOOTSTRAP_BLOCKS):
    """
    Блочный bootstrap для корреляции: учитывает автокорреляцию/перекрытие.
    Возвращает (corr_obs, ci_low, ci_high, p_value).
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]; y = y[mask]
    n = len(x)
    if n < block_size * 2:
        return np.nan, np.nan, np.nan, np.nan
    corr_obs = stats.spearmanr(x, y).correlation
    # блочный bootstrap
    n_blocks = n // block_size
    boot = []
    for _ in range(n_iter):
        starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s+block_size) for s in starts])
        xb = x[idx]; yb = y[idx]
        try:
            boot.append(stats.spearmanr(xb, yb).correlation)
        except Exception:
            pass
    if not boot:
        return corr_obs, np.nan, np.nan, np.nan
    boot = np.array(boot)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    # p-value: доля bootstrap, где |corr| >= |corr_obs| (two-sided)
    p = 2 * min((boot >= corr_obs).mean(), (boot <= corr_obs).mean())
    p = min(p, 1.0)
    return corr_obs, ci_low, ci_high, p

# ============================================================
# 1. ГЕОМЕТРИЯ RS
# ============================================================
def block_geometry(rs: pd.DataFrame, rep: Report):
    rep.emit(section("1. ГЕОМЕТРИЯ RS"))

    # --- 1a. Распределение RS по всей панели (raw + winsorized) ---
    all_rs = rs.values.flatten()
    all_rs = all_rs[~np.isnan(all_rs)]

    # winsorize на 1%/99% для описательных статистик
    q01, q99 = np.quantile(all_rs, [0.01, 0.99])
    wins = np.clip(all_rs, q01, q99)

    from scipy.stats import median_abs_deviation
    mad = median_abs_deviation(all_rs)
    iqr = np.quantile(all_rs, 0.75) - np.quantile(all_rs, 0.25)

    rep.emit(f"\n[1a. Распределение RS по всей панели]")
    rep.emit(f"  N = {len(all_rs):,}")
    rep.emit(f"  --- RAW (с выбросами) ---")
    rep.emit(f"  mean   = {all_rs.mean():+.6f}")
    rep.emit(f"  median = {np.median(all_rs):+.6f}")
    rep.emit(f"  std    = {all_rs.std():.6f}")
    rep.emit(f"  skew   = {stats.skew(all_rs):+.3f}  (артефакт выбросов)")
    rep.emit(f"  kurt   = {stats.kurtosis(all_rs):.3f}  (артефакт выбросов)")
    rep.emit(f"  --- WINSORIZED (1%/99%) ---")
    rep.emit(f"  mean   = {wins.mean():+.6f}")
    rep.emit(f"  median = {np.median(wins):+.6f}")
    rep.emit(f"  std    = {wins.std():.6f}")
    rep.emit(f"  skew   = {stats.skew(wins):+.3f}")
    rep.emit(f"  kurt   = {stats.kurtosis(wins):.3f}")
    rep.emit(f"  --- ROBUST ---")
    rep.emit(f"  MAD    = {mad:.6f}")
    rep.emit(f"  IQR    = {iqr:.6f}")
    for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
        rep.emit(f"  q{q:.2f} = {np.quantile(all_rs, q):+.6f}")

    plot_hist(pd.Series(wins), PLOTS_DIR / "rs_pooled_dist.png",
              "Распределение RS (winsorized 1%/99%)", "RS (20д)")

    save_df = pd.DataFrame([{
        "scope": "pooled_raw", "n": len(all_rs),
        "mean": all_rs.mean(), "median": np.median(all_rs), "std": all_rs.std(),
        "skew": stats.skew(all_rs), "kurt": stats.kurtosis(all_rs),
        "mad": mad, "iqr": iqr,
        "q01": np.quantile(all_rs, 0.01), "q05": np.quantile(all_rs, 0.05),
        "q95": np.quantile(all_rs, 0.95), "q99": np.quantile(all_rs, 0.99),
    }, {
        "scope": "pooled_wins", "n": len(wins),
        "mean": wins.mean(), "median": np.median(wins), "std": wins.std(),
        "skew": stats.skew(wins), "kurt": stats.kurtosis(wins),
        "mad": median_abs_deviation(wins),
        "iqr": np.quantile(wins, 0.75) - np.quantile(wins, 0.25),
        "q01": np.quantile(wins, 0.01), "q05": np.quantile(wins, 0.05),
        "q95": np.quantile(wins, 0.95), "q99": np.quantile(wins, 0.99),
    }])

    # --- 1b. Кросс-секционные статистики по датам ---
    rep.emit(f"\n[1b. Кросс-секционные статистики по датам]")
    cs = pd.DataFrame({
        "cs_mean": rs.mean(axis=1),
        "cs_median": rs.median(axis=1),
        "cs_std": rs.std(axis=1),
        "cs_mad": rs.apply(lambda row: median_abs_deviation(row.dropna())
                           if row.notna().sum() > 5 else np.nan, axis=1),
        "cs_skew": rs.apply(lambda row: stats.skew(row.dropna())
                            if row.notna().sum() > 5 else np.nan, axis=1),
        "cs_kurt": rs.apply(lambda row: stats.kurtosis(row.dropna())
                            if row.notna().sum() > 5 else np.nan, axis=1),
        "cs_share_positive": (rs > 0).sum(axis=1) / rs.notna().sum(axis=1),
        "cs_n": rs.notna().sum(axis=1),
    })
    cs = cs[cs["cs_n"] > 10]
    rep.emit(f"  дат с >=10 акций: {len(cs)}")
    rep.emit(f"  cs_std:  mean={cs['cs_std'].mean():.4f}  "
             f"min={cs['cs_std'].min():.4f}  max={cs['cs_std'].max():.4f}")
    rep.emit(f"  cs_mad:  mean={cs['cs_mad'].mean():.4f}  "
             f"min={cs['cs_mad'].min():.4f}  max={cs['cs_mad'].max():.4f}")
    rep.emit(f"  cs_share_positive: mean={cs['cs_share_positive'].mean():.3f}  "
             f"range=[{cs['cs_share_positive'].min():.3f} .. {cs['cs_share_positive'].max():.3f}]")

    plot_timeline(cs.index, cs["cs_std"], PLOTS_DIR / "cs_std_timeline.png",
                  "Кросс-секционная дисперсия RS (std по акциям)", "cs_std")
    plot_timeline(cs.index, cs["cs_mad"], PLOTS_DIR / "cs_mad_timeline.png",
                  "Кросс-секционная MAD RS (robust)", "cs_mad")
    plot_timeline(cs.index, cs["cs_share_positive"], PLOTS_DIR / "cs_share_pos_timeline.png",
                  "Доля акций, превосходящих IMOEX", "share")

    cs_out = cs.reset_index()
    cs_out.columns = [str(c) for c in cs_out.columns]
    with sqlite3.connect(OUT_SQL) as c:
        cs_out.to_sql("cs_stats_by_date", c, if_exists="replace", index=False)

    save_df = pd.concat([save_df, pd.DataFrame([{
        "scope": "cs_by_date", "n": len(cs),
        "mean": cs["cs_mean"].mean(), "median": cs["cs_median"].mean(),
        "std": cs["cs_std"].mean(),
        "skew": cs["cs_skew"].mean(), "kurt": cs["cs_kurt"].mean(),
        "mad": cs["cs_mad"].mean(), "iqr": np.nan,
        "q01": np.nan, "q05": np.nan, "q95": np.nan, "q99": np.nan,
    }])], ignore_index=True)

    with sqlite3.connect(OUT_SQL) as c:
        save_df.to_sql("rs_geometry", c, if_exists="replace", index=False)

    # --- 1c. Периоды высокой/низкой дифференциации ---
    rep.emit(f"\n[1c. Периоды высокой/низкой дифференциации]")
    cs_sorted = cs["cs_std"].dropna().sort_values()
    low = cs_sorted.head(10)
    high = cs_sorted.tail(10)
    rep.emit("  Топ-10 низкой дифференциации:")
    for d, v in low.items():
        rep.emit(f"    {d.date()}  cs_std={v:.4f}")
    rep.emit("  Топ-10 высокой дифференциации:")
    for d, v in high.items():
        rep.emit(f"    {d.date()}  cs_std={v:.4f}")

# ============================================================
# 2. СТРУКТУРА КРОСС-СЕКЦИОННОГО РАСПРЕДЕЛЕНИЯ
# ============================================================
def block_structure(rs: pd.DataFrame, rep: Report):
    rep.emit(section("2. СТРУКТУРА КРОСС-СЕКЦИОННОГО РАСПРЕДЕЛЕНИЯ"))

    # --- 2a. Концентрация: топ-10% акций дают какую долю положительного RS ---
    rep.emit("\n[2a. Концентрация RS]")
    conc_rows = []
    for date, row in rs.iterrows():
        x = row.dropna()
        if len(x) < 10: continue
        pos = x[x > 0]
        if len(pos) < 5: continue
        total_pos = pos.sum()
        top10 = pos.nlargest(max(1, int(len(x)*0.1))).sum()
        conc = top10 / total_pos if total_pos > 0 else np.nan
        conc_rows.append({"date": date, "n": len(x), "top10_share": conc})
    conc_df = pd.DataFrame(conc_rows)
    if len(conc_df):
        rep.emit(f"  top10_share of positive RS: median={conc_df['top10_share'].median():.3f}  "
                 f"std={conc_df['top10_share'].std():.3f}")
        with sqlite3.connect(OUT_SQL) as c:
            conc_df.to_sql("concentration", c, if_exists="replace", index=False)

    # --- 2b. Кластеризация акций: пробуем, но с честной диагностикой ---
    # При PC1 < 20% кластеризация акций почти всегда вырождается.
    # Сначала проверяем: есть ли структура?
    rep.emit("\n[2b. Проверка: есть ли кластерная структура у акций?]")
    valid = [c for c in rs.columns if rs[c].dropna().shape[0] >= MIN_BARS]
    rs_sub = rs[valid]

    X = rs_sub.fillna(rs_sub.mean())
    X = X.dropna(how="all", axis=1)
    valid = X.columns.tolist()
    n_stocks = X.shape[1]

    if n_stocks < 10:
        rep.emit("  слишком мало акций, пропуск")
        return

    # PCA по акциям: транспонируем, чтобы каждая акция была точкой
    Xs = StandardScaler().fit_transform(X.T)   # (n_stocks, n_dates)
    pca = PCA(n_components=min(5, n_stocks, X.shape[0]))
    Z = pca.fit_transform(Xs)                  # (n_stocks, n_components)
    ev = pca.explained_variance_ratio_
    rep.emit(f"  PCA по акциям: PC1={ev[0]:.3f}  PC2={ev[1]:.3f}  PC3={ev[2]:.3f}")

    # Диагностика: если PC1 < 20%, кластеры будут случайными
    if ev[0] < 0.20:
        rep.emit(f"  PC1 = {ev[0]:.3f} < 0.20 — структура слабая, кластеры будут нестабильны.")
        rep.emit(f"  Пропускаем KMeans, вместо этого — топы по отдельным осям.")
        # Топ-10 акций по PC1 (самые «выраженные» по главной оси)
        pc1_loadings = pd.Series(Z[:, 0], index=valid)
        rep.emit("\n  Топ-10 акций по |PC1| (самые выраженные):")
        for t, v in pc1_loadings.abs().nlargest(10).items():
            rep.emit(f"    {t:<10} PC1_loading={pc1_loadings[t]:+.3f}")
        with sqlite3.connect(OUT_SQL) as c:
            pd.DataFrame({"ticker": valid, "PC1": Z[:, 0],
                          "PC2": Z[:, 1] if Z.shape[1] > 1 else np.nan,
                          "PC3": Z[:, 2] if Z.shape[1] > 2 else np.nan}).to_sql(
                "stock_pca_loadings", c, if_exists="replace", index=False)
        # кластеры не сохраняем — их нет
    else:
        # Если структура есть — KMeans
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        best_k, best_score, best_labels = 2, -1, None
        for k in [2, 3, 4]:
            km = KMeans(n_clusters=k, n_init=20, random_state=42)
            labels = km.fit_predict(Z[:, :3])
            if len(set(labels)) < 2: continue
            score = silhouette_score(Z[:, :3], labels)
            rep.emit(f"    k={k}: silhouette={score:.3f}")
            if score > best_score:
                best_score, best_k, best_labels = score, k, labels
        if best_labels is not None:
            for cl in sorted(set(best_labels)):
                members = [valid[i] for i in range(n_stocks) if best_labels[i] == cl]
                rep.emit(f"    Cluster {cl}: n={len(members)}  "
                         f"примеры: {', '.join(members[:8])}")
            with sqlite3.connect(OUT_SQL) as c:
                pd.DataFrame({"ticker": valid, "cluster": best_labels}).to_sql(
                    "rs_clusters", c, if_exists="replace", index=False)
        else:
            rep.emit("  KMeans не дал осмысленных кластеров")

    # --- 2c. Смена лидеров (rank autocorr на лаге HORIZON) ---
    rep.emit("\n[2c. Смена лидеров (rank autocorr на лаге HORIZON=20)]")
    rank_ac = []
    dates = rs.index
    for i in range(0, len(dates) - HORIZON, HORIZON):
        prev = rs.iloc[i].dropna()
        cur = rs.iloc[i + HORIZON].dropna()
        common = prev.index.intersection(cur.index)
        if len(common) < 10: continue
        c = stats.spearmanr(prev[common], cur[common]).correlation
        rank_ac.append({"date": dates[i], "rank_corr": c})
    rank_df = pd.DataFrame(rank_ac)
    if len(rank_df):
        rep.emit(f"  rank autocorr (lag=20d): median={rank_df['rank_corr'].median():.3f}  "
                 f"std={rank_df['rank_corr'].std():.3f}  n={len(rank_df)}")
        rep.emit(f"  (случайное ожидание: 0.000)")
        with sqlite3.connect(OUT_SQL) as c:
            rank_df.to_sql("rank_autocorr", c, if_exists="replace", index=False)

# ============================================================
# 3. УСТОЙЧИВОСТЬ ОТНОСИТЕЛЬНОЙ СИЛЫ
# ============================================================
def block_persistence(rs: pd.DataFrame, rep: Report):
    rep.emit(section("3. УСТОЙЧИВОСТЬ ОТНОСИТЕЛЬНОЙ СИЛЫ"))

    # --- 3a. Автокорреляция RS на лагах 1..60 ---
    rep.emit("\n[3a. Автокорреляция RS по лагам]")
    # для каждой акции: corr(RS_t, RS_{t+lag}), затем медиана по акциям
    valid = [c for c in rs.columns if rs[c].dropna().shape[0] >= MIN_BARS]
    lags = [1, 5, 10, 20, 40, 60]
    ac_rows = []
    for lag in lags:
        cors = []
        for t in valid:
            s = rs[t].dropna()
            if len(s) < lag + 20: continue
            c = s.autocorr(lag=lag)
            if not np.isnan(c): cors.append(c)
        if cors:
            ac_rows.append({"lag": lag, "median_ac": np.median(cors),
                            "mean_ac": np.mean(cors), "n_stocks": len(cors)})
    ac_df = pd.DataFrame(ac_rows)
    rep.emit(ac_df.round(4).to_string(index=False))
    with sqlite3.connect(OUT_SQL) as c:
        ac_df.to_sql("rs_autocorr_by_lag", c, if_exists="replace", index=False)

    # --- 3b. Persistence квантильных рангов ---
    rep.emit("\n[3b. Persistence квантильных рангов]")
    # для каждого дня: rank акций по RS (внутри дня)
    # затем смотрим, какая доля top-20% остаётся в top-20% через 20 дней
    top_frac = 0.20
    persist_rows = []
    dates = rs.index
    for i in range(0, len(dates) - HORIZON, HORIZON):
        row1 = rs.iloc[i].dropna()
        row2 = rs.iloc[i + HORIZON].dropna() if i + HORIZON < len(dates) else None
        if row2 is None: continue
        common = row1.index.intersection(row2.index)
        if len(common) < 20: continue
        n_top = max(1, int(len(common) * top_frac))
        top1 = set(row1[common].nlargest(n_top).index)
        top2 = set(row2[common].nlargest(n_top).index)
        bot1 = set(row1[common].nsmallest(n_top).index)
        bot2 = set(row2[common].nsmallest(n_top).index)
        persist_rows.append({
            "date": dates[i],
            "top_persist": len(top1 & top2) / n_top,
            "bot_persist": len(bot1 & bot2) / n_top,
            "n": len(common),
        })
    p_df = pd.DataFrame(persist_rows)
    if len(p_df):
        rep.emit(f"  top-20% persistence (через 20д): median={p_df['top_persist'].median():.3f}")
        rep.emit(f"  bot-20% persistence (через 20д): median={p_df['bot_persist'].median():.3f}")
        rep.emit(f"  (случайное ожидание: {top_frac:.3f})")
        with sqlite3.connect(OUT_SQL) as c:
            p_df.to_sql("quantile_persistence", c, if_exists="replace", index=False)

# ============================================================
# 4. CROSS-SECTIONAL DEPENDENCE
# ============================================================
def block_dependence(rs: pd.DataFrame, rep: Report):
    rep.emit(section("4. CROSS-SECTIONAL DEPENDENCE"))

    valid = [c for c in rs.columns if rs[c].dropna().shape[0] >= MIN_BARS]
    rs_sub = rs[valid].dropna(how="all", axis=0).dropna(how="all", axis=1)
    rs_sub = rs_sub.dropna(how="all", axis=1)

    rep.emit(f"\n[4a. Попарные корреляции RS]")
    rep.emit(f"  акций: {len(valid)}")
    corr = rs_sub.corr(method="spearman", min_periods=100)
    tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    vals = tri.values.flatten()
    vals = vals[~np.isnan(vals)]
    rep.emit(f"  пар: {len(vals)}")
    rep.emit(f"  mean corr    = {vals.mean():+.3f}")
    rep.emit(f"  median corr  = {np.median(vals):+.3f}")
    rep.emit(f"  std corr     = {vals.std():.3f}")
    rep.emit(f"  доля |corr|>0.3: {(np.abs(vals)>0.3).mean():.1%}")
    plot_hist(pd.Series(vals), PLOTS_DIR / "pairwise_corr_rs.png",
              "Распределение попарных корреляций RS", "corr")

    with sqlite3.connect(OUT_SQL) as c:
        corr.reset_index().to_sql("rs_corr_matrix", c, if_exists="replace", index=False)
        pd.DataFrame([{
            "n_pairs": len(vals), "mean": vals.mean(), "median": np.median(vals),
            "std": vals.std(), "share_gt_03": (np.abs(vals)>0.3).mean(),
        }]).to_sql("rs_corr_summary", c, if_exists="replace", index=False)

    # --- 4b. PCA ---
    rep.emit("\n[4b. PCA кросс-секционной структуры]")
    X = rs_sub.fillna(rs_sub.mean())
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(10, Xs.shape[1]))
    Z = pca.fit_transform(Xs)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)

    rep.emit(f"  PC1: {ev[0]:.3f}  cumulative {cum[0]:.3f}")
    for i in range(1, min(5, len(ev))):
        rep.emit(f"  PC{i+1}: {ev[i]:.3f}  cumulative {cum[i]:.3f}")

    # ИСПРАВЛЕНО: корректный подсчёт компонент
    rep.emit(f"\n  Сколько компонент нужно для порога:")
    for thr in [0.3, 0.5, 0.7, 0.9]:
        idx = np.where(cum >= thr)[0]
        if len(idx) == 0:
            rep.emit(f"    для {thr:.0%}: НЕ ДОСТИГАЕТСЯ (max {cum[-1]:.3f})")
        else:
            rep.emit(f"    для {thr:.0%}: {idx[0]+1} компонент")

    with sqlite3.connect(OUT_SQL) as c:
        pd.DataFrame({"pc": [f"PC{i+1}" for i in range(len(ev))],
                      "explained_var": ev,
                      "cumulative": cum}).to_sql(
            "rs_pca_variance", c, if_exists="replace", index=False)

# ============================================================
# 5. ИСТОРИЧЕСКАЯ ИНФОРМАЦИЯ (фичи)
# ============================================================
def build_historical_features(prices: pd.DataFrame, imoex: pd.Series) -> dict:
    """
    Строит базовые семейства исторических фич для каждой акции.
    Все фичи — на дату t, без будущего.
    Возвращает dict: name -> wide DataFrame (index=date, columns=tickers).
    """
    feats = {}
    # returns
    ret1 = prices.pct_change(1)
    ret5 = prices.pct_change(5)
    ret20 = prices.pct_change(20)
    ret60 = prices.pct_change(60)

    # momentum (relative to IMOEX)
    imoex_ret20 = imoex.pct_change(20).reindex(prices.index)
    imoex_ret60 = imoex.pct_change(60).reindex(prices.index)
    feats["mom_20_rel"] = ret20.sub(imoex_ret20, axis=0)
    feats["mom_60_rel"] = ret60.sub(imoex_ret60, axis=0)

    # position in own range (52w)
    roll_max = prices.rolling(252, min_periods=100).max()
    roll_min = prices.rolling(252, min_periods=100).min()
    feats["price_position"] = (prices - roll_min) / (roll_max - roll_min)

    # volatility
    feats["vol_20"] = ret1.rolling(20).std()
    feats["vol_60"] = ret1.rolling(60).std()
    # relative vol (vol vs cross-sectional median)
    feats["vol_20_rel"] = feats["vol_20"].div(feats["vol_20"].median(axis=1), axis=0)

    # skewness / kurtosis (60d)
    feats["skew_60"] = ret1.rolling(60).skew()
    feats["kurt_60"] = ret1.rolling(60).kurt()

    # beta к IMOEX (rolling 60d)
    imoex_ret1 = imoex.pct_change(1).reindex(prices.index)
    def rolling_beta(y, x, window=60):
        cov = y.rolling(window).cov(x)
        var = x.rolling(window).var()
        return cov.div(var, axis=0)
    feats["beta_60"] = rolling_beta(ret1, imoex_ret1, 60)

    # efficiency (net move / sum abs move) — 20d
    def efficiency(px, window=20):
        net = px.diff(window).abs()
        path = px.diff().abs().rolling(window).sum()
        return net / path
    feats["efficiency_20"] = efficiency(prices, 20)

    # cross-sectional rank по momentum
    feats["mom_20_rank"] = feats["mom_20_rel"].rank(axis=1, pct=True)

    return feats

# ============================================================
# 6. PREDICTIVE EDA
# ============================================================
def block_predictive(prices: pd.DataFrame, imoex: pd.Series, rs: pd.DataFrame, rep: Report):
    rep.emit(section("6. PREDICTIVE EDA (без ML)"))

    feats = build_historical_features(prices, imoex)
    rep.emit(f"\n  построено семейств фич: {len(feats)}")
    for name in feats:
        rep.emit(f"    {name}")

    # --- 6a. Spearman IC с блочным bootstrap ---
    rep.emit("\n[6a. Spearman IC (блочный bootstrap, p-value)]")
    ic_rows = []
    for fname, fdf in feats.items():
        # выравниваем по датам и акциям
        common_dates = fdf.index.intersection(rs.index)
        common_tickers = fdf.columns.intersection(rs.columns)
        if len(common_dates) < 100 or len(common_tickers) < 20:
            continue
        # для каждой даты: cross-sectional Spearman между feature и rs
        ics = []
        for d in common_dates:
            f = fdf.loc[d, common_tickers]
            r = rs.loc[d, common_tickers]
            mask = f.notna() & r.notna()
            if mask.sum() < 20: continue
            ic = stats.spearmanr(f[mask], r[mask]).correlation
            if not np.isnan(ic): ics.append(ic)
        if len(ics) < 50: continue
        ics = np.array(ics)
        # блочный bootstrap по временной оси (не по акциям)
        # (упрощённо: используем блоки по 20 дат)
        n = len(ics)
        boot = []
        for _ in range(BOOTSTRAP_BLOCKS):
            starts = np.random.randint(0, max(1, n - BLOCK_SIZE), size=n // BLOCK_SIZE)
            idx = np.concatenate([np.arange(s, min(s+BLOCK_SIZE, n)) for s in starts])
            boot.append(ics[idx].mean())
        boot = np.array(boot)
        mean_ic = ics.mean()
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        p = 2 * min((boot >= 0).mean(), (boot <= 0).mean())
        # stability: доля подпериодов с IC того же знака
        n_third = n // 3
        signs = []
        for k in range(3):
            seg = ics[k*n_third:(k+1)*n_third]
            if len(seg): signs.append(np.sign(seg.mean()))
        stable = len(set(signs)) == 1
        ic_rows.append({
            "feature": fname, "n_dates": n,
            "ic_mean": mean_ic, "ic_std": ics.std(),
            "ic_ci_low": ci_low, "ic_ci_high": ci_high,
            "p_value": p, "stable_3_subperiods": stable,
            "sign_consistent": stable,
        })
        rep.emit(f"  {fname:<18}: IC={mean_ic:+.4f}  "
                 f"CI=[{ci_low:+.4f}, {ci_high:+.4f}]  p={p:.3f}  "
                 f"stable={'✓' if stable else '✗'}")

    ic_df = pd.DataFrame(ic_rows).sort_values("ic_mean", key=abs, ascending=False)
    with sqlite3.connect(OUT_SQL) as c:
        ic_df.to_sql("feature_ic", c, if_exists="replace", index=False)

    # --- 6b. Условное распределение RS по квантилям фичи ---
    rep.emit("\n[6b. Условное распределение RS по квантилям (топ-3 фичи)]")
    top_feats = ic_df.head(3)["feature"].tolist()
    for fname in top_feats:
        fdf = feats[fname]
        common_dates = fdf.index.intersection(rs.index)
        common_tickers = fdf.columns.intersection(rs.columns)
        f = fdf.loc[common_dates, common_tickers]
        r = rs.loc[common_dates, common_tickers]
        # разбиваем на 5 квантилей по f (кросс-секционно, по каждой дате)
        q_rows = []
        for d in common_dates:
            fv = f.loc[d]; rv = r.loc[d]
            mask = fv.notna() & rv.notna()
            if mask.sum() < 50: continue
            fv = fv[mask]; rv = rv[mask]
            try:
                q = pd.qcut(fv, 5, labels=False, duplicates="drop")
            except Exception:
                continue
            for qi in range(5):
                sel = rv[q == qi]
                if len(sel) > 0:
                    q_rows.append({"date": d, "quantile": qi, "mean_rs": sel.mean()})
        if q_rows:
            qdf = pd.DataFrame(q_rows)
            agg = qdf.groupby("quantile")["mean_rs"].median()
            rep.emit(f"\n  {fname}:")
            for qi, v in agg.items():
                rep.emit(f"    Q{qi+1}: median mean_rs = {v:+.4f}")

# ============================================================
# MAIN
# ============================================================
def main():
    rep = Report()
    rep.emit("# Кросс-секционный EDA: RS vs IMOEX (только IS)\n")
    rep.emit(f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n")

    split = load_split()
    rep.emit(f"split_date = {split}")
    rep.emit(f"Горизонт RS: {HORIZON} дней")
    rep.emit(f"VAL не трогаем\n")

    prices = load_prices(split)
    imoex = load_imoex(split)
    rep.emit(f"prices: {prices.shape[0]} дат × {prices.shape[1]} акций")
    rep.emit(f"IMOEX: {len(imoex)} дат")
    rep.emit(f"IS: {prices.index.min().date()} → {prices.index.max().date()}\n")

    # --- RS ---
    rep.emit(section("ПОСТРОЕНИЕ RS"))
    rs = compute_rs(prices, imoex, HORIZON)
    # ограничиваем RS: только даты, где t+20 тоже в IS
    # (compute_rs уже даёт NaN там, где нет будущего)
    valid_rs = rs.dropna(how="all")
    rep.emit(f"дат с RS: {len(valid_rs)}")
    rep.emit(f"диапазон: {valid_rs.index.min().date()} → {valid_rs.index.max().date()}")
    rep.emit(f"акций: {rs.shape[1]}")

    # сохраняем RS
    with sqlite3.connect(OUT_SQL) as c:
        rs_out = rs.reset_index()
        rs_out.columns = [str(x) for x in rs_out.columns]
        rs_out.to_sql("rs_matrix", c, if_exists="replace", index=False)

    # --- блоки ---
    block_geometry(rs, rep)
    block_structure(rs, rep)
    block_persistence(rs, rep)
    block_dependence(rs, rep)
    block_predictive(prices, imoex, rs, rep)

    rep.emit(section("ИТОГ"))
    rep.emit(f"SQL: {OUT_SQL}")
    rep.emit(f"Report: {OUT_MD}")
    rep.emit(f"Plots: {PLOTS_DIR}")
    with sqlite3.connect(OUT_SQL) as c:
        for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
            n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            rep.emit(f"  {t:<30} {n:>6} строк")
    rep.emit("\nГрафики (тёмный фон, монохром):")
    for p in sorted(PLOTS_DIR.glob("*.png")):
        rep.emit(f"  {p.name}")

    rep.save(OUT_MD)


if __name__ == "__main__":
    main()