# research/features/research_deep.py
# ============================================================
# Углублённое исследование (только IS).
#
# 1. Геометрия распределений: левый/правый хвосты, quantile asymmetry,
#    tail index по порогам, стабильность skew/kurt, fit Normal/t/Skewed-t.
# 2. Память и зависимость: ACF ret/abs/sq, VR на горизонтах, DFA,
#    устойчивость Hurst, спектр.
# 3. Условная динамика риска: GARCH, GJR, EGARCH, leverage, устойчивость.
# 4. Нестационарность: rolling moments, change-point, structural breaks,
#    сравнение периодов.
#
# Запуск: python research_deep.py
# ============================================================
import json
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, signal

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.stattools import acf, adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.regression.linear_model import OLS
    SM_OK = True
except ImportError:
    SM_OK = False
    print("WARN: statsmodels")

try:
    from arch import arch_model
    ARCH_OK = True
except ImportError:
    ARCH_OK = False
    print("WARN: arch (pip install arch)")

try:
    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    CUSUM_OK = True
except ImportError:
    CUSUM_OK = False

# ============================================================
# ПУТИ И КОНСТАНТЫ
# ============================================================
ROOT = Path(__file__).parent.parent
STOCKS_DB = ROOT / "dataset" / "sql" / "stocks.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"
SPLIT_JSON = ROOT / "backtests" / "split_info.json"

OUT_SQL = Path(__file__).parent / "sql" / "research_deep.sqlite"
OUT_SQL.parent.mkdir(exist_ok=True)
OUT_MD = Path(__file__).parent / "research_deep_report.md"

GLOBAL = ["IMOEX", "RGBI", "GOLD", "BRENT", "CNYRUB_TOM"]
MIN_BARS = 500
MAX_HEAVY = 40          # для GARCH, DFA, change-point
TAIL_QUANTILES = [0.90, 0.95, 0.975, 0.99]   # пороги для tail index
VR_HORIZONS = [2, 5, 10, 20, 60]              # Variance Ratio горизонты


# ============================================================
# HELPERS
# ============================================================
def section(t):
    return f"\n{'='*70}\n{t}\n{'='*70}"

class Report:
    def __init__(self): self.lines = []
    def emit(self, s=""):
        print(s); self.lines.append(str(s))
    def save(self, p):
        p.write_text("\n".join(self.lines), encoding="utf-8")
        print(f"\nОтчёт: {p}")

def load_split():
    return json.loads(SPLIT_JSON.read_text())["split_date"]

def load_returns(split_date):
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

def load_global(split_date):
    conn = sqlite3.connect(INDEXES_DB)
    df = pd.read_sql("SELECT ticker, date, close FROM indexes WHERE date < ?",
                     conn, params=(split_date,)).sort_values(["ticker","date"])
    conn.close()
    df["ret"] = df.groupby("ticker")["close"].pct_change()
    w = df.pivot(index="date", columns="ticker", values="ret")
    w.index = pd.to_datetime(w.index)
    return w[[c for c in GLOBAL if c in w.columns]].sort_index()

def save(name, df):
    with sqlite3.connect(OUT_SQL) as c:
        df.to_sql(name, c, if_exists="replace", index=False)


# ============================================================
# 1. ГЕОМЕТРИЯ РАСПРЕДЕЛЕНИЙ
# ============================================================
def block_distribution_geometry(stock, glob, rep):
    rep.emit(section("1. ГЕОМЕТРИЯ РАСПРЕДЕЛЕНИЙ"))

    rows = []
    # --- 1a. Левый/правый хвосты + quantile asymmetry ---
    rep.emit("\n[1a. Левый/правый хвосты, quantile asymmetry]")
    for label, df in [("STOCKS", stock), ("GLOBAL", glob)]:
        for col in df.columns:
            x = df[col].dropna()
            if len(x) < MIN_BARS: continue
            q = np.quantile(x, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            # хвостовые «длины» от медианы
            left_1 = q[3] - q[0]   # median - q01
            right_1 = q[6] - q[3]  # q99 - median
            left_5 = q[3] - q[1]
            right_5 = q[5] - q[3]
            # quantile asymmetry (Bowley skew): (q75 + q25 - 2*q50)/(q75 - q25)
            bowley = (q[4] + q[2] - 2*q[3]) / (q[4] - q[2]) if (q[4]-q[2])>0 else np.nan
            # tail ratio right/left на 1% и 5%
            tr_1 = right_1 / left_1 if left_1 > 0 else np.nan
            tr_5 = right_5 / left_5 if left_5 > 0 else np.nan
            rows.append({"scope": label, "ticker": col, "n": len(x),
                         "left_tail_1pct": left_1, "right_tail_1pct": right_1,
                         "tail_ratio_1pct": tr_1,
                         "left_tail_5pct": left_5, "right_tail_5pct": right_5,
                         "tail_ratio_5pct": tr_5,
                         "bowley_skew": bowley})
        sub = pd.DataFrame([r for r in rows if r["scope"]==label])
        if len(sub):
            rep.emit(f"  {label}: median tail_ratio_1pct={sub['tail_ratio_1pct'].median():.3f}  "
                     f"median bowley={sub['bowley_skew'].median():+.3f}")
    save("dist_tails", pd.DataFrame(rows))

    # --- 1b. Tail index по разным порогам (Hill) ---
    rep.emit("\n[1b. Tail index (Hill α) по порогам]")
    hill_rows = []
    for label, df in [("STOCKS", stock), ("GLOBAL", glob)]:
        for col in df.columns:
            x = df[col].dropna()
            if len(x) < MIN_BARS: continue
            absx = np.abs(x.values)
            y = np.sort(absx)[::-1]
            for q in TAIL_QUANTILES:
                k = int((1-q) * len(y))
                k = max(k, 15); k = min(k, len(y)-1)
                if k > 10 and y[k] > 0:
                    hill = 1.0 / np.mean(np.log(y[:k] / y[k]))
                else:
                    hill = np.nan
                hill_rows.append({"scope": label, "ticker": col,
                                  "threshold_q": q, "k": k, "hill_alpha": hill})
    hdf = pd.DataFrame(hill_rows)
    save("tail_index_by_threshold", hdf)
    for label in ["STOCKS", "GLOBAL"]:
        sub = hdf[hdf["scope"]==label]
        for q in TAIL_QUANTILES:
            s = sub[sub["threshold_q"]==q]["hill_alpha"]
            if len(s): rep.emit(f"  {label} q={q:.3f}: median α={s.median():.3f}")

    # --- 1c. Стабильность skew/kurt (rolling 252) ---
    rep.emit("\n[1c. Стабильность skew/kurt (rolling 252)]")
    stab_rows = []
    for label, df in [("STOCKS", stock), ("GLOBAL", glob)]:
        for col in df.columns:
            x = df[col].dropna()
            if len(x) < 500: continue
            rs = x.rolling(252).skew()
            rk = x.rolling(252).kurt()
            stab_rows.append({"scope": label, "ticker": col,
                              "skew_mean": rs.mean(), "skew_std": rs.std(),
                              "skew_min": rs.min(), "skew_max": rs.max(),
                              "kurt_mean": rk.mean(), "kurt_std": rk.std(),
                              "kurt_max": rk.max()})
        sub = pd.DataFrame([r for r in stab_rows if r["scope"]==label])
        if len(sub):
            rep.emit(f"  {label}: skew std={sub['skew_std'].median():.3f}  "
                     f"kurt std={sub['kurt_std'].median():.1f}")
    save("skew_kurt_stability", pd.DataFrame(stab_rows))

    # --- 1d. Fit Normal / Student-t / Skewed-t (AIC, KS) ---
    rep.emit("\n[1d. Fit: Normal vs Student-t vs Skewed-t]")
    fit_rows = []
    for label, df in [("STOCKS", stock), ("GLOBAL", glob)]:
        for col in df.columns:
            x = df[col].dropna().values
            if len(x) < MIN_BARS: continue
            # Normal
            mu, sig = x.mean(), x.std()
            ll_n = np.sum(stats.norm.logpdf(x, mu, sig))
            aic_n = 2*2 - 2*ll_n
            # Student-t
            try:
                nu, loc, scale = stats.t.fit(x)
                ll_t = np.sum(stats.t.logpdf(x, nu, loc, scale))
                aic_t = 2*3 - 2*ll_t
            except Exception:
                nu, aic_t = np.nan, np.nan
            # Skewed-t (если есть scipy.stats.skewnorm как прокси;
            # для настоящего skewed-t нужен отдельный fit — используем scipy.stats.t как базу)
            # Примечание: в scipy нет готового skewed-t, поэтому сравниваем Normal vs t.
            # Skewed-t можно добавить через библиотеку `skewt` или ручной MLE.
            # Здесь — только Normal vs Student-t как минимальный набор.
            fit_rows.append({"scope": label, "ticker": col, "n": len(x),
                             "aic_normal": aic_n, "aic_t": aic_t,
                             "t_df": nu,
                             "t_better": int(aic_t < aic_n) if not np.isnan(aic_t) else np.nan})
    fdf = pd.DataFrame(fit_rows)
    save("dist_fit", fdf)
    for label in ["STOCKS", "GLOBAL"]:
        sub = fdf[fdf["scope"]==label]
        if len(sub):
            rep.emit(f"  {label}: доля t лучше Normal: {sub['t_better'].mean():.1%}  "
                     f"median df={sub['t_df'].median():.1f}")


# ============================================================
# 2. ПАМЯТЬ И ЗАВИСИМОСТЬ
# ============================================================
def block_memory_dependence(stock, glob, rep):
    rep.emit(section("2. ПАМЯТЬ И ЗАВИСИМОСТЬ"))

    # --- 2a. ACF ret / abs / sq ---
    rep.emit("\n[2a. ACF(1..10) для ret, |ret|, ret²]")
    acf_rows = []
    for label, df in [("STOCKS", stock), ("GLOBAL", glob)]:
        for col in df.columns:
            x = df[col].dropna()
            if len(x) < MIN_BARS: continue
            for name, series in [("ret", x), ("abs", x.abs()), ("sq", x**2)]:
                a = acf(series, nlags=10, fft=False)
                row = {"scope": label, "ticker": col, "kind": name}
                for i in range(1, 11):
                    row[f"acf{i}"] = a[i] if i < len(a) else np.nan
                acf_rows.append(row)
    adf = pd.DataFrame(acf_rows)
    save("acf_ret_abs_sq", adf)
    for label in ["STOCKS", "GLOBAL"]:
        for kind in ["ret", "abs", "sq"]:
            sub = adf[(adf["scope"]==label) & (adf["kind"]==kind)]
            if len(sub):
                rep.emit(f"  {label} {kind}: median |ACF(1)|={sub['acf1'].abs().median():.4f}  "
                         f"|ACF(10)|={sub['acf10'].abs().median():.4f}")

    # --- 2b. Short vs long memory (Ljung-Box на разных лагах + ACF decay) ---
    rep.emit("\n[2b. Short vs long memory]")
    mem_rows = []
    for label, df in [("STOCKS", stock), ("GLOBAL", glob)]:
        for col in df.columns:
            x = df[col].dropna()
            if len(x) < MIN_BARS: continue
            for kind, series in [("ret", x), ("abs", x.abs())]:
                a = acf(series, nlags=50, fft=False)
                # скорость затухания: лаг, где |ACF| < 0.05 впервые
                below = np.where(np.abs(a[1:]) < 0.05)[0]
                decay_lag = below[0]+1 if len(below) else 50
                mem_rows.append({"scope": label, "ticker": col, "kind": kind,
                                 "acf_decay_lag": decay_lag,
                                 "acf5": a[5] if len(a)>5 else np.nan,
                                 "acf20": a[20] if len(a)>20 else np.nan,
                                 "acf50": a[50] if len(a)>50 else np.nan})
    mdf = pd.DataFrame(mem_rows)
    save("memory_short_long", mdf)
    for label in ["STOCKS", "GLOBAL"]:
        for kind in ["ret", "abs"]:
            sub = mdf[(mdf["scope"]==label) & (mdf["kind"]==kind)]
            if len(sub):
                rep.emit(f"  {label} {kind}: median decay_lag={sub['acf_decay_lag'].median():.0f}  "
                         f"median ACF(20)={sub['acf20'].median():.4f}")

    # --- 2c. Variance Ratio на горизонтах ---
    rep.emit("\n[2c. Variance Ratio на горизонтах]")
    vr_rows = []
    for label, df in [("STOCKS", stock), ("GLOBAL", glob)]:
        for col in df.columns:
            x = df[col].dropna().values
            if len(x) < MIN_BARS: continue
            row = {"scope": label, "ticker": col, "n": len(x)}
            for q in VR_HORIZONS:
                if len(x) < q+1:
                    row[f"vr_{q}"] = np.nan; continue
                var1 = np.var(x, ddof=1)
                xq = pd.Series(x).rolling(q).sum().dropna().values
                varq = np.var(xq, ddof=1) / q
                row[f"vr_{q}"] = varq/var1 if var1>0 else np.nan
            vr_rows.append(row)
    vdf = pd.DataFrame(vr_rows)
    save("variance_ratio", vdf)
    for label in ["STOCKS", "GLOBAL"]:
        sub = vdf[vdf["scope"]==label]
        if len(sub):
            for q in VR_HORIZONS:
                c = f"vr_{q}"
                if c in sub: rep.emit(f"  {label} VR({q}): median={sub[c].median():.3f}")

    # --- 2d. DFA + устойчивость Hurst (по окнам) ---
    rep.emit("\n[2d. DFA и устойчивость Hurst]")
    def dfa(x, order=1, max_lag=50):
        x = np.asarray(x); n = len(x)
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

    heavy_cols = sorted([c for c in stock.columns if stock[c].dropna().shape[0]>=MIN_BARS],
                        key=lambda c: stock[c].dropna().shape[0], reverse=True)[:MAX_HEAVY]
    hurst_rows = []
    for col in heavy_cols:
        x = stock[col].dropna().values
        h_full = dfa(x)
        # устойчивость: DFA на 3-х подокнах
        n = len(x); third = n//3
        h1 = dfa(x[:third]); h2 = dfa(x[third:2*third]); h3 = dfa(x[2*third:])
        hurst_rows.append({"scope":"STOCKS","ticker":col,"hurst_full":h_full,
                           "h_win1":h1,"h_win2":h2,"h_win3":h3,
                           "hurst_std":np.nanstd([h1,h2,h3])})
    for col in glob.columns:
        x = glob[col].dropna().values
        if len(x) < MIN_BARS: continue
        h_full = dfa(x)
        n = len(x); third = n//3
        h1 = dfa(x[:third]); h2 = dfa(x[third:2*third]); h3 = dfa(x[2*third:])
        hurst_rows.append({"scope":"GLOBAL","ticker":col,"hurst_full":h_full,
                           "h_win1":h1,"h_win2":h2,"h_win3":h3,
                           "hurst_std":np.nanstd([h1,h2,h3])})
    hdf = pd.DataFrame(hurst_rows)
    save("hurst_stability", hdf)
    for label in ["STOCKS","GLOBAL"]:
        sub = hdf[hdf["scope"]==label]
        if len(sub):
            rep.emit(f"  {label}: median Hurst={sub['hurst_full'].median():.3f}  "
                     f"median std по окнам={sub['hurst_std'].median():.3f}")

    # --- 2e. Спектральный анализ (пики) ---
    rep.emit("\n[2e. Спектральный анализ (top-5 пиков)]")
    spec_rows = []
    for col in list(stock.columns[:MAX_HEAVY]) + list(glob.columns):
        x = (stock[col] if col in stock else glob[col]).dropna().values
        if len(x) < MIN_BARS: continue
        f, P = signal.periodogram(x, fs=1.0)
        # топ-5 пиков по мощности (исключая f=0)
        idx = np.argsort(P[1:])[::-1][:5] + 1
        for rank, i in enumerate(idx, 1):
            spec_rows.append({"ticker": col, "rank": rank,
                              "freq": f[i], "period_days": 1.0/f[i] if f[i]>0 else np.nan,
                              "power": P[i]})
    save("spectral_peaks", pd.DataFrame(spec_rows))
    rep.emit("  → таблица spectral_peaks (топ-5 частот на тикер)")


# ============================================================
# 3. УСЛОВНАЯ ДИНАМИКА РИСКА
# ============================================================
def block_conditional_risk(stock, glob, rep):
    rep.emit(section("3. УСЛОВНАЯ ДИНАМИКА РИСКА"))

    if not ARCH_OK:
        rep.emit("  SKIP: arch не установлен")
        return

    heavy = sorted([c for c in stock.columns if stock[c].dropna().shape[0]>=MIN_BARS],
                   key=lambda c: stock[c].dropna().shape[0], reverse=True)[:MAX_HEAVY]
    cols = heavy + list(glob.columns)

    rows = []
    for col in cols:
        x = (stock[col] if col in stock else glob[col]).dropna()
        if len(x) < MIN_BARS: continue
        x_pct = x * 100  # arch чувствителен к масштабу
        res = {"ticker": col, "scope": "STOCKS" if col in stock else "GLOBAL", "n": len(x)}

        # GARCH(1,1)
        try:
            am = arch_model(x_pct, vol="Garch", p=1, q=1, dist="t")
            r = am.fit(disp="off", show_warning=False)
            res["garch_alpha"] = r.params.get("alpha[1]", np.nan)
            res["garch_beta"] = r.params.get("beta[1]", np.nan)
            res["garch_persist"] = res["garch_alpha"] + res["garch_beta"]
            res["garch_loglik"] = r.loglikelihood
        except Exception: pass

        # GJR-GARCH (асимметрия)
        try:
            am = arch_model(x_pct, vol="Garch", p=1, o=1, q=1, dist="t")
            r = am.fit(disp="off", show_warning=False)
            res["gjr_gamma"] = r.params.get("gamma[1]", np.nan)  # leverage
            res["gjr_persist"] = (r.params.get("alpha[1]",0) +
                                  r.params.get("gamma[1]",0)/2 +
                                  r.params.get("beta[1]",0))
        except Exception: pass

        # EGARCH (лог-дисперсия, асимметрия)
        try:
            am = arch_model(x_pct, vol="EGARCH", p=1, o=1, q=1, dist="t")
            r = am.fit(disp="off", show_warning=False)
            res["egarch_gamma"] = r.params.get("gamma[1]", np.nan)  # асимметрия
        except Exception: pass

        rows.append(res)

    df = pd.DataFrame(rows)
    save("garch_models", df)

    for scope in ["STOCKS", "GLOBAL"]:
        sub = df[df["scope"]==scope]
        if len(sub):
            rep.emit(f"\n[{scope}]")
            rep.emit(f"  GARCH persist: median={sub['garch_persist'].median():.4f}")
            rep.emit(f"  GJR gamma (leverage): median={sub['gjr_gamma'].median():+.4f}")
            rep.emit(f"  EGARCH gamma (asym): median={sub['egarch_gamma'].median():+.4f}")

    # --- устойчивость параметров GARCH по подпериодам ---
    rep.emit("\n[Устойчивость GARCH по 3 подпериодам]")
    stab_rows = []
    focus = ["IMOEX", "SBER"] if "SBER" in stock else ["IMOEX"]
    for col in focus:
        x = (stock[col] if col in stock else glob[col]).dropna() * 100
        n = len(x); third = n//3
        for i, seg in enumerate([x[:third], x[third:2*third], x[2*third:]], 1):
            try:
                am = arch_model(seg, vol="Garch", p=1, q=1, dist="t")
                r = am.fit(disp="off", show_warning=False)
                stab_rows.append({"ticker": col, "window": i,
                                  "alpha": r.params.get("alpha[1]", np.nan),
                                  "beta": r.params.get("beta[1]", np.nan),
                                  "persist": r.params.get("alpha[1]",0)+r.params.get("beta[1]",0)})
            except Exception: pass
    sdf = pd.DataFrame(stab_rows)
    save("garch_stability", sdf)
    for col in focus:
        sub = sdf[sdf["ticker"]==col]
        if len(sub):
            rep.emit(f"  {col}: persist по окнам = {sub['persist'].round(3).tolist()}")


# ============================================================
# 4. НЕСТАЦИОНАРНОСТЬ И СТРУКТУРНЫЕ ИЗМЕНЕНИЯ
# ============================================================
def block_nonstationarity(stock, glob, rep):
    rep.emit(section("4. НЕСТАЦИОНАРНОСТЬ И СТРУКТУРНЫЕ ИЗМЕНЕНИЯ"))

    # --- 4a. Rolling moments ---
    rep.emit("\n[4a. Rolling moments (252d) для IMOEX и SBER]")
    focus = [("IMOEX", glob["IMOEX"])] if "IMOEX" in glob else []
    for t in ["SBER", "GAZP", "LKOH"]:
        if t in stock: focus.append((t, stock[t])); break
    roll_rows = []
    for name, x in focus:
        x = x.dropna()
        if len(x) < 500: continue
        rm = x.rolling(252).mean()
        rs = x.rolling(252).std()
        rk = x.rolling(252).skew()
        rku = x.rolling(252).kurt()
        roll_rows.append({"ticker": name, "n": len(x),
                          "mean_min": rm.min(), "mean_max": rm.max(), "mean_std": rm.std(),
                          "std_min": rs.min(), "std_max": rs.max(),
                          "std_ratio": rs.max()/rs.min() if rs.min()>0 else np.nan,
                          "skew_min": rk.min(), "skew_max": rk.max(),
                          "kurt_max": rku.max()})
    save("rolling_moments", pd.DataFrame(roll_rows))
    for r in roll_rows:
        rep.emit(f"  {r['ticker']}: std ratio={r['std_ratio']:.2f}  "
                 f"skew [{r['skew_min']:.2f}..{r['skew_max']:.2f}]  "
                 f"kurt max={r['kurt_max']:.1f}")

    # --- 4b. Change-point detection (CUSUM) ---
    rep.emit("\n[4b. Change-point detection (CUSUM)]")
    cp_rows = []
    for label, df in [("STOCKS", stock), ("GLOBAL", glob)]:
        cols = [c for c in df.columns if df[c].dropna().shape[0]>=MIN_BARS][:MAX_HEAVY]
        for col in cols:
            x = df[col].dropna().values
            if len(x) < 200: continue
            try:
                res = OLS(x, np.ones(len(x))).fit()
                cusum = breaks_cusumolsresid(res.resid)
                cp_rows.append({"scope": label, "ticker": col,
                                "cusum_stat": float(cusum[0]), "cusum_p": float(cusum[1])})
            except Exception: pass
    cpdf = pd.DataFrame(cp_rows)
    save("cusum_breaks", cpdf)
    for label in ["STOCKS","GLOBAL"]:
        sub = cpdf[cpdf["scope"]==label]
        if len(sub):
            rep.emit(f"  {label}: доля p<0.05 (есть сдвиг): {(sub['cusum_p']<0.05).mean():.1%}")

    # --- 4c. Сравнение периодов ---
    rep.emit("\n[4c. Сравнение статистик по 3 равным периодам]")
    period_rows = []
    for name, x in focus:
        x = x.dropna()
        n = len(x); third = n//3
        segs = [("P1", x.iloc[:third]), ("P2", x.iloc[third:2*third]), ("P3", x.iloc[2*third:])]
        for pname, seg in segs:
            period_rows.append({"ticker": name, "period": pname,
                                "n": len(seg), "mean": seg.mean(), "std": seg.std(),
                                "skew": stats.skew(seg), "kurt": stats.kurtosis(seg)})
    pdf = pd.DataFrame(period_rows)
    save("period_comparison", pdf)
    for name, _ in focus:
        sub = pdf[pdf["ticker"]==name]
        rep.emit(f"\n  {name}:")
        for _, r in sub.iterrows():
            rep.emit(f"    {r['period']}: mean={r['mean']:+.6f}  std={r['std']:.6f}  "
                     f"skew={r['skew']:+.2f}  kurt={r['kurt']:.1f}")

    # --- 4d. Взаимное изменение: корреляция rolling std vs rolling skew ---
    rep.emit("\n[4d. Связь rolling std и rolling skew (IMOEX)]")
    if "IMOEX" in glob:
        x = glob["IMOEX"].dropna()
        rs = x.rolling(252).std()
        rk = x.rolling(252).skew()
        valid = pd.concat([rs, rk], axis=1).dropna()
        if len(valid) > 10:
            corr = valid.iloc[:,0].corr(valid.iloc[:,1])
            rep.emit(f"  corr(rolling std, rolling skew) = {corr:+.3f}")


# ============================================================
# MAIN
# ============================================================
def main():
    rep = Report()
    rep.emit("# Углублённое исследование (IS)\n")
    rep.emit(f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n")

    split = load_split()
    rep.emit(f"split_date = {split} (VAL не трогаем)\n")

    stock = load_returns(split)
    glob = load_global(split)
    rep.emit(f"stocks: {stock.shape[0]} дат × {stock.shape[1]} тикеров")
    rep.emit(f"global: {glob.shape[0]} дат × {glob.shape[1]} инструментов")
    rep.emit(f"IS: {stock.index.min().date()} → {stock.index.max().date()}")

    block_distribution_geometry(stock, glob, rep)
    block_memory_dependence(stock, glob, rep)
    block_conditional_risk(stock, glob, rep)
    block_nonstationarity(stock, glob, rep)

    rep.emit(section("ИТОГ"))
    rep.emit(f"SQL: {OUT_SQL}")
    rep.emit(f"Report: {OUT_MD}")
    with sqlite3.connect(OUT_SQL) as c:
        for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
            n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            rep.emit(f"  {t:<25} {n:>6} строк")
    rep.save(OUT_MD)


if __name__ == "__main__":
    main()