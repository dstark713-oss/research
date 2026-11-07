# research/features/eda.py
# ============================================================
# EDA ядро: статистический портрет рынка на IS.
# ВАЖНО: работает ТОЛЬКО на IS (date < split_date).
# VAL не трогается.
#
# 8 блоков:
#   1. Distributional properties
#   2. Extreme Value Theory
#   3. Temporal dependence
#   4. Volatility econometrics
#   5. Stationarity
#   6. Long memory
#   7. Non-stationarity over time
#   8. Global factor econometrics
#
# Выход:
#   features/sql/eda.sqlite   — машинночитаемые таблицы
#   features/eda_report.md    — человекочитаемый отчёт
#
# Запуск: python eda.py
# ============================================================
import json
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# --- опциональные зависимости (проверяются внутри) ---
try:
    from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    SM_OK = True
except ImportError:
    SM_OK = False
    print("WARN: statsmodels не установлен — часть тестов пропущена")

try:
    from arch import arch_model
    ARCH_OK = True
except ImportError:
    ARCH_OK = False
    print("WARN: arch не установлен — GARCH пропущен (pip install arch)")

try:
    from statsmodels.tsa.stattools import PhillipsPerron
    PP_OK = True
except ImportError:
    PP_OK = False

# ============================================================
# ПУТИ
# ============================================================
ROOT = Path(__file__).parent.parent
STOCKS_DB = ROOT / "dataset" / "sql" / "stocks.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"
SPLIT_JSON = ROOT / "backtests" / "split_info.json"

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(exist_ok=True)
OUT_SQL = OUT_DIR / "sql" / "eda.sqlite"
OUT_SQL.parent.mkdir(exist_ok=True)
OUT_MD = OUT_DIR / "eda_report.md"

GLOBAL_TICKERS = ["IMOEX", "RGBI", "GOLD", "BRENT", "CNYRUB_TOM"]
MAX_STOCKS_FULL = 50   # для тяжёлых тестов (GARCH, ADF) — только топ-N по истории
MIN_BARS = 500         # минимум баров, чтобы тикер попал в анализ


# ============================================================
# HELPERS
# ============================================================
def section(title):
    line = "=" * 70
    return f"\n{line}\n{title}\n{line}"


class Report:
    def __init__(self):
        self.lines = []
    def emit(self, s=""):
        print(s)
        self.lines.append(str(s))
    def save(self, path):
        path.write_text("\n".join(self.lines), encoding="utf-8")
        print(f"\nОтчёт: {path}")


def load_split_date() -> str:
    if not SPLIT_JSON.exists():
        raise FileNotFoundError(f"нет {SPLIT_JSON}. Сначала запусти backtests/split.py")
    info = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    return info["split_date"]


def load_stock_returns(split_date: str) -> pd.DataFrame:
    """
    Возвращает DataFrame: index=date (только IS), columns=tickers, values=ret.
    Берём returns, если есть; иначе считаем из prices.
    """
    conn = sqlite3.connect(STOCKS_DB)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}

    if "returns" in tables:
        df = pd.read_sql(
            "SELECT ticker, date, ret FROM returns WHERE date < ?",
            conn, params=(split_date,),
        )
    else:
        prices = pd.read_sql(
            "SELECT ticker, date, close FROM prices WHERE date < ?",
            conn, params=(split_date,),
        )
        prices = prices.sort_values(["ticker", "date"])
        prices["ret"] = prices.groupby("ticker")["close"].pct_change()
        df = prices[["ticker", "date", "ret"]].dropna()

    conn.close()

    wide = df.pivot(index="date", columns="ticker", values="ret")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def load_global_returns(split_date: str) -> pd.DataFrame:
    conn = sqlite3.connect(INDEXES_DB)
    df = pd.read_sql(
        "SELECT ticker, date, close FROM indexes WHERE date < ?",
        conn, params=(split_date,),
    )
    conn.close()

    df = df.sort_values(["ticker", "date"])
    df["ret"] = df.groupby("ticker")["close"].pct_change()
    wide = df.pivot(index="date", columns="ticker", values="ret")
    wide.index = pd.to_datetime(wide.index)
    # оставляем только нужные глобальные
    cols = [c for c in GLOBAL_TICKERS if c in wide.columns]
    return wide[cols].sort_index()


def save_table(name: str, df: pd.DataFrame):
    with sqlite3.connect(OUT_SQL) as conn:
        df.to_sql(name, conn, if_exists="replace", index=False)


# ============================================================
# 1. DISTRIBUTIONAL PROPERTIES
# ============================================================
def block_distributions(stock_ret: pd.DataFrame, global_ret: pd.DataFrame, rep: Report):
    rep.emit(section("1. DISTRIBUTIONAL PROPERTIES"))

    rows = []
    for label, df in [("STOCKS", stock_ret), ("GLOBAL", global_ret)]:
        rep.emit(f"\n[{label}]")
        for col in df.columns:
            x = df[col].dropna()
            if len(x) < MIN_BARS:
                continue
            # базовые моменты
            mean = x.mean()
            std = x.std()
            skew = stats.skew(x)
            kurt = stats.kurtosis(x)           # excess kurtosis (норма=0)
            # квантили
            q01, q05, q25, q50, q75, q95, q99 = np.quantile(
                x, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
            )
            # хвостовые отношения (tail heaviness)
            iqr = q75 - q25
            tail_ratio = (q99 - q01) / iqr if iqr > 0 else np.nan
            # Jarque-Bera (нормальность)
            jb_stat, jb_p = stats.jarque_bera(x)
            # Anderson-Darling
            try:
                ad = stats.anderson(x, dist="norm")
                ad_stat = ad.statistic
            except Exception:
                ad_stat = np.nan
            # доля |x| > 3σ, > 5σ
            z = (x - mean) / std if std > 0 else pd.Series(0, index=x.index)
            share_3s = float((z.abs() > 3).mean())
            share_5s = float((z.abs() > 5).mean())
            # положительные vs отрицательные
            pos = x[x > 0]
            neg = x[x < 0]
            rows.append({
                "scope": label, "ticker": col,
                "n": len(x), "mean": mean, "std": std,
                "skew": skew, "kurt_excess": kurt,
                "q01": q01, "q05": q05, "q25": q25, "q50": q50,
                "q75": q75, "q95": q95, "q99": q99,
                "tail_ratio": tail_ratio,
                "jb_stat": jb_stat, "jb_p": jb_p,
                "ad_stat": ad_stat,
                "share_3s": share_3s, "share_5s": share_5s,
                "mean_pos": pos.mean() if len(pos) else np.nan,
                "mean_neg": neg.mean() if len(neg) else np.nan,
                "share_pos": len(pos) / len(x),
            })
        # краткая сводка
        sub = pd.DataFrame([r for r in rows if r["scope"] == label])
        if len(sub):
            rep.emit(f"  тикеров: {len(sub)}")
            rep.emit(f"  median skew:      {sub['skew'].median():.3f}")
            rep.emit(f"  median kurt:      {sub['kurt_excess'].median():.3f}")
            rep.emit(f"  median JB p:      {sub['jb_p'].median():.4f}  "
                     f"(доля p<0.05: {(sub['jb_p']<0.05).mean():.2%})")
            rep.emit(f"  median share>3σ:  {sub['share_3s'].median():.4f}")
            rep.emit(f"  median share>5σ:  {sub['share_5s'].median():.4f}")

    out = pd.DataFrame(rows)
    save_table("distributions", out)
    rep.emit(f"\n→ таблица distributions: {len(out)} строк")


# ============================================================
# 2. EXTREME VALUE THEORY
# ============================================================
def block_evt(stock_ret: pd.DataFrame, global_ret: pd.DataFrame, rep: Report):
    rep.emit(section("2. EXTREME VALUE THEORY (EVT)"))

    rows = []
    for label, df in [("STOCKS", stock_ret), ("GLOBAL", global_ret)]:
        rep.emit(f"\n[{label}]")
        for col in df.columns:
            x = df[col].dropna()
            if len(x) < MIN_BARS:
                continue
            # Hill estimator для правого хвоста (tail index α)
            # α = 1/ξ, ξ — tail index. Меньше α → тяжелее хвост.
            # Используем верхние k = 5% наблюдений.
            y = np.sort(np.abs(x.values))[::-1]
            k = max(int(0.05 * len(y)), 20)
            k = min(k, len(y) - 1)
            if k > 10 and y[k] > 0:
                hill = 1.0 / np.mean(np.log(y[:k] / y[k]))
            else:
                hill = np.nan

            # Пиковая нагруженность (POT) — превышения над порогом u = q95
            u = np.quantile(x, 0.95)
            exceed = x[x > u] - u
            # средний эксцесс превышения (mean excess)
            mean_excess = exceed.mean() if len(exceed) else np.nan

            # частота экстремумов (|z| > 4)
            z = (x - x.mean()) / x.std() if x.std() > 0 else x * 0
            n_extreme_4s = int((z.abs() > 4).sum())
            n_extreme_5s = int((z.abs() > 5).sum())

            rows.append({
                "scope": label, "ticker": col,
                "n": len(x),
                "hill_alpha_5pct": hill,
                "n_exceed_q95": len(exceed),
                "mean_excess_q95": mean_excess,
                "n_extreme_4s": n_extreme_4s,
                "n_extreme_5s": n_extreme_5s,
                "max_abs_ret": float(x.abs().max()),
            })
        sub = pd.DataFrame([r for r in rows if r["scope"] == label])
        if len(sub):
            rep.emit(f"  median Hill α (5%): {sub['hill_alpha_5pct'].median():.3f}")
            rep.emit(f"  median mean excess q95: {sub['mean_excess_q95'].median():.4f}")
            rep.emit(f"  тикеров с ≥1 экстремумом >4σ: "
                     f"{(sub['n_extreme_4s']>0).sum()} / {len(sub)}")

    out = pd.DataFrame(rows)
    save_table("extremes", out)
    rep.emit(f"\n→ таблица extremes: {len(out)} строк")
    rep.emit("  Hill α < 4 → тяжёлые хвосты (нет конечной дисперсии/4-го момента)")
    rep.emit("  Hill α < 2 → нет конечной дисперсии (экстремально тяжёлые)")


# ============================================================
# 3. TEMPORAL DEPENDENCE
# ============================================================
def block_temporal(stock_ret: pd.DataFrame, global_ret: pd.DataFrame, rep: Report):
    rep.emit(section("3. TEMPORAL DEPENDENCE"))

    rows = []
    for label, df in [("STOCKS", stock_ret), ("GLOBAL", global_ret)]:
        rep.emit(f"\n[{label}]")
        for col in df.columns:
            x = df[col].dropna()
            if len(x) < MIN_BARS:
                continue

            # ACF на лагах 1..5
            acf_vals = acf(x, nlags=5, fft=False)
            # PACF на лагах 1..5
            pacf_vals = pacf(x, nlags=5, method="ywm")
            # Ljung-Box на 10 лагах
            lb = acorr_ljungbox(x, lags=[10], return_df=True)
            lb_stat = float(lb["lb_stat"].iloc[0])
            lb_p = float(lb["lb_pvalue"].iloc[0])

            # Runs test: серии одинаковых знаков
            signs = np.sign(x.values)
            signs = signs[signs != 0]
            if len(signs) > 0:
                runs = 1 + int((signs[1:] != signs[:-1]).sum())
                n_pos = int((signs > 0).sum())
                n_neg = int((signs < 0).sum())
                n = len(signs)
                if n_pos > 0 and n_neg > 0:
                    exp_runs = 1 + 2 * n_pos * n_neg / n
                    var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n * n * (n - 1))
                    z_runs = (runs - exp_runs) / np.sqrt(var_runs) if var_runs > 0 else np.nan
                else:
                    z_runs = np.nan
            else:
                runs, z_runs = np.nan, np.nan

            # Variance Ratio (VR) на 5 и 20 лагах
            def variance_ratio(x, q):
                x = np.asarray(x)
                n = len(x)
                if n < q + 1:
                    return np.nan
                var_1 = np.var(x, ddof=1)
                # q-периодная доходность
                x_q = pd.Series(x).rolling(q).sum().dropna().values
                var_q = np.var(x_q, ddof=1) / q
                return var_q / var_1 if var_1 > 0 else np.nan

            vr5 = variance_ratio(x, 5)
            vr20 = variance_ratio(x, 20)

            rows.append({
                "scope": label, "ticker": col,
                "n": len(x),
                "acf1": acf_vals[1] if len(acf_vals) > 1 else np.nan,
                "acf2": acf_vals[2] if len(acf_vals) > 2 else np.nan,
                "acf5": acf_vals[5] if len(acf_vals) > 5 else np.nan,
                "pacf1": pacf_vals[1] if len(pacf_vals) > 1 else np.nan,
                "lb_stat_10": lb_stat,
                "lb_p_10": lb_p,
                "runs_z": z_runs,
                "vr5": vr5,
                "vr20": vr20,
            })
        sub = pd.DataFrame([r for r in rows if r["scope"] == label])
        if len(sub):
            rep.emit(f"  median |ACF(1)|: {sub['acf1'].abs().median():.4f}")
            rep.emit(f"  медиана VR(5):   {sub['vr5'].median():.3f}  (1 = random walk)")
            rep.emit(f"  медиана VR(20):  {sub['vr20'].median():.3f}")
            rep.emit(f"  доля p(Ljung-Box)<0.05: {(sub['lb_p_10']<0.05).mean():.2%}")
            rep.emit(f"  доля |z_runs|>1.96:      {(sub['runs_z'].abs()>1.96).mean():.2%}")

    out = pd.DataFrame(rows)
    save_table("temporal", out)
    rep.emit(f"\n→ таблица temporal: {len(out)} строк")


# ============================================================
# 4. VOLATILITY ECONOMETRICS
# ============================================================
def block_volatility(stock_ret: pd.DataFrame, global_ret: pd.DataFrame, rep: Report):
    rep.emit(section("4. VOLATILITY ECONOMETRICS"))

    # ARCH-LM на всех, GARCH — только на топ-N по длине
    rows = []
    for label, df in [("STOCKS", stock_ret), ("GLOBAL", global_ret)]:
        rep.emit(f"\n[{label}]")
        # отбираем тикеры для GARCH: самые длинные
        sub_cols = [c for c in df.columns if df[c].dropna().shape[0] >= MIN_BARS]
        sub_cols_garch = sorted(sub_cols, key=lambda c: df[c].dropna().shape[0], reverse=True)[:MAX_STOCKS_FULL]

        for col in sub_cols:
            x = df[col].dropna()
            # ARCH-LM на 5 лагах
            try:
                arch_stat, arch_p, _, _ = het_arch(x.values, nlags=5)
            except Exception:
                arch_stat, arch_p = np.nan, np.nan

            # асимметрия волатильности: std(neg)/std(pos)
            pos = x[x > 0]
            neg = x[x < 0]
            vol_asym = neg.std() / pos.std() if pos.std() > 0 else np.nan

            # downside / upside vol
            down_vol = neg.std() if len(neg) > 1 else np.nan
            up_vol = pos.std() if len(pos) > 1 else np.nan

            # GARCH(1,1) — только для отобранных
            garch_alpha = garch_beta = garch_omega = np.nan
            garch_persist = np.nan
            if ARCH_OK and col in sub_cols_garch:
                try:
                    am = arch_model(x * 100, vol="Garch", p=1, q=1, dist="t")
                    res = am.fit(disp="off", show_warning=False)
                    garch_alpha = float(res.params.get("alpha[1]", np.nan))
                    garch_beta = float(res.params.get("beta[1]", np.nan))
                    garch_omega = float(res.params.get("omega", np.nan))
                    garch_persist = garch_alpha + garch_beta
                except Exception:
                    pass

            rows.append({
                "scope": label, "ticker": col,
                "n": len(x),
                "arch_lm_stat": arch_stat, "arch_lm_p": arch_p,
                "vol_asym": vol_asym,
                "down_vol": down_vol, "up_vol": up_vol,
                "garch_alpha": garch_alpha, "garch_beta": garch_beta,
                "garch_omega": garch_omega, "garch_persist": garch_persist,
            })
        sub = pd.DataFrame([r for r in rows if r["scope"] == label])
        if len(sub):
            rep.emit(f"  доля p(ARCH-LM)<0.05: {(sub['arch_lm_p']<0.05).mean():.2%}")
            rep.emit(f"  медиана vol_asym (neg/pos): {sub['vol_asym'].median():.3f}")
            garch_ok = sub["garch_persist"].dropna()
            if len(garch_ok):
                rep.emit(f"  GARCH(1,1) persistence: median={garch_ok.median():.4f}  "
                         f"(>0.95 → долгая память волатильности)")

    out = pd.DataFrame(rows)
    save_table("volatility", out)
    rep.emit(f"\n→ таблица volatility: {len(out)} строк")


# ============================================================
# 5. STATIONARITY
# ============================================================
def block_stationarity(stock_ret: pd.DataFrame, global_ret: pd.DataFrame, rep: Report):
    rep.emit(section("5. STATIONARITY (ADF / KPSS)"))

    rows = []
    for label, df in [("STOCKS", stock_ret), ("GLOBAL", global_ret)]:
        rep.emit(f"\n[{label}]")
        sub_cols = [c for c in df.columns if df[c].dropna().shape[0] >= MIN_BARS]
        sub_cols_pp = sub_cols[:MAX_STOCKS_FULL]  # PP тяжёлый

        for col in sub_cols:
            x = df[col].dropna()
            # ADF (H0: есть единичный корень → нестационарно)
            try:
                adf_stat, adf_p, _, _, _, _ = adfuller(x, autolag="AIC")
            except Exception:
                adf_stat, adf_p = np.nan, np.nan
            # KPSS (H0: стационарно)
            try:
                kpss_stat, kpss_p, _, _ = kpss(x, regression="c", nlags="auto")
            except Exception:
                kpss_stat, kpss_p = np.nan, np.nan

            # PP (Phillips-Perron) — если statsmodels поддерживает
            pp_stat = pp_p = np.nan
            if PP_OK and col in sub_cols_pp:
                try:
                    pp = PhillipsPerron(x, autolag="AIC")
                    pp_stat = float(pp.stat)
                    pp_p = float(pp.pvalue)
                except Exception:
                    pass

            rows.append({
                "scope": label, "ticker": col, "n": len(x),
                "adf_stat": adf_stat, "adf_p": adf_p,
                "kpss_stat": kpss_stat, "kpss_p": kpss_p,
                "pp_stat": pp_stat, "pp_p": pp_p,
                "adf_reject_5": int(adf_p < 0.05) if not np.isnan(adf_p) else np.nan,
                "kpss_reject_5": int(kpss_p < 0.05) if not np.isnan(kpss_p) else np.nan,
            })

        sub = pd.DataFrame([r for r in rows if r["scope"] == label])
        if len(sub):
            rep.emit(f"  доля ADF p<0.05 (стационарны):  "
                     f"{(sub['adf_p']<0.05).mean():.2%}")
            rep.emit(f"  доля KPSS p<0.05 (отвергли стационарность): "
                     f"{(sub['kpss_p']<0.05).mean():.2%}")
            rep.emit(f"  ADF и KPSS согласны (стационарно): "
                     f"{((sub['adf_p']<0.05) & (sub['kpss_p']>=0.05)).mean():.2%}")
            rep.emit(f"  ADF и KPSS согласны (нестационарно): "
                     f"{((sub['adf_p']>=0.05) & (sub['kpss_p']<0.05)).mean():.2%}")

    out = pd.DataFrame(rows)
    save_table("stationarity", out)
    rep.emit(f"\n→ таблица stationarity: {len(out)} строк")


# ============================================================
# 6. LONG MEMORY
# ============================================================
def block_long_memory(stock_ret: pd.DataFrame, global_ret: pd.DataFrame, rep: Report):
    rep.emit(section("6. LONG MEMORY (Hurst / DFA)"))

    def hurst_rs(x, max_lag=50):
        """Hurst через R/S analysis."""
        x = np.asarray(x)
        n = len(x)
        if n < max_lag * 2:
            return np.nan
        lags = np.arange(10, max_lag)
        rs = []
        for lag in lags:
            n_chunks = n // lag
            if n_chunks < 2:
                continue
            vals = []
            for i in range(n_chunks):
                chunk = x[i*lag:(i+1)*lag]
                mean = chunk.mean()
                z = chunk - mean
                cum = np.cumsum(z)
                R = cum.max() - cum.min()
                S = chunk.std(ddof=1)
                if S > 0:
                    vals.append(R / S)
            if vals:
                rs.append(np.mean(vals))
        if len(rs) < 3:
            return np.nan
        try:
            slope, _ = np.polyfit(np.log(lags[:len(rs)]), np.log(rs), 1)
            return slope
        except Exception:
            return np.nan

    def dfa(x, order=1, max_lag=50):
        """Detrended Fluctuation Analysis."""
        x = np.asarray(x)
        n = len(x)
        y = np.cumsum(x - x.mean())
        lags = np.unique(np.floor(np.logspace(np.log10(4), np.log10(min(max_lag, n//4)), 10)).astype(int))
        F = []
        for lag in lags:
            if lag < 2:
                continue
            n_chunks = n // lag
            if n_chunks < 2:
                continue
            rms = []
            for i in range(n_chunks):
                seg = y[i*lag:(i+1)*lag]
                t = np.arange(lag)
                coef = np.polyfit(t, seg, order)
                trend = np.polyval(coef, t)
                rms.append(np.sqrt(np.mean((seg - trend) ** 2)))
            if rms:
                F.append((lag, np.mean(rms)))
        if len(F) < 3:
            return np.nan
        lags_arr = np.array([f[0] for f in F])
        F_arr = np.array([f[1] for f in F])
        try:
            slope, _ = np.polyfit(np.log(lags_arr), np.log(F_arr), 1)
            return slope
        except Exception:
            return np.nan

    rows = []
    for label, df in [("STOCKS", stock_ret), ("GLOBAL", global_ret)]:
        rep.emit(f"\n[{label}]")
        sub_cols = [c for c in df.columns if df[c].dropna().shape[0] >= MIN_BARS]
        sub_cols = sub_cols[:MAX_STOCKS_FULL]  # тяжёлые вычисления

        for col in sub_cols:
            x = df[col].dropna().values
            h_rs = hurst_rs(x)
            h_dfa = dfa(x)
            # по абсолютным доходностям — долгая память в волатильности
            h_rs_abs = hurst_rs(np.abs(x))
            rows.append({
                "scope": label, "ticker": col, "n": len(x),
                "hurst_rs": h_rs, "hurst_dfa": h_dfa,
                "hurst_rs_abs": h_rs_abs,
                "interpretation": (
                    "mean-reverting" if h_rs and h_rs < 0.45 else
                    "random walk" if h_rs and h_rs < 0.55 else
                    "trending/persistent" if h_rs else "n/a"
                ),
            })
        sub = pd.DataFrame([r for r in rows if r["scope"] == label])
        if len(sub):
            rep.emit(f"  median Hurst (R/S):     {sub['hurst_rs'].median():.3f}")
            rep.emit(f"  median Hurst (DFA):     {sub['hurst_dfa'].median():.3f}")
            rep.emit(f"  median Hurst |ret| (R/S): {sub['hurst_rs_abs'].median():.3f}")
            rep.emit("  H≈0.5 — random walk; H<0.5 — антиперсистентность; H>0.5 — трендовость")

    out = pd.DataFrame(rows)
    save_table("long_memory", out)
    rep.emit(f"\n→ таблица long_memory: {len(out)} строк")


# ============================================================
# 7. NON-STATIONARITY OVER TIME
# ============================================================
def block_rolling(stock_ret: pd.DataFrame, global_ret: pd.DataFrame, rep: Report):
    rep.emit(section("7. NON-STATIONARITY OVER TIME (rolling / breaks)"))

    # rolling статистики для IMOEX и для одного «среднего» тикера
    focus = []
    if "IMOEX" in global_ret.columns:
        focus.append(("IMOEX", global_ret["IMOEX"]))
    # добавляем SBER / GAZP, если есть
    for t in ["SBER", "GAZP", "LKOH"]:
        if t in stock_ret.columns:
            focus.append((t, stock_ret[t]))
            break

    rows = []
    for name, x in focus:
        x = x.dropna()
        if len(x) < 500:
            continue
        w = 252  # окно 1 год
        roll_mean = x.rolling(w).mean()
        roll_std = x.rolling(w).std()
        roll_skew = x.rolling(w).skew()
        roll_kurt = x.rolling(w).kurt()

        # CUSUM structural break test
        cusum_p = np.nan
        if SM_OK:
            try:
                res = OLS(x.values, np.ones(len(x))).fit()
                cusum = breaks_cusumolsresid(res.resid)
                cusum_p = float(cusum[1])
            except Exception:
                pass

        rows.append({
            "ticker": name,
            "n": len(x),
            "roll_mean_min": roll_mean.min(),
            "roll_mean_max": roll_mean.max(),
            "roll_mean_std": roll_mean.std(),
            "roll_std_min": roll_std.min(),
            "roll_std_max": roll_std.max(),
            "roll_std_max_min_ratio": roll_std.max() / roll_std.min() if roll_std.min() > 0 else np.nan,
            "roll_skew_min": roll_skew.min(),
            "roll_skew_max": roll_skew.max(),
            "roll_kurt_max": roll_kurt.max(),
            "cusum_p": cusum_p,
        })
        rep.emit(f"\n[{name}]")
        rep.emit(f"  rolling mean (252d): [{roll_mean.min():.6f} .. {roll_mean.max():.6f}]")
        rep.emit(f"  rolling std  (252d): [{roll_std.min():.6f} .. {roll_std.max():.6f}]  "
                 f"ratio max/min = {rows[-1]['roll_std_max_min_ratio']:.2f}")
        rep.emit(f"  rolling skew (252d): [{roll_skew.min():.3f} .. {roll_skew.max():.3f}]")
        rep.emit(f"  rolling kurt (252d): max = {roll_kurt.max():.3f}")
        if not np.isnan(cusum_p):
            rep.emit(f"  CUSUM p-value: {cusum_p:.4f}  "
                     f"({'есть структурный сдвиг' if cusum_p<0.05 else 'нет сдвига'})")

    out = pd.DataFrame(rows)
    save_table("rolling", out)
    rep.emit(f"\n→ таблица rolling: {len(out)} строк")


# ============================================================
# 8. GLOBAL FACTOR ECONOMETRICS
# ============================================================
def block_factors(stock_ret: pd.DataFrame, global_ret: pd.DataFrame, rep: Report):
    rep.emit(section("8. GLOBAL FACTOR ECONOMETRICS"))

    # выравниваем по датам
    common_idx = stock_ret.index.intersection(global_ret.index)
    if len(common_idx) < MIN_BARS:
        rep.emit("  SKIP: мало общих дат между stocks и global")
        return

    Y = stock_ret.loc[common_idx]
    X = global_ret.loc[common_idx]

    # оставляем тикеры с достаточной историей
    valid = [c for c in Y.columns if Y[c].dropna().shape[0] >= MIN_BARS][:MAX_STOCKS_FULL]
    Y = Y[valid]

    # 1) корреляции
    rep.emit("\n[Корреляции stocks vs global]")
    corr_rows = []
    for t in valid:
        for g in X.columns:
            sub = pd.concat([Y[t], X[g]], axis=1).dropna()
            if len(sub) < MIN_BARS:
                continue
            c = sub.iloc[:, 0].corr(sub.iloc[:, 1])
            corr_rows.append({"ticker": t, "factor": g, "corr": c, "n": len(sub)})
    corr_df = pd.DataFrame(corr_rows)

    # 2) регрессии с HAC (Newey-West)
    rep.emit("\n[Регрессии ret ~ factors (HAC)]")
    reg_rows = []
    for t in valid:
        sub = pd.concat([Y[t], X], axis=1).dropna()
        if len(sub) < MIN_BARS:
            continue
        y = sub.iloc[:, 0].values
        Xm = sub.iloc[:, 1:].values
        # добавляем константу
        Xm_c = np.column_stack([np.ones(len(Xm)), Xm])
        try:
            res = OLS(y, Xm_c).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            # beta для каждого фактора
            names = ["const"] + list(X.columns)
            betas = dict(zip(names, res.params))
            tstats = dict(zip(names, res.tvalues))
            row = {"ticker": t, "n": len(sub), "r2": res.rsquared}
            for g in X.columns:
                row[f"beta_{g}"] = betas.get(g, np.nan)
                row[f"tstat_{g}"] = tstats.get(g, np.nan)
            reg_rows.append(row)
        except Exception:
            pass

    reg_df = pd.DataFrame(reg_rows)

    # сводка
    if len(corr_df):
        for g in X.columns:
            sub = corr_df[corr_df["factor"] == g]
            if len(sub):
                rep.emit(f"  corr vs {g:<10}: median={sub['corr'].median():+.3f}  "
                         f"std={sub['corr'].std():.3f}  "
                         f"доля |corr|>0.3: {(sub['corr'].abs()>0.3).mean():.1%}")

    if len(reg_df):
        rep.emit(f"\n  медиана R²: {reg_df['r2'].median():.3f}")
        for g in X.columns:
            col = f"beta_{g}"
            if col in reg_df:
                sub = reg_df[col].dropna()
                tcol = f"tstat_{g}"
                tsub = reg_df[tcol].dropna()
                if len(sub):
                    rep.emit(f"  beta_{g:<10}: median={sub.median():+.4f}  "
                             f"доля |t|>1.96: {(tsub.abs()>1.96).mean():.1%}")

    save_table("factors_corr", corr_df)
    save_table("factors_reg", reg_df)
    rep.emit(f"\n→ таблицы factors_corr ({len(corr_df)}), factors_reg ({len(reg_df)})")

    # 3) корреляции между самими глобальными факторами
    rep.emit("\n[Корреляции между глобальными факторами]")
    gcorr = X.corr()
    rep.emit(gcorr.round(3).to_string())


# ============================================================
# MAIN
# ============================================================
def main():
    rep = Report()
    rep.emit("# EDA: статистический портрет рынка (IS)\n")
    rep.emit(f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n")

    if not STOCKS_DB.exists() or not INDEXES_DB.exists():
        rep.emit("FATAL: нет одной из баз (stocks.sqlite / indexes.sqlite)")
        rep.save(OUT_MD)
        return

    split_date = load_split_date()
    rep.emit(f"split_date = {split_date} (VAL не трогаем)\n")

    rep.emit(section("ЗАГРУЗКА"))
    stock_ret = load_stock_returns(split_date)
    global_ret = load_global_returns(split_date)
    rep.emit(f"stocks:  {stock_ret.shape[0]} дат × {stock_ret.shape[1]} тикеров (IS)")
    rep.emit(f"global:  {global_ret.shape[0]} дат × {global_ret.shape[1]} инструментов (IS)")
    if stock_ret.shape[0] > 0:
        rep.emit(f"  IS диапазон: {stock_ret.index.min().date()} → {stock_ret.index.max().date()}")

    # --- 8 блоков ---
    block_distributions(stock_ret, global_ret, rep)
    block_evt(stock_ret, global_ret, rep)
    block_temporal(stock_ret, global_ret, rep)
    block_volatility(stock_ret, global_ret, rep)
    block_stationarity(stock_ret, global_ret, rep)
    block_long_memory(stock_ret, global_ret, rep)
    block_rolling(stock_ret, global_ret, rep)
    block_factors(stock_ret, global_ret, rep)

    rep.emit(section("ИТОГ"))
    rep.emit(f"SQL:     {OUT_SQL}")
    rep.emit(f"Report:  {OUT_MD}")
    rep.emit("\nТаблицы в eda.sqlite:")
    with sqlite3.connect(OUT_SQL) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for (t,) in tables:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            rep.emit(f"  {t:<20} {n:>5} строк")

    rep.save(OUT_MD)


if __name__ == "__main__":
    main()